"""Indexador batch de chunks semanticos no Qdrant.

Escaneia output/ em busca de *_content_list_v2.json, gera chunks com MinerUChunker,
calcula embeddings densos + esparsos (lexical_weights) via BAAI/bge-m3 (FlagEmbedding) e faz upsert na collection.

Uso:
    python -m backend.indexing.index_chunks
    python -m backend.indexing.index_chunks --reset
    python -m backend.indexing.index_chunks --url http://localhost:6333
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import psutil
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, \
    SparseIndexParams, SparseVectorParams, VectorParams


from backend.indexing.chunks import MinerUChunker
from backend.services.embedder import BgeM3EmbedderService
from backend.core.config import QDRANT_COLLECTION, QDRANT_TIMEOUT_SECONDS, QDRANT_URL, OUTPUT_DIR
from backend.core.schemas import DocumentMetadata

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

COLLECTION_NAME = QDRANT_COLLECTION  # alinhado com a busca (backend.core.config)
# QDRANT_URL vem de backend.core.config (override por env). Antes era hardcoded aqui,
# o que fazia o worker de GPU ignorar o env e sempre usar o IP fixo da rede.
CONTENT_LIST_SUFFIX = "_content_list_v2.json"
EMBEDDING_REPORT_PATH = Path("relatorio_embeddings.csv")

# bge-m3: dense (1024d) + sparse (lexical_weights) — tudo via FlagEmbedding
DENSE_MODEL = "BAAI/bge-m3"
BATCH_SIZE = 32

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("mineru.indexer")


# ---------------------------------------------------------------------------
# Model availability check
# ---------------------------------------------------------------------------

def check_models_available(embedder: BgeM3EmbedderService) -> None:
    """Informa sobre o status local do modelo."""
    print("\nVerificando disponibilidade dos modelos...")
    if embedder.is_cached_locally():
        print(f"  ✅   [Dense + Sparse] {DENSE_MODEL} já está no cache local.")
    else:
        print(f"  ℹ️   [Dense + Sparse] {DENSE_MODEL}  —  Download automático ativado via HuggingFace Hub.")
    print("  Modelos prontos para uso.\n")


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------

def _ram_mb() -> float:
    """Retorna uso atual de RAM do processo em MB."""
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


def _vram_mb() -> float:
    """Retorna VRAM alocada em MB (GPU 0). Retorna 0.0 se CUDA indisponivel."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated(0) / 1024 / 1024
    except Exception:
        pass
    return 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_json_files(output_dir: Path) -> list[Path]:
    """Retorna todos os content_list_v2.json em output/."""
    log.info("Iniciando varredura de JSONs em '%s'...", output_dir)
    files = [
        path for path in output_dir.rglob(f"*{CONTENT_LIST_SUFFIX}")
        if path.is_file() and path.name.endswith(CONTENT_LIST_SUFFIX)
    ]
    log.info("Varredura concluida: %d arquivo(s) encontrado(s).", len(files))
    return sorted(files)


def progress_bar(current: int, total: int, width: int = 28) -> str:
    if total <= 0:
        return "[" + ("-" * width) + "]"
    filled = int((current / total) * width)
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def resolve_md_path(json_path: Path) -> str:
    """Deriva o path web do .md correspondente ao JSON."""
    # Estrutura comum: output/<doc>/<method>/<doc>_content_list_v2.json
    # O markdown pode ficar no diretório do método ou no diretório do documento.
    method_dir = json_path.parent
    doc_dir = json_path.parent.parent
    doc_name = doc_dir.name

    candidates = [
        method_dir / f"{doc_name}.md",
        doc_dir / f"{doc_name}.md",
    ]
    for md_candidate in candidates:
        if md_candidate.exists():
            rel = md_candidate.relative_to(OUTPUT_DIR).as_posix()
            return f"/output/{rel}"

    for search_dir in (method_dir, doc_dir):
        mds = sorted(search_dir.glob("*.md"))
        if mds:
            rel = mds[0].relative_to(OUTPUT_DIR).as_posix()
            log.debug("Markdown fallback encontrado para '%s': %s", json_path.name, rel)
            return f"/output/{rel}"

    mds = sorted(doc_dir.rglob("*.md"))
    if mds:
        rel = mds[0].relative_to(OUTPUT_DIR).as_posix()
        log.debug("Markdown recursivo encontrado para '%s': %s", json_path.name, rel)
        return f"/output/{rel}"

    log.warning("Nenhum markdown encontrado para '%s'. Usando path de diretorio.", json_path.name)
    return f"/output/{doc_name}/"


def resolve_llm_metadata_path(json_path: Path) -> Path:
    """Deriva o path do JSON de metadados LLM ao lado do markdown do documento."""
    doc_dir = json_path.parent.parent
    doc_name = doc_dir.name
    return json_path.parent / f"{doc_name}_metadata_llm.json"


def load_llm_metadata_payload(json_path: Path) -> dict:
    """Carrega metadados gerados pela LLM, quando existirem, para anexar ao payload."""
    meta_path = resolve_llm_metadata_path(json_path)
    if not meta_path.exists():
        return {}

    try:
        raw_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        metadata = DocumentMetadata.model_validate(raw_meta)
    except Exception as exc:
        log.warning("Falha ao carregar metadados LLM de '%s': %s", meta_path, exc)
        return {}

    payload = metadata.model_dump(exclude_none=True)
    payload["metadata_source"] = "llm"
    return payload


def setup_collection(client: QdrantClient, reset: bool, dense_dim: int) -> None:
    """Cria (ou recria) a collection com vetores densos e esparsos."""
    log.info("Verificando collection '%s'...", COLLECTION_NAME)
    exists = client.collection_exists(COLLECTION_NAME)

    if exists and reset:
        print(f"  ♻️  Recriando collection '{COLLECTION_NAME}'...")
        log.info("Collection existe e --reset foi informado. Recriando collection.")
        client.delete_collection(COLLECTION_NAME)
        exists = False

    if not exists:
        print(f"  ✨ Criando collection '{COLLECTION_NAME}'...")
        log.info("Criando collection com dense=%d e sparse habilitado.", dense_dim)
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "dense": VectorParams(size=dense_dim, distance=Distance.COSINE),
             
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    index=SparseIndexParams(on_disk=False)
                )
            },
        )
        print(f"  Collection criada (dense: {dense_dim}d, sparse habilitado).")
        log.info("Collection criada com sucesso.")
    else:
        print(f"  ℹ️  Collection '{COLLECTION_NAME}' já existe — adicionando pontos.")
        log.info("Collection existente sera reutilizada.")


# ---------------------------------------------------------------------------
# Indexador
# ---------------------------------------------------------------------------

def _lexical_to_sparse(lexical_weights: dict) -> tuple[list[int], list[float]]:
    """Converte lexical_weights do bge-m3 para (indices, values) do Qdrant.

    lexical_weights: {token_id_str: weight_float}
    """
    indices = [int(k) for k in lexical_weights.keys()]
    values  = [float(v) for v in lexical_weights.values()]
    return indices, values


def index_document(
    client: QdrantClient,
    json_path: Path,
    chunker: MinerUChunker,
    embedder: BgeM3EmbedderService,
    item_uuid: str = "",
    item_handle: str = "",
    task_id: str | None = None,
    *,
    doc_id_override: str | None = None,
    doc_path_override: str | None = None,
    llm_payload_override: dict | None = None,
    chunks_jsonl_path: Path | None = None,
    upsert_batch_size: int | None = None,
) -> dict:
    """Processa um JSON, gera embeddings e upsert no Qdrant.

    Retorna dict com metricas do documento (n_chunks, tempos, memoria, etc.).

    Os `*_override` permitem que o pipeline v2 (artefatos no MinIO) forneça doc_id,
    doc_path e payload da LLM explicitamente — sem depender do layout local
    output/<doc>/<metodo>/ (resolve_md_path/load_llm_metadata_payload). Quando
    `chunks_jsonl_path` é informado, os chunks são persistidos em JSONL (uma linha
    por chunk) ANTES do embedding — para posterior upload ao MinIO/auditoria. O
    upsert no Qdrant é feito em batches de `upsert_batch_size` (config.QDRANT_UPSERT_BATCH_SIZE).
    """
    if not json_path.name.endswith(CONTENT_LIST_SUFFIX):
        raise ValueError(
            f"Arquivo invalido para chunking: '{json_path.name}'. "
            f"Apenas arquivos com sufixo '{CONTENT_LIST_SUFFIX}' sao aceitos."
        )

    doc_id   = doc_id_override or json_path.parent.parent.name
    doc_path = doc_path_override or resolve_md_path(json_path)
    log.info("Processando documento '%s' (%s)", doc_id, json_path)

    ts_inicio = datetime.now(timezone.utc).isoformat(timespec="seconds")
    ram_inicio = _ram_mb()
    vram_inicio = _vram_mb()
    t_total = time.perf_counter()

    # --- Chunking ---
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    llm_metadata_payload = (
        llm_payload_override if llm_payload_override is not None
        else load_llm_metadata_payload(json_path)
    )
    log.debug("JSON carregado para '%s'. Blocos: %d", doc_id, len(data) if isinstance(data, list) else -1)

    t_chunk = time.perf_counter()
    chunks = chunker.process(data, doc_id=doc_id)
    chunk_time_s = time.perf_counter() - t_chunk
    log.info("Chunking concluido para '%s': %d chunk(s) em %.3fs.", doc_id, len(chunks), chunk_time_s)

    # Persistencia dos chunks em JSONL (uma linha por chunk) — leitura incremental,
    # reindexacao e auditoria. Feito ANTES do embedding para nao segurar memoria.
    if chunks_jsonl_path is not None:
        chunks_jsonl_path = Path(chunks_jsonl_path)
        chunks_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with open(chunks_jsonl_path, "w", encoding="utf-8") as jf:
            for c in chunks:
                jf.write(json.dumps(c.model_dump(), ensure_ascii=False) + "\n")
        log.info("Chunks de '%s' salvos em JSONL: %s (%d linha(s)).", doc_id, chunks_jsonl_path, len(chunks))

    # Resumo de filtros de qualidade
    rejection_info = chunker.rejection_summary
    if rejection_info["total_rejected"] > 0:
        log.info(
            "Filtros de qualidade '%s': %d rejeitado(s) — %s",
            doc_id, rejection_info["total_rejected"], rejection_info["by_reason"],
        )

    if not chunks:
        print(f"    ⚠️  Nenhum chunk gerado para '{doc_id}'")
        log.warning("Nenhum chunk gerado para '%s'.", doc_id)
        return {
            "doc_id": doc_id,
            "timestamp": ts_inicio,
            "n_chunks": 0,
            "total_chars": 0,
            "chunk_time_s": round(chunk_time_s, 3),
            "embed_time_s": 0.0,
            "upsert_time_s": 0.0,
            "total_time_s": 0.0,
            "chars_per_s": 0.0,
            "ram_delta_mb": 0.0,
            "ram_peak_mb": round(ram_inicio, 1),
            "vram_peak_mb": round(vram_inicio, 1),
            "avg_sparse_tokens": 0.0,
            "status": "vazio",
        }

    texts      = [c.content for c in chunks]
    total_chars = sum(len(t) for t in texts)

    # --- Embedding ---
    # Só ESTE trecho toca a GPU: o embedder adquire o lock internamente (dense +
    # sparse + transferência p/ CPU). Chunking (acima) e build de pontos + upsert
    # no Qdrant (abaixo) ficam FORA do lock. document_id vai ao owner (diagnóstico).
    log.info("Gerando embeddings dense + sparse (bge-m3) para '%s' (%d textos, batch=%d)...", doc_id, len(texts), BATCH_SIZE)
    t_embed = time.perf_counter()
    dense_vecs, lexical_list = embedder.embed_documents(texts, batch_size=BATCH_SIZE, document_id=doc_id, task_id=task_id)
    embed_time_s = time.perf_counter() - t_embed
    if len(dense_vecs) > 0:
        log.debug("Dimensao dense para '%s': %d", doc_id, len(dense_vecs[0]))
    log.info("Embeddings concluidos para '%s' em %.3fs.", doc_id, embed_time_s)

    # Densidade media do sparse vector (numero de tokens com peso > 0)
    avg_sparse_tokens = sum(len(lw) for lw in lexical_list) / len(lexical_list) if lexical_list else 0.0

    ram_pos  = _ram_mb()
    vram_pos = _vram_mb()

    # --- Build points ---
    points: list[PointStruct] = []
    for chunk, dense, lw in zip(chunks, dense_vecs, lexical_list):
        sparse_indices, sparse_values = _lexical_to_sparse(lw)
        
        payload = chunk.metadata.model_dump(exclude_none=True)
        payload.update(llm_metadata_payload)
        payload["content"] = chunk.content
        payload["doc_name"] = Path(doc_path).name if doc_path.endswith(".md") else doc_id
        payload["doc_path"] = doc_path
        # Vínculo com o item DSpace (quando a ingestão vem de um item-UUID).
        # Mesmos campos usados pelo filtro ?uuid= da busca e pelo backfill add_item_uuid.py.
        if item_uuid:
            payload["item_uuid"] = item_uuid
        if item_handle:
            payload["item_handle"] = item_handle
        
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "dense": dense,
                    "sparse": {
                        "indices": sparse_indices,
                        "values":  sparse_values,
                    },
                },
                payload=payload,
            )
        )
    log.info("Pontos preparados para '%s': %d", doc_id, len(points))

    # --- Upsert (em batches, para nao mandar um payload gigante de uma vez) ---
    t_upsert = time.perf_counter()
    batch = upsert_batch_size or 100
    for start in range(0, len(points), batch):
        client.upsert(collection_name=COLLECTION_NAME, points=points[start:start + batch])
    upsert_time_s = time.perf_counter() - t_upsert
    total_time_s  = time.perf_counter() - t_total
    log.info("Upsert concluido para '%s' em %.3fs (%d ponto(s), batch=%d).", doc_id, upsert_time_s, len(points), batch)

    return {
        "doc_id":            doc_id,
        "timestamp":         ts_inicio,
        "n_chunks":          len(points),
        "total_chars":       total_chars,
        "chunk_time_s":      round(chunk_time_s, 3),
        "embed_time_s":      round(embed_time_s, 3),
        "upsert_time_s":     round(upsert_time_s, 3),
        "total_time_s":      round(total_time_s, 3),
        "chars_per_s":       round(total_chars / embed_time_s, 1) if embed_time_s > 0 else 0.0,
        "ram_delta_mb":      round(ram_pos - ram_inicio, 1),
        "ram_peak_mb":       round(max(ram_inicio, ram_pos), 1),
        "vram_peak_mb":      round(max(vram_inicio, vram_pos), 1),
        "avg_sparse_tokens": round(avg_sparse_tokens, 1),
        "status":            "ok",
    }


def is_document_indexed(client: QdrantClient, doc_id: str) -> bool:
    """Retorna True se o Qdrant já contém pontos com esse doc_id."""
    try:
        result = client.count(
            collection_name=COLLECTION_NAME,
            count_filter=Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
            ),
            exact=False,
        )
        return result.count > 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Indexação de documento único (usada pelo pipeline da API / worker)
# ---------------------------------------------------------------------------

_api_client: QdrantClient | None = None
_api_chunker: MinerUChunker | None = None
_api_embedder: BgeM3EmbedderService | None = None
_api_ready = False


def _get_indexer() -> tuple[QdrantClient, MinerUChunker, BgeM3EmbedderService]:
    """Constrói (uma vez) os singletons de indexação. O embedder reusa o _SHARED_MODEL global."""
    global _api_client, _api_chunker, _api_embedder, _api_ready
    if not _api_ready:
        _api_client = QdrantClient(url=QDRANT_URL, timeout=QDRANT_TIMEOUT_SECONDS or None)
        _api_embedder = BgeM3EmbedderService()
        _api_embedder.load_model(require_cache=False)  # reusa o modelo já carregado, se houver
        _api_chunker = MinerUChunker()
        setup_collection(_api_client, reset=False, dense_dim=_api_embedder.get_dense_dimension())
        _api_ready = True
    return _api_client, _api_chunker, _api_embedder


def index_single_document(json_path: Path, item_uuid: str = "", item_handle: str = "", task_id: str | None = None) -> dict:
    """Indexa um único documento no Qdrant (caminho da API). Reusa singletons e
    remove pontos anteriores do mesmo doc_id (evita duplicação em re-upload).

    item_uuid/item_handle, se informados, são gravados no payload dos chunks
    (vínculo com o item DSpace na ingestão a partir de um item-UUID)."""
    client, chunker, embedder = _get_indexer()
    json_path = Path(json_path).resolve()  # caminhos do output são absolutos (config.OUTPUT_DIR)
    doc_id = json_path.parent.parent.name
    # Dedup por doc_name (o chunker grava doc_id com sufixo .pdf; doc_name = "<stem>.md"
    # é determinístico e bate com o payload). Remove pontos antigos antes de re-indexar.
    doc_name = f"{doc_id}.md"
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=Filter(
            must=[FieldCondition(key="doc_name", match=MatchValue(value=doc_name))]
        ),
    )
    log.info("Pontos anteriores de '%s' removidos antes da re-indexação (se houver).", doc_name)
    return index_document(client, json_path, chunker, embedder, item_uuid=item_uuid, item_handle=item_handle, task_id=task_id)


def index_materialized_document(
    json_path: Path,
    *,
    doc_id: str,
    doc_path: str = "",
    llm_metadata_payload: dict | None = None,
    item_uuid: str = "",
    item_handle: str = "",
    bitstream_uuid: str | None = None,
    document_checksum: str = "",
    source_artifact_uri: str = "",
    document_title: str | None = None,
    markdown_path: Path | None = None,
    task_id: str | None = None,
    chunks_jsonl_path: Path | None = None,
    chunking_report_path: Path | None = None,
    upsert_batch_size: int | None = None,
    strategy: str | None = None,
) -> dict:
    """Indexa um content_list_v2.json JÁ MATERIALIZADO localmente a partir do MinIO
    (pipeline v2), com doc_id/doc_path/payload LLM explícitos — sem depender do
    layout output/<doc>/<metodo>/.

    Se `CHUNKING_STRATEGY` (ou `strategy`) for `structural_tokens`/`legacy_chars`, usa
    o novo caminho estrutural (StructuralTokenChunker/LegacyCharacterChunker via
    interface comum); caso contrário, cai no caminho legado direto (MinerUChunker).
    """
    from backend.core import config as settings

    strategy = (strategy or settings.CHUNKING_STRATEGY).strip().lower()
    return _index_structural_document(
        json_path, doc_id=doc_id, doc_path=doc_path,
        llm_metadata_payload=llm_metadata_payload or {},
        item_uuid=item_uuid, item_handle=item_handle, bitstream_uuid=bitstream_uuid,
        document_checksum=document_checksum, source_artifact_uri=source_artifact_uri,
        document_title=document_title, markdown_path=markdown_path, task_id=task_id,
        chunks_jsonl_path=chunks_jsonl_path, chunking_report_path=chunking_report_path,
        upsert_batch_size=upsert_batch_size or settings.QDRANT_UPSERT_BATCH_SIZE,
        strategy=strategy,
    )


def _index_structural_document(
    json_path: Path,
    *,
    doc_id: str,
    doc_path: str,
    llm_metadata_payload: dict,
    item_uuid: str,
    item_handle: str,
    bitstream_uuid: str | None,
    document_checksum: str,
    source_artifact_uri: str,
    document_title: str | None,
    markdown_path: Path | None,
    task_id: str | None,
    chunks_jsonl_path: Path | None,
    chunking_report_path: Path | None,
    upsert_batch_size: int,
    strategy: str,
) -> dict:
    """Caminho estrutural por tokens (§24).

    Ordem: parse → chunk → persistência JSONL → embeddings (ÚNICO trecho sob o lock
    da GPU) → build de pontos → upsert dos NOVOS → só então remove versões antigas
    (§27: uma falha antes do upsert não apaga os chunks ativos anteriores).
    """
    import uuid as _uuid
    from backend.core import config as settings
    from backend.indexing.chunk_models import ChunkingError
    from backend.indexing.chunks import get_chunker
    from backend.indexing.document_blocks import MinerUDocumentParser

    client, _, embedder = _get_indexer()
    json_path = Path(json_path).resolve()
    doc_path = doc_path or f"{doc_id}.md"
    doc_name = Path(doc_path).name if doc_path.endswith(".md") else f"{doc_id}.md"

    ts_inicio = datetime.now(timezone.utc).isoformat(timespec="seconds")
    ram_inicio = _ram_mb()
    t_total = time.perf_counter()

    # --- Parse estrutural (CPU, FORA do lock da GPU) ---
    t_parse = time.perf_counter()
    parser = MinerUDocumentParser(
        remove_repeated_headers=settings.CHUNK_REMOVE_REPEATED_HEADERS,
        remove_repeated_footers=settings.CHUNK_REMOVE_REPEATED_FOOTERS,
        keep_footnotes=settings.CHUNK_KEEP_FOOTNOTES,
        normalize_text=settings.CHUNK_NORMALIZE_TEXT,
        reconstruct_cross_page=settings.MINERU_RECONSTRUCT_CROSS_PAGE_PARAGRAPHS,
        chart_mode=settings.CHUNK_CHART_MODE,
        image_mode=settings.CHUNK_IMAGE_MODE,
        chart_min_confidence=settings.CHUNK_CHART_MIN_CONFIDENCE,
        visual_llm=settings.CHUNK_VISUAL_LLM,
    )
    try:
        blocks, structure_source = parser.parse_json_file(json_path)
        if not blocks and markdown_path and Path(markdown_path).exists():
            blocks, structure_source = parser.parse_markdown(Path(markdown_path).read_text(encoding="utf-8"))
    except ChunkingError as exc:
        if markdown_path and Path(markdown_path).exists():
            log.warning("[index] parse JSON falhou (%s) — fallback markdown para '%s'.", exc, doc_id)
            blocks, structure_source = parser.parse_markdown(Path(markdown_path).read_text(encoding="utf-8"))
        else:
            raise
    parse_time_s = time.perf_counter() - t_parse
    title = document_title or parser.document_title

    # --- Chunking (CPU, FORA do lock da GPU) ---
    t_chunk = time.perf_counter()
    chunker = get_chunker(strategy)
    result = chunker.chunk(
        blocks, document_id=doc_id, document_title=title,
        document_checksum=document_checksum, item_uuid=item_uuid,
        bitstream_uuid=bitstream_uuid, source_artifact_uri=source_artifact_uri,
        structure_source=structure_source,
    )
    result.metrics.parse_time_s = parse_time_s
    result.metrics.cross_page_merges = getattr(parser, "cross_page_merges", 0)
    chunk_time_s = time.perf_counter() - t_chunk
    chunks = result.chunks
    log.info("[index] '%s': %d chunk(s) (%s) em %.3fs.", doc_id, len(chunks), strategy, chunk_time_s)

    # --- Persistência JSONL + relatório ANTES do embedding (§23) ---
    if chunks_jsonl_path is not None:
        chunks_jsonl_path = Path(chunks_jsonl_path)
        chunks_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with open(chunks_jsonl_path, "w", encoding="utf-8") as jf:
            for c in chunks:
                jf.write(json.dumps(c.model_dump(), ensure_ascii=False) + "\n")
        log.info("[index] chunks de '%s' salvos em JSONL: %s (%d linha(s)).", doc_id, chunks_jsonl_path, len(chunks))
    report = result.report()
    if chunking_report_path is not None:
        chunking_report_path = Path(chunking_report_path)
        chunking_report_path.parent.mkdir(parents=True, exist_ok=True)
        chunking_report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if not chunks:
        log.warning("[index] nenhum chunk gerado para '%s'.", doc_id)
        return {
            "doc_id": doc_id, "timestamp": ts_inicio, "n_chunks": 0, "total_chars": 0,
            "chunk_time_s": round(chunk_time_s, 3), "embed_time_s": 0.0, "upsert_time_s": 0.0,
            "total_time_s": round(time.perf_counter() - t_total, 3), "chars_per_s": 0.0,
            "ram_delta_mb": 0.0, "ram_peak_mb": round(ram_inicio, 1), "vram_peak_mb": 0.0,
            "avg_sparse_tokens": 0.0, "status": "vazio", "chunking_strategy": strategy,
            "chunking_report": report,
        }

    # --- Texto a embeddar (contextualized_text | text) ---
    field = settings.EMBEDDING_TEXT_FIELD
    texts = [c.embedding_input(field) for c in chunks]
    total_chars = sum(len(t) for t in texts)

    # --- Embedding (ÚNICO trecho sob o lock da GPU) ---
    t_embed = time.perf_counter()
    dense_vecs, lexical_list = embedder.embed_documents(
        texts, batch_size=settings.EMBEDDING_BATCH_SIZE, document_id=doc_id, task_id=task_id
    )
    embed_time_s = time.perf_counter() - t_embed
    avg_sparse_tokens = sum(len(lw) for lw in lexical_list) / len(lexical_list) if lexical_list else 0.0
    ram_pos = _ram_mb()

    # --- Build de pontos (FORA do lock) ---
    document_version = f"{document_checksum}:{result.chunking_config_hash}"
    point_ns = _uuid.uuid5(_uuid.NAMESPACE_URL, "evidencia_pipe/chunk")
    points: list[PointStruct] = []
    for chunk, dense, lw in zip(chunks, dense_vecs, lexical_list):
        sparse_indices, sparse_values = _lexical_to_sparse(lw)
        payload = _structural_payload(
            chunk, doc_id=doc_id, doc_name=doc_name, doc_path=doc_path,
            item_uuid=item_uuid, item_handle=item_handle, document_version=document_version,
            llm_metadata_payload=llm_metadata_payload,
        )
        points.append(PointStruct(
            id=str(_uuid.uuid5(point_ns, chunk.chunk_id)),
            vector={"dense": dense, "sparse": {"indices": sparse_indices, "values": sparse_values}},
            payload=payload,
        ))

    # --- Upsert dos NOVOS pontos (§27: só depois de gerar/persistir/embeddar) ---
    t_upsert = time.perf_counter()
    for start in range(0, len(points), upsert_batch_size):
        client.upsert(collection_name=COLLECTION_NAME, points=points[start:start + upsert_batch_size])
    upsert_time_s = time.perf_counter() - t_upsert

    # --- Remove versões ANTIGAS do documento (mesmo doc_name, outra document_version).
    # Feito APÓS o upsert confirmado dos novos — nunca antes (§27). Falha aqui é
    # best-effort (não invalida a indexação nova).
    try:
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=Filter(
                must=[FieldCondition(key="doc_name", match=MatchValue(value=doc_name))],
                must_not=[FieldCondition(key="document_version", match=MatchValue(value=document_version))],
            ),
        )
    except Exception as exc:
        log.warning("[index] limpeza de versões antigas de '%s' falhou (best-effort): %s", doc_name, exc)

    total_time_s = time.perf_counter() - t_total
    log.info("[index] '%s' indexado: %d ponto(s), embed=%.2fs upsert=%.2fs.", doc_id, len(points), embed_time_s, upsert_time_s)
    return {
        "doc_id": doc_id, "timestamp": ts_inicio, "n_chunks": len(points), "total_chars": total_chars,
        "chunk_time_s": round(chunk_time_s, 3), "embed_time_s": round(embed_time_s, 3),
        "upsert_time_s": round(upsert_time_s, 3), "total_time_s": round(total_time_s, 3),
        "chars_per_s": round(total_chars / embed_time_s, 1) if embed_time_s > 0 else 0.0,
        "ram_delta_mb": round(ram_pos - ram_inicio, 1), "ram_peak_mb": round(max(ram_inicio, ram_pos), 1),
        "vram_peak_mb": round(_vram_mb(), 1), "avg_sparse_tokens": round(avg_sparse_tokens, 1),
        "status": "ok", "chunking_strategy": strategy, "chunking_version": result.chunking_version,
        "chunking_config_hash": result.chunking_config_hash, "tokenizer_name": result.tokenizer_name,
        "structure_source": structure_source, "chunking_report": report,
    }


def _structural_payload(
    chunk, *, doc_id: str, doc_name: str, doc_path: str, item_uuid: str,
    item_handle: str, document_version: str, llm_metadata_payload: dict,
) -> dict:
    """Payload do Qdrant para um StructuralChunk (§26). Sem embeddings/imagens/JSON
    MinerU completo/objetos não serializáveis."""
    from backend.core import config as settings

    payload = {
        "chunk_id": chunk.chunk_id,
        "chunk_index": chunk.chunk_index,
        "document_id": chunk.document_id,
        "item_uuid": item_uuid or chunk.item_uuid,
        "bitstream_uuid": chunk.bitstream_uuid,
        "text": chunk.text,
        "document_title": chunk.document_title,
        "section_title": chunk.section_title,
        "section_path": chunk.section_path,
        "content_type": chunk.content_type,
        "token_count": chunk.token_count,
        "embedding_token_count": chunk.embedding_token_count,
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "page_numbers": chunk.page_numbers,
        # --- Política v2 (§22): classificação/recuperação + página impressa ---
        "normalized_content_type": chunk.normalized_content_type,
        "section_kind": chunk.section_kind,
        "retrieval_profile": chunk.retrieval_profile or None,
        "searchable_by_default": chunk.searchable_by_default,
        "ranking_weight": chunk.ranking_weight,
        "is_table": chunk.is_table,
        "is_chart": chunk.is_chart,
        "is_reference": chunk.is_reference,
        "is_appendix": chunk.is_appendix,
        # §10/§11/§12: sinais de validação visual (observabilidade/filtro).
        "table_quality_score": (chunk.metadata or {}).get("table_quality_score"),
        "chart_data_confidence": (chunk.metadata or {}).get("chart_data_confidence"),
        "image_kind": (chunk.metadata or {}).get("image_kind"),
        "printed_page_start": chunk.printed_page_start,
        "printed_page_end": chunk.printed_page_end,
        "printed_page_numbers": chunk.printed_page_numbers or None,
        "semantic_completeness": chunk.semantic_completeness,
        "cross_page_merged": chunk.cross_page_merged,
        "quality_score": chunk.quality_score,
        "source_block_ids": chunk.block_ids or None,
        "chunking_strategy": chunk.chunking_strategy,
        "chunking_version": chunk.chunking_version,
        "chunking_config_hash": chunk.chunking_config_hash,
        "tokenizer_name": chunk.tokenizer_name,
        "source_artifact_uri": chunk.source_artifact_uri,
        "document_checksum": chunk.document_checksum,
        "document_version": document_version,
        "active": True,
        "pipeline_version": settings.PIPELINE_VERSION,
        # Compatibilidade com a busca existente (filtros por doc_name/doc_id/page/section).
        "content": chunk.text,
        "doc_id": f"{doc_id}.pdf",
        "doc_name": doc_name,
        "doc_path": doc_path,
        "page": chunk.page_start,
        "section": chunk.section_title,
        "type": chunk.content_type,
    }
    if item_handle:
        payload["item_handle"] = item_handle
    payload.update(llm_metadata_payload)
    return {k: v for k, v in payload.items() if v is not None}


def push_llm_metadata_to_qdrant(doc_id: str, payload: dict) -> int:
    """Propaga um payload de metadados LLM JÁ CARREGADO para os pontos existentes
    do documento no Qdrant, via set_payload (merge de chaves, sem re-embedar).

    É o mecanismo que DESACOPLA o enrich da indexação: o índice não depende do LLM
    e, quando o enrich roda (antes ou depois da indexação), os metadados chegam
    aqui como um merge de payload — sem re-embedar nem reindexar.

    Retorna o número de pontos atualizados (0 se o doc ainda não foi indexado ou
    se não há payload a propagar)."""
    if not payload:
        return 0

    client, _, _ = _get_indexer()
    doc_name = f"{doc_id}.md"
    selector = Filter(must=[FieldCondition(key="doc_name", match=MatchValue(value=doc_name))])
    n = client.count(collection_name=COLLECTION_NAME, count_filter=selector, exact=True).count
    if n == 0:
        log.info("push_llm_metadata_to_qdrant: doc '%s' ainda não indexado (0 pontos).", doc_id)
        return 0

    client.set_payload(collection_name=COLLECTION_NAME, payload=payload, points=selector, wait=True)
    log.info("push_llm_metadata_to_qdrant: metadados LLM propagados a %d ponto(s) de '%s'.", n, doc_id)
    return n


def sync_llm_metadata_to_qdrant(doc_id: str) -> int:
    """Variante que lê os metadados LLM do layout local (<doc>_metadata_llm.json)
    e os propaga ao Qdrant. Usada pelo fluxo legado (enrich_job). O fluxo v2 (MinIO)
    usa push_llm_metadata_to_qdrant com o payload já em mãos.

    Retorna o número de pontos atualizados (0 se o doc ainda não foi indexado
    ou se não há metadados LLM a propagar)."""
    matches = list((OUTPUT_DIR / doc_id).rglob(f"{doc_id}{CONTENT_LIST_SUFFIX}"))
    if not matches:
        log.warning("sync_llm_metadata_to_qdrant: content_list de '%s' não encontrado.", doc_id)
        return 0

    payload = load_llm_metadata_payload(matches[0])
    if not payload:
        log.info("sync_llm_metadata_to_qdrant: sem metadados LLM para '%s' — nada a propagar.", doc_id)
        return 0

    return push_llm_metadata_to_qdrant(doc_id, payload)


EMBEDDING_CSV_HEADERS = [
    "doc_id", "timestamp", "n_chunks", "total_chars",
    "chunk_time_s", "embed_time_s", "upsert_time_s", "total_time_s",
    "chars_per_s", "ram_delta_mb", "ram_peak_mb", "vram_peak_mb",
    "avg_sparse_tokens", "status",
]


def save_embedding_report(metrics_list: list[dict]) -> None:
    """Salva o relatorio de embeddings em CSV (append ou cria)."""
    file_exists = EMBEDDING_REPORT_PATH.exists()
    with open(EMBEDDING_REPORT_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EMBEDDING_CSV_HEADERS, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerows(metrics_list)
    log.info("Relatorio de embeddings salvo em '%s' (%d entradas).", EMBEDDING_REPORT_PATH, len(metrics_list))


def index_all(client: QdrantClient, reset: bool, embedder: BgeM3EmbedderService, dense_dim: int, enable_llm_repair: bool = False, llm_model: str = "gemma4:latest") -> None:
    """Indexa todos os documentos encontrados em output/."""
    log.info("Inicio da indexacao em lote.")
    json_files = find_json_files(OUTPUT_DIR)

    if not json_files:
        print(f"⚠️  Nenhum arquivo content_list_v2.json encontrado em '{OUTPUT_DIR}/'")
        print("   Execute o MinerU primeiro (python main.py) para gerar os JSONs.")
        log.warning("Sem arquivos para indexar. Encerrando sem indexacao.")
        sys.exit(0)

    print(f"\n{len(json_files)} documento(s) encontrado(s) em '{OUTPUT_DIR}/'")
    if enable_llm_repair:
        print(f"  🧠 Reparo via LLM habilitado (modelo: {llm_model})")
    print()
    setup_collection(client, reset, dense_dim=dense_dim)

    # Limpa o CSV se existir e reset foi solicitado
    if reset and EMBEDDING_REPORT_PATH.exists():
        EMBEDDING_REPORT_PATH.unlink()
        log.info("Relatorio de embeddings anterior removido (--reset).")

    chunker = MinerUChunker(enable_llm_repair=enable_llm_repair, llm_model=llm_model)
    log.info("Chunker inicializado: max_chunk_chars=%d overlap_chars=%d llm_repair=%s", chunker.max_chunk_chars, chunker.overlap_chars, enable_llm_repair)

    all_metrics: list[dict] = []
    total_chunks = 0
    errors = 0
    skipped = 0
    for i, json_path in enumerate(json_files, 1):
        doc_name = json_path.parent.parent.name
        bar = progress_bar(i - 1, len(json_files))
        print(f"\r{bar} {i-1}/{len(json_files)} | {doc_name}", end="", flush=True)
        if not reset and is_document_indexed(client, doc_name):
            skipped += 1
            log.info("Documento '%s' ja indexado — pulando.", doc_name)
            continue
        try:
            metrics = index_document(client, json_path, chunker, embedder)
            n = metrics["n_chunks"]
            total_chunks += n
            all_metrics.append(metrics)
            log.info("Documento '%s' indexado com sucesso (%d chunk(s)).", doc_name, n)
        except Exception as exc:
            errors += 1
            print(f"\nErro ao indexar '{doc_name}': {exc}")
            log.exception("Falha ao indexar documento '%s'.", doc_name)
            all_metrics.append({
                "doc_id": doc_name, "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "n_chunks": 0, "total_chars": 0,
                "chunk_time_s": 0.0, "embed_time_s": 0.0, "upsert_time_s": 0.0, "total_time_s": 0.0,
                "chars_per_s": 0.0, "ram_delta_mb": 0.0, "ram_peak_mb": 0.0, "vram_peak_mb": 0.0,
                "avg_sparse_tokens": 0.0, "status": f"erro: {exc}",
            })

    print(f"\r{progress_bar(len(json_files), len(json_files))} {len(json_files)}/{len(json_files)} | concluido")

    if all_metrics:
        save_embedding_report(all_metrics)

    print(f"\n{'=' * 55}")
    print(f"Total: {total_chunks} chunks | {len(json_files) - errors - skipped} ok | {skipped} pulado(s) | {errors} erro(s)")
    if enable_llm_repair:
        repair_info = chunker.rejection_summary
        print(f"Chunks reparados via LLM: {repair_info.get('total_repaired', 0)}")
    print(f"Relatorio salvo em: {EMBEDDING_REPORT_PATH}")
    print(f"Collection: '{COLLECTION_NAME}' no Qdrant ({QDRANT_URL})")
    print(f"{'=' * 55}\n")
    log.info(
        "Indexacao concluida. total_chunks=%d docs_ok=%d docs_pulados=%d docs_erro=%d collection=%s",
        total_chunks,
        len(json_files) - errors - skipped,
        skipped,
        errors,
        COLLECTION_NAME,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Indexa chunks semanticos no Qdrant usando BAAI/bge-m3 (dense + sparse)."
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Recria a collection antes de indexar (apaga dados existentes)."
    )
    parser.add_argument(
        "--url", default=QDRANT_URL,
        help=f"URL do Qdrant. Padrao: {QDRANT_URL}"
    )
    parser.add_argument(
        "--llm-repair", action="store_true",
        help="Habilita reparo de chunks via LLM local (Ollama) para corrigir OCR."
    )
    parser.add_argument(
        "--llm-model", default="gemma4:latest",
        help="Modelo Ollama para reparo. Padrao: gemma4:latest"
    )
    
    args = parser.parse_args()
    log.info("Parametros recebidos: reset=%s url=%s llm_repair=%s llm_model=%s", args.reset, args.url, args.llm_repair, args.llm_model)

    print(f"\nConectando ao Qdrant em {args.url}...")
    try:
        client = QdrantClient(url=args.url, timeout=QDRANT_TIMEOUT_SECONDS or None)
        client.get_collections()
        print("Conexao OK")
        log.info("Conexao com Qdrant estabelecida com sucesso.")
    except Exception as exc:
        print(f"Falha ao conectar: {exc}")
        print("Verifique se o Qdrant esta rodando: docker compose up -d")
        log.exception("Nao foi possivel conectar ao Qdrant.")
        sys.exit(1)

    embedder = BgeM3EmbedderService()
    check_models_available(embedder)

    print(f"\nCarregando modelo {DENSE_MODEL} (dense + sparse)...")
    if not embedder.load_model(require_cache=False):
        print("Falha ao carregar o modelo.")
        sys.exit(1)

    # Detecta a dimensao real do vetor denso para evitar mismatch na collection.
    dense_dim = embedder.get_dense_dimension()
    log.info("Dimensao dense detectada: %d", dense_dim)

    log.info("Iniciando indexacao geral.")
    index_all(
        client, reset=args.reset, embedder=embedder, dense_dim=dense_dim,
        enable_llm_repair=args.llm_repair, llm_model=args.llm_model,
    )
    log.info("Processo finalizado com sucesso.")


if __name__ == "__main__":
    main()
