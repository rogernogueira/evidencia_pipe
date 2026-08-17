"""Lógica das etapas do pipeline (download → MinerU → índice; enrich desacoplado).

Estas funções são PURAS em relação ao Celery: recebem/retornam um PipelineContext
leve e falam apenas com o ArtifactStore (MinIO) e o ManifestRepository. São usadas
pelas tasks Celery (backend/tasks.py), garantindo UMA fonte oficial de artefatos.

Contrato de cada etapa:
  1. abre o manifesto no MinIO (ou o cria, no download);
  2. baixa/lê só os objetos necessários para um tempdir exclusivo;
  3. executa o processamento;
  4. grava os novos artefatos no MinIO (SHA-256 explícito + metadados mínimos);
  5. atualiza o manifesto sob lock curto;
  6. remove o tempdir;
  7. retorna um novo PipelineContext (a etapa de indexação retorna um resumo).

Nenhuma etapa retorna markdown, JSON MinerU, chunks, embeddings, imagens ou bytes.
"""

from __future__ import annotations

import shutil
import tempfile
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from backend.core import config as settings
from backend.core.logger import log
from backend.core.schemas import (
    ART_CHUNKING_REPORT,
    ART_CHUNKS,
    ART_EMBEDDING_REPORT,
    ART_LLM_RAW_RESPONSE,
    ART_METADATA_CANDIDATES,
    ART_MINERU_CONTENT_LIST,
    ART_MINERU_IMAGES,
    ART_MINERU_IMAGES_MANIFEST,
    ART_MINERU_MARKDOWN,
    ART_MINERU_METRICS,
    ART_SOURCE_PDF,
    ArtifactReference,
    CTX_STAGE_DOWNLOADED,
    CTX_STAGE_ENRICHED,
    CTX_STAGE_EXTRACTED,
    PipelineContext,
    STAGE_DOWNLOAD,
    STAGE_ENRICHMENT,
    STAGE_INDEXING,
    STAGE_MINERU,
)
from backend.services.artifact_store import get_artifact_store, sanitize_component
from backend.services.manifest_repository import get_manifest_repository

# Namespace fixo para derivar pipeline_id determinístico (idempotência §32).
_PIPELINE_NS = uuid.uuid5(uuid.NAMESPACE_URL, "evidencia_pipe/pipeline")


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now():
    return datetime.now(timezone.utc)


def _derive_pipeline_id(item_uuid: str, bs_uuid: str | None, document_id: str) -> str:
    return str(uuid.uuid5(_PIPELINE_NS, f"{item_uuid or ''}|{bs_uuid or ''}|{document_id}"))


def _artifact_metadata(*, pipeline_id, document_id, job_id, artifact_name, stage) -> dict:
    """Metadados mínimos gravados no objeto (§17) — sem credenciais nem conteúdo."""
    return {
        "pipeline-id": pipeline_id,
        "document-id": document_id,
        "job-id": job_id,
        "artifact-name": artifact_name,
        "pipeline-version": settings.PIPELINE_VERSION,
        "stage": stage,
        "created-at": _iso_now(),
    }


@contextmanager
def _work_dir(prefix: str) -> Iterator[Path]:
    """Diretório temporário exclusivo, sempre removido ao final (salvo depuração)."""
    base = Path(settings.ARTIFACT_TEMP_DIR)
    base.mkdir(parents=True, exist_ok=True)
    d = Path(tempfile.mkdtemp(prefix=f"{prefix}-", dir=base))
    keep = False
    try:
        yield d
    except BaseException:
        keep = settings.ARTIFACT_KEEP_LOCAL_TEMP_ON_FAILURE
        raise
    finally:
        if keep:
            log.warning("[pipeline] preservando tempdir de depuração (não retornado à API): %s", d)
        else:
            shutil.rmtree(d, ignore_errors=True)


def _context(pipeline_id, job_id, document_id, item_uuid, bs_uuid, manifest_uri, stage, force, warnings=0) -> PipelineContext:
    return PipelineContext(
        pipeline_id=uuid.UUID(pipeline_id),
        job_id=job_id,
        item_uuid=item_uuid or "",
        bitstream_uuid=bs_uuid or None,
        document_id=document_id,
        artifact_manifest_uri=manifest_uri,
        current_stage=stage,
        pipeline_version=settings.PIPELINE_VERSION,
        warnings_count=warnings,
        force=force,
    )


# ==========================================================================
# Etapa 1 — download do bitstream DSpace → MinIO
# ==========================================================================

def stage_download(
    *, bs_uuid: str, filename: str, job_id: str,
    item_uuid: str = "", item_handle: str = "", force: bool = False,
) -> PipelineContext:
    from backend.services.dspace_service import download_bitstream

    store = get_artifact_store()
    repo = get_manifest_repository()

    document_id = sanitize_component(job_id, field="job_id")
    pipeline_id = _derive_pipeline_id(item_uuid, bs_uuid, document_id)
    repo.create(
        pipeline_id=pipeline_id, job_id=job_id, document_id=document_id,
        item_uuid=item_uuid, bitstream_uuid=bs_uuid,
    )
    manifest = repo.load(pipeline_id, document_id)
    manifest_uri = repo.manifest_uri(pipeline_id, document_id)
    src_key = store.artifact_key(pipeline_id, document_id, "source", "original.pdf")

    reuse = (
        not force
        and manifest.is_stage_completed(STAGE_DOWNLOAD)
        and ART_SOURCE_PDF in manifest.artifacts
        and store.exists(src_key)
    )
    if reuse:
        log.info("[download] reutilizando source_pdf de '%s' (retry idempotente).", document_id)
        return _context(pipeline_id, job_id, document_id, item_uuid, bs_uuid, manifest_uri,
                        CTX_STAGE_DOWNLOADED, force, manifest.warnings.__len__())

    repo.start_stage(pipeline_id, document_id, STAGE_DOWNLOAD)
    page_count = 0
    with _work_dir("download") as wd:
        local = download_bitstream(bs_uuid, wd, filename=f"{document_id}.pdf")
        ref = store.put_file(
            src_key, local, content_type="application/pdf", name=ART_SOURCE_PDF,
            metadata=_artifact_metadata(pipeline_id=pipeline_id, document_id=document_id,
                                        job_id=job_id, artifact_name=ART_SOURCE_PDF, stage=STAGE_DOWNLOAD),
        )
        try:
            from backend.services.mineru_service import count_pages

            page_count = count_pages(local)
        except Exception:
            page_count = 0

    with repo.update(pipeline_id, document_id) as m:
        m.item_handle = item_handle or m.item_handle
        m.artifacts[ART_SOURCE_PDF] = ref
        if page_count:
            m.metrics["page_count"] = page_count
        st = m.stage(STAGE_DOWNLOAD)
        st.status = "COMPLETED"
        st.completed_at = _now()
        m.status = "RUNNING"

    return _context(pipeline_id, job_id, document_id, item_uuid, bs_uuid, manifest_uri,
                    CTX_STAGE_DOWNLOADED, force)


# ==========================================================================
# Etapa 2 — extração MinerU (usa arquivo local; produtos vão ao MinIO)
# ==========================================================================

def _upload_mineru_images(store, pipeline_id, document_id, job_id, images_dir: Path):
    """Sobe as imagens sob mineru/images/ e grava um índice images_manifest.json.
    O manifesto principal aponta para o prefixo (object_count/total_size) e o índice,
    evitando inchar o manifesto com uma referência por imagem (§20)."""
    if images_dir is None or not images_dir.is_dir():
        return None, None
    files = sorted(p for p in images_dir.iterdir() if p.is_file())
    if not files:
        return None, None

    index = []
    total = 0
    prefix_key = store.artifact_key(pipeline_id, document_id, "mineru", "images")
    for i, img in enumerate(files, 1):
        # Nome sanitizado e determinístico, preservando a extensão.
        ext = "".join(img.suffixes[-1:]) or ".bin"
        safe_name = f"image_{i:04d}{ext}"
        key = store.artifact_key(pipeline_id, document_id, "mineru", "images", safe_name)
        ctype = _guess_content_type(img.name)
        ref = store.put_file(key, img, content_type=ctype, name=safe_name,
                             metadata=_artifact_metadata(pipeline_id=pipeline_id, document_id=document_id,
                                                         job_id=job_id, artifact_name="mineru_image", stage=STAGE_MINERU))
        total += ref.size_bytes
        index.append({"original_name": img.name, "object_key": key, "sha256": ref.sha256,
                      "size_bytes": ref.size_bytes, "content_type": ctype})

    idx_ref = store.put_json(
        store.artifact_key(pipeline_id, document_id, "mineru", "images_manifest.json"),
        {"count": len(index), "total_size_bytes": total, "images": index},
        name=ART_MINERU_IMAGES_MANIFEST,
    )
    prefix_ref = ArtifactReference(
        name=ART_MINERU_IMAGES,
        uri=store.build_uri(prefix_key + "/"),
        bucket=store.bucket,
        object_key=prefix_key + "/",
        content_type="application/x-directory-prefix",
        object_count=len(index),
        total_size_bytes=total,
    )
    return prefix_ref, idx_ref


def _guess_content_type(name: str) -> str:
    import mimetypes

    return mimetypes.guess_type(name)[0] or "application/octet-stream"


def stage_mineru(ctx: PipelineContext, *, task_id: str | None = None) -> PipelineContext:
    from backend.services.mineru_service import process_pdf

    store = get_artifact_store()
    repo = get_manifest_repository()
    pipeline_id = str(ctx.pipeline_id)
    document_id = ctx.document_id
    manifest = repo.load_from_uri(ctx.artifact_manifest_uri)

    if (not ctx.force and manifest.is_stage_completed(STAGE_MINERU)
            and ART_MINERU_CONTENT_LIST in manifest.artifacts):
        log.info("[mineru] reutilizando extração de '%s' (retry idempotente).", document_id)
        return _replace_stage(ctx, CTX_STAGE_EXTRACTED)

    src_ref = manifest.artifacts.get(ART_SOURCE_PDF)
    if src_ref is None:
        raise RuntimeError(f"source_pdf ausente no manifesto de {document_id}.")

    repo.start_stage(pipeline_id, document_id, STAGE_MINERU)
    refs: dict[str, ArtifactReference] = {}
    proc_result: dict = {}
    with _work_dir("mineru") as wd:
        pdf_local = wd / f"{document_id}.pdf"
        store.download_to_file(src_ref.object_key, pdf_local, expected_sha256=src_ref.sha256 or None)
        out_dir = wd / "out"
        out_dir.mkdir()

        proc_result = process_pdf(pdf_local, out_dir, task_id=task_id, document_id=document_id)
        if proc_result.get("status") != "Sucesso":
            raise RuntimeError(f"MinerU falhou para {document_id}")

        doc_out = out_dir / document_id
        json_matches = list(doc_out.rglob(f"{document_id}_content_list_v2.json"))
        if not json_matches:
            raise RuntimeError(f"content_list não encontrado após extração de {document_id}")
        json_path = json_matches[0]
        md_matches = list(doc_out.rglob("*.md"))
        md_path = md_matches[0] if md_matches else None
        images_dirs = list(doc_out.rglob("images"))
        images_dir = images_dirs[0] if images_dirs else None

        meta_kw = dict(pipeline_id=pipeline_id, document_id=document_id, job_id=manifest.job_id, stage=STAGE_MINERU)
        refs[ART_MINERU_CONTENT_LIST] = store.put_file(
            store.artifact_key(pipeline_id, document_id, "mineru", "content_list_v2.json"),
            json_path, content_type="application/json", name=ART_MINERU_CONTENT_LIST,
            metadata=_artifact_metadata(artifact_name=ART_MINERU_CONTENT_LIST, **meta_kw),
        )
        if md_path is not None:
            refs[ART_MINERU_MARKDOWN] = store.put_file(
                store.artifact_key(pipeline_id, document_id, "mineru", "document.md"),
                md_path, content_type="text/markdown; charset=utf-8", name=ART_MINERU_MARKDOWN,
                metadata=_artifact_metadata(artifact_name=ART_MINERU_MARKDOWN, **meta_kw),
            )
        refs[ART_MINERU_METRICS] = store.put_json(
            store.artifact_key(pipeline_id, document_id, "mineru", "processing_metrics.json"),
            proc_result, name=ART_MINERU_METRICS,
            metadata=_artifact_metadata(artifact_name=ART_MINERU_METRICS, **meta_kw),
        )
        img_prefix_ref, img_index_ref = _upload_mineru_images(store, pipeline_id, document_id, manifest.job_id, images_dir)

    with repo.update(pipeline_id, document_id) as m:
        m.artifacts.update(refs)
        if img_prefix_ref is not None:
            m.artifacts[ART_MINERU_IMAGES] = img_prefix_ref
        if img_index_ref is not None:
            m.artifacts[ART_MINERU_IMAGES_MANIFEST] = img_index_ref
        if proc_result.get("quantidade_paginas"):
            m.metrics["page_count"] = proc_result["quantidade_paginas"]
        m.metrics["images_count"] = proc_result.get("quantidade_imagens_extraidas", 0)
        st = m.stage(STAGE_MINERU)
        st.status = "COMPLETED"
        st.completed_at = _now()

    return _replace_stage(ctx, CTX_STAGE_EXTRACTED)


# ==========================================================================
# Etapa 3 — enriquecimento por LLM (best-effort)
# ==========================================================================

def stage_enrich(ctx: PipelineContext) -> PipelineContext:
    from backend.services import llm_enrich_service as llm_enrich

    store = get_artifact_store()
    repo = get_manifest_repository()
    pipeline_id = str(ctx.pipeline_id)
    document_id = ctx.document_id
    warnings = ctx.warnings_count

    if not llm_enrich.is_available():
        log.info("[enrich] provedor LLM não configurado (LLM_ENRICH_API_KEY ausente) — pulando enrich de '%s'.", document_id)
        return _replace_stage(ctx, CTX_STAGE_ENRICHED)

    manifest = repo.load_from_uri(ctx.artifact_manifest_uri)
    if (not ctx.force and manifest.is_stage_completed(STAGE_ENRICHMENT)
            and ART_METADATA_CANDIDATES in manifest.artifacts):
        log.info("[enrich] reutilizando metadados de '%s' (retry idempotente).", document_id)
        return _replace_stage(ctx, CTX_STAGE_ENRICHED)

    md_ref = manifest.artifacts.get(ART_MINERU_MARKDOWN)
    if md_ref is None:
        log.warning("[enrich] markdown ausente para '%s' — pulando enrich.", document_id)
        return _replace_stage(ctx, CTX_STAGE_ENRICHED)

    repo.start_stage(pipeline_id, document_id, STAGE_ENRICHMENT)
    try:
        markdown = store.read_text(md_ref.object_key, max_bytes=settings.LLM_MAX_INPUT_ARTIFACT_BYTES)
        raw_sink: list[str] = []
        meta = llm_enrich.enrich_markdown(markdown, doc_id=document_id, uuid=ctx.item_uuid, raw_sink=raw_sink)
        meta_ref = store.put_json(
            store.artifact_key(pipeline_id, document_id, "enrichment", "metadata_candidates.json"),
            meta.model_dump(), name=ART_METADATA_CANDIDATES,
            metadata=_artifact_metadata(pipeline_id=pipeline_id, document_id=document_id,
                                        job_id=manifest.job_id, artifact_name=ART_METADATA_CANDIDATES, stage=STAGE_ENRICHMENT),
        )
        raw_ref = None
        if settings.ARTIFACT_KEEP_LLM_RAW_RESPONSE and raw_sink:
            raw_ref = store.put_text(
                store.artifact_key(pipeline_id, document_id, "enrichment", "raw_response.json"),
                raw_sink[0], content_type="application/json", name=ART_LLM_RAW_RESPONSE,
            )
        with repo.update(pipeline_id, document_id) as m:
            m.artifacts[ART_METADATA_CANDIDATES] = meta_ref
            if raw_ref is not None:
                m.artifacts[ART_LLM_RAW_RESPONSE] = raw_ref
            if meta.revisar:
                m.warnings.append(f"enrichment: metadados de '{document_id}' marcados para revisão humana")
            st = m.stage(STAGE_ENRICHMENT)
            st.status = "COMPLETED"
            st.completed_at = _now()
        if meta.revisar:
            warnings += 1
        # Enrich DESACOPLADO da indexação: se o documento já estiver indexado (enrich
        # rodando como follow-up DEPOIS do índice), propaga os metadados ao Qdrant via
        # set_payload — sem re-embedar. Se ainda não indexado, é no-op (a propagação
        # ocorrerá quando o índice existir). Best-effort: nunca falha o estágio.
        if manifest.is_stage_completed(STAGE_INDEXING):
            _propagate_llm_metadata(document_id, meta)
    except Exception as exc:
        # Best-effort: registra aviso no manifesto e SEGUE para a indexação.
        log.error("[enrich] falhou para '%s': %s", document_id, exc)
        try:
            with repo.update(pipeline_id, document_id) as m:
                m.warnings.append(f"enrichment falhou: {type(exc).__name__}: {str(exc)[:200]}")
                st = m.stage(STAGE_ENRICHMENT)
                st.status = "FAILED"
                st.error = str(exc)[:500]
        except Exception:
            pass
        warnings += 1

    return _replace_stage(ctx, CTX_STAGE_ENRICHED, warnings=warnings)


# ==========================================================================
# Etapa 4 — chunking + embeddings + upsert Qdrant → resumo pequeno
# ==========================================================================

def stage_index(ctx: PipelineContext, *, task_id: str | None = None) -> dict:
    from backend.indexing.index_chunks import index_materialized_document

    store = get_artifact_store()
    repo = get_manifest_repository()
    pipeline_id = str(ctx.pipeline_id)
    document_id = ctx.document_id
    manifest = repo.load_from_uri(ctx.artifact_manifest_uri)

    def _summary(status, chunk_count, indexed_count, failed_count=0):
        return {
            "pipeline_id": pipeline_id,
            "document_id": document_id,
            "status": status,
            "chunk_count": chunk_count,
            "indexed_count": indexed_count,
            "failed_count": failed_count,
            "artifact_manifest_uri": ctx.artifact_manifest_uri,
            "qdrant_collection": settings.QDRANT_COLLECTION,
        }

    if not ctx.force and manifest.is_stage_completed(STAGE_INDEXING):
        log.info("[index] '%s' já indexado (retry idempotente).", document_id)
        n = manifest.metrics.get("chunk_count", 0)
        return _summary("completed", n, manifest.metrics.get("indexed_count", n))

    cl_ref = manifest.artifacts.get(ART_MINERU_CONTENT_LIST)
    if cl_ref is None:
        raise RuntimeError(f"content_list ausente no manifesto de {document_id}.")

    repo.start_stage(pipeline_id, document_id, STAGE_INDEXING)
    with _work_dir("index") as wd:
        # Materializa no layout esperado (<doc>/<metodo>/<doc>_content_list_v2.json).
        method_dir = wd / document_id / "hybrid_auto"
        method_dir.mkdir(parents=True)
        json_local = method_dir / f"{document_id}_content_list_v2.json"
        store.download_to_file(cl_ref.object_key, json_local, expected_sha256=cl_ref.sha256 or None)

        # A indexação é DESACOPLADA do enriquecimento por LLM: NÃO lê nem depende de
        # metadata_candidates. O título vem da estrutura do documento (parser); os
        # metadados LLM, quando houver, são propagados DEPOIS pelo stage_enrich via
        # set_payload (push_llm_metadata_to_qdrant), sem re-embedar.

        # Markdown como fallback de parsing (§7): materializa quando disponível.
        md_local = None
        md_ref = manifest.artifacts.get(ART_MINERU_MARKDOWN)
        if md_ref is not None:
            md_local = wd / f"{document_id}.md"
            try:
                store.download_to_file(md_ref.object_key, md_local, expected_sha256=md_ref.sha256 or None)
            except Exception as exc:
                log.warning("[index] markdown de fallback indisponível para '%s': %s", document_id, exc)
                md_local = None

        chunks_jsonl = wd / "chunks.jsonl"
        chunking_report_local = wd / "chunking_report.json"
        result = index_materialized_document(
            json_local, doc_id=document_id, doc_path=f"{document_id}.md",
            item_uuid=ctx.item_uuid,
            item_handle=manifest.item_handle or "", bitstream_uuid=ctx.bitstream_uuid,
            document_checksum=(cl_ref.sha256 or ""), source_artifact_uri=cl_ref.uri,
            markdown_path=md_local, task_id=task_id,
            chunks_jsonl_path=chunks_jsonl, chunking_report_path=chunking_report_local,
            upsert_batch_size=settings.QDRANT_UPSERT_BATCH_SIZE,
        )
        chunks_ref = None
        if settings.ARTIFACT_KEEP_CHUNKS and chunks_jsonl.exists():
            chunks_ref = store.put_file(
                store.artifact_key(pipeline_id, document_id, "indexing", "chunks.jsonl"),
                chunks_jsonl, content_type="application/x-ndjson", name=ART_CHUNKS,
                metadata=_artifact_metadata(pipeline_id=pipeline_id, document_id=document_id,
                                            job_id=manifest.job_id, artifact_name=ART_CHUNKS, stage=STAGE_INDEXING),
            )
        # Relatório de chunking (§23) — separado do relatório de embeddings.
        chunking_report_ref = store.put_json(
            store.artifact_key(pipeline_id, document_id, "indexing", "chunking_report.json"),
            result.get("chunking_report", {}), name=ART_CHUNKING_REPORT,
            metadata=_artifact_metadata(pipeline_id=pipeline_id, document_id=document_id,
                                        job_id=manifest.job_id, artifact_name=ART_CHUNKING_REPORT, stage=STAGE_INDEXING),
        )
        # embedding_report NÃO carrega o chunking_report inteiro (evita duplicação).
        embed_summary = {k: v for k, v in result.items() if k != "chunking_report"}
        report_ref = store.put_json(
            store.artifact_key(pipeline_id, document_id, "indexing", "embedding_report.json"),
            embed_summary, name=ART_EMBEDDING_REPORT,
            metadata=_artifact_metadata(pipeline_id=pipeline_id, document_id=document_id,
                                        job_id=manifest.job_id, artifact_name=ART_EMBEDDING_REPORT, stage=STAGE_INDEXING),
        )

    n_chunks = int(result.get("n_chunks", 0)) if isinstance(result, dict) else 0
    chunking_report = result.get("chunking_report", {}) if isinstance(result, dict) else {}
    # 0 chunks COM blocos estruturais é anomalia (a política de seção/filtros descartou o
    # documento inteiro): registra no manifesto e devolve status próprio, para o job não
    # sair como sucesso silencioso com o índice vazio.
    block_count = int(chunking_report.get("document_block_count", 0) or 0)
    empty_index = n_chunks == 0 and block_count > 0
    empty_reason = ""
    if empty_index:
        empty_reason = (
            f"indexação vazia: {block_count} bloco(s) estruturais e 0 chunk(s) — "
            f"pulados={chunking_report.get('skipped_by_reason') or {}} "
            f"rejeitados={chunking_report.get('rejected_by_reason') or {}}"
        )
        log.error("[index] '%s': %s", document_id, empty_reason)
    with repo.update(pipeline_id, document_id) as m:
        if chunks_ref is not None:
            m.artifacts[ART_CHUNKS] = chunks_ref
        m.artifacts[ART_CHUNKING_REPORT] = chunking_report_ref
        m.artifacts[ART_EMBEDDING_REPORT] = report_ref
        m.metrics["chunk_count"] = n_chunks
        m.metrics["indexed_count"] = n_chunks
        m.metrics["chunking_strategy"] = result.get("chunking_strategy")
        m.metrics["chunking_config_hash"] = result.get("chunking_config_hash")
        m.metrics["tokenizer_name"] = result.get("tokenizer_name")
        m.metrics["structure_source"] = chunking_report.get("structure_source")
        if empty_index:
            m.warnings.append(empty_reason)
        st = m.stage(STAGE_INDEXING)
        st.status = "COMPLETED"
        st.completed_at = _now()
        m.status = "COMPLETED"

    if empty_index:
        summary = _summary("empty", n_chunks, n_chunks)
        summary["index_warning"] = empty_reason
        return summary
    return _summary("completed", n_chunks, n_chunks)


# ==========================================================================
# util
# ==========================================================================

def _replace_stage(ctx: PipelineContext, stage: str, *, warnings: int | None = None) -> PipelineContext:
    return ctx.model_copy(update={
        "current_stage": stage,
        "warnings_count": ctx.warnings_count if warnings is None else warnings,
    })


def _propagate_llm_metadata(document_id: str, meta) -> None:
    """Propaga (best-effort) os metadados LLM aos pontos já indexados no Qdrant via
    set_payload. Usado pelo enrich desacoplado (rodando após a indexação)."""
    try:
        from backend.indexing.index_chunks import push_llm_metadata_to_qdrant

        payload = meta.model_dump(exclude_none=True)
        payload["metadata_source"] = "llm"
        push_llm_metadata_to_qdrant(document_id, payload)
    except Exception as exc:
        log.warning("[enrich] falha ao propagar metadados ao Qdrant para '%s': %s", document_id, exc)


def build_context_from_manifest(
    manifest_uri: str, *, stage: str, force: bool = False,
) -> PipelineContext:
    """Reconstrói um PipelineContext leve a partir do manifesto — usado por
    follow-ups DESACOPLADOS (ex.: o enrich disparado após a indexação), que recebem
    apenas a URI do manifesto e não o contexto original da chain."""
    repo = get_manifest_repository()
    m = repo.load_from_uri(manifest_uri)
    return PipelineContext(
        pipeline_id=uuid.UUID(m.pipeline_id),
        job_id=m.job_id,
        item_uuid=m.item_uuid or "",
        bitstream_uuid=m.bitstream_uuid or None,
        document_id=m.document_id,
        artifact_manifest_uri=manifest_uri,
        current_stage=stage,
        pipeline_version=settings.PIPELINE_VERSION,
        force=force,
    )
