import csv
import traceback
from pathlib import Path

from backend.core.logger import log
from backend.services.job_store import set_status

LOG_PROCESS_FILE = Path("relatorio_processamento.csv")
LOG_EMBED_FILE = Path("relatorio_embeddings.csv")

def write_process_log(result: dict):
    file_exists = LOG_PROCESS_FILE.exists()
    with open(LOG_PROCESS_FILE, 'a', newline='', encoding='utf-8') as f:
        fieldnames = [
            "arquivo", "tempo_processamento_s", "quantidade_paginas", 
            "quantidade_imagens_extraidas", "quantidade_tabelas_extraidas",
            "ram_max_mb", "vram_max_mb", "status"
        ]
        # extrasaction="ignore": tolera chaves extras no dict (ex.: métricas novas)
        # sem quebrar o relatório CSV legado.
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)

def write_embed_log(result: dict):
    file_exists = LOG_EMBED_FILE.exists()
    with open(LOG_EMBED_FILE, 'a', newline='', encoding='utf-8') as f:
        fieldnames = [
            "doc_id", "timestamp", "n_chunks", "total_chars", "chunk_time_s",
            "embed_time_s", "upsert_time_s", "total_time_s", "chars_per_s",
            "ram_delta_mb", "ram_peak_mb", "vram_peak_mb", "avg_sparse_tokens",
            "status"
        ]
        # extrasaction="ignore": o result do chunker estrutural traz chaves extras
        # (chunking_strategy/version/config_hash, tokenizer_name, structure_source,
        # chunking_report) que não entram neste CSV flat — ignora em vez de quebrar.
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)

def run_pdf_pipeline(pdf_path: Path, item_uuid: str = "", item_handle: str = ""):
    """Fluxo LEGADO (síncrono, em background) — agora usando o MESMO MinIO das tasks
    Celery, para não manter duas fontes de artefatos (§42).

    O PDF já existe localmente; ele é gravado como source_pdf no MinIO e as etapas
    2-4 reutilizam exatamente a lógica de backend/services/pipeline_stages.py
    (MinerU → enrich → index), com os artefatos indo ao MinIO e o manifesto sendo a
    fonte de descoberta. A coordenação de GPU continua sendo feita próximo ao recurso
    (process_pdf e embedder), sem lock aninhado.
    """
    # Import tardio: evita puxar o SDK do MinIO em contextos que só usam os CSVs.
    from backend.services import pipeline_stages as stages

    pdf_path = Path(pdf_path)
    job_id = pdf_path.stem
    log.info("Pipeline Background (legado→MinIO) Iniciado: %s", pdf_path.name)
    set_status(job_id, "processando", filename=pdf_path.name)
    try:
        ctx = stages.stage_ingest_local_pdf(pdf_path, item_uuid=item_uuid, item_handle=item_handle)
        set_status(job_id, "processando", stage="download",
                   pipeline_id=str(ctx.pipeline_id), document_id=ctx.document_id,
                   artifact_manifest_uri=ctx.artifact_manifest_uri)

        # 2. Extração MinerU → MinIO
        set_status(job_id, "processando", stage="mineru")
        ctx = stages.stage_mineru(ctx)

        # 3. Indexação/embedding → Qdrant (best-effort quanto ao índice). A indexação
        # é DESACOPLADA do LLM: roda ANTES do enrich e não depende dele.
        set_status(job_id, "processando", stage="index")
        try:
            summary = stages.stage_index(ctx)
            set_status(job_id, "concluido", stage="index",
                       n_chunks=summary.get("chunk_count"),
                       indexed_count=summary.get("indexed_count"),
                       index_error=None)
            log.info("Pipeline Background Concluído para: %s", pdf_path.name)
        except Exception as idx_exc:
            log.error("Indexação falhou para %s: %s", pdf_path.name, idx_exc)
            log.error(traceback.format_exc())
            set_status(job_id, "concluido", stage="index", index_error=str(idx_exc))

        # 4. Enriquecimento por LLM (best-effort, DESACOPLADO) → MinIO. Roda APÓS a
        # indexação e propaga os metadados ao Qdrant (set_payload). Sem provedor
        # configurado é no-op; falhas aqui não afetam o índice já concluído.
        set_status(job_id, "processando", stage="llm")
        ctx = stages.stage_enrich(ctx)

    except Exception as e:
        log.error("Pipeline Background falhou criticamente: %s", e)
        log.error(traceback.format_exc())
        set_status(job_id, "erro", error=str(e))
