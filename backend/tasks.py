"""Tasks Celery do pipeline de ingestão.

Uma chain por PDF: baixar_dspace → extrair_mineru → enrich_llm → indexar_qdrant.
O retorno de cada task (um dict `ctx`) é o 1º argumento da próxima, propagando
job_id/item_uuid/paths entre os estágios sem estado global.

Semântica de erro (fiel ao run_pdf_pipeline original):
  - download/mineru falham  → status "erro" e a chain para (estágios seguintes não rodam).
  - enrich falha            → best-effort: loga, marca llm_error e SEGUE para a indexação.
  - index falha             → status "concluido" + index_error (o markdown já é válido).
"""

import traceback
import urllib.error
from pathlib import Path

from backend.celery_app import app
from backend.core.config import OUTPUT_DIR
from backend.core.logger import log
from backend.services.mineru_service import process_pdf
from backend.services.dspace_service import download_bitstream
from backend.services import llm_enrich_service as llm_enrich
from backend.services.job_store import set_status, find_markdown, markdown_url

# PDFs baixados do DSpace (mesmo diretório usado pelo endpoint antigo).
DATA_DIR = Path("data")


def _job_id_of(args, kwargs):
    """Extrai o job_id de uma task: kwarg explícito (baixar_dspace) ou ctx["job_id"]
    (primeiro posicional nas demais)."""
    if kwargs.get("job_id"):
        return kwargs["job_id"]
    if args and isinstance(args[0], dict):
        return args[0].get("job_id")
    return None


class PipelineTask(app.Task):
    """Base das tasks do pipeline: em falha não tratada, marca o job como "erro"
    (senão o status congelaria no último estágio e a chain pararia em silêncio)."""

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        job_id = _job_id_of(args, kwargs)
        if job_id:
            stage = self.name.rsplit(".", 1)[-1]
            set_status(job_id, "erro", stage=stage, error=f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------
# Estágio 1 — download do bitstream (fila: download)
# --------------------------------------------------------------------------
@app.task(bind=True, base=PipelineTask, max_retries=3, default_retry_delay=15)
def baixar_dspace(self, bs_uuid, filename, job_id, item_uuid="", item_handle=""):
    set_status(
        job_id, "processando", stage="download", filename=filename,
        source="dspace-item", item_uuid=item_uuid, item_handle=item_handle,
        bitstream_uuid=bs_uuid,
    )
    try:
        pdf_path = download_bitstream(bs_uuid, DATA_DIR, filename=filename)
    except (urllib.error.HTTPError, urllib.error.URLError) as e:
        try:
            raise self.retry(exc=e)  # transiente → backoff/retry
        except self.MaxRetriesExceededError:
            set_status(job_id, "erro", stage="download", error=f"download falhou: {e}")
            raise
    except ValueError as e:  # conteúdo não é PDF → retry não ajuda
        set_status(job_id, "erro", stage="download", error=str(e))
        raise
    return {"job_id": job_id, "pdf_path": str(pdf_path),
            "item_uuid": item_uuid, "item_handle": item_handle}


# --------------------------------------------------------------------------
# Estágio 2 — extração MinerU (fila: extract)
# --------------------------------------------------------------------------
@app.task(base=PipelineTask)
def extrair_mineru(ctx):
    from backend.services.pipeline_worker import write_process_log

    job_id = ctx["job_id"]
    pdf_path = Path(ctx["pdf_path"])
    set_status(job_id, "processando", stage="mineru")

    proc_result = process_pdf(pdf_path, OUTPUT_DIR)
    write_process_log(proc_result)
    if proc_result.get("status") != "Sucesso":
        set_status(job_id, "erro", stage="mineru", error="Falha na extração do MinerU")
        raise RuntimeError(f"MinerU falhou para {job_id}")

    json_matches = list((OUTPUT_DIR / job_id).rglob(f"{job_id}_content_list_v2.json"))
    if not json_matches:
        set_status(job_id, "erro", stage="mineru", error="JSON de conteúdo não encontrado após extração")
        raise RuntimeError(f"content_list não encontrado para {job_id}")

    md_path = find_markdown(job_id)
    ctx["json_path"] = str(json_matches[0])
    ctx["md_url"] = markdown_url(md_path) if md_path else None
    return ctx


# --------------------------------------------------------------------------
# Estágio 3 — enriquecimento por LLM (fila: llm) — best-effort
# --------------------------------------------------------------------------
@app.task(base=PipelineTask)
def enrich_llm(ctx):
    job_id = ctx["job_id"]
    if not llm_enrich.is_available():
        log.info("enrich_llm: DEEPSEEK_API_KEY ausente — pulando enrich de %s.", job_id)
        return ctx

    set_status(job_id, "processando", stage="llm", md_url=ctx.get("md_url"))
    try:
        meta = llm_enrich.enrich_job(job_id, uuid=ctx.get("item_uuid", ""))
        set_status(
            job_id, "processando", stage="llm", md_url=ctx.get("md_url"),
            metadata_json=meta.arquivo_json, metadata_revisar=meta.revisar,
        )
    except Exception as exc:
        log.error("enrich_llm falhou para %s: %s", job_id, exc)
        log.error(traceback.format_exc())
        set_status(job_id, "processando", stage="llm", md_url=ctx.get("md_url"), llm_error=str(exc))
    return ctx  # SEMPRE segue: a indexação não pode ser bloqueada pela LLM


# --------------------------------------------------------------------------
# Estágio 4 — embedding + upsert no Qdrant (fila: gpu, concurrency=1)
# --------------------------------------------------------------------------
@app.task(base=PipelineTask)
def indexar_qdrant(ctx):
    from backend.indexing.index_chunks import index_single_document
    from backend.services.pipeline_worker import write_embed_log

    job_id = ctx["job_id"]
    set_status(job_id, "processando", stage="index", md_url=ctx.get("md_url"))
    try:
        embed_result = index_single_document(
            Path(ctx["json_path"]),
            item_uuid=ctx.get("item_uuid", ""),
            item_handle=ctx.get("item_handle", ""),
        )
        write_embed_log(embed_result)
        set_status(
            job_id, "concluido", stage="index", md_url=ctx.get("md_url"),
            n_chunks=embed_result.get("n_chunks") if isinstance(embed_result, dict) else None,
        )
        log.info("Pipeline concluído para %s.", job_id)
    except Exception as exc:
        log.error("Indexação falhou para %s: %s", job_id, exc)
        log.error(traceback.format_exc())
        # Markdown já é válido — conclui com aviso de índice (fiel ao comportamento original).
        set_status(job_id, "concluido", stage="index", md_url=ctx.get("md_url"), index_error=str(exc))
    return ctx
