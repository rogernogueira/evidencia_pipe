"""Tasks Celery do pipeline de ingestão (v2 — artefatos no MinIO).

Chain OBRIGATÓRIA por PDF: baixar_dspace → extrair_mineru → indexar_qdrant.
Antes dela, só na ingestão de ITEM, pode haver um estágio 0 (resolver_item_dspace):
quando o item ainda não está disponível no DSpace, a API não devolve mais 502 — ela
enfileira essa task, que reconsulta o DSpace com backoff até os PDFs aparecerem e só
então dispara uma chain por PDF.
O enriquecimento por LLM é DESACOPLADO: NÃO faz parte da chain obrigatória. Ele roda
como follow-up OPCIONAL (enrich_after_index) disparado APÓS a indexação — ou sob
demanda pelo endpoint /api/files/enrich/{job_id} — e propaga os metadados ao Qdrant
via set_payload, sem re-embedar. Assim a indexação nunca espera nem depende do LLM.

A chain transporta SOMENTE um PipelineContext leve (identificadores + URI do
manifesto). Todo conteúdo (PDF, markdown, JSON MinerU, chunks, imagens, relatórios)
vive no MinIO; cada task abre o manifesto, baixa só o necessário, processa, grava os
novos artefatos e atualiza o manifesto. A lógica de cada etapa fica em
backend/services/pipeline_stages.py (reutilizada pelo fluxo legado).

Antes de retornar, cada task valida o payload de saída (tamanho + chaves proibidas)
via validate_chain_payload_size — barreira anti-regressão (§26/§27).

Semântica de erro (fiel ao pipeline original):
  - item indisponível       → espera na fila (backoff) e, esgotadas as tentativas,
                              status "erro" no registro `item:{uuid}`.
  - download/mineru falham  → status "erro" e a chain para.
  - index falha             → status "concluido" + index_error (o markdown já é válido).
  - enrich (fora da chain)  → best-effort: nunca altera o status já "concluido".
"""

import traceback
import urllib.error

from backend.celery_app import app
from backend.core import config as settings
from backend.core.logger import log
from backend.core.schemas import CTX_STAGE_INDEXED, PipelineContext, validate_chain_payload_size
from backend.services import ingest_service as ingest
from backend.services import pipeline_stages as stages
from backend.services.dspace_service import item_ainda_indisponivel, resolve_item_pdfs
from backend.services.job_store import add_failed, clear_failed, set_status


def _job_id_of(args, kwargs):
    """Extrai o job_id de uma task: kwarg explícito (baixar_dspace) ou ctx["job_id"]
    (primeiro posicional nas demais)."""
    if kwargs.get("job_id"):
        return kwargs["job_id"]
    if args and isinstance(args[0], dict):
        return args[0].get("job_id")
    return None


def _parse_context(ctx) -> PipelineContext:
    if isinstance(ctx, PipelineContext):
        return ctx
    return PipelineContext.model_validate(ctx)


def _finalize(payload: dict) -> dict:
    """Valida o payload de saída (tamanho + chaves proibidas) antes de devolvê-lo à
    chain. Só registra o TAMANHO — nunca o conteúdo (evita vazar dado sensível)."""
    size = validate_chain_payload_size(
        payload,
        settings.CELERY_CHAIN_MAX_PAYLOAD_BYTES,
        enforce_lightweight=settings.CELERY_CHAIN_ENFORCE_LIGHTWEIGHT_CONTEXT,
    )
    log.debug("[chain] payload de saída: %d bytes", size)
    return payload


def _status_extra(ctx: PipelineContext) -> dict:
    return {
        "pipeline_id": str(ctx.pipeline_id),
        "document_id": ctx.document_id,
        "artifact_manifest_uri": ctx.artifact_manifest_uri,
    }


class PipelineTask(app.Task):
    """Base das tasks: em falha não tratada, marca o job como "erro" (senão o status
    congelaria no último estágio e a chain pararia em silêncio)."""

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        job_id = _job_id_of(args, kwargs)
        if job_id:
            stage = self.name.rsplit(".", 1)[-1]
            set_status(job_id, "erro", stage=stage, error=f"{type(exc).__name__}: {exc}")
            add_failed(job_id)  # entra na fila de falhas (reprocessável)


# --------------------------------------------------------------------------
# Estágio 0 (só na ingestão de ITEM) — resolver os PDFs no DSpace (fila: download)
#
# Existe para o caso em que o item AINDA não está disponível para download (em
# submissão/workflow, embargo, DSpace fora do ar). Antes disso a API respondia 502 e
# o pedido se perdia; agora ela enfileira esta task, que tenta de novo com backoff
# (DSPACE_ITEM_RETRY_*) e só desiste ao esgotar as tentativas — aí o item vai para a
# fila de falhas. Cada tentativa é uma task nova (kwarg `attempt`), e não um retry do
# Celery, para o mesmo caminho servir à API e ao worker.
# --------------------------------------------------------------------------
@app.task(bind=True)
def resolver_item_dspace(self, item_uuid, force=False, attempt=1):
    key = ingest.item_job_key(item_uuid)
    set_status(
        key, "processando", stage=ingest.PENDING_STAGE, source=ingest.PENDING_SOURCE,
        item_uuid=item_uuid, force=force, attempt=attempt,
        max_attempts=settings.DSPACE_ITEM_RETRY_MAX_ATTEMPTS,
    )
    try:
        pdfs = resolve_item_pdfs(item_uuid)
    except Exception as exc:
        esgotou = attempt >= settings.DSPACE_ITEM_RETRY_MAX_ATTEMPTS
        if item_ainda_indisponivel(exc) and not esgotou:
            espera = ingest.schedule_item_resolution(
                item_uuid, force=force, attempt=attempt, error=f"{type(exc).__name__}: {exc}"
            )
            return {"item_uuid": item_uuid, "status": ingest.PENDING_STAGE, **espera}

        motivo = ("indisponível no DSpace após "
                  f"{attempt} tentativa(s)" if item_ainda_indisponivel(exc) else "erro definitivo")
        log.error("[dspace item] desistindo de %s (%s): %s", item_uuid, motivo, exc)
        set_status(
            key, "erro", stage=ingest.PENDING_STAGE, item_uuid=item_uuid, attempt=attempt,
            error=f"Item {motivo} — {type(exc).__name__}: {exc}",
            next_retry_at=None, next_retry_in_seconds=None,
        )
        add_failed(key)  # reenfileirável por POST /api/files/reprocess/item:{uuid}
        raise

    jobs = ingest.enqueue_item_pdfs(item_uuid, pdfs, force)
    set_status(
        key, ingest.RESOLVED_STATUS, stage=ingest.PENDING_STAGE, item_uuid=item_uuid,
        attempt=attempt, n_pdfs=len(jobs), job_ids=[j["job_id"] for j in jobs],
        error=None, last_error=None, next_retry_at=None, next_retry_in_seconds=None,
    )
    clear_failed(key)
    log.info("[dspace item] %s disponível na tentativa %d — %d PDF(s) enfileirado(s).",
             item_uuid, attempt, len(jobs))
    return {"item_uuid": item_uuid, "status": "na_fila", "jobs": len(jobs),
            "job_ids": [j["job_id"] for j in jobs]}


# --------------------------------------------------------------------------
# Estágio 1 — download do bitstream (fila: download)
# --------------------------------------------------------------------------
@app.task(bind=True, base=PipelineTask, max_retries=3, default_retry_delay=15)
def baixar_dspace(self, bs_uuid, filename, job_id, item_uuid="", item_handle="", force=False):
    set_status(
        job_id, "processando", stage="download", filename=filename,
        source="dspace-item", item_uuid=item_uuid, item_handle=item_handle,
        bitstream_uuid=bs_uuid,
    )
    try:
        ctx = stages.stage_download(
            bs_uuid=bs_uuid, filename=filename, job_id=job_id,
            item_uuid=item_uuid, item_handle=item_handle, force=force,
        )
    except (urllib.error.HTTPError, urllib.error.URLError) as e:
        try:
            raise self.retry(exc=e)  # transiente → backoff/retry
        except self.MaxRetriesExceededError:
            set_status(job_id, "erro", stage="download", error=f"download falhou: {e}")
            raise
    except ValueError as e:  # conteúdo não é PDF → retry não ajuda
        set_status(job_id, "erro", stage="download", error=str(e))
        raise

    set_status(job_id, "processando", stage="download", **_status_extra(ctx))
    return _finalize(ctx.to_message())


# --------------------------------------------------------------------------
# Estágio 2 — extração MinerU (fila: extract)
# --------------------------------------------------------------------------
@app.task(bind=True, base=PipelineTask)
def extrair_mineru(self, ctx):
    ctx = _parse_context(ctx)
    set_status(ctx.job_id, "processando", stage="mineru", **_status_extra(ctx))
    try:
        ctx = stages.stage_mineru(ctx, task_id=self.request.id)
    except Exception as exc:
        set_status(ctx.job_id, "erro", stage="mineru", error=f"{type(exc).__name__}: {exc}")
        _record_stage_failure(ctx, "mineru", exc)
        raise
    return _finalize(ctx.to_message())


# --------------------------------------------------------------------------
# Enriquecimento por LLM (fila: llm) — DESACOPLADO, best-effort
#
# enrich_llm continua disponível para quem quiser encadear o enrich ANTES da
# indexação (recebe/retorna PipelineContext). NÃO é mais parte da chain obrigatória.
# --------------------------------------------------------------------------
@app.task(base=PipelineTask)
def enrich_llm(ctx):
    ctx = _parse_context(ctx)
    set_status(ctx.job_id, "processando", stage="llm", **_status_extra(ctx))
    # stage_enrich é best-effort: trata erros internamente e SEMPRE segue.
    ctx = stages.stage_enrich(ctx)
    if ctx.warnings_count:
        set_status(ctx.job_id, "processando", stage="llm", warnings_count=ctx.warnings_count)
    return _finalize(ctx.to_message())


@app.task
def enrich_after_index(summary):
    """Follow-up DESACOPLADO do enrich, disparado APÓS a indexação (fila: llm).

    Recebe o `summary` retornado por indexar_qdrant (que contém a URI do manifesto),
    reconstrói o contexto a partir do manifesto e roda o enrich, que propaga os
    metadados ao Qdrant (set_payload). É totalmente best-effort: NUNCA altera o
    status já "concluido" do job nem quebra nada — o índice já está pronto e é
    autoritativo antes deste follow-up rodar. Retorna o `summary` intacto."""
    try:
        manifest_uri = (summary or {}).get("artifact_manifest_uri")
        if not manifest_uri:
            log.warning("[enrich follow-up] sem artifact_manifest_uri no summary — pulando enrich.")
            return summary
        ctx = stages.build_context_from_manifest(manifest_uri, stage=CTX_STAGE_INDEXED)
        ctx = stages.stage_enrich(ctx)
        if ctx.warnings_count:
            set_status(ctx.job_id, "concluido", stage="llm", warnings_count=ctx.warnings_count)
    except Exception as exc:  # pragma: no cover — barreira best-effort
        log.warning("[enrich follow-up] enrich desacoplado falhou (best-effort): %s", exc)
    return summary


# --------------------------------------------------------------------------
# Indexação — chunking + embedding + upsert no Qdrant (fila: gpu, concurrency=1)
# --------------------------------------------------------------------------
@app.task(bind=True, base=PipelineTask)
def indexar_qdrant(self, ctx):
    ctx = _parse_context(ctx)
    set_status(ctx.job_id, "processando", stage="index", **_status_extra(ctx))
    try:
        summary = stages.stage_index(ctx, task_id=self.request.id)
        # Índice vazio com blocos estruturais não é sucesso: vai para a fila de falhas
        # (com index_error) para aparecer em /failures e poder ser reprocessado.
        empty = summary.get("status") == "empty"
        set_status(
            ctx.job_id, "concluido", stage="index",
            n_chunks=summary.get("chunk_count"),
            indexed_count=summary.get("indexed_count"),
            # limpa erro de índice de uma tentativa anterior (reprocess)
            index_error=summary.get("index_warning") if empty else None,
            **_status_extra(ctx),
        )
        if empty:
            log.error("Indexação vazia para %s: %s", ctx.job_id, summary.get("index_warning"))
            add_failed(ctx.job_id)
        else:
            clear_failed(ctx.job_id)  # indexou com sucesso → sai da fila de falhas
            log.info("Pipeline concluído para %s.", ctx.job_id)
        return _finalize(summary)
    except Exception as exc:
        log.error("Indexação falhou para %s: %s", ctx.job_id, exc)
        log.error(traceback.format_exc())
        _record_stage_failure(ctx, "indexing", exc)
        # Não re-levanta (markdown já é válido) → on_failure não dispara; entra na
        # fila de falhas explicitamente para permitir reprocessar só a indexação.
        add_failed(ctx.job_id)
        # Markdown/artefatos já são válidos — conclui com aviso de índice
        # (fiel ao comportamento original), sem retornar conteúdo grande.
        set_status(ctx.job_id, "concluido", stage="index", index_error=str(exc), **_status_extra(ctx))
        return _finalize({
            "pipeline_id": str(ctx.pipeline_id),
            "document_id": ctx.document_id,
            "status": "completed_with_index_error",
            "index_error": str(exc)[:300],
            "artifact_manifest_uri": ctx.artifact_manifest_uri,
        })


def _record_stage_failure(ctx: PipelineContext, stage: str, exc: Exception) -> None:
    """Registra a falha no manifesto (mensagem limitada, sem stack trace)."""
    try:
        from backend.services.manifest_repository import get_manifest_repository

        get_manifest_repository().fail_stage(
            str(ctx.pipeline_id), ctx.document_id, stage,
            error_type=type(exc).__name__, message=str(exc),
        )
    except Exception as inner:  # pragma: no cover
        log.warning("[chain] não foi possível registrar falha no manifesto: %s", inner)
