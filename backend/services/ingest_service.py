"""Enfileiramento da ingestão — lógica compartilhada entre a API e os workers.

Isto não vive em backend/api/routes/files.py porque a task `resolver_item_dspace`
(backend/tasks.py) precisa do MESMO enfileiramento: quando o item ainda não está
disponível no DSpace, quem resolve os PDFs e dispara as chains é o worker, não a
requisição HTTP. O import de `backend.tasks` é TARDIO (dentro das funções) para não
fechar ciclo — backend.tasks importa este módulo no topo.

Acompanhamento do ITEM (não do documento): enquanto o DSpace não devolve os PDFs, não
existe job por PDF para registrar status. Por isso o item ganha um registro próprio no
job_store, com a chave `item:{uuid}` (ver `item_job_key`) e `source=dspace-item-pending`
— ele aparece em GET /api/files/active com stage `aguardando_dspace` e, se as tentativas
se esgotarem, em GET /api/files/failures. Quando os PDFs finalmente saem, esse registro
vira `resolvido` (status terminal PRÓPRIO: não é `concluido`, para não poluir
GET /api/files/succeeded, que é a lista de documentos indexados) e o acompanhamento
passa a ser feito job a job.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path

from backend.core import config as settings
from backend.core.logger import log
from backend.services import llm_enrich_service as llm_enrich
from backend.services.job_store import clear_failed, set_status

# Prefixo da chave de acompanhamento do item (o `{uuid}` é o do ITEM, não do bitstream).
ITEM_KEY_PREFIX = "item:"
# Marca o registro como acompanhamento de item pendente (e não um documento).
PENDING_SOURCE = "dspace-item-pending"
# Estágio exibido em /active e /failures enquanto o item não sai do DSpace.
PENDING_STAGE = "aguardando_dspace"
# Status terminal do registro do item quando os PDFs foram resolvidos e enfileirados.
RESOLVED_STATUS = "resolvido"


def item_job_key(item_uuid: str) -> str:
    """Chave do registro de acompanhamento do ITEM no job_store."""
    return f"{ITEM_KEY_PREFIX}{item_uuid}"


def is_pending_item(job: dict | None) -> bool:
    """True se o registro é um acompanhamento de item pendente (não um documento)."""
    return bool(job) and job.get("source") == PENDING_SOURCE


def retry_delay_seconds(attempt: int) -> int:
    """Intervalo até a próxima tentativa, dado o número de tentativas JÁ feitas
    (1 = a primeira falhou). Backoff exponencial com teto:
    delay = min(DELAY * BACKOFF^(attempt-1), MAX_DELAY)."""
    fator = settings.DSPACE_ITEM_RETRY_BACKOFF ** max(0, attempt - 1)
    delay = settings.DSPACE_ITEM_RETRY_DELAY_SECONDS * fator
    return max(1, int(min(delay, settings.DSPACE_ITEM_RETRY_MAX_DELAY_SECONDS)))


# --------------------------------------------------------------------------
# Chain por documento (um PDF = um job)
# --------------------------------------------------------------------------
def build_chain(bs_uuid, filename, job_id, item_uuid, item_handle, force):
    """Cadeia OBRIGATÓRIA (3 estágios) — o download roda no worker e grava no MinIO.
    O enrich NÃO entra aqui: é anexado como follow-up opcional em `enqueue_chain`."""
    from celery import chain

    from backend.tasks import baixar_dspace, extrair_mineru, indexar_qdrant

    return chain(
        baixar_dspace.s(bs_uuid, filename, job_id=job_id, item_uuid=item_uuid,
                        item_handle=item_handle, force=force),
        extrair_mineru.s(),
        indexar_qdrant.s(),
    )


def enqueue_chain(bs_uuid, filename, job_id, item_uuid, item_handle, force):
    """Enfileira a chain obrigatória e, quando o enrich está habilitado e há provedor
    LLM configurado, anexa enrich_after_index como follow-up DESACOPLADO (link) que
    roda APÓS a indexação — o índice nunca espera nem depende do LLM."""
    from backend.tasks import enrich_after_index

    clear_failed(job_id)  # (re)enfileirar supera uma falha anterior
    sig = build_chain(bs_uuid, filename, job_id, item_uuid, item_handle, force)
    link = None
    if settings.LLM_ENRICH_AUTO and llm_enrich.is_available():
        link = enrich_after_index.s()
    sig.apply_async(link=link)


def enqueue_item_pdfs(item_uuid: str, pdfs: list[dict], force: bool) -> list[dict]:
    """Registra e enfileira um job por PDF do item. Devolve o resumo de cada job
    (o mesmo conteúdo de `jobs` na resposta do endpoint de ingestão de item)."""
    jobs = []
    for pdf in pdfs:
        bs_uuid = pdf["bitstream_uuid"]
        filename = pdf["filename"]
        item_handle = pdf.get("item_handle") or ""
        job_id = Path(filename).stem

        set_status(
            job_id, "na_fila", filename=filename,
            source="dspace-item", item_uuid=item_uuid, item_handle=item_handle,
            bitstream_uuid=bs_uuid,
        )
        enqueue_chain(bs_uuid, filename, job_id, item_uuid, item_handle, force)

        jobs.append({
            "job_id": job_id,
            "filename": filename,
            "bitstream_uuid": bs_uuid,
            "status_url": f"/api/files/status/{job_id}",
            "result_url": f"/api/files/result/{job_id}",
        })
    return jobs


# --------------------------------------------------------------------------
# Item ainda indisponível: espera na fila em vez de 502
# --------------------------------------------------------------------------
def schedule_item_resolution(item_uuid: str, *, force: bool, attempt: int, error: str = "",
                             delay: int | None = None) -> dict:
    """Agenda mais uma tentativa de resolver os PDFs do item e atualiza o registro de
    acompanhamento. `attempt` é o número de tentativas JÁ feitas — a primeira
    normalmente é a que a própria requisição HTTP acabou de fazer. `delay` sobrescreve
    o backoff (0 = tentar já; usado pelo reprocessamento manual).

    Devolve o resumo do agendamento (entra na resposta 202 do endpoint)."""
    from backend.tasks import resolver_item_dspace

    proxima = attempt + 1
    delay = retry_delay_seconds(attempt) if delay is None else max(0, int(delay))
    key = item_job_key(item_uuid)
    next_retry_at = (datetime.now(timezone.utc) + timedelta(seconds=delay)).isoformat()

    set_status(
        key, "na_fila", stage=PENDING_STAGE, source=PENDING_SOURCE,
        item_uuid=item_uuid, force=force,
        attempt=attempt, next_attempt=proxima,
        max_attempts=settings.DSPACE_ITEM_RETRY_MAX_ATTEMPTS,
        next_retry_in_seconds=delay, next_retry_at=next_retry_at,
        last_error=error[:300] or None,
    )
    clear_failed(key)  # reenfileirar supera uma desistência anterior
    resolver_item_dspace.apply_async(
        kwargs={"item_uuid": item_uuid, "force": force, "attempt": proxima},
        countdown=delay,
    )
    log.info("[dspace item] %s indisponível (tentativa %d/%d) — nova tentativa em %ds.",
             item_uuid, attempt, settings.DSPACE_ITEM_RETRY_MAX_ATTEMPTS, delay)
    return {
        "attempt": attempt,
        "next_attempt": proxima,
        "max_attempts": settings.DSPACE_ITEM_RETRY_MAX_ATTEMPTS,
        "next_retry_in_seconds": delay,
        "next_retry_at": next_retry_at,
    }


def item_wait_is_alive(job: dict | None) -> bool:
    """True se já existe uma espera EM ANDAMENTO para o item — evita duplicar a
    resolução (e, depois, as chains) quando o mesmo POST é repetido.

    Vale tanto para a espera agendada (`na_fila`) quanto para a tentativa em curso
    (`processando`). Uma espera cujo `next_retry_at` já passou há mais de um intervalo
    máximo é considerada órfã (worker reiniciado, mensagem perdida) e NÃO segura o
    reenfileiramento.
    """
    if not is_pending_item(job) or job.get("status") not in {"na_fila", "processando"}:
        return False
    try:
        previsto = datetime.fromisoformat(job.get("next_retry_at") or "")
    except ValueError:
        return False
    if previsto.tzinfo is None:
        previsto = previsto.replace(tzinfo=timezone.utc)
    limite = previsto + timedelta(seconds=settings.DSPACE_ITEM_RETRY_MAX_DELAY_SECONDS)
    return datetime.now(timezone.utc) <= limite


def item_wait_summary(job: dict) -> dict:
    """Resumo da espera em andamento (mesma forma do retorno de schedule_item_resolution)."""
    return {
        "attempt": job.get("attempt"),
        "next_attempt": job.get("next_attempt"),
        "max_attempts": job.get("max_attempts", settings.DSPACE_ITEM_RETRY_MAX_ATTEMPTS),
        "next_retry_in_seconds": job.get("next_retry_in_seconds"),
        "next_retry_at": job.get("next_retry_at"),
    }
