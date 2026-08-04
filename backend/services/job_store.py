"""Registro de status dos jobs de ingestão.

Com o Celery, o processo da API (produtor) e os workers (consumidores) são
processos distintos — um dict em memória não é compartilhado entre eles. Por isso
o status vive no Redis (DB separado do broker), com TTL. Se o Redis estiver
indisponível, cai para um dict em memória por-processo (degradação graciosa: a API
não quebra, mas o status entre processos deixa de ser compartilhado até o Redis
voltar). A consulta de markdown continua no filesystem (fonte de verdade do MinerU).
"""

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from backend.core.config import JOBSTORE_REDIS_URL, JOBSTORE_TTL, OUTPUT_DIR
from backend.core.logger import log

_KEY_PREFIX = "job:"
# Índice (fila leve) de jobs que precisam de atenção/reprocessamento: sorted set
# score=epoch, para listar do mais recente. Não é um DLQ de broker — é um registro
# consultável (GET /failures) que alimenta o reprocessamento manual (POST /reprocess).
_FAILED_ZSET = "jobs:failed"
# Índice dos jobs EM EXECUÇÃO (na fila ou processando): sorted set com score=epoch da
# entrada (não do último update), para listar do mais recente. Mantido pelo set_status:
# o job entra ao ficar na_fila/processando e sai ao concluir ou falhar. Sem isso a única
# forma de listar os ativos seria varrer todas as chaves job:* do Redis.
_ACTIVE_ZSET = "jobs:active"
_ACTIVE_STATUSES = frozenset({"na_fila", "processando"})
# Índice dos jobs BEM SUCEDIDOS: sorted set com score=epoch da conclusão, para listar
# os últimos. "Bem sucedido" é `concluido` SEM `index_error` — um job que extraiu mas
# falhou ao indexar também é `concluido` (com index_error) e pertence à fila de falhas,
# não a este índice. Re-enfileirar um job o remove daqui até concluir de novo.
_SUCCEEDED_ZSET = "jobs:succeeded"

# Fallback em memória (usado só se o Redis não responder).
_lock = threading.Lock()
_jobs: dict[str, dict] = {}
_failed: dict[str, float] = {}
_active: dict[str, float] = {}
_succeeded: dict[str, float] = {}

_redis = None
_redis_ready = False


def _get_redis():
    """Cliente Redis lazy. Retorna None se indisponível (aciona o fallback in-memory)."""
    global _redis, _redis_ready
    if _redis_ready:
        return _redis
    try:
        import redis  # import tardio: dependência opcional em dev sem Redis

        client = redis.Redis.from_url(JOBSTORE_REDIS_URL, decode_responses=True)
        client.ping()
        _redis = client
        log.info("job_store: usando Redis em %s", JOBSTORE_REDIS_URL)
    except Exception as exc:
        _redis = None
        log.warning("job_store: Redis indisponível (%s) — usando fallback em memória.", exc)
    _redis_ready = True
    return _redis


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def set_status(job_id: str, status: str, **extra) -> None:
    """Cria/atualiza (merge) o registro do job. `extra` são campos livres
    (filename, stage, item_uuid, n_chunks, error, ...)."""
    job = None
    client = _get_redis()
    if client is not None:
        try:
            key = _KEY_PREFIX + job_id
            raw = client.get(key)
            job = json.loads(raw) if raw else {"job_id": job_id}
            job["status"] = status
            job["updated_at"] = _now()
            job.update(extra)
            client.set(key, json.dumps(job, ensure_ascii=False), ex=JOBSTORE_TTL)
        except Exception as exc:
            job = None
            log.warning("job_store.set_status: falha no Redis para '%s' (%s) — fallback memória.", job_id, exc)

    if job is None:
        with _lock:
            job = _jobs.setdefault(job_id, {"job_id": job_id})
            job["status"] = status
            job["updated_at"] = _now()
            job.update(extra)
            job = dict(job)

    # Índices derivados a partir do registro MESCLADO: `extra` sozinho não basta, porque
    # um `index_error` gravado antes continua no registro (ex.: o follow-up de enrich
    # reescreve status=concluido sem repassar index_error).
    _track_active(job_id, _is_active(job))
    _track_succeeded(job_id, _is_succeeded(job))


def _is_active(job: dict) -> bool:
    return job.get("status") in _ACTIVE_STATUSES


def _is_succeeded(job: dict) -> bool:
    return job.get("status") == "concluido" and not job.get("index_error")


def _track_active(job_id: str, active: bool) -> None:
    """Mantém o índice de jobs em execução: entra em na_fila/processando (preservando o
    instante da primeira entrada) e sai em qualquer status terminal (concluido/erro)."""
    _update_index(_ACTIVE_ZSET, _active, job_id, active, keep_first=True, caller="_track_active")


def _track_succeeded(job_id: str, succeeded: bool) -> None:
    """Mantém o índice de jobs bem sucedidos: entra ao concluir sem erro de índice
    (score = instante da conclusão) e sai ao ser re-enfileirado ou ao falhar."""
    _update_index(_SUCCEEDED_ZSET, _succeeded, job_id, succeeded, keep_first=False, caller="_track_succeeded")


def _update_index(zset: str, mem_index: dict[str, float], job_id: str, member: bool,
                  *, keep_first: bool, caller: str) -> None:
    """Adiciona/remove o job de um índice. `keep_first` preserva o score da primeira
    entrada (idade); sem ele o score é atualizado a cada chamada (recência)."""
    now = time.time()
    client = _get_redis()
    if client is not None:
        try:
            if not member:
                client.zrem(zset, job_id)
            elif keep_first:
                client.zadd(zset, {job_id: now}, nx=True)
            else:
                client.zadd(zset, {job_id: now})
            return
        except Exception as exc:
            log.warning("job_store.%s: falha no Redis para '%s' (%s) — fallback memória.", caller, job_id, exc)
    with _lock:
        if not member:
            mem_index.pop(job_id, None)
        elif keep_first:
            mem_index.setdefault(job_id, now)
        else:
            mem_index[job_id] = now


def get_job(job_id: str) -> dict | None:
    client = _get_redis()
    if client is not None:
        try:
            raw = client.get(_KEY_PREFIX + job_id)
            return json.loads(raw) if raw else None
        except Exception as exc:
            log.warning("job_store.get_job: falha no Redis para '%s' (%s) — fallback memória.", job_id, exc)

    with _lock:
        job = _jobs.get(job_id)
        return dict(job) if job else None


def add_failed(job_id: str) -> None:
    """Adiciona o job ao índice de falhas (idempotente). Chamado quando um estágio
    falha ou conclui com erro de índice — sinaliza que precisa de reprocessamento."""
    client = _get_redis()
    if client is not None:
        try:
            client.zadd(_FAILED_ZSET, {job_id: time.time()})
            return
        except Exception as exc:
            log.warning("job_store.add_failed: falha no Redis para '%s' (%s) — fallback memória.", job_id, exc)
    with _lock:
        _failed[job_id] = time.time()


def clear_failed(job_id: str) -> None:
    """Remove o job do índice de falhas (após sucesso ou ao re-enfileirar)."""
    client = _get_redis()
    if client is not None:
        try:
            client.zrem(_FAILED_ZSET, job_id)
            return
        except Exception as exc:
            log.warning("job_store.clear_failed: falha no Redis para '%s' (%s) — fallback memória.", job_id, exc)
    with _lock:
        _failed.pop(job_id, None)


def _index_ids(zset: str, mem_index: dict[str, float], limit: int, caller: str) -> list[str]:
    """IDs de um índice de jobs (sorted set), do mais recente para o mais antigo."""
    client = _get_redis()
    if client is not None:
        try:
            return client.zrevrange(zset, 0, limit - 1)
        except Exception as exc:
            log.warning("job_store.%s: falha no Redis (%s) — fallback memória.", caller, exc)
    with _lock:
        return [k for k, _ in sorted(mem_index.items(), key=lambda kv: kv[1], reverse=True)][:limit]


def list_failed(limit: int = 100) -> list[dict]:
    """Lista os jobs no índice de falhas (mais recentes primeiro), com o registro
    completo de cada um. Jobs cujo registro já expirou (TTL) são podados do índice."""
    limit = max(1, limit)
    out: list[dict] = []
    for jid in _index_ids(_FAILED_ZSET, _failed, limit, "list_failed"):
        job = get_job(jid)
        if job is None:  # registro expirou → poda o índice
            clear_failed(jid)
            continue
        out.append(job)
    return out


def clear_active(job_id: str) -> None:
    """Remove o job do índice de execução (poda de entradas órfãs)."""
    _track_active(job_id, False)


def clear_succeeded(job_id: str) -> None:
    """Remove o job do índice de sucesso (poda de entradas órfãs)."""
    _track_succeeded(job_id, False)


def list_active(limit: int = 100) -> list[dict]:
    """Lista os jobs EM EXECUÇÃO — na fila ou processando — do mais recente para o mais
    antigo, com o registro completo de cada um. Poda do índice as entradas cujo registro
    expirou (TTL) e as que já não estão num status ativo (concluído/erro)."""
    return _list_indexed(_ACTIVE_ZSET, _active, limit, "list_active", _is_active, clear_active)


def list_succeeded(limit: int = 100) -> list[dict]:
    """Lista os **últimos jobs bem sucedidos** (concluídos sem `index_error`), do mais
    recente para o mais antigo, com o registro completo de cada um. Poda do índice as
    entradas cujo registro expirou (TTL) e as que já não estão em sucesso (ex.: job
    re-enfileirado ou reprocessado com erro)."""
    return _list_indexed(_SUCCEEDED_ZSET, _succeeded, limit, "list_succeeded", _is_succeeded, clear_succeeded)


def _list_indexed(zset, mem_index, limit, caller, is_valid, prune) -> list[dict]:
    """Resolve um índice de jobs em registros completos, podando as entradas órfãs
    (registro expirado por TTL) e as que já não satisfazem `is_valid`."""
    limit = max(1, limit)
    out: list[dict] = []
    for jid in _index_ids(zset, mem_index, limit, caller):
        job = get_job(jid)
        if job is None or not is_valid(job):
            prune(jid)
            continue
        out.append(job)
    return out


def find_markdown(job_id: str) -> Path | None:
    """Localiza o .md de saída do MinerU (output/<job_id>/<método>/<job_id>.md)."""
    doc_dir = OUTPUT_DIR / job_id
    if not doc_dir.exists():
        return None
    exact = list(doc_dir.rglob(f"{job_id}.md"))
    if exact:
        return exact[0]
    others = sorted(doc_dir.rglob("*.md"))
    return others[0] if others else None


def markdown_url(md_path: Path) -> str:
    return f"/output/{md_path.relative_to(OUTPUT_DIR).as_posix()}"
