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

# Fallback em memória (usado só se o Redis não responder).
_lock = threading.Lock()
_jobs: dict[str, dict] = {}
_failed: dict[str, float] = {}

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
            return
        except Exception as exc:
            log.warning("job_store.set_status: falha no Redis para '%s' (%s) — fallback memória.", job_id, exc)

    with _lock:
        job = _jobs.setdefault(job_id, {"job_id": job_id})
        job["status"] = status
        job["updated_at"] = _now()
        job.update(extra)


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


def list_failed(limit: int = 100) -> list[dict]:
    """Lista os jobs no índice de falhas (mais recentes primeiro), com o registro
    completo de cada um. Jobs cujo registro já expirou (TTL) são podados do índice."""
    limit = max(1, limit)
    client = _get_redis()
    if client is not None:
        try:
            ids = client.zrevrange(_FAILED_ZSET, 0, limit - 1)
        except Exception as exc:
            log.warning("job_store.list_failed: falha no Redis (%s) — fallback memória.", exc)
            ids = None
    else:
        ids = None
    if ids is None:
        with _lock:
            ids = [k for k, _ in sorted(_failed.items(), key=lambda kv: kv[1], reverse=True)][:limit]

    out: list[dict] = []
    for jid in ids:
        job = get_job(jid)
        if job is None:  # registro expirou → poda o índice
            clear_failed(jid)
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
