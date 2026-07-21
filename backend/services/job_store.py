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
from datetime import datetime, timezone
from pathlib import Path

from backend.core.config import JOBSTORE_REDIS_URL, JOBSTORE_TTL, OUTPUT_DIR
from backend.core.logger import log

_KEY_PREFIX = "job:"

# Fallback em memória (usado só se o Redis não responder).
_lock = threading.Lock()
_jobs: dict[str, dict] = {}

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
