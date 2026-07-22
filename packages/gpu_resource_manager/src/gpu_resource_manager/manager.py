"""GPUResourceManager — API pública estável da biblioteca.

Coordena exclusão mútua distribuída, fila global com prioridade + FIFO + aging,
lease com TTL/heartbeat e recuperação após falhas. Independente de Celery, FastAPI,
PyTorch, MinerU, BGE-M3 e Qdrant — conhece apenas Redis e o protocolo de coordenação.
"""

from __future__ import annotations

import json
import logging
import random
import time
from contextlib import contextmanager
from typing import Any, Iterator, Optional

from .config import GPUManagerConfig
from .exceptions import (
    GPUAcquisitionTimeout,
    GPUManagerError,
    GPURequestCancelled,
)
from .lease import GPULease
from .models import Events, GPURequest, OwnerInfo, utc_now_iso
from .queue import effective_priority
from .redis_backend import RedisBackend
from .status import approximate_position, read_queue, read_status

log = logging.getLogger("gpu_resource_manager")


class GPUResourceManager:
    """Ponto de entrada para adquirir/consultar recursos de GPU."""

    def __init__(
        self, config: GPUManagerConfig, *, backend: Optional[RedisBackend] = None
    ) -> None:
        self.config = config
        self.backend = backend if backend is not None else RedisBackend(config)

    # ------------------------------------------------------------------ #
    # Aquisição
    # ------------------------------------------------------------------ #
    @contextmanager
    def acquire(
        self,
        *,
        resource: Optional[str] = None,
        service: str,
        priority: Optional[int] = None,
        task_id: Optional[str] = None,
        document_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        wait_timeout_seconds: Optional[float] = None,
    ) -> Iterator[GPULease]:
        """Context manager síncrono: enfileira, aguarda a vez, adquire o lock,
        inicia o heartbeat e devolve o :class:`GPULease`. Libera ao sair (mesmo em
        exceção). Ver ``acquire_lease()``/``release_lease()`` para uso sem ``with``."""
        lease = self.acquire_lease(
            resource=resource, service=service, priority=priority,
            task_id=task_id, document_id=document_id, metadata=metadata,
            wait_timeout_seconds=wait_timeout_seconds,
        )
        try:
            yield lease
        finally:
            lease.release()

    def acquire_lease(
        self,
        *,
        resource: Optional[str] = None,
        service: str,
        priority: Optional[int] = None,
        task_id: Optional[str] = None,
        document_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        wait_timeout_seconds: Optional[float] = None,
    ) -> GPULease:
        """Versão imperativa de ``acquire`` (o chamador deve chamar ``lease.release()``)."""
        cfg = self.config
        res = cfg.resource(resource)
        prio = cfg.clamp_priority(priority)
        timeout = cfg.wait_timeout_seconds if wait_timeout_seconds is None else wait_timeout_seconds

        # Fail-closed: sem Redis, ninguém usa a GPU.
        self.backend.ensure_available()

        request = GPURequest.create(
            resource=res, service=service, priority=prio,
            task_id=task_id, document_id=document_id, metadata=metadata,
        )
        self._emit(Events.REQUEST_CREATED, request=request)
        log.info(
            "gpu request criado resource=%s service=%s priority=%d request_id=%s task_id=%s document_id=%s",
            res, service, prio, request.request_id, task_id, document_id,
        )

        seq = self.backend.enqueue(request, request_ttl_ms=cfg.request_ttl_seconds * 1000)
        request.sequence = seq
        self._emit(Events.REQUEST_WAITING, request=request)

        token = _new_token()
        started = time.monotonic()
        last_wait_log = started
        last_req_hb = started
        attempts = 0

        while True:
            attempts += 1
            now = time.time()
            code, detail, position = self.backend.try_acquire(
                resource=res, request_id=request.request_id, token=token, now=now,
                lock_ttl_ms=cfg.lock_ttl_seconds * 1000,
                owner_json=self._owner_json(request, token),
            )

            if code == 1:
                wait_time = time.monotonic() - started
                queue_size = len(self.backend.read_queue_raw(res, limit=1000))
                eff = effective_priority(cfg, prio, request.enqueued_at, now)
                lease = GPULease(
                    backend=self.backend, config=cfg, request=request, token=token,
                    wait_time_seconds=wait_time, acquisition_attempts=attempts,
                    queue_size_at_acquisition=queue_size, effective_priority=eff,
                )
                lease.start_heartbeat()
                log.info(
                    "gpu adquirido resource=%s service=%s request_id=%s token=%s "
                    "wait_time_seconds=%.2f attempts=%d effective_priority=%d",
                    res, service, request.request_id, token, wait_time, attempts, eff,
                )
                self._emit(Events.LOCK_ACQUIRED, request=request,
                           wait_time_seconds=round(wait_time, 3), attempts=attempts,
                           effective_priority=eff)
                return lease

            if code == -1:
                # A solicitação sumiu da fila (TTL expirou / cancelada / órfã removida).
                raise GPURequestCancelled(
                    "solicitação removida da fila antes de adquirir (órfã/cancelada)",
                    resource=res, request_id=request.request_id,
                )

            # code == 0: aguardando (não é o próximo, ou lock ocupado).
            elapsed = time.monotonic() - started
            if elapsed >= timeout:
                self._on_timeout(request, token, elapsed, res)

            # Heartbeat da solicitação em espera (mantém a request key viva).
            mono = time.monotonic()
            if mono - last_req_hb >= cfg.request_heartbeat_seconds:
                last_req_hb = mono
                alive = self.backend.wait_heartbeat(
                    resource=res, request_id=request.request_id,
                    ttl_ms=cfg.request_ttl_seconds * 1000,
                )
                if alive != 1:
                    # Perdemos a request key (limpeza concorrente): re-enfileira.
                    log.warning(
                        "gpu espera: request key expirou — re-enfileirando resource=%s request_id=%s",
                        res, request.request_id,
                    )
                    request.enqueued_at = time.time()
                    seq = self.backend.enqueue(
                        request, request_ttl_ms=cfg.request_ttl_seconds * 1000
                    )
                    request.sequence = seq

            # Log periódico de espera.
            if mono - last_wait_log >= cfg.wait_log_interval_seconds:
                last_wait_log = mono
                self._log_waiting(request, res, elapsed, position, detail)

            self._sleep_with_jitter()

    # ------------------------------------------------------------------ #
    def _on_timeout(self, request: GPURequest, token: str, elapsed: float, res: str) -> None:
        # Remove a solicitação (fila + metadados) e informa o estado atual.
        owner = None
        ttl = None
        position = None
        try:
            self.backend.cancel(resource=res, request_id=request.request_id)
            owner = self.backend.read_owner(res)
            _, ttl = self.backend.read_lock(res)
            position = approximate_position(self.backend, self.config, res, request.request_id)
        except Exception:  # pragma: no cover - não mascarar o timeout
            pass
        log.error(
            "gpu TIMEOUT resource=%s service=%s request_id=%s priority=%d wait_time_seconds=%.1f",
            res, request.service, request.request_id, request.priority, elapsed,
        )
        self._emit(Events.LOCK_TIMEOUT, request=request, wait_time_seconds=round(elapsed, 3))
        raise GPUAcquisitionTimeout(
            f"timeout ({elapsed:.0f}s) aguardando o recurso '{res}'",
            resource=res, service=request.service, request_id=request.request_id,
            priority=request.priority, wait_time_seconds=elapsed,
            current_owner=owner, lock_ttl_seconds=ttl, queue_position=position,
        )

    def _log_waiting(self, request, res, elapsed, position, detail) -> None:
        owner = None
        ttl = None
        try:
            owner = self.backend.read_owner(res)
            _, ttl = self.backend.read_lock(res)
        except GPUManagerError:  # pragma: no cover
            pass
        eff = effective_priority(self.config, request.priority, request.enqueued_at)
        log.info(
            "gpu aguardando resource=%s service=%s request_id=%s priority=%d effective_priority=%d "
            "position~=%s wait_time_seconds=%.0f reason=%s owner_service=%s lock_ttl_seconds=%s",
            res, request.service, request.request_id, request.priority, eff,
            position, elapsed, detail,
            (owner or {}).get("service"), round(ttl, 1) if ttl else None,
        )
        self._emit(Events.REQUEST_WAITING, request=request,
                   wait_time_seconds=round(elapsed, 3), position=position,
                   effective_priority=eff)

    def _sleep_with_jitter(self) -> None:
        base = self.config.poll_interval_seconds
        jitter = random.uniform(0, self.config.poll_jitter_seconds) if self.config.poll_jitter_seconds else 0.0
        time.sleep(base + jitter)

    # ------------------------------------------------------------------ #
    # Consulta / administração
    # ------------------------------------------------------------------ #
    def get_status(self, *, resource: Optional[str] = None):
        res = self.config.resource(resource)
        return read_status(self.backend, self.config, res)

    def get_queue(self, *, resource: Optional[str] = None, limit: int = 100):
        res = self.config.resource(resource)
        return read_queue(self.backend, self.config, res, limit=limit)

    def cancel_request(self, request_id: str, *, resource: Optional[str] = None) -> bool:
        """Remove uma solicitação da fila (idempotente). NÃO libera lock já adquirido.

        Se ``resource`` não for informado, usa o recurso padrão. (Para achar o
        recurso automaticamente, o chamador pode iterar seus recursos conhecidos.)
        """
        res = self.config.resource(resource)
        removed = self.backend.cancel(resource=res, request_id=request_id) == 1
        if removed:
            log.info("gpu request cancelada resource=%s request_id=%s", res, request_id)
            cb = self.config.on_event
            if cb is not None:
                try:
                    cb(Events.REQUEST_CANCELLED, {"resource": res, "request_id": request_id})
                except Exception:  # pragma: no cover
                    log.exception("on_event(request_cancelled) lançou exceção")
        return removed

    def healthcheck(self) -> dict[str, Any]:
        """Verifica o backend. Lança GPUBackendUnavailable se o Redis não responder."""
        self.backend.healthcheck()
        return {
            "status": "ok",
            "backend": "redis",
            "resource_default": self.config.resource_name,
            "checked_at": utc_now_iso(),
        }

    def close(self) -> None:
        self.backend.close()

    # ------------------------------------------------------------------ #
    def _owner_json(self, request: GPURequest, token: str) -> str:
        owner = OwnerInfo(
            token=token, request_id=request.request_id, resource=request.resource,
            service=request.service, priority=request.priority,
            hostname=request.hostname, pid=request.pid,
            acquired_at=utc_now_iso(), last_heartbeat=utc_now_iso(),
            task_id=request.task_id, document_id=request.document_id,
            metadata=request.metadata,
        )
        return json.dumps(owner.to_dict(), ensure_ascii=False)

    def _emit(self, event: str, *, request: Optional[GPURequest] = None, **extra: Any) -> None:
        cb = self.config.on_event
        if cb is None:
            return
        payload: dict[str, Any] = dict(extra)
        if request is not None:
            payload.update({
                "resource": request.resource, "service": request.service,
                "request_id": request.request_id, "priority": request.priority,
                "task_id": request.task_id, "document_id": request.document_id,
                "hostname": request.hostname, "pid": request.pid,
                "sequence": request.sequence,
            })
        try:
            cb(event, payload)
        except Exception:  # pragma: no cover
            log.exception("on_event(%s) lançou exceção", event)


def _new_token() -> str:
    import uuid

    return str(uuid.uuid4())
