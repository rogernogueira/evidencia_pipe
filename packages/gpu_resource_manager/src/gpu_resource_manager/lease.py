"""GPULease — o objeto retornado por ``manager.acquire()``.

Carrega comportamento (heartbeat em background, validação de propriedade, release)
e por isso vive separado dos modelos de dados puros. Um lease NÃO é reentrante:
adquirir o mesmo recurso duas vezes no mesmo fluxo é erro do chamador (ver README).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import TYPE_CHECKING, Any, Optional

from .exceptions import GPULockLostError, GPUReleaseError
from .models import Events, OwnerInfo, utc_now_iso

if TYPE_CHECKING:  # pragma: no cover
    from .config import GPUManagerConfig
    from .redis_backend import RedisBackend

log = logging.getLogger("gpu_resource_manager")


class GPULease:
    """Representa a posse ativa de um recurso de GPU.

    Atributos públicos estáveis: ``request_id``, ``token``, ``resource``,
    ``service``, ``priority``, ``acquired_at``, ``wait_time_seconds``, ``is_valid``.
    Métodos: ``ensure_valid()``, ``release()``, e o dict ``metrics``.
    """

    def __init__(
        self,
        *,
        backend: "RedisBackend",
        config: "GPUManagerConfig",
        request,  # GPURequest
        token: str,
        wait_time_seconds: float,
        acquisition_attempts: int,
        queue_size_at_acquisition: int,
        effective_priority: int,
    ) -> None:
        self._backend = backend
        self._config = config
        self._request = request

        self.request_id: str = request.request_id
        self.token: str = token
        self.resource: str = request.resource
        self.service: str = request.service
        self.priority: int = request.priority
        self.effective_priority: int = effective_priority
        self.task_id: Optional[str] = request.task_id
        self.document_id: Optional[str] = request.document_id
        self.metadata: dict[str, Any] = dict(request.metadata)

        self._acquired_monotonic = time.monotonic()
        self.acquired_at: str = utc_now_iso()
        self.wait_time_seconds: float = round(wait_time_seconds, 3)
        self._last_heartbeat: str = self.acquired_at

        # métricas expostas ao consumidor (ver §25)
        self.metrics: dict[str, Any] = {
            "wait_time_seconds": self.wait_time_seconds,
            "held_time_seconds": None,
            "acquisition_attempts": acquisition_attempts,
            "priority": self.priority,
            "effective_priority": effective_priority,
            "queue_size_at_acquisition": queue_size_at_acquisition,
        }

        self._lock_lost = False
        self._released = False
        self._state_lock = threading.Lock()
        self._stop = threading.Event()
        self._hb_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------ #
    @property
    def is_valid(self) -> bool:
        return not self._lock_lost and not self._released

    @property
    def held_time_seconds(self) -> float:
        return round(time.monotonic() - self._acquired_monotonic, 3)

    def ensure_valid(self) -> None:
        """Lança GPULockLostError se a propriedade foi perdida (ou o lease foi liberado)."""
        if self._lock_lost:
            raise GPULockLostError(
                "propriedade do lock perdida (heartbeat falhou ou TTL expirou)",
                resource=self.resource,
                request_id=self.request_id,
                token=self.token,
            )
        if self._released:
            raise GPULockLostError(
                "lease já liberado",
                resource=self.resource,
                request_id=self.request_id,
                token=self.token,
            )

    # ------------------------------------------------------------------ #
    def _owner_json(self) -> str:
        self._last_heartbeat = utc_now_iso()
        owner = OwnerInfo(
            token=self.token,
            request_id=self.request_id,
            resource=self.resource,
            service=self.service,
            priority=self.priority,
            hostname=self._request.hostname,
            pid=self._request.pid,
            acquired_at=self.acquired_at,
            last_heartbeat=self._last_heartbeat,
            task_id=self.task_id,
            document_id=self.document_id,
            metadata=self.metadata,
        )
        return json.dumps(owner.to_dict(), ensure_ascii=False)

    def _emit(self, event: str, **extra: Any) -> None:
        cb = self._config.on_event
        if cb is None:
            return
        payload = {
            "resource": self.resource,
            "service": self.service,
            "request_id": self.request_id,
            "priority": self.priority,
            "effective_priority": self.effective_priority,
            "task_id": self.task_id,
            "document_id": self.document_id,
            "hostname": self._request.hostname,
            "pid": self._request.pid,
            **extra,
        }
        try:
            cb(event, payload)
        except Exception:  # pragma: no cover
            log.exception("on_event(%s) lançou exceção", event)

    # ------------------------------------------------------------------ #
    # Heartbeat
    # ------------------------------------------------------------------ #
    def start_heartbeat(self) -> None:
        self._hb_thread = threading.Thread(
            target=self._heartbeat_loop,
            name=f"gpu-lease-hb-{self.resource}-{self.request_id[:8]}",
            daemon=True,
        )
        self._hb_thread.start()

    def _heartbeat_loop(self) -> None:
        interval = self._config.heartbeat_seconds
        ttl_ms = self._config.lock_ttl_seconds * 1000
        while not self._stop.wait(interval):
            try:
                ok = self._backend.renew(
                    resource=self.resource,
                    token=self.token,
                    ttl_ms=ttl_ms,
                    owner_json=self._owner_json(),
                )
            except Exception as exc:
                # Falha transitória de backend: loga e tenta de novo no próximo ciclo.
                log.warning(
                    "gpu heartbeat: renovação falhou (backend) resource=%s request_id=%s: %s",
                    self.resource, self.request_id, exc,
                )
                self._emit(Events.HEARTBEAT_FAILED, error=str(exc))
                continue
            if ok != 1:
                # Propriedade perdida: outro processo tem o lock ou o TTL expirou.
                with self._state_lock:
                    self._lock_lost = True
                log.error(
                    "gpu heartbeat: PROPRIEDADE PERDIDA resource=%s request_id=%s token=%s",
                    self.resource, self.request_id, self.token,
                )
                self._emit(Events.LOCK_LOST)
                return
            log.debug(
                "gpu heartbeat ok resource=%s request_id=%s held_time_seconds=%.1f",
                self.resource, self.request_id, self.held_time_seconds,
            )

    # ------------------------------------------------------------------ #
    # Release
    # ------------------------------------------------------------------ #
    def release(self) -> None:
        with self._state_lock:
            if self._released:
                return
            self._released = True
        self._stop.set()
        if self._hb_thread is not None:
            self._hb_thread.join(timeout=self._config.heartbeat_seconds + 5)

        held = self.held_time_seconds
        self.metrics["held_time_seconds"] = held

        if self._lock_lost:
            # Não somos mais o dono — não tentar liberar o lock de outro processo.
            log.warning(
                "gpu release: lock já perdido; não liberando resource=%s request_id=%s",
                self.resource, self.request_id,
            )
            self._emit(Events.LOCK_RELEASED, held_time_seconds=held, lock_was_lost=True)
            return

        try:
            released = self._backend.release(resource=self.resource, token=self.token)
        except Exception as exc:
            log.error(
                "gpu release: falha no backend resource=%s request_id=%s: %s",
                self.resource, self.request_id, exc,
            )
            raise GPUReleaseError(
                "falha ao liberar o lock", resource=self.resource,
                request_id=self.request_id, cause=str(exc),
            ) from exc

        log.info(
            "gpu released resource=%s service=%s request_id=%s token=%s held_time_seconds=%.2f",
            self.resource, self.service, self.request_id, self.token, held,
        )
        self._emit(Events.LOCK_RELEASED, held_time_seconds=held,
                   lock_was_lost=(released != 1))

    # ------------------------------------------------------------------ #
    # Context manager (o próprio lease pode ser usado com `with`, mas o fluxo
    # normal é `with manager.acquire(...) as lease:` — ver manager.py).
    # ------------------------------------------------------------------ #
    def __enter__(self) -> "GPULease":  # pragma: no cover - conveniência
        return self

    def __exit__(self, *exc) -> bool:  # pragma: no cover
        self.release()
        return False
