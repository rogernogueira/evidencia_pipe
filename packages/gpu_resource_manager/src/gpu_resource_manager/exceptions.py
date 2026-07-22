"""Exceções públicas da biblioteca.

Todas herdam de :class:`GPUManagerError` e expõem atributos estruturados úteis
para logs e APIs (nunca dependem de Celery/FastAPI). Instâncias carregam um
``details`` dict serializável para facilitar telemetria.
"""

from __future__ import annotations

from typing import Any, Optional


class GPUManagerError(Exception):
    """Base de todas as exceções da biblioteca."""

    def __init__(self, message: str, **details: Any) -> None:
        super().__init__(message)
        self.message = message
        # Atributos estruturados (sem segredos) para logs/telemetria.
        self.details: dict[str, Any] = {k: v for k, v in details.items() if v is not None}

    def __str__(self) -> str:  # pragma: no cover - trivial
        if self.details:
            return f"{self.message} ({self.details})"
        return self.message


class GPUInvalidConfiguration(GPUManagerError):
    """Configuração inválida (TTL <= heartbeat, prioridade fora do intervalo, etc.)."""


class GPUBackendUnavailable(GPUManagerError):
    """O Redis está indisponível. Modo fail-closed: o cliente NÃO pode usar a GPU."""


class GPUAcquisitionTimeout(GPUManagerError):
    """Espera pela GPU excedeu ``wait_timeout_seconds``."""

    def __init__(
        self,
        message: str,
        *,
        resource: str,
        service: str,
        request_id: str,
        priority: int,
        wait_time_seconds: float,
        current_owner: Optional[dict[str, Any]] = None,
        lock_ttl_seconds: Optional[float] = None,
        queue_position: Optional[int] = None,
    ) -> None:
        super().__init__(
            message,
            resource=resource,
            service=service,
            request_id=request_id,
            priority=priority,
            wait_time_seconds=round(wait_time_seconds, 3),
            current_owner=current_owner,
            lock_ttl_seconds=lock_ttl_seconds,
            queue_position=queue_position,
        )
        self.resource = resource
        self.service = service
        self.request_id = request_id
        self.priority = priority
        self.wait_time_seconds = wait_time_seconds
        self.current_owner = current_owner
        self.lock_ttl_seconds = lock_ttl_seconds
        self.queue_position = queue_position


class GPULockLostError(GPUManagerError):
    """A propriedade do lock foi perdida (heartbeat falhou / TTL expirou / roubo)."""

    def __init__(self, message: str, *, resource: str, request_id: str, token: str) -> None:
        super().__init__(message, resource=resource, request_id=request_id, token=token)
        self.resource = resource
        self.request_id = request_id
        self.token = token


class GPURequestCancelled(GPUManagerError):
    """A solicitação foi cancelada (via ``cancel_request`` ou órfã removida) antes de adquirir."""

    def __init__(self, message: str, *, resource: str, request_id: str) -> None:
        super().__init__(message, resource=resource, request_id=request_id)
        self.resource = resource
        self.request_id = request_id


class GPUReleaseError(GPUManagerError):
    """Falha ao liberar o lock (token não confere ou erro de backend)."""
