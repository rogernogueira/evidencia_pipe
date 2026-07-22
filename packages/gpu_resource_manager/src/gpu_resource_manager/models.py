"""Modelos de dados tipados (puros, sem lógica de backend).

Usados na API pública para status/fila/eventos. ``GPULease`` (com heartbeat) fica
em ``lease.py`` porque carrega comportamento e referência ao backend.
"""

from __future__ import annotations

import os
import socket
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


def new_uuid() -> str:
    return str(uuid.uuid4())


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def hostname() -> str:
    try:
        return socket.gethostname()
    except Exception:  # pragma: no cover
        return "unknown"


# Nomes de eventos emitidos via callback/hook (config.on_event).
class Events:
    REQUEST_CREATED = "request_created"
    REQUEST_WAITING = "request_waiting"
    LOCK_ACQUIRED = "lock_acquired"
    LOCK_RELEASED = "lock_released"
    LOCK_TIMEOUT = "lock_timeout"
    LOCK_LOST = "lock_lost"
    HEARTBEAT_FAILED = "heartbeat_failed"
    BACKEND_UNAVAILABLE = "backend_unavailable"
    REQUEST_CANCELLED = "request_cancelled"


@dataclass
class GPURequest:
    """Solicitação de GPU (o que é enfileirado)."""

    request_id: str
    resource: str
    service: str
    priority: int
    created_at: str
    enqueued_at: float  # epoch seconds (usado pelo aging, no relógio do cliente)
    hostname: str
    pid: int
    sequence: Optional[int] = None  # atribuído pelo Redis (INCR) no enqueue
    task_id: Optional[str] = None
    document_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        resource: str,
        service: str,
        priority: int,
        task_id: Optional[str] = None,
        document_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> "GPURequest":
        return cls(
            request_id=new_uuid(),
            resource=resource,
            service=service,
            priority=priority,
            created_at=utc_now_iso(),
            enqueued_at=time.time(),
            hostname=hostname(),
            pid=os.getpid(),
            task_id=task_id,
            document_id=document_id,
            metadata=dict(metadata or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OwnerInfo:
    """Metadados do dono atual do lock (diagnóstico; NÃO é a fonte de exclusão)."""

    token: str
    request_id: str
    resource: str
    service: str
    priority: int
    hostname: str
    pid: int
    acquired_at: str
    last_heartbeat: str
    task_id: Optional[str] = None
    document_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OwnerInfo":
        known = cls.__dataclass_fields__  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class QueueEntry:
    """Uma entrada da fila, já com prioridade efetiva e tempo de espera calculados."""

    request_id: str
    service: str
    priority: int
    effective_priority: int
    sequence: Optional[int]
    wait_time_seconds: float
    task_id: Optional[str] = None
    document_id: Optional[str] = None
    hostname: Optional[str] = None
    pid: Optional[int] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ResourceStatus:
    """Snapshot de status de um recurso (não altera o lock)."""

    resource: str
    locked: bool
    lock_ttl_seconds: Optional[float]
    owner: Optional[dict[str, Any]]
    queue_size: int
    next_request: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
