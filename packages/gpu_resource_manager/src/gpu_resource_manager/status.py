"""Montagem de snapshots de status (leitura pura, nunca altera o lock)."""

from __future__ import annotations

import time
from typing import Optional

from .config import GPUManagerConfig
from .models import ResourceStatus
from .queue import build_queue_entries
from .redis_backend import RedisBackend


def read_status(
    backend: RedisBackend, config: GPUManagerConfig, resource: str
) -> ResourceStatus:
    now = time.time()
    token, ttl = backend.read_lock(resource)
    owner = backend.read_owner(resource) if token else None
    raw = backend.read_queue_raw(resource, limit=1000)
    entries = build_queue_entries(config, raw, now=now)

    next_request = None
    if entries:
        top = entries[0]
        next_request = {
            "service": top.service,
            "priority": top.priority,
            "effective_priority": top.effective_priority,
            "wait_time_seconds": top.wait_time_seconds,
            "request_id": top.request_id,
        }

    return ResourceStatus(
        resource=resource,
        locked=token is not None,
        lock_ttl_seconds=round(ttl, 1) if ttl is not None else None,
        owner=owner,
        queue_size=len(entries),
        next_request=next_request,
    )


def read_queue(
    backend: RedisBackend, config: GPUManagerConfig, resource: str, limit: int = 100
) -> list[dict]:
    raw = backend.read_queue_raw(resource, limit=limit)
    entries = build_queue_entries(config, raw)
    return [e.to_dict() for e in entries[:limit]]


def approximate_position(
    backend: RedisBackend, config: GPUManagerConfig, resource: str, request_id: str
) -> Optional[int]:
    """Posição 1-based aproximada do request na ordem efetiva (None se ausente)."""
    raw = backend.read_queue_raw(resource, limit=1000)
    entries = build_queue_entries(config, raw)
    for i, e in enumerate(entries, 1):
        if e.request_id == request_id:
            return i
    return None
