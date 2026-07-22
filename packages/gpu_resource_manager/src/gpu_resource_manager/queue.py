"""Cálculo de prioridade efetiva (aging) e montagem das entradas da fila (client-side).

A seleção real do vencedor acontece no Lua (``TRY_ACQUIRE``) — estas funções são a
versão Python equivalente, usadas apenas para *leitura* (status/fila) e para os
logs de espera. Mantêm a mesma fórmula de aging para consistência de exibição.
"""

from __future__ import annotations

import math
import time
from typing import Any, Optional

from .config import GPUManagerConfig
from .models import QueueEntry


def effective_priority(
    config: GPUManagerConfig, base_priority: int, enqueued_at: float, now: Optional[float] = None
) -> int:
    """effective = max(min_eff, base - floor(wait/interval) * step)."""
    if not config.aging_enabled or config.aging_interval_seconds <= 0:
        return base_priority
    now = time.time() if now is None else now
    waited = max(0.0, now - enqueued_at)
    steps = math.floor(waited / config.aging_interval_seconds)
    eff = base_priority - steps * config.aging_step
    return max(config.min_effective_priority, eff)


def build_queue_entries(
    config: GPUManagerConfig, raw_requests: list[dict[str, Any]], now: Optional[float] = None
) -> list[QueueEntry]:
    """Converte requests brutos em QueueEntry ordenados por (efetiva, sequence)."""
    now = time.time() if now is None else now
    entries: list[QueueEntry] = []
    for req in raw_requests:
        base = int(req.get("priority", config.default_priority))
        enq = float(req.get("enqueued_at", now))
        eff = effective_priority(config, base, enq, now)
        entries.append(
            QueueEntry(
                request_id=req.get("request_id", ""),
                service=req.get("service", "?"),
                priority=base,
                effective_priority=eff,
                sequence=req.get("sequence"),
                wait_time_seconds=round(max(0.0, now - enq), 3),
                task_id=req.get("task_id"),
                document_id=req.get("document_id"),
                hostname=req.get("hostname"),
                pid=req.get("pid"),
                metadata=req.get("metadata", {}) or {},
            )
        )
    entries.sort(key=lambda e: (e.effective_priority, e.sequence if e.sequence is not None else 1 << 62))
    return entries
