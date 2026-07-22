"""Configuração da biblioteca — independente de qualquer framework.

`GPUManagerConfig` é um dataclass (sem dependência de validação externa: mantém o
footprint mínimo para reuso em outros repositórios). Pode ser criado pelo
construtor, por ``from_env()``, por ``from_dict()`` ou diretamente em testes.
Todos os campos são validados em ``__post_init__``.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, fields
from typing import Any, Callable, Optional

from .exceptions import GPUInvalidConfiguration

# Callback opcional de eventos: recebe (event_name, payload_dict). Não é serializado.
EventCallback = Callable[[str, dict[str, Any]], None]

_RESOURCE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_:.\-]*$")
_TRUE = {"1", "true", "yes", "on", "y", "t"}


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in _TRUE


@dataclass
class GPUManagerConfig:
    """Configuração imutável (por convenção) do gerenciador de GPU."""

    # --- núcleo ---
    enabled: bool = True
    redis_url: str = "redis://localhost:6379/2"
    resource_name: str = "gpu0"
    key_prefix: str = "gpu"

    # --- lease / heartbeat do lock ---
    lock_ttl_seconds: int = 300
    heartbeat_seconds: int = 30

    # --- espera ---
    wait_timeout_seconds: int = 1800
    poll_interval_seconds: float = 2.0
    poll_jitter_seconds: float = 1.0
    wait_log_interval_seconds: int = 30

    # --- solicitação em espera (órfãs) ---
    request_ttl_seconds: int = 120
    request_heartbeat_seconds: int = 30

    # --- prioridades ---
    default_priority: int = 50
    min_priority: int = 0
    max_priority: int = 1000

    # --- aging ---
    aging_enabled: bool = True
    aging_interval_seconds: int = 300
    aging_step: int = 1
    min_effective_priority: int = 0

    # --- redis client ---
    redis_socket_timeout_seconds: float = 5.0
    redis_connect_timeout_seconds: float = 5.0
    redis_health_check_interval_seconds: int = 30

    # --- observabilidade (não vem de env) ---
    on_event: Optional[EventCallback] = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._validate()

    # ------------------------------------------------------------------ #
    # Validação
    # ------------------------------------------------------------------ #
    def _validate(self) -> None:
        def positive(name: str, value: float) -> None:
            if value is None or value <= 0:
                raise GPUInvalidConfiguration(
                    f"'{name}' deve ser positivo", field=name, value=value
                )

        if not self.redis_url or "://" not in self.redis_url:
            raise GPUInvalidConfiguration(
                "redis_url inválida", field="redis_url", value=self.redis_url
            )
        if not _RESOURCE_RE.match(self.resource_name or ""):
            raise GPUInvalidConfiguration(
                "resource_name inválido (use [A-Za-z0-9_:.-])",
                field="resource_name",
                value=self.resource_name,
            )
        if not self.key_prefix:
            raise GPUInvalidConfiguration("key_prefix vazio", field="key_prefix")

        positive("lock_ttl_seconds", self.lock_ttl_seconds)
        positive("heartbeat_seconds", self.heartbeat_seconds)
        positive("wait_timeout_seconds", self.wait_timeout_seconds)
        positive("poll_interval_seconds", self.poll_interval_seconds)
        positive("request_ttl_seconds", self.request_ttl_seconds)
        positive("request_heartbeat_seconds", self.request_heartbeat_seconds)
        positive("aging_interval_seconds", self.aging_interval_seconds)

        if self.poll_jitter_seconds < 0:
            raise GPUInvalidConfiguration(
                "poll_jitter_seconds não pode ser negativo",
                field="poll_jitter_seconds",
                value=self.poll_jitter_seconds,
            )
        # TTL do lock deve ser maior que o heartbeat (senão expira entre renovações).
        if self.lock_ttl_seconds <= self.heartbeat_seconds:
            raise GPUInvalidConfiguration(
                "lock_ttl_seconds deve ser > heartbeat_seconds",
                lock_ttl_seconds=self.lock_ttl_seconds,
                heartbeat_seconds=self.heartbeat_seconds,
            )
        # TTL da solicitação em espera deve ser maior que o heartbeat de espera.
        if self.request_ttl_seconds <= self.request_heartbeat_seconds:
            raise GPUInvalidConfiguration(
                "request_ttl_seconds deve ser > request_heartbeat_seconds",
                request_ttl_seconds=self.request_ttl_seconds,
                request_heartbeat_seconds=self.request_heartbeat_seconds,
            )
        # Intervalo de prioridades coerente.
        if self.min_priority > self.max_priority:
            raise GPUInvalidConfiguration(
                "min_priority > max_priority",
                min_priority=self.min_priority,
                max_priority=self.max_priority,
            )
        if not (self.min_priority <= self.default_priority <= self.max_priority):
            raise GPUInvalidConfiguration(
                "default_priority fora de [min_priority, max_priority]",
                default_priority=self.default_priority,
                min_priority=self.min_priority,
                max_priority=self.max_priority,
            )
        if self.aging_step < 0:
            raise GPUInvalidConfiguration("aging_step negativo", value=self.aging_step)
        if self.min_effective_priority < self.min_priority:
            raise GPUInvalidConfiguration(
                "min_effective_priority < min_priority",
                min_effective_priority=self.min_effective_priority,
                min_priority=self.min_priority,
            )

    def clamp_priority(self, priority: Optional[int]) -> int:
        """Normaliza a prioridade para o intervalo configurado (default se None)."""
        if priority is None:
            priority = self.default_priority
        try:
            priority = int(priority)
        except (TypeError, ValueError):
            raise GPUInvalidConfiguration("priority não é inteiro", value=priority)
        return max(self.min_priority, min(self.max_priority, priority))

    # ------------------------------------------------------------------ #
    # Construtores alternativos
    # ------------------------------------------------------------------ #
    @classmethod
    def from_dict(cls, data: dict[str, Any], **overrides: Any) -> "GPUManagerConfig":
        known = {f.name for f in fields(cls)}
        merged = {k: v for k, v in {**data, **overrides}.items() if k in known}
        return cls(**merged)

    @classmethod
    def from_env(
        cls,
        env: Optional[dict[str, str]] = None,
        *,
        on_event: Optional[EventCallback] = None,
    ) -> "GPUManagerConfig":
        """Cria a configuração a partir de variáveis de ambiente ``GPU_MANAGER_*``.

        ``GPU_RESOURCE_NAME`` (sem prefixo ``GPU_MANAGER_``) nomeia o recurso padrão.
        """
        e = env if env is not None else os.environ

        def g(key: str, default: Optional[str] = None) -> Optional[str]:
            val = e.get(key)
            return val if val not in (None, "") else default

        def gi(key: str, default: int) -> int:
            raw = g(key)
            return int(raw) if raw is not None else default

        def gf(key: str, default: float) -> float:
            raw = g(key)
            return float(raw) if raw is not None else default

        return cls(
            enabled=_as_bool(g("GPU_MANAGER_ENABLED"), True),
            redis_url=g("GPU_MANAGER_REDIS_URL", "redis://localhost:6379/2"),
            resource_name=g("GPU_RESOURCE_NAME", "gpu0"),
            key_prefix=g("GPU_MANAGER_KEY_PREFIX", "gpu"),
            lock_ttl_seconds=gi("GPU_MANAGER_LOCK_TTL_SECONDS", 300),
            heartbeat_seconds=gi("GPU_MANAGER_HEARTBEAT_SECONDS", 30),
            wait_timeout_seconds=gi("GPU_MANAGER_WAIT_TIMEOUT_SECONDS", 1800),
            poll_interval_seconds=gf("GPU_MANAGER_POLL_INTERVAL_SECONDS", 2.0),
            poll_jitter_seconds=gf("GPU_MANAGER_POLL_JITTER_SECONDS", 1.0),
            wait_log_interval_seconds=gi("GPU_MANAGER_WAIT_LOG_INTERVAL_SECONDS", 30),
            request_ttl_seconds=gi("GPU_MANAGER_REQUEST_TTL_SECONDS", 120),
            request_heartbeat_seconds=gi("GPU_MANAGER_REQUEST_HEARTBEAT_SECONDS", 30),
            default_priority=gi("GPU_MANAGER_DEFAULT_PRIORITY", 50),
            min_priority=gi("GPU_MANAGER_MIN_PRIORITY", 0),
            max_priority=gi("GPU_MANAGER_MAX_PRIORITY", 1000),
            aging_enabled=_as_bool(g("GPU_MANAGER_AGING_ENABLED"), True),
            aging_interval_seconds=gi("GPU_MANAGER_AGING_INTERVAL_SECONDS", 300),
            aging_step=gi("GPU_MANAGER_AGING_STEP", 1),
            min_effective_priority=gi("GPU_MANAGER_MIN_EFFECTIVE_PRIORITY", 0),
            redis_socket_timeout_seconds=gf("GPU_MANAGER_REDIS_SOCKET_TIMEOUT_SECONDS", 5.0),
            redis_connect_timeout_seconds=gf("GPU_MANAGER_REDIS_CONNECT_TIMEOUT_SECONDS", 5.0),
            redis_health_check_interval_seconds=gi(
                "GPU_MANAGER_REDIS_HEALTH_CHECK_INTERVAL_SECONDS", 30
            ),
            on_event=on_event,
        )

    # ------------------------------------------------------------------ #
    # Derivação de chaves Redis (recursos nomeados — sem 'global' hardcoded)
    # ------------------------------------------------------------------ #
    def resource(self, resource_name: Optional[str] = None) -> str:
        return resource_name or self.resource_name

    def key_lock(self, resource: str) -> str:
        return f"{self.key_prefix}:{resource}:lock"

    def key_owner(self, resource: str) -> str:
        return f"{self.key_prefix}:{resource}:owner"

    def key_queue(self, resource: str) -> str:
        return f"{self.key_prefix}:{resource}:queue"

    def key_sequence(self, resource: str) -> str:
        return f"{self.key_prefix}:{resource}:sequence"

    def request_prefix(self, resource: str) -> str:
        return f"{self.key_prefix}:{resource}:request:"

    def key_request(self, resource: str, request_id: str) -> str:
        return f"{self.request_prefix(resource)}{request_id}"

    def key_metrics(self, resource: str) -> str:
        return f"{self.key_prefix}:{resource}:metrics"
