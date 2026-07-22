"""Backend Redis — encapsula o cliente e os scripts Lua registrados.

Fail-closed: qualquer falha de conectividade vira :class:`GPUBackendUnavailable`.
NÃO há fallback em memória (um fallback local quebraria a exclusão entre processos).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from . import lua_scripts as lua
from .config import GPUManagerConfig
from .exceptions import GPUBackendUnavailable

log = logging.getLogger("gpu_resource_manager")


class RedisBackend:
    """Thin wrapper sobre redis-py com os scripts Lua da biblioteca."""

    def __init__(self, config: GPUManagerConfig, client: Optional[Any] = None) -> None:
        self.config = config
        self._external_client = client is not None
        self._client = client
        self._scripts: dict[str, Any] = {}
        if client is not None:
            self._register_scripts()

    # ------------------------------------------------------------------ #
    def _connect(self) -> Any:
        try:
            import redis  # import tardio: só redis-py é dependência de runtime
        except ImportError as exc:  # pragma: no cover
            raise GPUBackendUnavailable(
                "redis-py não instalado", cause=str(exc)
            ) from exc

        cfg = self.config
        try:
            client = redis.Redis.from_url(
                cfg.redis_url,
                decode_responses=True,
                socket_timeout=cfg.redis_socket_timeout_seconds,
                socket_connect_timeout=cfg.redis_connect_timeout_seconds,
                health_check_interval=cfg.redis_health_check_interval_seconds,
            )
            return client
        except Exception as exc:
            raise GPUBackendUnavailable(
                "falha ao construir o cliente Redis", redis_url=_redact(cfg.redis_url)
            ) from exc

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = self._connect()
            self._register_scripts()
        return self._client

    def _register_scripts(self) -> None:
        c = self._client
        self._scripts = {
            "enqueue": c.register_script(lua.ENQUEUE),
            "try_acquire": c.register_script(lua.TRY_ACQUIRE),
            "release": c.register_script(lua.RELEASE),
            "renew": c.register_script(lua.RENEW),
            "wait_heartbeat": c.register_script(lua.WAIT_HEARTBEAT),
            "cancel": c.register_script(lua.CANCEL),
        }

    def _run(self, name: str, keys: list[str], args: list[Any]) -> Any:
        try:
            return self._scripts[name](keys=keys, args=args)
        except GPUBackendUnavailable:
            raise
        except Exception as exc:
            self._raise_unavailable(f"script '{name}' falhou", exc)

    def _raise_unavailable(self, msg: str, exc: Exception) -> None:
        log.error("gpu_resource_manager backend indisponível: %s (%s)", msg, exc)
        self._emit_backend_unavailable(msg, exc)
        raise GPUBackendUnavailable(msg, cause=str(exc)) from exc

    def _emit_backend_unavailable(self, msg: str, exc: Exception) -> None:
        cb = self.config.on_event
        if cb is not None:
            try:
                from .models import Events

                cb(Events.BACKEND_UNAVAILABLE, {"message": msg, "error": str(exc)})
            except Exception:  # pragma: no cover - callbacks não podem derrubar o fluxo
                log.exception("on_event(backend_unavailable) lançou exceção")

    # ------------------------------------------------------------------ #
    # Health
    # ------------------------------------------------------------------ #
    def healthcheck(self) -> bool:
        """Retorna True se o Redis respondeu ao PING. Lança GPUBackendUnavailable se não."""
        try:
            ok = bool(self.client.ping())
        except Exception as exc:
            self._raise_unavailable("PING falhou", exc)
        if not ok:  # pragma: no cover
            raise GPUBackendUnavailable("PING retornou falso")
        # garante que os scripts existem no servidor
        if not self._scripts:
            self._register_scripts()
        return True

    def ensure_available(self) -> None:
        self.healthcheck()

    # ------------------------------------------------------------------ #
    # Scripts
    # ------------------------------------------------------------------ #
    def enqueue(self, request, *, request_ttl_ms: int) -> int:
        cfg = self.config
        r = request.resource
        payload = json.dumps(request.to_dict(), ensure_ascii=False)
        seq = self._run(
            "enqueue",
            keys=[cfg.key_sequence(r), cfg.key_queue(r)],
            args=[
                request.request_id,
                payload,
                int(request.priority),
                int(request_ttl_ms),
                cfg.request_prefix(r),
                lua.SCORE_PRIORITY_FACTOR,
            ],
        )
        return int(seq)

    def try_acquire(
        self, *, resource: str, request_id: str, token: str, now: float,
        lock_ttl_ms: int, owner_json: str,
    ) -> tuple[int, str, int]:
        cfg = self.config
        res = self._run(
            "try_acquire",
            keys=[cfg.key_lock(resource), cfg.key_owner(resource), cfg.key_queue(resource)],
            args=[
                request_id, token, repr_float(now), int(lock_ttl_ms), owner_json,
                cfg.request_prefix(resource),
                "1" if cfg.aging_enabled else "0",
                int(cfg.aging_interval_seconds), int(cfg.aging_step),
                int(cfg.min_effective_priority),
            ],
        )
        code, detail, position = res[0], res[1], res[2]
        return int(code), _to_str(detail), int(position)

    def release(self, *, resource: str, token: str) -> int:
        cfg = self.config
        return int(self._run(
            "release",
            keys=[cfg.key_lock(resource), cfg.key_owner(resource)],
            args=[token],
        ))

    def renew(self, *, resource: str, token: str, ttl_ms: int, owner_json: str) -> int:
        cfg = self.config
        return int(self._run(
            "renew",
            keys=[cfg.key_lock(resource), cfg.key_owner(resource)],
            args=[token, int(ttl_ms), owner_json],
        ))

    def wait_heartbeat(self, *, resource: str, request_id: str, ttl_ms: int) -> int:
        cfg = self.config
        return int(self._run(
            "wait_heartbeat",
            keys=[],
            args=[cfg.key_request(resource, request_id), int(ttl_ms)],
        ))

    def cancel(self, *, resource: str, request_id: str) -> int:
        cfg = self.config
        return int(self._run(
            "cancel",
            keys=[cfg.key_queue(resource)],
            args=[request_id, cfg.key_request(resource, request_id)],
        ))

    # ------------------------------------------------------------------ #
    # Leitura (status/fila) — nunca altera o lock
    # ------------------------------------------------------------------ #
    def read_lock(self, resource: str) -> tuple[Optional[str], Optional[float]]:
        cfg = self.config
        try:
            pipe = self.client.pipeline()
            pipe.get(cfg.key_lock(resource))
            pipe.pttl(cfg.key_lock(resource))
            token, pttl = pipe.execute()
        except Exception as exc:
            self._raise_unavailable("leitura de lock falhou", exc)
        ttl_s = (pttl / 1000.0) if isinstance(pttl, int) and pttl >= 0 else None
        return token, ttl_s

    def read_owner(self, resource: str) -> Optional[dict[str, Any]]:
        try:
            raw = self.client.get(self.config.key_owner(resource))
        except Exception as exc:
            self._raise_unavailable("leitura de owner falhou", exc)
        return json.loads(raw) if raw else None

    def read_queue_raw(self, resource: str, limit: int = 100) -> list[dict[str, Any]]:
        """Lê a fila (com metadados de cada request key). Remove exibição de órfãos."""
        cfg = self.config
        try:
            ids = self.client.zrange(cfg.key_queue(resource), 0, max(0, limit - 1))
            if not ids:
                return []
            pipe = self.client.pipeline()
            for rid in ids:
                pipe.get(cfg.key_request(resource, rid))
            raws = pipe.execute()
        except Exception as exc:
            self._raise_unavailable("leitura de fila falhou", exc)
        out: list[dict[str, Any]] = []
        for raw in raws:
            if raw:
                try:
                    out.append(json.loads(raw))
                except Exception:  # pragma: no cover
                    continue
        return out

    def close(self) -> None:
        if self._client is not None and not self._external_client:
            try:
                self._client.close()
            except Exception:  # pragma: no cover
                pass


def repr_float(x: float) -> str:
    # Representação estável para passar ao Lua (evita notação científica surpresa).
    return format(float(x), ".6f")


def _to_str(v: Any) -> str:
    return v.decode() if isinstance(v, (bytes, bytearray)) else str(v)


def _redact(url: str) -> str:
    """Remove a senha de uma URL Redis para logs (nunca logar credenciais)."""
    try:
        if "@" in url and "://" in url:
            scheme, rest = url.split("://", 1)
            creds, host = rest.split("@", 1)
            user = creds.split(":", 1)[0]
            return f"{scheme}://{user}:***@{host}"
    except Exception:  # pragma: no cover
        pass
    return url
