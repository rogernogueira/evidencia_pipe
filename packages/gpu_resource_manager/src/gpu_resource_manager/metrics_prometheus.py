"""Integração Prometheus OPCIONAL (não é dependência obrigatória do core).

Uso::

    from gpu_resource_manager import GPUManagerConfig, GPUResourceManager
    from gpu_resource_manager.metrics_prometheus import PrometheusMetrics

    metrics = PrometheusMetrics()   # requer `pip install gpu-resource-manager[prometheus]`
    config = GPUManagerConfig.from_env(on_event=metrics.on_event)
    manager = GPUResourceManager(config)

Registra contadores/histogramas a partir dos eventos emitidos pelo manager.
Se ``prometheus_client`` não estiver instalado, o import falha com mensagem clara.
"""

from __future__ import annotations

from typing import Any

from .models import Events


class PrometheusMetrics:
    def __init__(self, namespace: str = "gpu_resource_manager", registry: Any = None) -> None:
        try:
            from prometheus_client import Counter, Histogram
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "prometheus_client não instalado. Use: pip install gpu-resource-manager[prometheus]"
            ) from exc

        kw = {"registry": registry} if registry is not None else {}
        labels = ["resource", "service"]
        self._acquired = Counter(f"{namespace}_lock_acquired_total", "Locks adquiridos", labels, **kw)
        self._released = Counter(f"{namespace}_lock_released_total", "Locks liberados", labels, **kw)
        self._timeouts = Counter(f"{namespace}_lock_timeout_total", "Timeouts de aquisição", labels, **kw)
        self._lost = Counter(f"{namespace}_lock_lost_total", "Propriedades perdidas", labels, **kw)
        self._cancelled = Counter(f"{namespace}_request_cancelled_total", "Solicitações canceladas", labels, **kw)
        self._backend_unavail = Counter(f"{namespace}_backend_unavailable_total", "Falhas de backend", **kw)
        self._wait_seconds = Histogram(f"{namespace}_wait_seconds", "Tempo de espera até adquirir", labels, **kw)
        self._held_seconds = Histogram(f"{namespace}_held_seconds", "Tempo de posse do lock", labels, **kw)

    def on_event(self, event: str, payload: dict[str, Any]) -> None:
        res = payload.get("resource", "?")
        svc = payload.get("service", "?")
        if event == Events.LOCK_ACQUIRED:
            self._acquired.labels(res, svc).inc()
            if payload.get("wait_time_seconds") is not None:
                self._wait_seconds.labels(res, svc).observe(payload["wait_time_seconds"])
        elif event == Events.LOCK_RELEASED:
            self._released.labels(res, svc).inc()
            if payload.get("held_time_seconds") is not None:
                self._held_seconds.labels(res, svc).observe(payload["held_time_seconds"])
        elif event == Events.LOCK_TIMEOUT:
            self._timeouts.labels(res, svc).inc()
        elif event == Events.LOCK_LOST:
            self._lost.labels(res, svc).inc()
        elif event == Events.REQUEST_CANCELLED:
            self._cancelled.labels(res, svc).inc()
        elif event == Events.BACKEND_UNAVAILABLE:
            self._backend_unavail.inc()
