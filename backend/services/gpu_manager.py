"""Instância compartilhada do GPUResourceManager para o evidencia_pipe.

Este módulo é o ÚNICO ponto de acoplamento entre o projeto e a biblioteca
`gpu_resource_manager`. Ele:
  - constrói a config a partir do `.env` já existente (backend.core.config);
  - integra o logger da biblioteca ao logging do projeto (backend.core.logger);
  - expõe um singleton lazy `get_gpu_manager()` reutilizado por MinerU, BGE-M3,
    tasks e endpoints.

A biblioteca continua independente: quem importa Celery/FastAPI/PyTorch é o
consumidor, nunca o núcleo do gpu_resource_manager.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

from gpu_resource_manager import GPUManagerConfig, GPUResourceManager

from backend.core import config as settings
from backend.core.logger import log

# Integração de logs (§24): o logger próprio da biblioteca propaga para o root
# logger já configurado por backend/core/logger.py (basicConfig). Só garantimos o
# nível; não mexemos no núcleo da lib.
logging.getLogger("gpu_resource_manager").setLevel(logging.INFO)

_manager: Optional[GPUResourceManager] = None
_lock = threading.Lock()


def _on_event(event: str, payload: dict) -> None:
    """Hook de observabilidade → integra os eventos ao logger do projeto (§25).

    Consumidores podem trocar este callback por métricas Prometheus
    (gpu_resource_manager.metrics_prometheus.PrometheusMetrics)."""
    log.debug("[gpu] event=%s %s", event, {k: payload.get(k) for k in ("resource", "service", "request_id")})


def build_config() -> GPUManagerConfig:
    """Constrói a GPUManagerConfig a partir das configurações do projeto (.env).

    `from_env` lê os tunables GPU_MANAGER_* diretamente do os.environ (populado pelo
    load_dotenv em backend.core.config). Em seguida sobrescrevemos redis_url e
    resource_name com os valores de backend.core.config, que já derivam o default do
    manager do MESMO REDIS_URL do projeto (base + '/2') quando GPU_MANAGER_REDIS_URL
    não é definido — assim não criamos um mecanismo paralelo de configuração."""
    base = GPUManagerConfig.from_env(on_event=_on_event)
    return GPUManagerConfig.from_dict(
        base.__dict__,
        redis_url=settings.GPU_MANAGER_REDIS_URL,
        resource_name=settings.GPU_RESOURCE_NAME,
        on_event=_on_event,
    )


def get_gpu_manager() -> GPUResourceManager:
    """Retorna o singleton do gerenciador (lazy). Fail-closed: se o Redis do
    manager estiver fora, as chamadas de acquire lançam GPUBackendUnavailable."""
    global _manager
    if _manager is None:
        with _lock:
            if _manager is None:
                cfg = build_config()
                _manager = GPUResourceManager(cfg)
                log.info(
                    "[gpu] GPUResourceManager pronto (resource=%s, redis=%s, enabled=%s)",
                    cfg.resource_name, cfg.redis_url, settings.GPU_MANAGER_ENABLED,
                )
    return _manager


def gpu_enabled() -> bool:
    return settings.GPU_MANAGER_ENABLED
