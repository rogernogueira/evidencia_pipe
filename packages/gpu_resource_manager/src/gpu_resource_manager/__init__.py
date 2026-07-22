"""gpu_resource_manager — coordenação distribuída de recursos de GPU sobre Redis.

API pública estável::

    from gpu_resource_manager import GPUResourceManager, GPUManagerConfig, GPUAcquisitionTimeout

    manager = GPUResourceManager(GPUManagerConfig.from_env())
    with manager.acquire(resource="gpu0", service="script", priority=10) as lease:
        executar_codigo_cuda()
        lease.ensure_valid()

O núcleo não importa Celery, FastAPI, PyTorch, MinerU, BGE-M3 ou Qdrant.
"""

from __future__ import annotations

import logging

from .config import GPUManagerConfig
from .exceptions import (
    GPUAcquisitionTimeout,
    GPUBackendUnavailable,
    GPUInvalidConfiguration,
    GPULockLostError,
    GPUManagerError,
    GPUReleaseError,
    GPURequestCancelled,
)
from .lease import GPULease
from .manager import GPUResourceManager
from .models import (
    Events,
    GPURequest,
    OwnerInfo,
    QueueEntry,
    ResourceStatus,
)

# Logger próprio da biblioteca; aplicações conectam seus próprios handlers.
# NullHandler evita "No handlers could be found" quando o consumidor não configura logging.
logging.getLogger("gpu_resource_manager").addHandler(logging.NullHandler())

__version__ = "0.1.0"

__all__ = [
    "GPUResourceManager",
    "GPUManagerConfig",
    "GPULease",
    "GPURequest",
    "OwnerInfo",
    "QueueEntry",
    "ResourceStatus",
    "Events",
    "GPUManagerError",
    "GPUAcquisitionTimeout",
    "GPUBackendUnavailable",
    "GPULockLostError",
    "GPUInvalidConfiguration",
    "GPURequestCancelled",
    "GPUReleaseError",
    "__version__",
]
