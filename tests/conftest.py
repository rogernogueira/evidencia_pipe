"""Fixtures dos testes de integração do evidencia_pipe (§40).

Usam fakeredis[lua] + mocks (subprocesso MinerU, PyTorch, BGE-M3) — sem GPU real.
NÃO importam FlagEmbedding/torch/qdrant reais: os pontos pesados são mockados.
"""

import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "packages", "gpu_resource_manager", "src"))

import fakeredis  # noqa: E402

from gpu_resource_manager import GPUManagerConfig, GPUResourceManager  # noqa: E402
from gpu_resource_manager.redis_backend import RedisBackend  # noqa: E402


@pytest.fixture
def gpu_manager_fake(monkeypatch):
    """Injeta um GPUResourceManager backed por fakeredis no singleton do projeto."""
    server = fakeredis.FakeServer()

    def make(**overrides):
        cfg = GPUManagerConfig(
            redis_url="redis://localhost:6379/2",
            resource_name="gpu0",
            lock_ttl_seconds=3, heartbeat_seconds=1,
            request_ttl_seconds=3, request_heartbeat_seconds=1,
            poll_interval_seconds=0.02, poll_jitter_seconds=0.0,
            wait_timeout_seconds=8,
            **overrides,
        )
        client = fakeredis.FakeStrictRedis(server=server, decode_responses=True)
        return GPUResourceManager(cfg, backend=RedisBackend(cfg, client=client))

    import backend.services.gpu_manager as gm

    primary = make()
    monkeypatch.setattr(gm, "_manager", primary, raising=False)
    monkeypatch.setattr(gm, "get_gpu_manager", lambda: primary)

    # expõe a fábrica p/ criar "outros processos" no mesmo fakeredis
    primary._make_peer = make  # type: ignore[attr-defined]
    return primary
