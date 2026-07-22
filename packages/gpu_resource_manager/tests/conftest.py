"""Fixtures de teste — usam fakeredis[lua] (Lua/cjson server-side, sem GPU/Redis real)."""

import os
import sys

import pytest

# Permite rodar os testes sem instalar o pacote (src layout).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import fakeredis  # noqa: E402

from gpu_resource_manager import GPUManagerConfig, GPUResourceManager  # noqa: E402
from gpu_resource_manager.redis_backend import RedisBackend  # noqa: E402


@pytest.fixture
def fake_server():
    """Um servidor fakeredis compartilhado (vários clientes veem o mesmo estado)."""
    return fakeredis.FakeServer()


def make_config(**overrides) -> GPUManagerConfig:
    base = dict(
        redis_url="redis://localhost:6379/2",
        heartbeat_seconds=1,
        lock_ttl_seconds=2,
        request_heartbeat_seconds=1,
        request_ttl_seconds=2,
        poll_interval_seconds=0.02,
        poll_jitter_seconds=0.0,
        wait_timeout_seconds=5,
        aging_interval_seconds=1,
        aging_step=5,
    )
    base.update(overrides)
    return GPUManagerConfig(**base)


@pytest.fixture
def manager_factory(fake_server):
    """Fábrica de managers que compartilham o mesmo fakeredis (processos lógicos distintos)."""
    created = []

    def _make(**overrides):
        cfg = make_config(**overrides)
        client = fakeredis.FakeStrictRedis(server=fake_server, decode_responses=True)
        mgr = GPUResourceManager(cfg, backend=RedisBackend(cfg, client=client))
        created.append(mgr)
        return mgr

    yield _make
    for m in created:
        try:
            m.close()
        except Exception:
            pass


@pytest.fixture
def manager(manager_factory):
    return manager_factory()
