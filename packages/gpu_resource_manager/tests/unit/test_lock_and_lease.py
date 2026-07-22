"""Cenários 1-8, 20, 22, 23, 25 (§39): lock, release, renew, expiração, ctx manager,
backend indisponível, status, metadados, lock perdido, sem fallback em memória."""

import time

import pytest

from gpu_resource_manager import (
    GPUBackendUnavailable,
    GPULockLostError,
    GPUManagerConfig,
    GPUResourceManager,
)


def test_acquire_free_lock_and_release_same_token(manager):
    with manager.acquire(service="a", priority=10) as lease:
        assert lease.is_valid
        assert lease.token
        st = manager.get_status()
        assert st.locked and st.owner["token"] == lease.token
    # liberado
    assert manager.get_status().locked is False


def test_release_rejected_for_other_token(manager):
    lease = manager.acquire_lease(service="a", priority=10)
    try:
        # tentar liberar com outro token não pode apagar o lock
        assert manager.backend.release(resource=manager.config.resource_name, token="outro-token") == 0
        assert manager.get_status().locked is True
    finally:
        lease.release()
    assert manager.get_status().locked is False


def test_renew_same_token_ok_other_token_fails(manager):
    lease = manager.acquire_lease(service="a", priority=10)
    res = manager.config.resource_name
    ttl_ms = manager.config.lock_ttl_seconds * 1000
    try:
        assert manager.backend.renew(resource=res, token=lease.token, ttl_ms=ttl_ms, owner_json="{}") == 1
        assert manager.backend.renew(resource=res, token="nope", ttl_ms=ttl_ms, owner_json="{}") == 0
    finally:
        lease.release()


def test_lock_expires_without_heartbeat(manager_factory):
    # emula kill -9: paramos o heartbeat e não liberamos; o TTL (2s) recupera o lock
    m = manager_factory(lock_ttl_seconds=2, heartbeat_seconds=1)
    lease = m.acquire_lease(service="a", priority=1)
    lease._stop.set()  # heartbeat não renova mais
    assert m.get_status().locked is True
    time.sleep(2.6)
    assert m.get_status().locked is False  # TTL expirou → recurso recuperado


def test_context_manager_releases_after_exception(manager):
    with pytest.raises(RuntimeError):
        with manager.acquire(service="a", priority=1):
            raise RuntimeError("boom")
    assert manager.get_status().locked is False


def test_backend_unavailable_is_fail_closed():
    # porta que recusa conexão → GPUBackendUnavailable (NUNCA fallback em memória)
    cfg = GPUManagerConfig(
        redis_url="redis://127.0.0.1:6390/2",
        redis_connect_timeout_seconds=0.3,
        redis_socket_timeout_seconds=0.3,
    )
    mgr = GPUResourceManager(cfg)
    with pytest.raises(GPUBackendUnavailable):
        mgr.healthcheck()
    with pytest.raises(GPUBackendUnavailable):
        mgr.acquire_lease(service="a", priority=1)


def test_status_reports_owner_and_ttl(manager):
    with manager.acquire(service="mineru", priority=20, task_id="t1", document_id="d1") as lease:
        st = manager.get_status()
        assert st.locked
        assert st.owner["service"] == "mineru"
        assert st.owner["task_id"] == "t1"
        assert st.owner["document_id"] == "d1"
        assert st.lock_ttl_seconds is not None and st.lock_ttl_seconds > 0


def test_generic_metadata_preserved(manager):
    md = {"operation": "x", "user": "svc"}
    with manager.acquire(service="ext", priority=10, metadata=md) as lease:
        assert lease.metadata["operation"] == "x"
        assert manager.get_status().owner["metadata"]["user"] == "svc"


def test_lock_lost_raises(manager):
    lease = manager.acquire_lease(service="z", priority=1)
    try:
        # rouba/expira o lock por baixo; o heartbeat (1s) deve detectar
        manager.backend.client.delete(manager.config.key_lock(manager.config.resource_name))
        time.sleep(1.6)
        with pytest.raises(GPULockLostError):
            lease.ensure_valid()
    finally:
        lease.release()


def test_no_inmemory_fallback_when_backend_dies(manager):
    # com backend vivo funciona; ao "matar" o cliente, novas operações levantam erro
    with manager.acquire(service="a", priority=1):
        pass
    # fecha o cliente e força falha
    manager.backend.client.connection_pool.disconnect()

    class _Boom:
        def __getattr__(self, _):
            raise ConnectionError("dead")

    manager.backend._client = _Boom()
    manager.backend._scripts = {}
    with pytest.raises(GPUBackendUnavailable):
        manager.healthcheck()
