"""Teste de integração com PROCESSOS independentes (não apenas threads) — §41.

Demonstra que a exclusão, a prioridade e a recuperação por TTL funcionam entre
processos distintos, cada um com seu próprio cliente Redis (como fariam contêineres
ou scripts externos diferentes). Requer um Redis REAL (ex.: o do docker-compose).

    GPU_MANAGER_REDIS_URL=redis://127.0.0.1:6379/2 pytest tests/integration -q

Sem Redis acessível, os testes são pulados (skip), não falham.
"""

import json
import multiprocessing as mp
import os
import time
import uuid

import pytest

from gpu_resource_manager import GPUManagerConfig, GPUResourceManager

REDIS_URL = os.getenv("GPU_MANAGER_REDIS_URL", "redis://127.0.0.1:6379/2")


def _redis_available() -> bool:
    try:
        import redis

        redis.Redis.from_url(REDIS_URL, socket_connect_timeout=0.5).ping()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _redis_available(), reason=f"Redis real indisponível em {REDIS_URL}")


@pytest.fixture
def resource_name():
    # recurso único por teste, para não colidir com dados existentes
    name = f"itest-{uuid.uuid4().hex[:8]}"
    yield name
    try:
        import redis

        c = redis.Redis.from_url(REDIS_URL)
        for k in c.scan_iter(f"gpu:{name}:*"):
            c.delete(k)
    except Exception:
        pass


def _cfg(resource_name, **kw):
    base = dict(
        redis_url=REDIS_URL, resource_name=resource_name,
        lock_ttl_seconds=3, heartbeat_seconds=1,
        request_ttl_seconds=3, request_heartbeat_seconds=1,
        poll_interval_seconds=0.05, poll_jitter_seconds=0.02,
        wait_timeout_seconds=20,
    )
    base.update(kw)
    return GPUManagerConfig(**base)


# ---- workers (top-level p/ serem "picklable" no spawn) -------------------- #
def _worker_hold(resource_name, service, priority, hold, start_delay, result_key):
    time.sleep(start_delay)
    import redis as _r

    cfg = _cfg(resource_name)
    mgr = GPUResourceManager(cfg)
    with mgr.acquire(service=service, priority=priority):
        t0 = time.time()
        time.sleep(hold)
        t1 = time.time()
    _r.Redis.from_url(REDIS_URL).rpush(result_key, json.dumps([service, t0, t1, priority]))
    mgr.close()


def _worker_die_holding(resource_name, ready_key):
    """Adquire e morre (os.kill) sem liberar — o TTL deve recuperar o recurso."""
    cfg = _cfg(resource_name)
    mgr = GPUResourceManager(cfg)
    lease = mgr.acquire_lease(service="crasher", priority=1)
    import redis as _r

    _r.Redis.from_url(REDIS_URL).set(ready_key, lease.token)
    os.kill(os.getpid(), 9)  # kill -9: nem heartbeat nem release rodam


def test_mutual_exclusion_and_priority_across_processes(resource_name):
    result_key = f"res:{resource_name}"
    ctx = mp.get_context("spawn")
    procs = [
        # dono inicial (mineru, prio 20) começa primeiro e segura 1.5s
        ctx.Process(target=_worker_hold, args=(resource_name, "mineru", 20, 1.5, 0.0, result_key)),
        # bge (prio 30) e ext (prio 10) chegam enquanto mineru segura
        ctx.Process(target=_worker_hold, args=(resource_name, "bge", 30, 0.4, 0.4, result_key)),
        ctx.Process(target=_worker_hold, args=(resource_name, "ext", 10, 0.4, 0.6, result_key)),
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=30)

    import redis

    raw = redis.Redis.from_url(REDIS_URL, decode_responses=True).lrange(result_key, 0, -1)
    runs = [json.loads(r) for r in raw]
    assert len(runs) == 3, runs
    runs.sort(key=lambda r: r[1])  # ordena por t0

    # 1) exclusão mútua: intervalos [t0,t1] não se sobrepõem
    for i in range(len(runs) - 1):
        assert runs[i][2] <= runs[i + 1][1] + 1e-3, ("OVERLAP", runs)

    order = [r[0] for r in runs]
    # 2) mineru rodou primeiro (já era dono); ext (prio10) antes de bge (prio30)
    assert order[0] == "mineru", order
    assert order.index("ext") < order.index("bge"), order


def test_ttl_recovers_lock_after_kill9(resource_name):
    ready_key = f"ready:{resource_name}"
    ctx = mp.get_context("spawn")
    p = ctx.Process(target=_worker_die_holding, args=(resource_name, ready_key))
    p.start()
    p.join(timeout=15)

    import redis

    c = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    # o lock foi adquirido...
    assert c.get(ready_key) is not None
    # ...e o TTL (3s) deve recuperá-lo mesmo sem release
    cfg = _cfg(resource_name)
    mgr = GPUResourceManager(cfg)
    lease = mgr.acquire_lease(service="successor", priority=1)  # espera o TTL expirar
    try:
        assert lease.is_valid
    finally:
        lease.release()
        mgr.close()
