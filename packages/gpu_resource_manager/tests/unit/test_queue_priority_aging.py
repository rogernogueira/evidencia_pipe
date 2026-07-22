"""Cenários 9-19, 21 (§39): timeout, cancelamento, órfãs, FIFO, prioridade,
não-preempção, aging, anti-starvation, atomicidade concorrente, isolamento de recursos."""

import threading
import time

import pytest

from gpu_resource_manager import GPUAcquisitionTimeout, GPURequestCancelled
from gpu_resource_manager.queue import effective_priority


def _acquire_and_hold(mgr, service, priority, hold, order, barrier=None):
    if barrier:
        barrier.wait()
    with mgr.acquire(service=service, priority=priority):
        order.append(("start", service, time.time()))
        time.sleep(hold)
        order.append(("end", service, time.time()))


def _no_overlap(order):
    iv = {}
    for kind, svc, ts in order:
        iv.setdefault(svc, {})[kind] = ts
    segs = sorted((v["start"], v["end"], s) for s, v in iv.items())
    return all(segs[i][1] <= segs[i + 1][0] + 1e-6 for i in range(len(segs) - 1)), segs


def test_wait_timeout_raises_and_cleans(manager_factory):
    holder = manager_factory()
    waiter = manager_factory(wait_timeout_seconds=1)
    lease = holder.acquire_lease(service="holder", priority=1)
    try:
        with pytest.raises(GPUAcquisitionTimeout) as ei:
            waiter.acquire_lease(service="late", priority=1)
        exc = ei.value
        assert exc.resource == waiter.config.resource_name
        assert exc.wait_time_seconds >= 1
        assert exc.current_owner and exc.current_owner["service"] == "holder"
        # solicitação foi removida da fila no timeout
        assert holder.get_status().queue_size == 0
    finally:
        lease.release()


def test_cancel_request_is_idempotent(manager_factory):
    holder = manager_factory()
    lease = holder.acquire_lease(service="holder", priority=1)
    got = {}

    def waiter():
        w = manager_factory(wait_timeout_seconds=8)
        try:
            w.acquire_lease(service="w", priority=1)
        except GPURequestCancelled:
            got["cancelled"] = True

    t = threading.Thread(target=waiter, daemon=True)
    t.start()
    # espera a solicitação aparecer na fila e cancela
    for _ in range(200):
        q = holder.get_queue()
        if q:
            break
        time.sleep(0.01)
    rid = holder.get_queue()[0]["request_id"]
    assert holder.cancel_request(rid) is True
    assert holder.cancel_request(rid) is False  # idempotente
    t.join(timeout=3)
    lease.release()
    assert got.get("cancelled") is True


def test_orphan_request_removed(manager):
    # injeta um membro na fila SEM request key (órfão) e outro válido
    res = manager.config.resource_name
    c = manager.backend.client
    c.zadd(manager.config.key_queue(res), {"orphan-id": 1})
    # uma tentativa de acquire de um request válido limpa o órfão
    lease = manager.acquire_lease(service="valid", priority=1)
    try:
        assert c.zscore(manager.config.key_queue(res), "orphan-id") is None
    finally:
        lease.release()


def test_fifo_same_priority(manager_factory):
    holder = manager_factory()
    lease = holder.acquire_lease(service="holder", priority=50)
    order = []
    threads = []
    # três esperando mesma prioridade; devem sair na ordem de chegada
    for name in ("first", "second", "third"):
        m = manager_factory(wait_timeout_seconds=10, aging_enabled=False)
        t = threading.Thread(target=_acquire_and_hold, args=(m, name, 50, 0.1, order), daemon=True)
        t.start()
        threads.append(t)
        time.sleep(0.15)  # garante ordem de enfileiramento
    time.sleep(0.2)
    lease.release()
    for t in threads:
        t.join(timeout=5)
    ok, segs = _no_overlap(order)
    assert ok, segs
    starts = [s for _, _, s in segs]
    assert starts == ["first", "second", "third"], starts


def test_higher_priority_wins_and_does_not_preempt(manager_factory):
    holder = manager_factory()
    lease = holder.acquire_lease(service="mineru", priority=20)  # já dono
    order = []
    order.append(("start", "mineru", time.time()))
    low = manager_factory(wait_timeout_seconds=10)
    high = manager_factory(wait_timeout_seconds=10)
    t_low = threading.Thread(target=_acquire_and_hold, args=(low, "bge", 30, 0.1, order), daemon=True)
    t_high = threading.Thread(target=_acquire_and_hold, args=(high, "ext", 10, 0.1, order), daemon=True)
    t_low.start(); time.sleep(0.1)
    t_high.start(); time.sleep(0.2)
    # não-preempção: mineru continua dono até liberar
    assert holder.get_status().owner["service"] == "mineru"
    order.append(("end", "mineru", time.time()))
    lease.release()
    t_low.join(timeout=5); t_high.join(timeout=5)
    ok, segs = _no_overlap(order)
    assert ok, segs
    starts = [s for _, _, s in segs]
    assert starts[0] == "mineru"
    assert starts.index("ext") < starts.index("bge"), starts  # prio 10 antes de 30


def test_aging_raises_old_low_priority(manager):
    cfg = manager.config
    now = 1000.0
    # baixa prioridade esperando muito supera alta recém-chegada
    old_low = effective_priority(cfg, base_priority=100, enqueued_at=now - 100, now=now)
    fresh_high = effective_priority(cfg, base_priority=30, enqueued_at=now, now=now)
    assert old_low < fresh_high  # aging trouxe a antiga para frente


def test_anti_starvation_end_to_end(manager_factory):
    # aging agressivo: uma tarefa de baixa prioridade presa atrás de várias de alta
    # acaba sendo escolhida. Verificamos via prioridade efetiva no snapshot da fila.
    holder = manager_factory()
    lease = holder.acquire_lease(service="holder", priority=1)
    low = manager_factory(wait_timeout_seconds=10, aging_interval_seconds=1, aging_step=50)
    order = []
    t = threading.Thread(target=_acquire_and_hold, args=(low, "low", 100, 0.05, order), daemon=True)
    t.start()
    time.sleep(1.2)  # deixa envelhecer > 1 intervalo
    q = holder.get_queue()
    assert q and q[0]["service"] == "low"
    assert q[0]["effective_priority"] < 100  # envelheceu
    lease.release()
    t.join(timeout=5)
    assert ("start", "low") in [(k, s) for k, s, _ in order]


def test_concurrent_acquire_mutual_exclusion(manager_factory):
    order = []
    barrier = threading.Barrier(2)
    ms = [manager_factory(wait_timeout_seconds=10) for _ in range(2)]
    threads = [
        threading.Thread(target=_acquire_and_hold, args=(ms[i], f"c{i}", 50, 0.2, order, barrier), daemon=True)
        for i in range(2)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=6)
    ok, segs = _no_overlap(order)
    assert ok, ("OVERLAP", segs)
    assert len(segs) == 2


def test_resource_isolation(manager):
    with manager.acquire(resource="gpu0", service="a", priority=1):
        assert manager.get_status(resource="gpu0").locked is True
        assert manager.get_status(resource="gpu1").locked is False
        # adquirir gpu1 é independente
        with manager.acquire(resource="gpu1", service="b", priority=1):
            assert manager.get_status(resource="gpu1").locked is True
