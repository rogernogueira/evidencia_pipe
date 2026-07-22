"""Testes da fila de falhas do job_store (índice de jobs reprocessáveis).

Exercita add/clear/list no fallback em memória (sem Redis), incluindo a poda de
entradas cujo registro de job já expirou.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def js(monkeypatch):
    import backend.services.job_store as job_store

    # Força o fallback em memória (sem Redis) e isola o estado por teste.
    monkeypatch.setattr(job_store, "_redis_ready", True)
    monkeypatch.setattr(job_store, "_redis", None)
    monkeypatch.setattr(job_store, "_jobs", {})
    monkeypatch.setattr(job_store, "_failed", {})
    return job_store


def test_add_list_orders_most_recent_first(js):
    js.set_status("a", "erro", stage="mineru", error="boom")
    js.add_failed("a")
    js.set_status("b", "erro", stage="index", error="x")
    js.add_failed("b")

    failed = js.list_failed()
    assert [j["job_id"] for j in failed] == ["b", "a"]  # mais recente primeiro
    assert failed[0]["status"] == "erro" and failed[0]["stage"] == "index"


def test_clear_failed_removes_from_index(js):
    js.set_status("a", "erro")
    js.add_failed("a")
    js.clear_failed("a")
    assert js.list_failed() == []


def test_list_prunes_stale_entries(js):
    # 'ghost' está no índice mas não tem registro de job (expirou) → deve ser podado.
    js.add_failed("ghost")
    js.set_status("real", "erro")
    js.add_failed("real")
    assert [j["job_id"] for j in js.list_failed()] == ["real"]
    assert "ghost" not in js._failed


def test_limit_is_respected(js):
    for i in range(5):
        js.set_status(f"j{i}", "erro")
        js.add_failed(f"j{i}")
    assert len(js.list_failed(limit=3)) == 3
