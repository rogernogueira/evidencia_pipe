"""Testes do índice de jobs em execução (job_store) e da rota GET /api/files/active.

O índice é exercitado no fallback em memória (sem Redis); a rota usa um app mínimo
com só o `files.router`, para não puxar a lifespan/StaticFiles do main.
"""

import os
import sys

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)


@pytest.fixture
def js(monkeypatch):
    import backend.services.job_store as job_store

    # Força o fallback em memória (sem Redis) e isola o estado por teste.
    monkeypatch.setattr(job_store, "_redis_ready", True)
    monkeypatch.setattr(job_store, "_redis", None)
    monkeypatch.setattr(job_store, "_jobs", {})
    monkeypatch.setattr(job_store, "_failed", {})
    monkeypatch.setattr(job_store, "_active", {})
    return job_store


def test_enqueued_and_processing_are_active(js):
    js.set_status("a", "na_fila", filename="a.pdf")
    js.set_status("b", "processando", stage="mineru", filename="b.pdf")

    active = js.list_active()
    assert [j["job_id"] for j in active] == ["b", "a"]  # mais recente primeiro
    assert active[0]["stage"] == "mineru"


def test_terminal_status_leaves_the_index(js):
    js.set_status("a", "na_fila")
    js.set_status("a", "processando", stage="index")
    assert [j["job_id"] for j in js.list_active()] == ["a"]

    js.set_status("a", "concluido", stage="index")
    assert js.list_active() == []

    js.set_status("b", "processando")
    js.set_status("b", "erro", error="boom")
    assert js.list_active() == []


def test_list_prunes_stale_entries(js):
    # 'ghost' está no índice mas não tem registro de job (TTL expirou) → é podado.
    js._active["ghost"] = 1.0
    js.set_status("real", "processando")
    assert [j["job_id"] for j in js.list_active()] == ["real"]
    assert "ghost" not in js._active


def test_limit_is_respected(js):
    for i in range(5):
        js.set_status(f"j{i}", "processando")
    assert len(js.list_active(limit=3)) == 3


def test_route_lists_ids_and_summary(js):
    from backend.api.routes import files as files_route

    js.set_status("a", "na_fila", filename="a.pdf", item_uuid="u-1")
    js.set_status("b", "processando", stage="mineru", filename="b.pdf")
    js.set_status("c", "concluido", stage="index")

    app = FastAPI()
    app.include_router(files_route.router)
    resp = TestClient(app).get("/api/files/active")

    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 2
    assert body["job_ids"] == ["b", "a"]
    assert body["jobs"][1] == {
        "job_id": "a", "status": "na_fila", "stage": None,
        "filename": "a.pdf", "item_uuid": "u-1",
        "updated_at": body["jobs"][1]["updated_at"],
    }
