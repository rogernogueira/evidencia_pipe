"""Testes do índice de jobs bem sucedidos (job_store) e da rota GET /api/files/succeeded.

Cobre o caso que distingue este índice do simples `status == "concluido"`: um job que
extraiu mas falhou ao indexar conclui COM `index_error` e NÃO é sucesso (pertence à fila
de falhas). Exercitado no fallback em memória (sem Redis).
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
    monkeypatch.setattr(job_store, "_succeeded", {})
    return job_store


def _run_ok(js, job_id, **extra):
    """Simula o caminho feliz do pipeline (na_fila → processando → concluido)."""
    js.set_status(job_id, "na_fila", filename=f"{job_id}.pdf", **extra)
    js.set_status(job_id, "processando", stage="index")
    js.set_status(job_id, "concluido", stage="index", n_chunks=12, indexed_count=12, index_error=None)


def test_lists_most_recent_first_and_leaves_active(js):
    _run_ok(js, "a")
    _run_ok(js, "b")

    assert [j["job_id"] for j in js.list_succeeded()] == ["b", "a"]  # mais recente primeiro
    assert js.list_active() == []  # concluir tira do índice de execução


def test_concluido_with_index_error_is_not_success(js):
    js.set_status("a", "processando", stage="index")
    js.set_status("a", "concluido", stage="index", index_error="Qdrant fora do ar")
    assert js.list_succeeded() == []

    # Reprocessado com sucesso (index_error limpo) → passa a contar como sucesso.
    js.set_status("a", "concluido", stage="index", index_error=None, n_chunks=3)
    assert [j["job_id"] for j in js.list_succeeded()] == ["a"]


def test_index_error_persisted_survives_a_later_status_write(js):
    """O follow-up de enrich reescreve status=concluido sem repassar index_error — o
    índice olha o registro mesclado, então o job segue fora do sucesso."""
    js.set_status("a", "concluido", stage="index", index_error="boom")
    js.set_status("a", "concluido", stage="llm", warnings_count=2)
    assert js.get_job("a")["index_error"] == "boom"
    assert js.list_succeeded() == []


def test_requeue_removes_from_success(js):
    _run_ok(js, "a")
    js.set_status("a", "na_fila")  # reprocessamento
    assert js.list_succeeded() == []
    assert [j["job_id"] for j in js.list_active()] == ["a"]


def test_error_removes_from_success(js):
    _run_ok(js, "a")
    js.set_status("a", "erro", stage="mineru", error="boom")
    assert js.list_succeeded() == []


def test_list_prunes_stale_entries(js):
    # 'ghost' está no índice mas não tem registro de job (TTL expirou) → é podado.
    js._succeeded["ghost"] = 1.0
    _run_ok(js, "real")
    assert [j["job_id"] for j in js.list_succeeded()] == ["real"]
    assert "ghost" not in js._succeeded


def test_limit_is_respected(js):
    for i in range(5):
        _run_ok(js, f"j{i}")
    assert len(js.list_succeeded(limit=3)) == 3


def test_route_lists_ids_and_summary(js):
    from backend.api.routes import files as files_route

    _run_ok(js, "a", item_uuid="u-1")
    js.set_status("a", "concluido", pipeline_id="p-1", document_id="a")
    js.set_status("b", "concluido", stage="index", index_error="boom")
    js.set_status("c", "processando", stage="mineru")

    app = FastAPI()
    app.include_router(files_route.router)
    resp = TestClient(app).get("/api/files/succeeded")

    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 1
    assert body["job_ids"] == ["a"]
    assert body["jobs"][0] == {
        "job_id": "a", "status": "concluido", "filename": "a.pdf", "item_uuid": "u-1",
        "chunk_count": 12, "indexed_count": 12, "artifact_id": "p-1/a",
        "updated_at": body["jobs"][0]["updated_at"],
    }
