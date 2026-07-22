"""Testes das rotas de busca semântica (backend/api/routes/search.py).

Sem Qdrant/bge-m3 reais: a dependência `get_semantic_search` é sobrescrita por um
fake assíncrono que registra as chamadas. Usa o TestClient (síncrono) do Starlette,
que executa os endpoints async sem precisar de pytest-asyncio. Montamos um app
mínimo só com o `search.router` para não puxar a lifespan/StaticFiles do main.
"""

import os
import sys

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from backend.api.dependencies import get_semantic_search  # noqa: E402
from backend.api.routes import search as search_route  # noqa: E402
from backend.core.config import QDRANT_COLLECTION  # noqa: E402
from backend.core.schemas import SearchResult  # noqa: E402


class FakeSemanticSearch:
    """Dublê assíncrono de SemanticSearch — registra chamadas e devolve o que for
    configurado, sem tocar Qdrant nem o embedder."""

    def __init__(self, *, connected=True, results=None, health=None):
        self._connected = connected
        self._results = results if results is not None else []
        self._health = health if health is not None else {}
        self.ensure_connected_calls = 0
        self.search_calls = []
        self.health_calls = 0

    async def ensure_connected(self):
        self.ensure_connected_calls += 1
        return self._connected

    async def search(self, query, limit=10, doc_id=None, uuid=None, type="hybrid"):
        self.search_calls.append(
            {"query": query, "limit": limit, "doc_id": doc_id, "uuid": uuid, "type": type}
        )
        return self._results

    async def health(self):
        self.health_calls += 1
        return self._health


def make_client(fake):
    """App mínimo com só o router de busca e a dependência sobrescrita."""
    app = FastAPI()
    app.include_router(search_route.router)
    app.dependency_overrides[get_semantic_search] = lambda: fake
    return TestClient(app)


def _sample_result():
    return SearchResult(
        doc_name="relatorio.md",
        doc_id="relatorio.pdf",
        doc_path="/output/relatorio/relatorio.md",
        section="Introdução",
        page=3,
        snippet="Trecho encontrado.",
        score=0.9876,
        type="paragraph",
        item_uuid="uuid-123",
        item_handle="123456789/1",
    )


# --------------------------------------------------------------------------
# GET /api/search/semantic
# --------------------------------------------------------------------------

def test_semantic_happy_path_returns_results():
    fake = FakeSemanticSearch(results=[_sample_result()])
    client = make_client(fake)

    resp = client.get("/api/search/semantic", params={"q": "orçamento"})

    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, list) and len(body) == 1
    assert body[0]["doc_id"] == "relatorio.pdf"
    assert body[0]["snippet"] == "Trecho encontrado."
    assert body[0]["score"] == 0.9876
    assert fake.ensure_connected_calls == 1


def test_semantic_forwards_query_params_to_search():
    fake = FakeSemanticSearch(results=[])
    client = make_client(fake)

    resp = client.get(
        "/api/search/semantic",
        params={"q": "meta", "limit": 25, "type": "dense", "doc_id": "a.pdf", "uuid": "u-9"},
    )

    assert resp.status_code == 200
    assert len(fake.search_calls) == 1
    call = fake.search_calls[0]
    assert call == {"query": "meta", "limit": 25, "doc_id": "a.pdf", "uuid": "u-9", "type": "dense"}


def test_semantic_empty_filters_become_none():
    """doc_id/uuid vazios (default) viram None ao chamar o serviço (`x or None`)."""
    fake = FakeSemanticSearch(results=[])
    client = make_client(fake)

    resp = client.get("/api/search/semantic", params={"q": "x"})

    assert resp.status_code == 200
    call = fake.search_calls[0]
    assert call["doc_id"] is None
    assert call["uuid"] is None


def test_semantic_uses_defaults_when_params_omitted():
    fake = FakeSemanticSearch(results=[])
    client = make_client(fake)

    resp = client.get("/api/search/semantic")

    assert resp.status_code == 200
    call = fake.search_calls[0]
    assert call["query"] == ""
    assert call["limit"] == 10
    assert call["type"] == "hybrid"


def test_semantic_unavailable_returns_503():
    fake = FakeSemanticSearch(connected=False)
    client = make_client(fake)

    resp = client.get("/api/search/semantic", params={"q": "x"})

    assert resp.status_code == 503
    body = resp.json()
    assert "error" in body
    assert QDRANT_COLLECTION in body["error"]
    # Não deve chegar a consultar quando indisponível.
    assert fake.search_calls == []


@pytest.mark.parametrize("limit", [0, -1, 51, 100])
def test_semantic_limit_out_of_range_is_422(limit):
    fake = FakeSemanticSearch(results=[])
    client = make_client(fake)

    resp = client.get("/api/search/semantic", params={"q": "x", "limit": limit})

    assert resp.status_code == 422
    assert fake.search_calls == []


@pytest.mark.parametrize("limit", [1, 50])
def test_semantic_limit_boundaries_ok(limit):
    fake = FakeSemanticSearch(results=[])
    client = make_client(fake)

    resp = client.get("/api/search/semantic", params={"q": "x", "limit": limit})

    assert resp.status_code == 200
    assert fake.search_calls[0]["limit"] == limit


def test_semantic_non_integer_limit_is_422():
    fake = FakeSemanticSearch(results=[])
    client = make_client(fake)

    resp = client.get("/api/search/semantic", params={"q": "x", "limit": "abc"})

    assert resp.status_code == 422
    assert fake.search_calls == []


# --------------------------------------------------------------------------
# GET /api/search/status
# --------------------------------------------------------------------------

def test_status_returns_health_payload():
    health = {
        "semantic": True,
        "qdrant_url": "http://qdrant:6333",
        "collection": QDRANT_COLLECTION,
        "dense_model": "BAAI/bge-m3",
    }
    fake = FakeSemanticSearch(health=health)
    client = make_client(fake)

    resp = client.get("/api/search/status")

    assert resp.status_code == 200
    assert resp.json() == health
    assert fake.health_calls == 1


def test_status_reflects_unavailable():
    fake = FakeSemanticSearch(health={"semantic": False})
    client = make_client(fake)

    resp = client.get("/api/search/status")

    assert resp.status_code == 200
    assert resp.json()["semantic"] is False
