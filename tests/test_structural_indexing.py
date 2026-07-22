"""Integração do caminho estrutural de indexação (§32): persistência JSONL,
payload do Qdrant, ordem upsert→delete (§27), ausência de chunks na chain.

Sem GPU e sem Qdrant real: embedder e client são fakes.
"""

import json
from pathlib import Path

import pytest

from backend.indexing import index_chunks
from backend.indexing.token_counter import WhitespaceTokenCounter, set_token_counter

_SAMPLE = Path("output/despesas-educacao-irpf-relatorio-de-avaliacao/hybrid_auto/"
               "despesas-educacao-irpf-relatorio-de-avaliacao_content_list_v2.json")

pytestmark = pytest.mark.skipif(not _SAMPLE.exists(), reason="amostra content_list ausente")


class _FakeEmbedder:
    def embed_documents(self, texts, batch_size=32, normalize=False, *, task_id=None, document_id=None):
        dense = [[0.1, 0.2, 0.3] for _ in texts]
        lexical = [{"1": 0.5, "2": 0.3} for _ in texts]
        return dense, lexical


class _FakeClient:
    def __init__(self):
        self.calls = []  # ordem de operações

    def upsert(self, collection_name, points):
        self.calls.append(("upsert", len(points)))
        self._last_points = points

    def delete(self, collection_name, points_selector):
        self.calls.append(("delete", points_selector))


@pytest.fixture
def fake_indexer(monkeypatch):
    set_token_counter(WhitespaceTokenCounter())
    client, embedder = _FakeClient(), _FakeEmbedder()
    monkeypatch.setattr(index_chunks, "_get_indexer", lambda: (client, None, embedder))
    yield client, embedder
    set_token_counter(None)


def test_structural_indexing_end_to_end(fake_indexer, tmp_path):
    client, _ = fake_indexer
    jsonl = tmp_path / "chunks.jsonl"
    report = tmp_path / "chunking_report.json"
    result = index_chunks.index_materialized_document(
        _SAMPLE, doc_id="despesas", doc_path="despesas.md",
        document_checksum="deadbeef", source_artifact_uri="minio://b/cl.json",
        chunks_jsonl_path=jsonl, chunking_report_path=report,
        strategy="structural_tokens",
    )
    # chunks persistidos em JSONL, uma linha por chunk, SEM embeddings
    assert jsonl.exists()
    lines = jsonl.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == result["n_chunks"] > 0
    first = json.loads(lines[0])
    assert "text" in first and not any(k in first for k in ("dense", "sparse", "embedding"))

    # relatório de chunking gerado
    assert report.exists()
    assert result["chunking_report"]["chunk_count"] == result["n_chunks"]
    assert result["chunking_strategy"] == "structural_tokens"

    # ordem §27: TODOS os upserts acontecem ANTES do delete de versões antigas
    ops = [c[0] for c in client.calls]
    assert "upsert" in ops and ops[-1] == "delete"
    assert ops.index("delete") == len(ops) - 1

    # payload do Qdrant contém metadados estruturais (§26) e não vaza objetos grandes
    p = client._last_points[0].payload
    for key in ("chunk_id", "content_type", "token_count", "section_path",
                "chunking_strategy", "document_version", "active", "page_numbers"):
        assert key in p
    assert p["active"] is True


def test_no_chunks_transported_in_chain_payload():
    """A validação anti-regressão rejeita chunks/markdown na chain (§26/§27)."""
    from backend.core.schemas import ForbiddenChainPayloadError, validate_chain_payload_size

    with pytest.raises(ForbiddenChainPayloadError):
        validate_chain_payload_size({"chunks": [1, 2, 3]}, 16384)
