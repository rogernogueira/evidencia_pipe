"""Testes de chunk_models (§32: IDs determinísticos, hash de config, ausência de
embeddings, serialização JSONL, fallback do texto de embedding)."""

import json

from backend.indexing.chunk_models import (
    StructuralChunk,
    compute_config_hash,
    make_chunk_id,
)


def test_config_hash_stable_and_order_independent():
    a = compute_config_hash({"x": 1, "y": 2})
    b = compute_config_hash({"y": 2, "x": 1})
    assert a == b
    assert compute_config_hash({"x": 1, "y": 3}) != a


def test_chunk_id_deterministic():
    kw = dict(document_id="doc", bitstream_uuid="bs", document_checksum="sha",
              chunking_version="v1", chunking_config_hash="h", chunk_index=3,
              normalized_text="um texto normal")
    assert make_chunk_id(**kw) == make_chunk_id(**kw)
    # muda com o índice
    kw2 = dict(kw, chunk_index=4)
    assert make_chunk_id(**kw2) != make_chunk_id(**kw)
    # robusto a reflow de whitespace
    kw3 = dict(kw, normalized_text="um   texto    normal")
    assert make_chunk_id(**kw3) == make_chunk_id(**kw)


def test_no_embeddings_in_chunk_dump():
    c = StructuralChunk(chunk_id="x", document_id="d", chunk_index=0, text="t",
                        token_count=1, character_count=1, content_type="paragraph")
    dumped = c.model_dump()
    assert not any(k in dumped for k in ("embedding", "dense", "sparse", "vector"))
    # roundtrip JSONL
    line = json.dumps(dumped, ensure_ascii=False)
    assert json.loads(line)["chunk_id"] == "x"


def test_embedding_input_fallback():
    c = StructuralChunk(chunk_id="x", document_id="d", chunk_index=0, text="corpo",
                        contextualized_text=None, token_count=1, character_count=5,
                        content_type="paragraph")
    assert c.embedding_input("contextualized_text") == "corpo"
    c.contextualized_text = "Documento: T\n\ncorpo"
    assert c.embedding_input("contextualized_text").startswith("Documento:")
    assert c.embedding_input("text") == "corpo"
