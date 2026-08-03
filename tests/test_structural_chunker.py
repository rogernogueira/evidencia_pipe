"""Testes do StructuralTokenChunker (§32).

Usa WhitespaceTokenCounter (1 token = 1 palavra) + config pequena para tornar os
limites determinísticos e sem GPU/transformers.
"""

import json

import pytest

from backend.indexing.chunk_models import DocumentBlock
from backend.indexing.structural_token_chunker import ChunkingConfig, StructuralTokenChunker
from backend.indexing.token_counter import WhitespaceTokenCounter


def _cfg(**kw):
    # Fixa os modos em v1 (isolamento de teste — não depende do .env ambiente).
    base = dict(target_tokens=10, max_tokens=20, min_tokens=3, overlap_tokens=3,
                max_overlap_tokens=6, table_max_tokens=15, list_max_tokens=12,
                force_split_above_tokens=40, chunking_version="test-v1",
                front_matter_mode="include", references_mode="separate",
                appendix_mode="flat", equation_mode="raw", table_mode="always")
    base.update(kw)
    return ChunkingConfig(**base)


def _chunker(**kw):
    return StructuralTokenChunker(config=_cfg(**kw), token_counter=WhitespaceTokenCounter())


def _para(text, order, section, page=1, level=1):
    return DocumentBlock(block_id=f"block-{order:04d}", block_type="paragraph", text=text,
                         order_index=order, page_number=page, section_path=section, heading_level=level)


def _words(n, prefix="palavra"):
    return " ".join(f"{prefix}{i}" for i in range(n))


def test_groups_paragraphs_up_to_target():
    blocks = [_para(_words(5), 0, ["S"]), _para(_words(6), 1, ["S"])]
    res = _chunker().chunk(blocks, document_id="d")
    # 5+6 = 11 >= target(10) → fecha em 1 chunk
    assert res.metrics.chunk_count == 1
    assert res.chunks[0].content_type == "paragraph"


def test_does_not_cross_sections():
    blocks = [_para(_words(4), 0, ["A"]), _para(_words(4), 1, ["B"])]
    res = _chunker().chunk(blocks, document_id="d")
    assert res.metrics.chunk_count == 2
    assert {c.section_title for c in res.chunks} == {"A", "B"}
    # sem overlap entre seções distintas
    assert all(c.overlap_token_count == 0 for c in res.chunks)


def test_oversized_paragraph_split_by_sentence():
    text = ". ".join(f"Frase numero {i} aqui tem varias palavras extras" for i in range(6)) + "."
    blocks = [_para(text, 0, ["S"])]
    res = _chunker().chunk(blocks, document_id="d")
    assert res.metrics.chunk_count >= 2
    assert any(c.split_method == "sentence" for c in res.chunks)
    assert res.metrics.sentence_split_count >= 1


def test_token_fallback_for_single_huge_sentence():
    # uma "sentença" única maior que max_tokens, sem pontuação → token_fallback
    blocks = [_para(_words(50), 0, ["S"])]
    res = _chunker().chunk(blocks, document_id="d")
    assert any(c.split_method == "token_fallback" for c in res.chunks)
    assert res.metrics.token_fallback_split_count >= 1
    # nenhum chunk textual comum acima do limite absoluto
    assert all(c.token_count <= 40 for c in res.chunks if c.content_type == "paragraph")


def test_small_table_preserved():
    tbl = DocumentBlock(block_id="block-0000", block_type="table",
                        text="Cap\n| A | B |\n| --- | --- |\n| 1 | 2 |", order_index=0,
                        page_number=3, section_path=["S"],
                        metadata={"table_caption": "Cap", "table_markdown": "| A | B |\n| --- | --- |\n| 1 | 2 |"})
    res = _chunker().chunk([tbl], document_id="d")
    tchunks = [c for c in res.chunks if c.content_type == "table"]
    assert len(tchunks) == 1
    assert tchunks[0].page_start == 3
    assert tchunks[0].metadata.get("is_table_continuation") is False


def test_large_table_split_repeats_header():
    rows = "\n".join(f"| r{i}a | r{i}b |" for i in range(20))
    md = "| A | B |\n| --- | --- |\n" + rows
    tbl = DocumentBlock(block_id="block-0000", block_type="table", text="Cap\n" + md,
                        order_index=0, page_number=2, section_path=["S"],
                        metadata={"table_caption": "Cap", "table_markdown": md})
    res = _chunker(table_max_tokens=12).chunk([tbl], document_id="d")
    tchunks = [c for c in res.chunks if c.content_type == "table"]
    assert len(tchunks) >= 2
    # cabeçalho repetido em cada parte
    assert all("| A | B |" in c.text for c in tchunks)
    # intervalo de linhas contíguo e continuação marcada
    assert tchunks[1].metadata.get("is_table_continuation") is True
    # sem overlap textual entre partes de tabela
    assert all(c.overlap_token_count == 0 for c in tchunks)


def test_list_preserved_and_continuation():
    items = "\n".join(f"- item {i} com texto" for i in range(12))
    lst = DocumentBlock(block_id="block-0000", block_type="list", text=items, order_index=0,
                        page_number=1, section_path=["S"],
                        metadata={"list_type": "unordered", "item_count": 12})
    res = _chunker(list_max_tokens=12).chunk([lst], document_id="d")
    lchunks = [c for c in res.chunks if c.content_type == "list"]
    assert len(lchunks) >= 2
    assert lchunks[1].metadata.get("is_list_continuation") is True


def test_structural_overlap_within_section():
    # dois parágrafos que fecham em chunks distintos e recebem overlap estrutural
    blocks = [_para(_words(12, "a"), 0, ["S"]), _para(_words(12, "b"), 1, ["S"])]
    res = _chunker(target_tokens=10, max_tokens=30, overlap_tokens=4, max_overlap_tokens=8).chunk(blocks, document_id="d")
    assert res.metrics.chunk_count == 2
    second = res.chunks[1]
    assert second.overlap_token_count > 0
    assert second.overlap_source_chunk_id == res.chunks[0].chunk_id
    assert "overlap_block_ids" in second.metadata


def test_references_separate_content_type():
    ref = DocumentBlock(block_id="block-0000", block_type="reference",
                        text="AUTOR, A. Obra completa com titulo longo. Editora, 2021.",
                        order_index=0, page_number=9, section_path=["Referências"])
    res = _chunker().chunk([ref], document_id="d")
    assert res.chunks[0].content_type == "references"
    assert res.chunks[0].metadata.get("is_reference_section") is True
    assert res.chunks[0].overlap_token_count == 0


def test_contextualized_text_and_min_max_limits():
    blocks = [_para(_words(8), 0, ["Cap 1", "Seção A"])]
    res = _chunker().chunk(blocks, document_id="d", document_title="Meu Documento")
    c = res.chunks[0]
    assert c.contextualized_text.startswith("Documento: Meu Documento")
    assert "Seção: Cap 1 > Seção A" in c.contextualized_text
    assert c.section_path == ["Cap 1", "Seção A"]
    assert c.embedding_token_count >= c.token_count


def test_deterministic_and_no_embeddings_serialization():
    blocks = [_para(_words(8), 0, ["S"])]
    ch = _chunker()
    r1 = ch.chunk(blocks, document_id="d", document_checksum="abc")
    r2 = ch.chunk(blocks, document_id="d", document_checksum="abc")
    assert [c.chunk_id for c in r1.chunks] == [c.chunk_id for c in r2.chunks]
    dumped = r1.chunks[0].model_dump()
    assert not any(k in dumped for k in ("embedding", "dense", "sparse", "vector"))
    json.dumps(dumped, ensure_ascii=False)  # não levanta


def test_hard_limit_enforced_for_common_text():
    # força um chunk de parágrafo acima do force_split_above_tokens → erro
    huge = DocumentBlock(block_id="block-0000", block_type="paragraph",
                         text=_words(100), order_index=0, section_path=["S"])
    ch = _chunker(max_tokens=200, force_split_above_tokens=50, target_tokens=40)
    with pytest.raises(Exception):
        ch.chunk([huge], document_id="d")
