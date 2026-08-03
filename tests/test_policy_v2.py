"""Testes da Política de Chunking v2 (§4/§5/§16/§21/§22).

Cobrem: classificador de seção (numeração + máquina de estados), perfis de
recuperação, enriquecimento do parser (printed_page_number, section_kind, siglas,
furniture) e os modos do chunker (front_matter/references metadata_only).

Usa WhitespaceTokenCounter para limites determinísticos, sem GPU/transformers.
"""

import pytest

from backend.indexing.chunk_models import (
    SECTION_ACRONYM_LIST,
    SECTION_ADMINISTRATIVE_APPENDIX,
    SECTION_ANALYTICAL_APPENDIX,
    SECTION_BIBLIOGRAPHY,
    SECTION_BODY,
    SECTION_FRONT_MATTER,
    SECTION_TABLE_OF_CONTENTS,
    DocumentBlock,
)
from backend.indexing.document_blocks import MinerUDocumentParser
from backend.indexing.retrieval_profile import (
    PROFILE_BIBLIOGRAPHIC,
    PROFILE_GENERAL,
    PROFILE_METHODOLOGICAL,
    PROFILE_QUANTITATIVE,
    classify_retrieval,
    detect_query_profile,
    profile_exclusions,
)
from backend.indexing.section_classifier import (
    SectionStateMachine,
    classify_appendix,
    infer_heading_level,
    parse_acronym_line,
    special_section_kind,
)
from backend.indexing.structural_token_chunker import ChunkingConfig, StructuralTokenChunker
from backend.indexing.token_counter import WhitespaceTokenCounter


# --------------------------------------------------------------------------
# §4 — Classificador de seção
# --------------------------------------------------------------------------

@pytest.mark.parametrize("text,level", [
    ("1 Introdução", 1),
    ("2 A dedução com despesa", 1),
    ("2.1 Aspecto tributário", 2),
    ("2.1.3 Progressividade", 3),
    ("Apêndice A", 1),                 # sem numeração → fallback 1
    ("Referências bibliográficas", 1),
])
def test_infer_heading_level(text, level):
    assert infer_heading_level(text, None) == level


def test_infer_level_prefers_numbering_over_mineru():
    # MinerU disse nível 2, mas a numeração diz 3.
    assert infer_heading_level("2.1.3 Progressividade", 2) == 3


def test_state_machine_full_document_flow():
    sm = SectionStateMachine()
    seq = [
        ("Relatório de Avaliação", SECTION_FRONT_MATTER),
        ("Coordenadores", SECTION_FRONT_MATTER),
        ("Sumário", SECTION_TABLE_OF_CONTENTS),
        ("Lista de siglas", SECTION_ACRONYM_LIST),
        ("1 Introdução", SECTION_BODY),
        ("2.1 Aspecto tributário", SECTION_BODY),
        ("Referências bibliográficas", SECTION_BIBLIOGRAPHY),
        ("Apêndice A – Modelo econométrico", SECTION_ANALYTICAL_APPENDIX),
        ("Apêndice B – Formulário de entrevista", SECTION_ADMINISTRATIVE_APPENDIX),
    ]
    for text, expected in seq:
        _, kind = sm.feed_heading(text, None)
        assert kind == expected, f"{text!r} → {kind} (esperado {expected})"


def test_nested_references_inside_appendix_does_not_leak():
    """Referências ANINHADA num apêndice não contamina o conteúdo seguinte (bug real)."""
    sm = SectionStateMachine()
    sm.feed_heading("1 Introdução", None)                       # body
    sm.feed_heading("Apêndice A – Modelo econométrico", None)   # analytical_appendix
    _, k_ref = sm.feed_heading("Referências", None)             # biblio (transiente)
    _, k_after = sm.feed_heading("A.5 Respostas às perguntas", None)
    assert k_ref == SECTION_BIBLIOGRAPHY
    assert k_after == SECTION_ANALYTICAL_APPENDIX               # voltou ao apêndice, não ficou biblio


def test_body_paragraph_before_and_after_first_numbered_heading():
    sm = SectionStateMachine()
    assert sm.current == SECTION_FRONT_MATTER          # capa
    sm.feed_heading("1 Introdução", None)
    assert sm.current == SECTION_BODY                   # corpo começou


def test_classify_appendix_administrative_by_keyword():
    assert classify_appendix("Apêndice B – Formulário de entrevista") == SECTION_ADMINISTRATIVE_APPENDIX
    assert classify_appendix("Apêndice C – Roteiro de aplicação") == SECTION_ADMINISTRATIVE_APPENDIX
    assert classify_appendix("Apêndice A – Modelo econométrico") == SECTION_ANALYTICAL_APPENDIX


def test_special_section_kind_none_for_plain_heading():
    assert special_section_kind("2.1 Aspecto tributário") is None
    assert special_section_kind("Lista de figuras") is not None


@pytest.mark.parametrize("line,expected", [
    ("IRPF — Imposto de Renda de Pessoa Física", ("IRPF", "Imposto de Renda de Pessoa Física")),
    ("PNE: Plano Nacional de Educação", ("PNE", "Plano Nacional de Educação")),
    ("- IPEA - Instituto de Pesquisa Econômica Aplicada", ("IPEA", "Instituto de Pesquisa Econômica Aplicada")),
])
def test_parse_acronym_line(line, expected):
    assert parse_acronym_line(line) == expected


def test_parse_acronym_line_rejects_prose():
    assert parse_acronym_line("Este é um parágrafo comum do corpo do texto.") is None


# --------------------------------------------------------------------------
# §21/§22 — Perfis de recuperação
# --------------------------------------------------------------------------

def test_classify_retrieval_body_paragraph_is_searchable():
    rc = classify_retrieval("paragraph", SECTION_BODY)
    assert rc.normalized_content_type == "body_paragraph"
    assert rc.searchable_by_default is True
    assert PROFILE_GENERAL in rc.retrieval_profile


def test_classify_retrieval_table_not_default_searchable_but_quantitative():
    rc = classify_retrieval("table", SECTION_BODY)
    assert rc.is_table is True
    assert rc.searchable_by_default is False
    assert PROFILE_QUANTITATIVE in rc.retrieval_profile


def test_classify_retrieval_reference_bibliographic_only():
    rc = classify_retrieval("references", SECTION_BIBLIOGRAPHY)
    assert rc.is_reference is True
    assert rc.searchable_by_default is False
    assert rc.retrieval_profile == [PROFILE_BIBLIOGRAPHIC]


def test_classify_retrieval_appendix_weights():
    an = classify_retrieval("paragraph", SECTION_ANALYTICAL_APPENDIX)
    assert an.ranking_weight == 0.85 and an.searchable_by_default is True
    ad = classify_retrieval("paragraph", SECTION_ADMINISTRATIVE_APPENDIX)
    assert ad.ranking_weight == 0.40 and ad.searchable_by_default is False


def test_classify_retrieval_front_matter_not_searchable():
    rc = classify_retrieval("paragraph", SECTION_FRONT_MATTER)
    assert rc.searchable_by_default is False and rc.retrieval_profile == []


@pytest.mark.parametrize("query,profile", [
    ("educação", PROFILE_GENERAL),
    ("qual foi o subsídio em 2019?", PROFILE_QUANTITATIVE),
    ("quantos declarantes tiveram dedução?", PROFILE_QUANTITATIVE),
    ("qual modelo econométrico foi utilizado?", PROFILE_METHODOLOGICAL),
    ("quais autores foram citados?", PROFILE_BIBLIOGRAPHIC),
    ("que autores embasaram a análise?", PROFILE_BIBLIOGRAPHIC),
    ("quais são as referências bibliográficas?", PROFILE_BIBLIOGRAPHIC),
])
def test_detect_query_profile(query, profile):
    assert detect_query_profile(query) == profile


@pytest.mark.parametrize("query", [
    # "autores"/"citados" como conteúdo NÃO é bibliográfico (falsos positivos q076/q046).
    "Qual grupo apresentava menor risco de evasão segundo os autores?",
    "de acordo com os autores, qual o efeito da política?",
    "os autores afirmam que a dedução é regressiva?",
    "Quais são os três objetos de avaliação citados?",
])
def test_content_framing_is_not_bibliographic(query):
    assert detect_query_profile(query) != PROFILE_BIBLIOGRAPHIC


def test_genuine_bibliographic_still_detected():
    assert detect_query_profile("quais obras citadas fundamentam o estudo?") == PROFILE_BIBLIOGRAPHIC


def test_profile_exclusions_general_requires_searchable():
    assert profile_exclusions(PROFILE_GENERAL).require_searchable_by_default is True
    assert profile_exclusions(PROFILE_BIBLIOGRAPHIC).require_is_reference is True


# --------------------------------------------------------------------------
# §2/§5/§6 — Enriquecimento do parser
# --------------------------------------------------------------------------

def _page(*blocks):
    return list(blocks)


def _title(text, level=1):
    return {"type": "title", "content": {"title_content": [{"type": "text", "content": text}], "level": level}}


def _para(text):
    return {"type": "paragraph", "content": {"paragraph_content": [{"type": "text", "content": text}]}}


def _page_number(n):
    return {"type": "page_number", "content": {"page_number_content": [{"type": "text", "content": str(n)}]}}


def test_parser_captures_printed_page_number():
    data = [
        _page(_title("Capa")),                                  # page_index 0
        _page(_para("Conteúdo da página."), _page_number(12)),  # page_index 1 → impressa 12
    ]
    parser = MinerUDocumentParser()
    blocks, _ = parser.parse_json(data)
    para = next(b for b in blocks if b.block_type == "paragraph")
    assert para.page_index == 1 and para.page_number == 2
    assert para.printed_page_number == 12


def test_parser_assigns_section_kind_and_acronyms():
    data = [_page(
        _title("Lista de siglas"),
        _para("IRPF — Imposto de Renda de Pessoa Física"),
        _title("1 Introdução"),
        _para("A política de dedução é analisada neste relatório introdutório."),
    )]
    parser = MinerUDocumentParser()
    blocks, _ = parser.parse_json(data)
    intro = next(b for b in blocks if "introdutório" in b.text)
    assert intro.section_kind == SECTION_BODY
    assert parser.acronyms.get("IRPF") == "Imposto de Renda de Pessoa Física"


def test_parser_marks_raw_text():
    data = [_page(_para("Texto de exemplo do corpo."))]
    parser = MinerUDocumentParser()
    blocks, _ = parser.parse_json(data)
    assert blocks[0].raw_text == blocks[0].text


# --------------------------------------------------------------------------
# §5/§15 — Modos do chunker
# --------------------------------------------------------------------------

def _cfg(**kw):
    # Fixa os modos em v1 (isolamento de teste — não depende do .env ambiente).
    base = dict(target_tokens=10, max_tokens=20, min_tokens=3, overlap_tokens=3,
                max_overlap_tokens=6, table_max_tokens=15, list_max_tokens=12,
                force_split_above_tokens=40, chunking_version="v2test",
                front_matter_mode="include", references_mode="separate",
                appendix_mode="flat", equation_mode="raw", table_mode="always")
    base.update(kw)
    return ChunkingConfig(**base)


def _chunker(**kw):
    return StructuralTokenChunker(config=_cfg(**kw), token_counter=WhitespaceTokenCounter())


def _block(text, order, section, kind, btype="paragraph", level=1):
    return DocumentBlock(block_id=f"block-{order:04d}", block_type=btype, text=text,
                         order_index=order, page_number=1, section_path=section,
                         section_kind=kind, heading_level=level)


def _mixed_blocks():
    return [
        _block("Sumário de navegação com muitas palavras aqui.", 0, ["Sumário"], SECTION_TABLE_OF_CONTENTS),
        _block("Parágrafo real do corpo com bastante conteúdo textual analítico.", 1, ["1 Intro"], SECTION_BODY),
        _block("Autor, A. Obra citada em referência bibliográfica completa aqui.", 2,
               ["Referências"], SECTION_BIBLIOGRAPHY, btype="reference"),
    ]


def test_chunker_v1_default_keeps_everything():
    res = _chunker().chunk(_mixed_blocks(), document_id="d")
    kinds = {c.section_kind for c in res.chunks}
    # v1 default (front_matter include, references separate): mantém corpo + navegação + refs.
    assert SECTION_BODY in kinds
    assert SECTION_TABLE_OF_CONTENTS in kinds


def test_chunker_v2_metadata_only_drops_front_and_refs():
    res = _chunker(front_matter_mode="metadata_only", references_mode="metadata_only").chunk(
        _mixed_blocks(), document_id="d"
    )
    kinds = {c.section_kind for c in res.chunks}
    assert kinds == {SECTION_BODY}                      # só o corpo sobra
    body = next(c for c in res.chunks if c.section_kind == SECTION_BODY)
    assert body.searchable_by_default is True
    assert body.normalized_content_type == "body_paragraph"


def test_chunker_populates_v2_payload_fields():
    res = _chunker().chunk(_mixed_blocks(), document_id="d")
    body = next(c for c in res.chunks if c.section_kind == SECTION_BODY)
    assert body.retrieval_profile and PROFILE_GENERAL in body.retrieval_profile
    assert body.is_table is False and body.is_reference is False


def test_chunker_skips_non_chunkable_furniture():
    blocks = [
        _block("Cabeçalho repetido do documento", 0, ["S"], SECTION_BODY, btype="header"),
        _block("Parágrafo de corpo com conteúdo suficiente para virar chunk.", 1, ["S"], SECTION_BODY),
    ]
    blocks[0].chunkable = False  # marcado como furniture pelo parser
    res = _chunker().chunk(blocks, document_id="d")
    assert all("Cabeçalho repetido" not in c.text for c in res.chunks)


# --------------------------------------------------------------------------
# §8 — Normalização de texto (Fase 2)
# --------------------------------------------------------------------------

from backend.indexing.text_normalization import normalize_text  # noqa: E402


def test_normalize_collapses_spaces_and_dehyphenates():
    assert normalize_text("Palavra   com  espaços") == "Palavra com espaços"
    assert normalize_text("educa-\ncao superior") == "educacao superior"


def test_normalize_fixes_space_before_punctuation():
    assert normalize_text("frase completa . Depois") == "frase completa. Depois"


def test_normalize_preserves_numbers_dates_currency():
    # A correção de espaço-antes-de-pontuação NÃO pode tocar números (§8).
    assert normalize_text("valor de 1 . 713 aqui") == "valor de 1 . 713 aqui"
    assert normalize_text("R$ 30.000,00 em 12/2019") == "R$ 30.000,00 em 12/2019"


def test_normalize_canonicalizes_quotes():
    assert normalize_text("aspas “curvas” aqui") == 'aspas "curvas" aqui'


# --------------------------------------------------------------------------
# §7 — Reconstrução entre páginas (Fase 2)
# --------------------------------------------------------------------------

from backend.indexing.cross_page_reconstruction import (  # noqa: E402
    block_pages,
    can_merge,
    reconstruct_cross_page,
)


def _pblock(text, order, page, section=("S",), kind=SECTION_BODY, bbox=None):
    return DocumentBlock(block_id=f"block-{order:04d}", block_type="paragraph", text=text,
                         raw_text=text, order_index=order, page_number=page,
                         section_path=list(section), section_kind=kind, bbox=bbox)


def test_can_merge_across_pages_continuation():
    a = _pblock("…base de cálculo de R$ 30.000,00 –", 0, 17)
    b = _pblock("R$ 1.713,58). Caso tenham pago…", 1, 18)
    assert can_merge(a, b) is True


def test_no_merge_when_first_ends_conclusively():
    a = _pblock("Fim da ideia completa.", 0, 17)
    b = _pblock("nova continuação em minúscula", 1, 18)
    assert can_merge(a, b) is False


def test_no_merge_same_page_or_different_section():
    a = _pblock("segue sem pontuação final", 0, 17)
    assert can_merge(a, _pblock("continua aqui", 1, 17)) is False        # mesma página
    assert can_merge(a, _pblock("continua aqui", 1, 18, section=("Outra",))) is False


def test_reconstruct_merges_and_tracks_provenance():
    blocks = [
        _pblock("…de R$ 30.000,00 –", 0, 17),
        _pblock("R$ 1.713,58). Caso tenham pago…", 1, 18),
        _pblock("Parágrafo independente e completo.", 2, 18),
    ]
    out, merges = reconstruct_cross_page(blocks)
    assert merges == 1 and len(out) == 2
    merged = out[0]
    assert merged.cross_page_merged is True
    assert merged.source_block_ids == ["block-0000", "block-0001"]
    assert block_pages(merged) == [17, 18]
    assert "R$ 30.000,00 – R$ 1.713,58" in merged.text


def test_reconstruct_dehyphenates_word_break_across_page():
    blocks = [_pblock("consti-", 0, 5), _pblock("tuição federal prevê", 1, 6)]
    out, merges = reconstruct_cross_page(blocks)
    assert merges == 1 and out[0].text.startswith("constituição federal")


# --------------------------------------------------------------------------
# §9 — equation_confidence (Fase 2)
# --------------------------------------------------------------------------

def test_equation_confidence_real_vs_ordinal():
    conf = MinerUDocumentParser._equation_confidence
    assert conf(r"subsidio = \underbrace{(aliq \times bc - deduc)}") >= 0.8
    assert conf(r"3 ^ { o }") <= 0.3
    assert conf(r"5 \textdegree") <= 0.3


def test_parser_flags_populate_raw_and_normalized():
    data = [_page(_para("texto   com    espaços  redundantes."))]
    parser = MinerUDocumentParser(normalize_text=True)
    blocks, _ = parser.parse_json(data)
    b = blocks[0]
    assert b.raw_text == "texto   com    espaços  redundantes."
    assert b.text == "texto com espaços redundantes."


# --------------------------------------------------------------------------
# §10/§11/§12 — Validação visual (Fase 3)
# --------------------------------------------------------------------------

from backend.indexing.visual_validation import (  # noqa: E402
    chart_data_confidence,
    classify_image,
    image_is_indexable,
    mermaid_to_text,
    table_quality_score,
    table_row_representation,
)


def test_chart_geo_incoherence_zeroes_confidence():
    conf, reason = chart_data_confidence(
        "Distribuição de subsídio por unidade federativa do Brasil", "Fonte: RFB",
        "Colômbia 30; Argentina 25; Chile 20; Peru 15")
    assert conf == 0.0 and reason == "geo_incoherent_foreign_countries"


def test_chart_coherent_has_high_confidence():
    conf, _ = chart_data_confidence("Cálculo do subsídio por faixa", "", "aliq 7,5% base 33.919")
    assert conf >= 0.8


def test_table_quality_context_matters():
    html = "<table><tr><td>Ano</td><td>Subsídio</td></tr><tr><td>2019</td><td>4,15</td></tr></table>"
    good, _ = table_quality_score(html, "Tabela 4: subsídio", "Fonte: SECAP", "body")
    poor, _ = table_quality_score(html, "", "", "table_of_contents")
    assert good >= 0.75 and poor < good


def test_table_row_representation():
    md = "| Ano | Subsídio |\n| --- | --- |\n| 2019 | R$ 4,15 bi |"
    rep = table_row_representation(md, "Tabela 4")
    assert "Ano: 2019" in rep and "Subsídio: R$ 4,15 bi" in rep


def test_image_classification_and_indexability():
    assert classify_image("Figura 6", "```mermaid\ngraph LR\n A-->B", "") == "flowchart"
    assert classify_image("", "", "images/logo_gov.jpg") == "logo"
    assert classify_image("", "", "images/x.jpg") == "decorative"
    assert image_is_indexable("flowchart", "Figura 6", "Diagrama de fluxo…") is True
    assert image_is_indexable("logo", "Marca", "algo") is False


def test_mermaid_to_text_extracts_labels():
    txt = mermaid_to_text('graph LR\n A["Gestão fiscal"] --> B["Execução"]')
    assert "Gestão fiscal" in txt and "Execução" in txt


def _chart_block(order, caption, body, section=("2 Análise",), kind=SECTION_BODY):
    return DocumentBlock(
        block_id=f"block-{order:04d}", block_type="figure_caption",
        text=f"{caption}\n{body}", order_index=order, page_number=1,
        section_path=list(section), section_kind=kind,
        metadata={"origin": "chart", "chart_data_confidence": 0.0, "figure_caption": caption},
    )


def test_chunker_conditional_table_skips_low_quality():
    # Tabela sem legenda numa seção de navegação → quality baixa → conditional pula.
    b = DocumentBlock(
        block_id="block-0001", block_type="table", text="| a | b |\n| 1 | 2 |",
        order_index=1, page_number=1, section_path=["Sumário"],
        section_kind=SECTION_TABLE_OF_CONTENTS,
        metadata={"table_markdown": "| a | b |\n| 1 | 2 |", "table_html": ""})
    # Sem legenda e fora do corpo → score ~0.6; com limiar policy-strict (0.75) é pulada.
    res = _chunker(table_mode="conditional", table_min_quality=0.75).chunk([b], document_id="d")
    assert res.metrics.rejected_by_reason.get("table_low_quality", 0) == 1
    assert not res.chunks


def test_chunker_conditional_keeps_table_with_caption_in_body():
    b = DocumentBlock(
        block_id="block-0001", block_type="table",
        text="Tabela 4: subsídio\n| Ano | Subsídio |\n| --- | --- |\n| 2019 | 4,15 |",
        order_index=1, page_number=1, section_path=["2 Análise"], section_kind=SECTION_BODY,
        metadata={"table_markdown": "| Ano | Subsídio |\n| --- | --- |\n| 2019 | 4,15 |",
                  "table_caption": "Tabela 4: subsídio", "table_footnote": "Fonte: SECAP",
                  "table_html": "<table><tr><td>Ano</td><td>Subsídio</td></tr><tr><td>2019</td><td>4,15</td></tr></table>"})
    res = _chunker(table_mode="conditional", table_min_quality=0.75).chunk([b], document_id="d")
    assert len(res.chunks) == 1 and res.chunks[0].is_table
