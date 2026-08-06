"""Testes do parser de sumário (`toc_parser`) e do validador (`toc_validator`).

Diagnóstico puro: nenhum destes módulos altera a classificação de seções — os testes
verificam que as divergências são DETECTADAS, não que sejam corrigidas.

O caso de referência é o real: em exames-educ-basica_relatorio-de-avaliacao um
fragmento "EVEX?" virou `title` nível 2 no meio das referências, desempilhou
"Referências bibliográficas" e reclassificou 14 chunks de bibliografia como corpo.
"""

import pytest

from backend.indexing.chunk_models import SECTION_BIBLIOGRAPHY, SECTION_BODY
from backend.indexing.document_blocks import MinerUDocumentParser
from backend.indexing.toc_parser import parse_toc
from backend.indexing.toc_validator import (
    FINDING_FALSE_HEADING,
    FINDING_NOT_IN_TOC,
    FINDING_TOC_ORPHAN,
    normalize_for_match,
    validate_document,
)


# --- Construtores de blocos MinerU ----------------------------------------

def _page(*blocks):
    return list(blocks)


def _title(text, level=2, bbox=None):
    return {
        "type": "title",
        "content": {"title_content": [{"type": "text", "content": text}], "level": level},
        "bbox": bbox or [112, 100, 900, 120],
    }


def _para(text, bbox=None):
    return {
        "type": "paragraph",
        "content": {"paragraph_content": [{"type": "text", "content": text}]},
        "bbox": bbox or [112, 200, 900, 240],
    }


def _index(*lines):
    return {
        "type": "index",
        "content": {
            "list_type": "text_list",
            "list_items": [
                {"item_type": "text", "item_content": [{"type": "text", "content": ln}]}
                for ln in lines
            ],
        },
        "bbox": [112, 100, 900, 800],
    }


_SUMARIO = _index(
    "1 Introdução 5",
    "2 Metodologia 10",
    "2.1 Coleta de dados . 11",
    "2.2 Análise ..... 14",
    "3 Conclusão 20",
    "Referências bibliográficas ..... 25",
)


# --- toc_parser ------------------------------------------------------------

def test_parse_toc_extrai_entradas_com_niveis_e_paginas():
    toc = parse_toc([_page(_SUMARIO)])
    assert toc.found and toc.source == "index_block"
    assert [(e.title, e.level, e.page) for e in toc.entries] == [
        ("1 Introdução", 1, 5),
        ("2 Metodologia", 1, 10),
        ("2.1 Coleta de dados", 2, 11),
        ("2.2 Análise", 2, 14),
        ("3 Conclusão", 1, 20),
        ("Referências bibliográficas", 1, 25),
    ]


def test_parse_toc_marca_nivel_suposto_para_entrada_sem_numeracao():
    toc = parse_toc([_page(_SUMARIO)])
    numeradas = [e for e in toc.entries if e.numbered]
    refs = next(e for e in toc.entries if e.title.startswith("Referências"))
    assert all(not e.level_inferred for e in numeradas)
    assert refs.level_inferred is True  # nível 1 é SUPOSIÇÃO, não fato do sumário


def test_parse_toc_une_entrada_quebrada_em_varias_linhas():
    toc = parse_toc([_page(_index(
        "1 Introdução 5",
        "2 Uso dos dados e informações do Saeb pelos gestores estaduais e",
        "municipais das redes públicas 12",
        "3 Conclusão 20",
        "4 Anexos",
        "31",
    ))])
    titulos = [e.title for e in toc.entries]
    assert titulos[1] == (
        "2 Uso dos dados e informações do Saeb pelos gestores estaduais e "
        "municipais das redes públicas"
    )
    assert toc.entries[1].page == 12
    # Última entrada: página numa linha isolada.
    assert (toc.entries[3].title, toc.entries[3].page) == ("4 Anexos", 31)


def test_parse_toc_descarta_lista_de_ilustracoes():
    toc = parse_toc([_page(_index(
        "Tabela 1 – Questões de avaliação . 9",
        "Tabela 2 – Acesso ao Boletim .. 14",
        "Gráfico 3 – Níveis de concordância . 17",
        "Quadro 1– Parâmetros da avaliação 11",
    ))])
    assert not toc.found
    assert toc.blocks_rejected_illustration == 1


def test_parse_toc_descarta_bloco_pequeno_demais():
    toc = parse_toc([_page(_index("1 Introdução 5", "2 Fim 9"))])
    assert not toc.found and toc.blocks_rejected_too_small == 1


def test_parse_toc_fallback_para_secao_sumario_sem_bloco_index():
    data = [_page(
        _title("Sumário"),
        {"type": "list", "content": {"list_type": "text_list", "list_items": [
            {"item_type": "text", "item_content": [{"type": "text", "content": ln}]}
            for ln in ("1 Introdução 5", "2 Metodologia 10", "3 Conclusão 20")
        ]}},
    )]
    toc = parse_toc(data)
    assert toc.found and toc.source == "toc_section" and len(toc.entries) == 3


def test_parse_toc_vazio_quando_nao_ha_sumario():
    assert not parse_toc([_page(_title("1 Introdução"), _para("Texto."))]).found


# --- normalização e casamento ---------------------------------------------

@pytest.mark.parametrize("a,b", [
    ("Referências bibliográficas", "REFERÊNCIAS BIBLIOGRÁFICAS"),
    ("Análise ..... ", "analise"),
    ("2.1 Coleta de dados", "2 1 coleta de dados"),
])
def test_normalize_for_match(a, b):
    assert normalize_for_match(a) == normalize_for_match(b)


# --- toc_validator: gate ---------------------------------------------------

def _analyze(data, document_id="doc"):
    blocks, _ = MinerUDocumentParser().parse_json(data)
    return validate_document(blocks, parse_toc(data), document_id=document_id)


def test_gate_sem_sumario_nao_acusa_nada():
    r = _analyze([_page(_title("EVEX?"), _para("Texto qualquer do documento."))])
    assert r.reliable is False
    assert r.findings == []
    assert "sumário não encontrado" in r.gate_reason


def test_gate_sumario_que_nao_reencontra_os_titulos_e_descartado():
    """Sumário legítimo, mas de OUTRO documento: taxa de casamento baixa → não julga."""
    data = [
        _page(_SUMARIO),
        _page(_title("Alfa"), _para("Um."), _title("Beta"), _para("Dois."),
              _title("Gama"), _para("Três.")),
    ]
    r = _analyze(data)
    assert r.reliable is False
    assert r.findings == []
    assert "taxa de casamento" in r.gate_reason


# --- toc_validator: detecções ---------------------------------------------

def _documento_com_evex():
    """Reproduz o caso real: fragmento curto e estreito no meio das referências."""
    return [
        _page(_SUMARIO),
        _page(_title("1 Introdução"), _para("Parágrafo da introdução do relatório.")),
        _page(_title("2 Metodologia"), _para("Descrição da metodologia adotada.")),
        _page(_title("2.1 Coleta de dados"), _para("Como os dados foram coletados.")),
        _page(_title("2.2 Análise"), _para("Como os dados foram analisados.")),
        _page(_title("3 Conclusão"), _para("Considerações finais do trabalho.")),
        _page(
            _title("Referências bibliográficas"),
            _para("ALVES, A. Título da obra. Editora, 2020."),
            _title("EVEX?", bbox=[114, 361, 169, 378]),          # ← fragmento espúrio
            _para("BRASIL. Lei nº 1, de 2021. Diário Oficial da União."),
            _para("COSTA, C. Outra obra citada. Editora, 2019."),
        ),
    ]


def test_detecta_titulo_espurio_ausente_do_sumario():
    r = _analyze(_documento_com_evex())
    assert r.reliable is True
    suspeitos = r.of_kind(FINDING_FALSE_HEADING)
    assert [f.heading_text for f in suspeitos] == ["EVEX?"]
    assert suspeitos[0].bbox_width_ratio < 0.35   # bbox estreito é parte da evidência


def test_mede_quebra_do_estado_de_secao_causada_pelo_titulo_espurio():
    r = _analyze(_documento_com_evex())
    evex = r.of_kind(FINDING_FALSE_HEADING)[0]
    assert evex.section_kind_before == SECTION_BIBLIOGRAPHY
    assert evex.section_kind_after == SECTION_BODY
    assert evex.breaks_section_state is True
    assert evex.blocks_affected == 2              # as duas referências seguintes
    assert r.blocks_wrongly_reclassified == 2


def test_detecta_nivel_divergente_do_sumario():
    """MinerU dá nível 2 a "Referências bibliográficas"; o sumário mostra que é nível 1."""
    r = _analyze(_documento_com_evex())
    mismatches = r.level_mismatches()
    assert [f.heading_text for f in mismatches] == ["Referências bibliográficas"]
    assert (mismatches[0].heading_level, mismatches[0].toc_level) == (2, 1)
    # Repousa na suposição de nível para entrada sem numeração — fica marcado.
    assert mismatches[0].toc_level_inferred is True
    assert r.level_mismatches(inferred=False) == []


def test_titulo_numerado_nao_casa_com_entrada_de_numeracao_diferente():
    """'4.6 Operação nº 2639576' não é a entrada '5.4 Operação nº 2639576'."""
    data = [
        _page(_index(
            "1 Introdução 5", "2 Metodologia 10", "3 Conclusão 20",
            "5.4 Operação nº 2639576 . 30", "5.5 Operação nº 3009064 . 32",
        )),
        _page(_title("1 Introdução"), _para("Introdução do relatório avaliativo.")),
        _page(_title("2 Metodologia"), _para("Metodologia utilizada na avaliação.")),
        _page(_title("3 Conclusão"), _para("Conclusões da avaliação realizada.")),
        _page(_title("4.6 Operação nº 2639576"), _para("Detalhamento da operação.")),
    ]
    r = _analyze(data)
    nao_casados = [f.heading_text for f in r.of_kind(FINDING_NOT_IN_TOC)]
    assert "4.6 Operação nº 2639576" in nao_casados
    # E, por não ter casado, não gera divergência de nível espúria.
    assert r.level_mismatches() == []


def test_titulos_auto_identificados_nao_sao_acusados():
    """'Sumário'/'Lista de tabelas' são curtos e não constam do sumário — mas são reais."""
    data = [
        _page(_SUMARIO),
        _page(_title("Sumário", bbox=[114, 100, 200, 120])),
        _page(_title("Lista de tabelas", bbox=[114, 100, 250, 120])),
        _page(_title("1 Introdução"), _para("Parágrafo da introdução do relatório.")),
        _page(_title("2 Metodologia"), _para("Descrição da metodologia adotada.")),
        _page(_title("2.1 Coleta de dados"), _para("Como os dados foram coletados.")),
        _page(_title("2.2 Análise"), _para("Como os dados foram analisados.")),
        _page(_title("3 Conclusão"), _para("Considerações finais do trabalho.")),
    ]
    r = _analyze(data)
    assert r.reliable is True
    assert r.of_kind(FINDING_FALSE_HEADING) == []


def test_reporta_entrada_do_sumario_sem_titulo_correspondente():
    data = _documento_com_evex()
    # Remove a página da conclusão: a entrada "3 Conclusão" fica órfã.
    data = [p for p in data if not any(
        b.get("type") == "title"
        and b["content"]["title_content"][0]["content"] == "3 Conclusão" for b in p
    )]
    r = _analyze(data)
    orfas = [f.toc_title for f in r.of_kind(FINDING_TOC_ORPHAN)]
    assert orfas == ["3 Conclusão"]


def test_section_path_poluido_e_contabilizado_mesmo_sem_quebra_de_secao():
    """Título espúrio no corpo não muda `section_kind`, mas suja o `section_path`
    — que entra no texto embedado quando CHUNK_EMBED_SECTION_CONTEXT=true."""
    data = [
        _page(_SUMARIO),
        _page(_title("1 Introdução"), _para("Parágrafo da introdução do relatório.")),
        _page(_title("2 Metodologia"),
              _title("Legenda:", bbox=[114, 300, 170, 318]),   # ← fragmento no corpo
              _para("Primeiro parágrafo após o fragmento."),
              _para("Segundo parágrafo após o fragmento.")),
        _page(_title("2.1 Coleta de dados"), _para("Como os dados foram coletados.")),
        _page(_title("2.2 Análise"), _para("Como os dados foram analisados.")),
        _page(_title("3 Conclusão"), _para("Considerações finais do trabalho.")),
    ]
    r = _analyze(data)
    legenda = next(f for f in r.of_kind(FINDING_FALSE_HEADING) if f.heading_text == "Legenda:")
    assert legenda.breaks_section_state is False       # body → body
    assert r.blocks_wrongly_reclassified == 0
    assert r.blocks_with_polluted_section_path == 2    # mas dois blocos herdam "Legenda:"


def test_validador_nao_altera_os_blocos():
    data = _documento_com_evex()
    blocks, _ = MinerUDocumentParser().parse_json(data)
    antes = [(b.block_id, b.section_kind, tuple(b.section_path), b.heading_level) for b in blocks]
    validate_document(blocks, parse_toc(data), document_id="doc")
    depois = [(b.block_id, b.section_kind, tuple(b.section_path), b.heading_level) for b in blocks]
    assert antes == depois
