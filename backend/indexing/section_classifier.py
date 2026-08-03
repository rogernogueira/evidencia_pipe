"""Classificação da seção documental (política v2, §4).

A hierarquia NÃO depende só do `level` do MinerU: ela é inferida também pela
**numeração** do título (`2.1.3 Progressividade` → nível 3). Uma **máquina de estados**
mantém a seção corrente (`section_kind`) enquanto se percorre o documento em ordem:

    front_matter → body → bibliography → analytical/administrative_appendix
    (e as seções de navegação: table_of_contents, list_of_*, acronym_list)

Este módulo é puro (sem GPU, sem I/O). É consumido pelo `MinerUDocumentParser`.
"""

from __future__ import annotations

import re
from typing import Optional

from backend.indexing.chunk_models import (
    SECTION_ACRONYM_LIST,
    SECTION_ADMINISTRATIVE_APPENDIX,
    SECTION_ANALYTICAL_APPENDIX,
    SECTION_BIBLIOGRAPHY,
    SECTION_BODY,
    SECTION_FRONT_MATTER,
    SECTION_LIST_OF_CHARTS,
    SECTION_LIST_OF_FIGURES,
    SECTION_LIST_OF_TABLES,
    SECTION_TABLE_OF_CONTENTS,
)

# --- Numeração (define o nível hierárquico) --------------------------------
# Captura "1", "2.1", "2.1.3" no início do título, seguido de espaço e conteúdo.
_NUMBERING_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)\.?\s+\S")

# --- Rótulos de seções especiais (case-insensitive) ------------------------
_REFERENCES_RE = re.compile(
    r"^\s*(refer[êe]ncias?|bibliografia|references|bibliography|obras\s+citadas)\b",
    re.IGNORECASE,
)
_APPENDIX_RE = re.compile(r"^\s*(ap[êe]ndices?|anexos?)\b", re.IGNORECASE)
_TOC_RE = re.compile(r"^\s*(sum[áa]rio|[íi]ndice(?!\s+de)|conte[úu]do)\b", re.IGNORECASE)
_LIST_FIGURES_RE = re.compile(r"lista\s+de\s+figuras|[íi]ndice\s+de\s+figuras", re.IGNORECASE)
_LIST_TABLES_RE = re.compile(
    r"lista\s+de\s+(tabelas|quadros)|[íi]ndice\s+de\s+(tabelas|quadros)", re.IGNORECASE
)
_LIST_CHARTS_RE = re.compile(r"lista\s+de\s+gr[áa]ficos|[íi]ndice\s+de\s+gr[áa]ficos", re.IGNORECASE)
_ACRONYM_RE = re.compile(
    r"lista\s+de\s+(siglas|abreviaturas|abreviaç[õo]es|s[íi]mbolos)"
    r"|siglas\s+e\s+abreviaturas|lista\s+de\s+acr[ôo]nimos",
    re.IGNORECASE,
)

# Palavras que caracterizam um apêndice ADMINISTRATIVO (§16). Sem elas, o apêndice é
# tratado como ANALÍTICO por padrão (busca com peso menor).
_ADMIN_APPENDIX_KEYWORDS = re.compile(
    r"\b(instrumento|formul[áa]rio|question[áa]rio|roteiro|termo\s+de\s+refer[êe]ncia|"
    r"participantes?|entrevistad|lista\s+de\s+presen|documentos?\s+consultad|"
    r"cronograma|matriz\s+de|of[íi]cio|portaria|ata)\b",
    re.IGNORECASE,
)

# Linha de dicionário de siglas: "IRPF — Imposto…", "IRPF: Imposto…", "IRPF - Imposto…".
_ACRONYM_LINE_RE = re.compile(
    r"^\s*(?:[-•*]\s*)?([A-ZÀ-Ú][A-ZÀ-Ú0-9./&-]{1,15})\s*(?:[—–:\-]|\s{2,})\s*([A-Za-zÀ-ÿ].{3,})$"
)


def infer_heading_level(text: str, fallback_level: Optional[int]) -> int:
    """Nível hierárquico do título: pela numeração quando houver (§4), senão o `level`
    do MinerU (ou 1). `2.1.3 …` → 3; `Apêndice A` → 1."""
    m = _NUMBERING_RE.match(text or "")
    if m:
        # nº de componentes numéricos (1 → nível 1, 2.1 → nível 2, 2.1.3 → nível 3).
        return min(len(m.group(1).split(".")), 6)
    if isinstance(fallback_level, int) and fallback_level >= 1:
        return fallback_level
    return 1


def _is_numbered(text: str) -> bool:
    return bool(_NUMBERING_RE.match(text or ""))


def classify_appendix(text: str) -> str:
    """analytical_appendix (padrão) × administrative_appendix (§16), por palavra-chave."""
    if _ADMIN_APPENDIX_KEYWORDS.search(text or ""):
        return SECTION_ADMINISTRATIVE_APPENDIX
    return SECTION_ANALYTICAL_APPENDIX


def special_section_kind(text: str) -> Optional[str]:
    """Retorna o `section_kind` quando o título INICIA uma seção especial/navegação;
    None para um título comum (numerado ou de corpo)."""
    t = text or ""
    if _REFERENCES_RE.match(t):
        return SECTION_BIBLIOGRAPHY
    if _LIST_FIGURES_RE.search(t):
        return SECTION_LIST_OF_FIGURES
    if _LIST_TABLES_RE.search(t):
        return SECTION_LIST_OF_TABLES
    if _LIST_CHARTS_RE.search(t):
        return SECTION_LIST_OF_CHARTS
    if _ACRONYM_RE.search(t):
        return SECTION_ACRONYM_LIST
    if _TOC_RE.match(t):
        return SECTION_TABLE_OF_CONTENTS
    if _APPENDIX_RE.match(t):
        return classify_appendix(t)
    return None


# Seções "grudentas" — uma vez iniciadas, persistem por títulos sem número/rótulo até
# o próximo título especial ou numerado.
_STICKY_KINDS = frozenset({
    SECTION_BIBLIOGRAPHY, SECTION_ANALYTICAL_APPENDIX, SECTION_ADMINISTRATIVE_APPENDIX,
})


class SectionStateMachine:
    """Percorre os títulos em ordem e mantém o `section_kind` corrente (§4).

    Uso pelo parser:
        sm = SectionStateMachine()
        for bloco:
            if heading: level, kind = sm.feed_heading(text, mineru_level)
            else:       kind = sm.current
    """

    def __init__(self) -> None:
        # Antes do primeiro título numerado, o conteúdo é pré-textual (capa, ficha…).
        self.current: str = SECTION_FRONT_MATTER
        self._body_started = False

    def feed_heading(self, text: str, mineru_level: Optional[int]) -> tuple[int, str]:
        """Atualiza o estado com um título e retorna (nível_efetivo, section_kind)."""
        level = infer_heading_level(text, mineru_level)
        special = special_section_kind(text)

        if special is not None:
            # Seção especial/navegação inicia aqui.
            self.current = special
        elif _is_numbered(text):
            # Primeiro título numerado encerra o front-matter e inicia o corpo.
            self._body_started = True
            self.current = SECTION_BODY
        else:
            # Título comum sem número: mantém seção grudenta (biblio/apêndice); caso
            # contrário, corpo (se já iniciado) ou front-matter (pré-textual).
            if self.current in _STICKY_KINDS:
                pass  # mantém
            elif self._body_started:
                self.current = SECTION_BODY
            else:
                self.current = SECTION_FRONT_MATTER
        return level, self.current


def parse_acronym_line(text: str) -> Optional[tuple[str, str]]:
    """Extrai (sigla, expansão) de uma linha do glossário (§5), ou None.

    Ex.: "IRPF — Imposto de Renda de Pessoa Física" → ("IRPF", "Imposto de Renda…").
    """
    m = _ACRONYM_LINE_RE.match((text or "").strip())
    if not m:
        return None
    sigla = m.group(1).strip(" .-—–:")
    expansao = m.group(2).strip()
    if len(sigla) < 2 or not expansao:
        return None
    return sigla, expansao
