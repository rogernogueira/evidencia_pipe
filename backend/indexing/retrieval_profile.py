"""Perfis de recuperação (política v2, §21/§22).

Traduz `(content_type, section_kind)` do chunk em:
  - `normalized_content_type` (§3): body_paragraph, structured_table, navigation_list…
  - `retrieval_profile[]`: em quais perfis de busca o chunk pode aparecer;
  - `searchable_by_default`: se entra na busca geral por padrão (§21);
  - `ranking_weight`: peso de ranqueamento (apêndice analítico 0,85; administrativo 0,40);
  - flags `is_table/is_chart/is_reference/is_appendix` (atalhos de filtro no Qdrant).

Também classifica a INTENÇÃO da consulta (`detect_query_profile`) e descreve o filtro
de cada perfil (`profile_exclusions`), consumidos pela busca (qdrant_client).

Módulo puro (sem GPU, sem Qdrant) — a montagem do Filter fica no qdrant_client.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from backend.indexing.chunk_models import (
    NAVIGATION_SECTION_KINDS,
    NON_DEFAULT_SEARCHABLE_SECTION_KINDS,
    SECTION_ADMINISTRATIVE_APPENDIX,
    SECTION_ANALYTICAL_APPENDIX,
    SECTION_BIBLIOGRAPHY,
    SECTION_FRONT_MATTER,
)

# Perfis de busca (§21).
PROFILE_GENERAL = "general"
PROFILE_QUANTITATIVE = "quantitative"
PROFILE_METHODOLOGICAL = "methodological"
PROFILE_BIBLIOGRAPHIC = "bibliographic"
ALL_PROFILES = (PROFILE_GENERAL, PROFILE_QUANTITATIVE, PROFILE_METHODOLOGICAL, PROFILE_BIBLIOGRAPHIC)


@dataclass
class RetrievalClass:
    """Resultado da classificação de recuperação de um chunk."""

    normalized_content_type: str
    retrieval_profile: list[str] = field(default_factory=list)
    searchable_by_default: bool = True
    ranking_weight: Optional[float] = None
    is_table: bool = False
    is_chart: bool = False
    is_reference: bool = False
    is_appendix: bool = False


def classify_retrieval(
    content_type: str,
    section_kind: Optional[str],
    *,
    origin: Optional[str] = None,
    chart_data_confident: bool = False,
) -> RetrievalClass:
    """Classifica um chunk para recuperação (§21/§22).

    `content_type`: o do chunker (paragraph/table/list/figure_caption/references/…).
    `origin`: dica do parser (ex.: 'chart' para gráficos vindos de `chart`).
    `chart_data_confident`: se os dados do gráfico foram validados (§11).
    """
    sk = section_kind
    is_reference = content_type == "references" or sk == SECTION_BIBLIOGRAPHY
    is_appendix = sk in (SECTION_ANALYTICAL_APPENDIX, SECTION_ADMINISTRATIVE_APPENDIX)
    is_table = content_type == "table"
    is_chart = origin == "chart" or content_type == "chart"

    # --- normalized_content_type (§3) ---
    if is_reference:
        nct = "bibliographic_reference"
    elif is_table:
        nct = "structured_table"
    elif is_chart:
        nct = "chart"
    elif content_type == "list":
        nct = "navigation_list" if sk in NAVIGATION_SECTION_KINDS else "semantic_list"
    elif content_type == "figure_caption":
        nct = "figure_caption"
    elif content_type in ("formula", "equation"):
        nct = "equation"
    elif content_type == "paragraph":
        nct = "body_paragraph"
    else:
        nct = content_type or "unknown"

    # --- ranking_weight (§16) ---
    weight: Optional[float] = None
    if sk == SECTION_ANALYTICAL_APPENDIX:
        weight = 0.85
    elif sk == SECTION_ADMINISTRATIVE_APPENDIX:
        weight = 0.40

    # --- searchable_by_default (§21): entra na busca GERAL por padrão? ---
    searchable = True
    if sk in NON_DEFAULT_SEARCHABLE_SECTION_KINDS:
        searchable = False           # front-matter, navegação, bibliografia, apêndice admin
    if is_reference or is_table or is_chart:
        searchable = False           # dados visuais/tabulares e referências fora da geral
    if is_chart and not chart_data_confident:
        searchable = False           # gráfico incoerente nunca é recuperável (§11)

    # --- retrieval_profile[] ---
    profiles: list[str] = []
    if is_reference:
        profiles = [PROFILE_BIBLIOGRAPHIC]
    elif sk in (SECTION_FRONT_MATTER,) or sk in NAVIGATION_SECTION_KINDS:
        profiles = []                # preservado p/ navegação/metadados; não recuperável
    elif sk == SECTION_ADMINISTRATIVE_APPENDIX:
        profiles = []                # fora da busca padrão
    elif is_table:
        profiles = [PROFILE_QUANTITATIVE, PROFILE_METHODOLOGICAL]
    elif is_chart:
        profiles = [PROFILE_QUANTITATIVE] if chart_data_confident else []
    elif nct == "equation":
        profiles = [PROFILE_METHODOLOGICAL]
    elif sk == SECTION_ANALYTICAL_APPENDIX:
        profiles = [PROFILE_GENERAL, PROFILE_METHODOLOGICAL]
    elif nct == "semantic_list":
        profiles = [PROFILE_GENERAL]
    elif nct == "body_paragraph":
        profiles = [PROFILE_GENERAL, PROFILE_QUANTITATIVE, PROFILE_METHODOLOGICAL]
    else:
        profiles = [PROFILE_GENERAL] if searchable else []

    return RetrievalClass(
        normalized_content_type=nct,
        retrieval_profile=profiles,
        searchable_by_default=searchable,
        ranking_weight=weight,
        is_table=is_table,
        is_chart=is_chart,
        is_reference=is_reference,
        is_appendix=is_appendix,
    )


# ---------------------------------------------------------------------------
# Intenção da consulta (§21)
# ---------------------------------------------------------------------------

_QUANT_RE = re.compile(
    r"\b(quant[oa]s?|qual\s+foi|quanto\s+custou|valor(es)?|percentu|propor[çc]|m[ée]dia|total|"
    r"n[úu]mero\s+de)\b|\bR\$|\d%|%|\b(19|20)\d{2}\b",
    re.IGNORECASE,
)
_METHOD_RE = re.compile(
    r"\b(metodolog|m[ée]todo|modelo|econom[ée]tric|regress|estimativ|amostra|"
    r"vari[áa]vel|c[áa]lculo|como\s+(foi|foram|se)\s+\w*(calcul|estim|avali|model))\b",
    re.IGNORECASE,
)
# Bibliográfico SÓ quando a pergunta é sobre autoria/citação/referências em si — não
# quando "autores" é só moldura de uma pergunta de conteúdo ("segundo os autores…").
_BIBLIO_RE = re.compile(
    r"quais?\s+(?:os\s+|as\s+)?autor|\bque\s+autores?\b|\bautoria\b"
    r"|quem\s+(?:escreveu|é\s+o\s+autor|são\s+os\s+autores)"
    # "citad" só conta como citação quando junto de termo acadêmico (evita "objetos citados").
    r"|(?:obras?|trabalhos?|estudos?|autores?|refer[êe]ncias?|artigos?|pesquisas?)\s+citad"
    r"|refer[êe]ncias?\s+bibliogr|\bbibliografia\b|fonte\s+bibliogr|cita[çc][õo]es\s+bibliogr",
    re.IGNORECASE,
)
# Moldura de conteúdo com "autores" (NÃO é bibliográfica): "segundo/conforme/de acordo
# com/para/na visão dos autores", "os autores afirmam/apontam/concluem/mostram".
_AUTHOR_FRAMING_RE = re.compile(
    r"(segundo|conforme|de\s+acordo\s+com|para|na\s+vis[ãa]o\s+d[eo]s?|"
    r"de\s+autores?\s+como)\s+os?\s+autor"
    r"|os\s+autores?\s+(afirm|apont|conclu|mostr|defend|identific|observ|sugere|indic)",
    re.IGNORECASE,
)


def detect_query_profile(query: str) -> str:
    """Classifica a intenção da consulta em um perfil (§21). Padrão: general.

    "segundo os autores" e afins são moldura de conteúdo — NÃO tornam a consulta
    bibliográfica (evita falso positivo que filtraria a busca para só referências)."""
    q = query or ""
    if _BIBLIO_RE.search(q) and not _AUTHOR_FRAMING_RE.search(q):
        return PROFILE_BIBLIOGRAPHIC
    if _METHOD_RE.search(q):
        return PROFILE_METHODOLOGICAL
    if _QUANT_RE.search(q):
        return PROFILE_QUANTITATIVE
    return PROFILE_GENERAL


@dataclass
class ProfileExclusions:
    """Descreve como filtrar a busca para um perfil, em termos de payload.

    - `require_searchable_by_default`: exige `searchable_by_default=true`.
    - `require_is_reference`: exige `is_reference=true` (perfil bibliográfico).
    - `exclude_section_kinds`: `section_kind` a excluir.
    - `exclude_is_reference`: excluir chunks de referência.
    """

    require_searchable_by_default: bool = False
    require_is_reference: bool = False
    exclude_section_kinds: tuple[str, ...] = ()
    exclude_is_reference: bool = False


def profile_exclusions(profile: str) -> ProfileExclusions:
    """Regras de filtro por perfil (§21), consumidas pela busca."""
    nav_and_front = tuple(NAVIGATION_SECTION_KINDS | {SECTION_FRONT_MATTER})
    if profile == PROFILE_BIBLIOGRAPHIC:
        return ProfileExclusions(require_is_reference=True)
    if profile == PROFILE_QUANTITATIVE:
        return ProfileExclusions(
            exclude_section_kinds=nav_and_front + (SECTION_ADMINISTRATIVE_APPENDIX, SECTION_BIBLIOGRAPHY),
            exclude_is_reference=True,
        )
    if profile == PROFILE_METHODOLOGICAL:
        return ProfileExclusions(
            exclude_section_kinds=nav_and_front + (SECTION_ADMINISTRATIVE_APPENDIX, SECTION_BIBLIOGRAPHY),
            exclude_is_reference=True,
        )
    # general (padrão): só o que é recuperável por padrão.
    return ProfileExclusions(require_searchable_by_default=True)
