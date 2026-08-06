"""Extração do SUMÁRIO (índice estrutural) do documento MinerU — só leitura.

Motivação: a hierarquia de títulos hoje depende do `level` do MinerU quando o título
não tem numeração ([section_classifier.infer_heading_level]). Esse `level` é ruidoso —
um fragmento solto na página pode virar `title` nível 2 e desempilhar uma seção real
(caso `EVEX?` em exames-educ-basica, que apagou "Referências bibliográficas" da pilha).

O sumário do próprio documento é uma fonte independente e confiável dessa hierarquia:
lista os títulos REAIS com nível (pela numeração/indentação) e página. Este módulo o
extrai; quem valida é `toc_validator`.

Este módulo é PURO (sem GPU, sem I/O, sem LLM) e NÃO altera a classificação de seções —
serve ao relatório de divergências. Formato de entrada: a mesma lista-de-páginas do
`content_list_v2.json` consumida por `MinerUDocumentParser`.

Fontes de linhas do sumário, nesta ordem:
    1. blocos `type: "index"` (é como o MinerU marca sumário e listas de ilustrações);
    2. blocos `list`/`paragraph` sob um título "Sumário"/"Índice" (fallback).

Listas de ilustrações ("Lista de Tabelas", "Lista de Gráficos") também vêm como
`index` — são descartadas por heurística de conteúdo, não pelo título do bloco.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from backend.indexing.chunk_models import SECTION_TABLE_OF_CONTENTS
from backend.indexing.section_classifier import infer_heading_level, special_section_kind

# Blocos que o MinerU tipa como índice (sumário OU lista de ilustrações).
_INDEX_BLOCK_TYPE = "index"

# Entrada de lista de ilustrações: "Tabela 3 – …", "Gráfico 12 - …", "Quadro 1– …".
_ILLUSTRATION_RE = re.compile(
    r"^\s*(tabelas?|quadros?|gr[áa]ficos?|figuras?|mapas?|imagens?|box|ilustra[çc][ãa]o)\s*\d",
    re.IGNORECASE,
)

# "Título ........ 83" | "Título . 83" | "Título 83" (pontos-guia opcionais).
_ENTRY_RE = re.compile(r"^(?P<title>.*?)[\s.·…_–-]*[\s.](?P<page>\d{1,4})\s*$")
# Linha que é SÓ a página (continuação de uma entrada quebrada em várias linhas).
_PAGE_ONLY_RE = re.compile(r"^[\s.·…_–-]*(?P<page>\d{1,4})\s*$")
# Numeração no início do título ("2.1.3 …") — define o nível hierárquico.
_NUMBERING_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)\.?\s+\S")

# Um bloco `index` só é aceito como sumário estrutural se tiver ao menos estas
# entradas parseadas e se a maioria NÃO for entrada de ilustração.
_MIN_ENTRIES_PER_BLOCK = 3
_MAX_ILLUSTRATION_FRACTION = 0.5


def _inline_text(items: Any) -> str:
    """Concatena itens inline `{type, content}` — versão local, sem depender do parser."""
    if not isinstance(items, list):
        return ""
    parts = [
        it.get("content", "")
        for it in items
        if isinstance(it, dict) and isinstance(it.get("content"), str)
    ]
    return "".join(parts).strip()


def _block_lines(block: dict) -> list[str]:
    """Linhas de texto de um bloco `index`/`list`/`paragraph`, na ordem original."""
    content = block.get("content")
    if not isinstance(content, dict):
        return []
    lines: list[str] = []
    for item in content.get("list_items", []) or []:
        if isinstance(item, dict):
            text = _inline_text(item.get("item_content", []))
            if text:
                lines.append(text)
    if lines:
        return lines
    for key in ("paragraph_content", "index_content"):
        text = _inline_text(content.get(key, []))
        if text:
            lines.extend(text.splitlines())
    return [ln for ln in (line.strip() for line in lines) if ln]


@dataclass(frozen=True)
class TocEntry:
    """Uma linha do sumário já reconstruída (quebras de linha unidas)."""

    title: str
    page: int
    level: int
    numbering: Optional[str]     # "2.1.3" quando numerada; None quando não
    source_page_index: int       # página do PDF onde a linha do sumário aparece
    line_span: int = 1           # nº de linhas do JSON unidas para formar a entrada
    # True quando o nível NÃO veio da numeração e sim do default (entrada sem número
    # no sumário → assume-se seção de topo, como "Referências"/"Apêndice A"). É uma
    # SUPOSIÇÃO: o bloco `index` do MinerU não preserva a indentação original.
    level_inferred: bool = False

    @property
    def numbered(self) -> bool:
        return self.numbering is not None


@dataclass
class TocIndex:
    """Resultado da extração: entradas + estatísticas para o gate de confiabilidade."""

    entries: list[TocEntry] = field(default_factory=list)
    index_blocks_seen: int = 0
    blocks_used: int = 0
    blocks_rejected_illustration: int = 0
    blocks_rejected_too_small: int = 0
    lines_unparsed: int = 0
    source: str = "none"         # "index_block" | "toc_section" | "none"

    @property
    def found(self) -> bool:
        return bool(self.entries)

    @property
    def numbered_entries(self) -> int:
        return sum(1 for e in self.entries if e.numbered)


def _parse_lines(lines: list[str], page_index: int) -> tuple[list[TocEntry], int]:
    """Converte linhas cruas em entradas, unindo as que quebraram no meio do título.

    Uma entrada só fecha quando aparece a página. Linhas sem página são acumuladas
    como continuação do título. Retorna (entradas, linhas_não_aproveitadas)."""
    entries: list[TocEntry] = []
    buffer: list[str] = []
    unparsed = 0

    def flush(title_tail: str, page: int) -> None:
        parts = buffer + ([title_tail] if title_tail else [])
        title = " ".join(" ".join(parts).split())
        if not title:
            return
        m = _NUMBERING_RE.match(title)
        numbering = m.group(1) if m else None
        entries.append(
            TocEntry(
                title=title,
                page=page,
                level=infer_heading_level(title, None),
                numbering=numbering,
                source_page_index=page_index,
                line_span=len(parts),
                level_inferred=numbering is None,
            )
        )

    for line in lines:
        line = line.strip()
        if not line:
            continue
        page_only = _PAGE_ONLY_RE.match(line)
        if page_only and buffer:
            # Linha que carrega apenas o número da página da entrada anterior.
            flush("", int(page_only.group("page")))
            buffer = []
            continue
        entry = _ENTRY_RE.match(line)
        if entry:
            flush(entry.group("title"), int(entry.group("page")))
            buffer = []
            continue
        # Sem página: é continuação do título da próxima linha.
        buffer.append(line)

    if buffer:
        unparsed += len(buffer)
    return entries, unparsed


def _looks_like_illustration_list(entries: list[TocEntry]) -> bool:
    """True quando a maioria das entradas é 'Tabela N –', 'Gráfico N –' etc."""
    if not entries:
        return True
    hits = sum(1 for e in entries if _ILLUSTRATION_RE.match(e.title))
    return (hits / len(entries)) > _MAX_ILLUSTRATION_FRACTION


def _iter_blocks(pages: list) -> Any:
    """(page_index, block) de todas as páginas, na ordem do documento."""
    for page_index, page_blocks in enumerate(pages):
        if not isinstance(page_blocks, list):
            continue
        for block in page_blocks:
            if isinstance(block, dict):
                yield page_index, block


def parse_toc(data: Any) -> TocIndex:
    """Extrai o sumário estrutural do content_list do MinerU.

    Aceita a lista-de-páginas (formato normal) ou uma lista plana de blocos."""
    if not isinstance(data, list):
        return TocIndex()
    pages = data if (data and isinstance(data[0], list)) else [data]

    toc = TocIndex()
    collected: list[TocEntry] = []

    # --- Fonte 1: blocos `index` -----------------------------------------
    for page_index, block in _iter_blocks(pages):
        if block.get("type") != _INDEX_BLOCK_TYPE:
            continue
        toc.index_blocks_seen += 1
        entries, unparsed = _parse_lines(_block_lines(block), page_index)
        if len(entries) < _MIN_ENTRIES_PER_BLOCK:
            toc.blocks_rejected_too_small += 1
            continue
        if _looks_like_illustration_list(entries):
            toc.blocks_rejected_illustration += 1
            continue
        toc.blocks_used += 1
        toc.lines_unparsed += unparsed
        collected.extend(entries)

    if collected:
        toc.entries = collected
        toc.source = "index_block"
        return toc

    # --- Fonte 2 (fallback): blocos sob um título "Sumário"/"Índice" ------
    in_toc_section = False
    for page_index, block in _iter_blocks(pages):
        b_type = block.get("type")
        if b_type == "title":
            title = _inline_text((block.get("content") or {}).get("title_content", []))
            in_toc_section = special_section_kind(title) == SECTION_TABLE_OF_CONTENTS
            continue
        if not in_toc_section or b_type not in ("list", "paragraph"):
            continue
        entries, unparsed = _parse_lines(_block_lines(block), page_index)
        if len(entries) < _MIN_ENTRIES_PER_BLOCK or _looks_like_illustration_list(entries):
            continue
        toc.blocks_used += 1
        toc.lines_unparsed += unparsed
        collected.extend(entries)

    if collected:
        toc.entries = collected
        toc.source = "toc_section"
    return toc
