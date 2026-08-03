"""Reconstrução de parágrafos partidos entre páginas (política v2, §7).

O MinerU entrega blocos por página; um parágrafo pode terminar numa página e
continuar na seguinte:

    página p:    "… incidiria sobre a base de cálculo de R$ 30.000,00 –"
    página p+1:  "R$ 1.713,58). Caso tenham pago …"

Esta etapa ocorre ANTES da contagem de tokens: funde blocos adjacentes que
claramente formam um só parágrafo, preservando a rastreabilidade
(`cross_page_merged`, `source_block_ids`, páginas de origem).

Opera sobre `list[DocumentBlock]` já ordenados e com `section_path`/`section_kind`
atribuídos. Módulo puro (sem GPU, sem I/O).
"""

from __future__ import annotations

import re
from typing import Optional

from backend.indexing.chunk_models import BLOCK_PARAGRAPH, DocumentBlock

# Pontuação conclusiva (§7): se o bloco termina assim, NÃO é continuação.
_CONCLUSIVE_END_RE = re.compile(r"[.!?:;”\"')\]]\s*$")
# Início que indica continuação (§7): minúscula, dígito, moeda, %, abre-parêntese, traço.
_CONTINUATION_START_RE = re.compile(r"^\s*(?:R\$|US\$|[a-zà-ÿ0-9(%,\-–—])")


def _ends_conclusive(text: str) -> bool:
    return bool(_CONCLUSIVE_END_RE.search(text or ""))


def _starts_continuation(text: str) -> bool:
    return bool(_CONTINUATION_START_RE.match(text or ""))


def _bbox_compatible(a: DocumentBlock, b: DocumentBlock, tol: float) -> bool:
    """Margens horizontais compatíveis (§7). Se faltar bbox em algum, não bloqueia."""
    if not a.bbox or not b.bbox or len(a.bbox) < 4 or len(b.bbox) < 4:
        return True
    width = max(a.bbox[2], b.bbox[2], 1.0)
    return abs(a.bbox[0] - b.bbox[0]) <= max(60.0, tol * width)


def _hyphen_break(text: str) -> bool:
    """Termina com hífen de quebra de palavra (letra + '-')."""
    t = (text or "").rstrip()
    return bool(re.search(r"[A-Za-zÀ-ÿ]-$", t))


def can_merge(a: DocumentBlock, b: DocumentBlock, *, bbox_tol: float = 0.12) -> bool:
    """Decide se `b` continua o parágrafo `a` na página seguinte (§7)."""
    if a.block_type != BLOCK_PARAGRAPH or b.block_type != BLOCK_PARAGRAPH:
        return False
    if a.section_path != b.section_path or a.section_kind != b.section_kind:
        return False
    # Precisa cruzar página (b numa página posterior).
    if a.page_number is None or b.page_number is None or b.page_number <= a.page_number:
        return False
    if _ends_conclusive(a.text):
        return False
    if not _starts_continuation(b.text):
        return False
    return _bbox_compatible(a, b, bbox_tol)


def _merge(a: DocumentBlock, b: DocumentBlock) -> DocumentBlock:
    """Funde `b` em `a`, preservando rastreabilidade (§7)."""
    if _hyphen_break(a.text):
        # Remove o hífen de quebra e junta sem espaço (educa- + ção → educação).
        text = a.text.rstrip()[:-1] + b.text.lstrip()
        raw = (a.raw_text or a.text).rstrip()[:-1] + (b.raw_text or b.text).lstrip()
    else:
        text = f"{a.text.rstrip()} {b.text.lstrip()}"
        raw = f"{(a.raw_text or a.text).rstrip()} {(b.raw_text or b.text).lstrip()}"

    prior_pages = a.metadata.get("cross_page_numbers") or [a.page_number]
    cross_pages = sorted({p for p in (*prior_pages, b.page_number) if p is not None})
    prior_printed = a.metadata.get("cross_printed_page_numbers") or (
        [a.printed_page_number] if a.printed_page_number is not None else []
    )
    printed = sorted({p for p in (*prior_printed, b.printed_page_number) if p is not None})
    src = (a.source_block_ids or [a.block_id]) + [b.block_id]

    return a.model_copy(update={
        "text": text,
        "raw_text": raw,
        "cross_page_merged": True,
        "source_block_ids": src,
        "metadata": {
            **a.metadata,
            "cross_page_numbers": cross_pages,
            "cross_printed_page_numbers": printed,
        },
    })


def reconstruct_cross_page(
    blocks: list[DocumentBlock], *, bbox_tol: float = 0.12
) -> tuple[list[DocumentBlock], int]:
    """Funde parágrafos partidos entre páginas (§7).

    Retorna (blocos_reconstruídos, nº_de_merges). Encadeia merges: A+B, depois (A+B)+C.
    """
    out: list[DocumentBlock] = []
    merges = 0
    for b in blocks:
        prev: Optional[DocumentBlock] = out[-1] if out else None
        if prev is not None and can_merge(prev, b, bbox_tol=bbox_tol):
            out[-1] = _merge(prev, b)
            merges += 1
        else:
            out.append(b)
    return out, merges


def block_pages(b: DocumentBlock) -> list[int]:
    """Todas as páginas cobertas por um bloco (inclui as de reconstrução, §7)."""
    pages = set(b.metadata.get("cross_page_numbers") or [])
    if b.page_number is not None:
        pages.add(b.page_number)
    return sorted(pages)


def block_printed_pages(b: DocumentBlock) -> list[int]:
    """Números impressos cobertos por um bloco (inclui reconstrução, §7)."""
    pages = set(b.metadata.get("cross_printed_page_numbers") or [])
    if b.printed_page_number is not None:
        pages.add(b.printed_page_number)
    return sorted(pages)
