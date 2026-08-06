"""Validação dos títulos detectados contra o SUMÁRIO do documento — só diagnóstico.

NÃO altera a classificação de seções. Compara o que `MinerUDocumentParser` detectou
como `title` com o que o sumário do próprio documento declara (via `toc_parser`), e
reporta as divergências:

    likely_false_heading      título detectado que NÃO existe no sumário e tem cara de
                              fragmento (curto, sem numeração) — ex.: "EVEX?"
    heading_not_in_toc        título fora do sumário, mas plausível (sumários costumam
                              omitir subníveis profundos) — informativo
    level_mismatch            nível efetivo usado na pilha ≠ nível declarado no sumário
    toc_entry_without_heading entrada do sumário sem título correspondente detectado

Para cada `likely_false_heading` também é medido o IMPACTO: se o `section_kind` antes
do título era transiente (bibliografia/navegação) e virou `body` depois, os blocos
seguintes foram reclassificados indevidamente. Esse é o dano concreto do caso `EVEX?`.

Gate de confiabilidade: sem sumário, com sumário pequeno demais ou com baixa taxa de
casamento, o relatório se declara NÃO confiável (`reliable=False`) — um sumário mal
extraído não pode condenar títulos bons.
"""

from __future__ import annotations

import re
import statistics
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Optional

from backend.indexing.chunk_models import (
    BLOCK_HEADING,
    NAVIGATION_SECTION_KINDS,
    SECTION_BIBLIOGRAPHY,
    SECTION_BODY,
    DocumentBlock,
)
from backend.indexing.section_classifier import special_section_kind
from backend.indexing.toc_parser import TocEntry, TocIndex

# --- Limiares (todos conservadores: na dúvida, NÃO acusa) -------------------
FUZZY_MATCH_THRESHOLD = 0.85     # similaridade mínima para casar título ↔ sumário
PREFIX_MATCH_MIN_CHARS = 12      # prefixo curto demais casa qualquer coisa
MIN_TOC_ENTRIES = 5              # abaixo disso o sumário não é base para julgar nada
MIN_MATCH_RATE = 0.6             # % de entradas do sumário que precisam ser achadas
SHORT_HEADING_MAX_CHARS = 15     # "EVEX?" = 5, "Legenda:" = 8; "b) Infraestrutura…" = 27 escapa
SHORT_HEADING_MAX_WORDS = 2
NARROW_BBOX_RATIO = 0.35         # largura < 35% da mediana dos títulos válidos

# Estados transientes: um título espúrio que os encerra reclassifica o que vem depois.
_TRANSIENT_KINDS = frozenset({SECTION_BIBLIOGRAPHY}) | NAVIGATION_SECTION_KINDS

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_NUMBERING_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)\.?\s+")

FINDING_FALSE_HEADING = "likely_false_heading"
FINDING_NOT_IN_TOC = "heading_not_in_toc"
FINDING_LEVEL_MISMATCH = "level_mismatch"
FINDING_TOC_ORPHAN = "toc_entry_without_heading"


def normalize_for_match(text: str) -> str:
    """Forma canônica para comparação: sem acento, sem pontuação, minúscula, 1 espaço."""
    folded = unicodedata.normalize("NFKD", text or "")
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
    folded = _PUNCT_RE.sub(" ", folded.lower())
    return " ".join(folded.split())


def _strip_numbering(text: str) -> str:
    return _NUMBERING_RE.sub("", text or "").strip()


def _numbering_of(text: str) -> Optional[str]:
    m = _NUMBERING_RE.match(text or "")
    return m.group(1) if m else None


def _is_short(text: str) -> bool:
    t = (text or "").strip()
    return len(t) <= SHORT_HEADING_MAX_CHARS and len(t.split()) <= SHORT_HEADING_MAX_WORDS


def _bbox_width(block: DocumentBlock) -> Optional[float]:
    bbox = block.bbox
    if not bbox or len(bbox) < 4:
        return None
    return float(bbox[2]) - float(bbox[0])


@dataclass
class Finding:
    """Uma divergência entre os títulos detectados e o sumário."""

    kind: str
    document_id: str
    heading_text: str = ""
    heading_page: Optional[int] = None
    heading_level: Optional[int] = None      # nível EFETIVO (o que a pilha usa hoje)
    block_id: str = ""
    toc_title: str = ""
    toc_level: Optional[int] = None
    toc_page: Optional[int] = None
    # True quando o nível do sumário veio do default (entrada sem numeração), e não
    # da numeração — a divergência repousa numa suposição, não num fato.
    toc_level_inferred: bool = False
    detail: str = ""
    # Impacto no estado de seção (só em likely_false_heading).
    section_kind_before: Optional[str] = None
    section_kind_after: Optional[str] = None
    blocks_affected: int = 0
    bbox_width_ratio: Optional[float] = None

    @property
    def breaks_section_state(self) -> bool:
        return (
            self.section_kind_before in _TRANSIENT_KINDS
            and self.section_kind_after == SECTION_BODY
        )


@dataclass
class TocValidationReport:
    document_id: str
    toc: TocIndex
    findings: list[Finding] = field(default_factory=list)
    headings_total: int = 0
    headings_matched: int = 0
    toc_entries_matched: int = 0
    reliable: bool = False
    gate_reason: str = ""

    @property
    def match_rate(self) -> float:
        total = len(self.toc.entries)
        return (self.toc_entries_matched / total) if total else 0.0

    def of_kind(self, kind: str) -> list[Finding]:
        return [f for f in self.findings if f.kind == kind]

    def level_mismatches(self, *, inferred: Optional[bool] = None) -> list[Finding]:
        """Divergências de nível; `inferred=False` filtra as que se apoiam em numeração
        do sumário (fato) em vez do nível suposto para entradas sem número."""
        out = self.of_kind(FINDING_LEVEL_MISMATCH)
        if inferred is None:
            return out
        return [f for f in out if f.toc_level_inferred is inferred]

    @property
    def blocks_wrongly_reclassified(self) -> int:
        """Blocos que herdaram `body` por causa de um título espúrio."""
        return sum(f.blocks_affected for f in self.findings if f.breaks_section_state)

    @property
    def blocks_with_polluted_section_path(self) -> int:
        """Blocos cujo `section_path` recebeu um título espúrio.

        Dano de segunda ordem, independente do `section_kind`: com
        `CHUNK_EMBED_SECTION_CONTEXT=true` o `section_path` entra no texto embedado
        ("Seção: 5 Conclusão > EVEX?"), então lixo na pilha degrada o vetor."""
        return sum(f.blocks_affected for f in self.of_kind(FINDING_FALSE_HEADING))


class _TocMatcher:
    """Casa títulos com entradas do sumário. Cada entrada casa no máximo uma vez."""

    def __init__(self, entries: list[TocEntry]) -> None:
        self.entries = entries
        self._by_numbering: dict[str, TocEntry] = {}
        self._norm: list[tuple[str, TocEntry]] = []
        for e in entries:
            if e.numbering and e.numbering not in self._by_numbering:
                self._by_numbering[e.numbering] = e
            self._norm.append((normalize_for_match(_strip_numbering(e.title)), e))
        self.used: set[int] = set()

    def _compatible(self, entry: TocEntry, numbering: Optional[str]) -> bool:
        """Numeração é discriminante: '4.6.1.1 Operação nº 2639576' NÃO é a entrada
        '5.4.3 Operação nº 2639576', por mais parecido que o texto seja."""
        if numbering and entry.numbering and numbering != entry.numbering:
            return False
        return id(entry) not in self.used

    def match(self, heading: str) -> Optional[TocEntry]:
        numbering = _numbering_of(heading)
        target = normalize_for_match(_strip_numbering(heading))
        if not target:
            return None

        # 1. Numeração idêntica é o sinal mais forte que existe.
        if numbering:
            entry = self._by_numbering.get(numbering)
            if entry is not None and id(entry) not in self.used:
                return self._take(entry)

        # 2. Igualdade exata da forma canônica.
        for norm, entry in self._norm:
            if norm and norm == target and self._compatible(entry, numbering):
                return self._take(entry)

        # 3. Prefixo (título truncado no corpo, ou entrada quebrada no sumário).
        for norm, entry in self._norm:
            if len(norm) < PREFIX_MATCH_MIN_CHARS or not self._compatible(entry, numbering):
                continue
            if norm.startswith(target[:PREFIX_MATCH_MIN_CHARS]) or target.startswith(
                norm[:PREFIX_MATCH_MIN_CHARS]
            ):
                if SequenceMatcher(None, norm, target).ratio() >= 0.6:
                    return self._take(entry)

        # 4. Fuzzy — absorve ruído de OCR ("Alfabetizaçao" vs "Alfabetização").
        best: Optional[TocEntry] = None
        best_ratio = FUZZY_MATCH_THRESHOLD
        for norm, entry in self._norm:
            if not norm or not self._compatible(entry, numbering):
                continue
            ratio = SequenceMatcher(None, norm, target).ratio()
            if ratio >= best_ratio:
                best, best_ratio = entry, ratio
        return self._take(best) if best is not None else None

    def _take(self, entry: TocEntry) -> TocEntry:
        self.used.add(id(entry))
        return entry


def _section_state_impact(
    blocks: list[DocumentBlock], heading_index: int
) -> tuple[Optional[str], Optional[str], int]:
    """(kind_antes, kind_depois, nº de blocos até o próximo título) para um título."""
    before: Optional[str] = None
    for b in reversed(blocks[:heading_index]):
        if b.block_type != BLOCK_HEADING:
            before = b.section_kind
            break
    heading = blocks[heading_index]
    after = heading.section_kind
    affected = 0
    for b in blocks[heading_index + 1:]:
        if b.block_type == BLOCK_HEADING:
            break
        affected += 1
    return before, after, affected


def validate_document(
    blocks: list[DocumentBlock], toc: TocIndex, *, document_id: str = ""
) -> TocValidationReport:
    """Compara os títulos detectados com o sumário. Não modifica `blocks`."""
    report = TocValidationReport(document_id=document_id, toc=toc)
    headings = [(i, b) for i, b in enumerate(blocks) if b.block_type == BLOCK_HEADING]
    report.headings_total = len(headings)

    if not toc.found:
        report.gate_reason = "sumário não encontrado"
        return report
    if len(toc.entries) < MIN_TOC_ENTRIES:
        report.gate_reason = f"sumário com apenas {len(toc.entries)} entrada(s)"
        return report

    matcher = _TocMatcher(toc.entries)
    matched: list[tuple[int, DocumentBlock, TocEntry]] = []
    unmatched: list[tuple[int, DocumentBlock]] = []
    for index, block in headings:
        entry = matcher.match(block.text)
        if entry is None:
            unmatched.append((index, block))
        else:
            matched.append((index, block, entry))

    report.headings_matched = len(matched)
    report.toc_entries_matched = len(matcher.used)

    # Gate: sumário que não reencontra os próprios títulos não serve de árbitro.
    if report.match_rate < MIN_MATCH_RATE:
        report.gate_reason = (
            f"taxa de casamento {report.match_rate:.0%} < {MIN_MATCH_RATE:.0%} "
            f"({report.toc_entries_matched}/{len(toc.entries)} entradas)"
        )
        return report
    report.reliable = True
    report.gate_reason = "ok"

    # Largura mediana dos títulos CONFIRMADOS — referência para detectar fragmento.
    widths = [w for _, b, _ in matched if (w := _bbox_width(b)) is not None and w > 0]
    median_width = statistics.median(widths) if widths else None

    for index, block in unmatched:
        ratio = None
        width = _bbox_width(block)
        if median_width and width is not None:
            ratio = width / median_width
        numbered = _numbering_of(block.text) is not None
        # "Sumário", "Lista de tabelas", "Referências", "Apêndice A" se auto-identificam
        # e legitimamente NÃO constam do sumário — nunca são fragmentos.
        self_identifying = special_section_kind(block.text) is not None
        suspect = not numbered and not self_identifying and (
            _is_short(block.text) or (ratio is not None and ratio < NARROW_BBOX_RATIO)
        )
        kind = FINDING_FALSE_HEADING if suspect else FINDING_NOT_IN_TOC
        finding = Finding(
            kind=kind,
            document_id=document_id,
            heading_text=block.text,
            heading_page=block.page_number,
            heading_level=block.heading_level,
            block_id=block.block_id,
            detail=(
                "ausente do sumário; "
                + ("sem numeração, " if not numbered else "")
                + f"{len(block.text)} char(s)"
                + (f", bbox {ratio:.0%} da mediana" if ratio is not None else "")
            ),
            bbox_width_ratio=ratio,
        )
        if kind == FINDING_FALSE_HEADING:
            before, after, affected = _section_state_impact(blocks, index)
            finding.section_kind_before = before
            finding.section_kind_after = after
            finding.blocks_affected = affected
        report.findings.append(finding)

    for _, block, entry in matched:
        if block.heading_level is not None and block.heading_level != entry.level:
            report.findings.append(
                Finding(
                    kind=FINDING_LEVEL_MISMATCH,
                    document_id=document_id,
                    heading_text=block.text,
                    heading_page=block.page_number,
                    heading_level=block.heading_level,
                    block_id=block.block_id,
                    toc_title=entry.title,
                    toc_level=entry.level,
                    toc_page=entry.page,
                    toc_level_inferred=entry.level_inferred,
                    detail=(
                        f"nível efetivo {block.heading_level} ≠ sumário {entry.level}"
                        + (" (nível do sumário SUPOSTO: entrada sem numeração)"
                           if entry.level_inferred else " (numeração do sumário)")
                    ),
                )
            )

    for entry in toc.entries:
        if id(entry) not in matcher.used:
            report.findings.append(
                Finding(
                    kind=FINDING_TOC_ORPHAN,
                    document_id=document_id,
                    toc_title=entry.title,
                    toc_level=entry.level,
                    toc_page=entry.page,
                    detail="entrada do sumário sem título detectado no corpo",
                )
            )

    return report
