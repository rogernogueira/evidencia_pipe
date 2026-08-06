#!/usr/bin/env python
"""Relatório de divergências entre os TÍTULOS detectados e o SUMÁRIO do documento.

DIAGNÓSTICO APENAS — não altera classificação, não reindexa, não escreve no Qdrant.
Serve para medir, no acervo inteiro, o problema encontrado em
exames-educ-basica_relatorio-de-avaliacao: o fragmento "EVEX?" virou `title` nível 2,
desempilhou "Referências bibliográficas" e fez 14 chunks de bibliografia serem
indexados como corpo (`searchable_by_default=true`).

Roda só em CPU: parser estrutural + `toc_parser`/`toc_validator`. Sem GPU, sem
embedding, sem LLM (o gate visual do parser é desligado de propósito — não influencia
títulos e deixaria o relatório caro e não-determinístico).

Uso (a partir de /app/evidencia_pipe):
    uv run python scripts/toc_heading_report.py                    # acervo inteiro
    uv run python scripts/toc_heading_report.py --limit 5          # amostra
    uv run python scripts/toc_heading_report.py --only <document_id>
    uv run python scripts/toc_heading_report.py --local caminho/content_list_v2.json
    uv run python scripts/toc_heading_report.py --out-dir relatorios/

Saídas (CSV, em --out-dir, default = diretório atual):
    toc_report_documentos.csv   uma linha por documento (gate, taxas, impacto)
    toc_report_divergencias.csv uma linha por divergência
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.core import config as settings
from backend.core.schemas import ART_MINERU_CONTENT_LIST, ArtifactManifest
from backend.indexing.document_blocks import MinerUDocumentParser
from backend.indexing.toc_parser import parse_toc
from backend.indexing.toc_validator import (
    FINDING_FALSE_HEADING,
    FINDING_LEVEL_MISMATCH,
    FINDING_NOT_IN_TOC,
    FINDING_TOC_ORPHAN,
    TocValidationReport,
    validate_document,
)
from backend.services.artifact_store import get_artifact_store

DOC_COLUMNS = [
    "document_id", "reliable", "gate_reason", "toc_source", "toc_entries",
    "headings_total", "headings_matched", "toc_entries_matched", "match_rate",
    "likely_false_headings", "headings_breaking_section_state",
    "blocks_wrongly_reclassified", "blocks_with_polluted_section_path",
    "level_mismatches", "level_mismatches_numbered",
    "toc_orphans", "headings_not_in_toc",
]

FINDING_COLUMNS = [
    "document_id", "kind", "heading_text", "heading_page", "heading_level", "block_id",
    "toc_title", "toc_level", "toc_level_inferred", "toc_page",
    "section_kind_before", "section_kind_after",
    "blocks_affected", "breaks_section_state", "bbox_width_ratio", "detail",
]


def _build_parser() -> MinerUDocumentParser:
    """Mesmos flags do indexador ([index_chunks]), menos o gate visual por LLM."""
    return MinerUDocumentParser(
        remove_repeated_headers=settings.CHUNK_REMOVE_REPEATED_HEADERS,
        remove_repeated_footers=settings.CHUNK_REMOVE_REPEATED_FOOTERS,
        keep_footnotes=settings.CHUNK_KEEP_FOOTNOTES,
        normalize_text=settings.CHUNK_NORMALIZE_TEXT,
        reconstruct_cross_page=settings.MINERU_RECONSTRUCT_CROSS_PAGE_PARAGRAPHS,
        chart_mode=settings.CHUNK_CHART_MODE,
        image_mode=settings.CHUNK_IMAGE_MODE,
        chart_min_confidence=settings.CHUNK_CHART_MIN_CONFIDENCE,
        visual_llm=False,
    )


def analyze(data, document_id: str) -> TocValidationReport:
    blocks, _ = _build_parser().parse_json(data)
    return validate_document(blocks, parse_toc(data), document_id=document_id)


def _doc_row(r: TocValidationReport) -> dict:
    breaking = [f for f in r.findings if f.breaks_section_state]
    return {
        "document_id": r.document_id,
        "reliable": r.reliable,
        "gate_reason": r.gate_reason,
        "toc_source": r.toc.source,
        "toc_entries": len(r.toc.entries),
        "headings_total": r.headings_total,
        "headings_matched": r.headings_matched,
        "toc_entries_matched": r.toc_entries_matched,
        "match_rate": f"{r.match_rate:.3f}",
        "likely_false_headings": len(r.of_kind(FINDING_FALSE_HEADING)),
        "headings_breaking_section_state": len(breaking),
        "blocks_wrongly_reclassified": r.blocks_wrongly_reclassified,
        "blocks_with_polluted_section_path": r.blocks_with_polluted_section_path,
        "level_mismatches": len(r.level_mismatches()),
        "level_mismatches_numbered": len(r.level_mismatches(inferred=False)),
        "toc_orphans": len(r.of_kind(FINDING_TOC_ORPHAN)),
        "headings_not_in_toc": len(r.of_kind(FINDING_NOT_IN_TOC)),
    }


def _finding_rows(r: TocValidationReport) -> list[dict]:
    rows = []
    for f in r.findings:
        rows.append({
            "document_id": f.document_id,
            "kind": f.kind,
            "heading_text": f.heading_text,
            "heading_page": f.heading_page if f.heading_page is not None else "",
            "heading_level": f.heading_level if f.heading_level is not None else "",
            "block_id": f.block_id,
            "toc_title": f.toc_title,
            "toc_level": f.toc_level if f.toc_level is not None else "",
            "toc_level_inferred": f.toc_level_inferred,
            "toc_page": f.toc_page if f.toc_page is not None else "",
            "section_kind_before": f.section_kind_before or "",
            "section_kind_after": f.section_kind_after or "",
            "blocks_affected": f.blocks_affected,
            "breaks_section_state": f.breaks_section_state,
            "bbox_width_ratio": (
                f"{f.bbox_width_ratio:.3f}" if f.bbox_width_ratio is not None else ""
            ),
            "detail": f.detail,
        })
    return rows


def _iter_manifests(store):
    prefix = f"{settings.MINIO_ARTIFACT_PREFIX.strip('/')}/"
    for ref in store.list_prefix(prefix):
        if ref.object_key.endswith("/manifest.json"):
            yield ref.object_key


def _collect_from_minio(only: str, limit: int) -> list[TocValidationReport]:
    store = get_artifact_store()
    manifests: list[ArtifactManifest] = []
    for key in _iter_manifests(store):
        try:
            m = ArtifactManifest(**store.read_json(key))
        except Exception as exc:  # noqa: BLE001 — manifesto corrompido não trava o relatório
            print(f"  ! manifesto ilegível {key}: {exc}", flush=True)
            continue
        if only and m.document_id != only:
            continue
        if ART_MINERU_CONTENT_LIST not in m.artifacts:
            continue
        manifests.append(m)
    if limit:
        manifests = manifests[:limit]

    print(f"Documentos com content_list do MinerU: {len(manifests)}", flush=True)
    reports: list[TocValidationReport] = []
    for i, m in enumerate(manifests, 1):
        ref = m.artifacts[ART_MINERU_CONTENT_LIST]
        try:
            data = store.read_json(ref.object_key)
            reports.append(analyze(data, m.document_id))
        except Exception as exc:  # noqa: BLE001
            print(f"  ✗ [{i}/{len(manifests)}] {m.document_id}: ERRO {exc}", flush=True)
            continue
        r = reports[-1]
        flag = "⚠" if r.blocks_wrongly_reclassified else " "
        print(
            f"  {flag} [{i}/{len(manifests)}] {r.document_id}: "
            f"sumário={len(r.toc.entries)} títulos={r.headings_total} "
            f"casou={r.match_rate:.0%} suspeitos={len(r.of_kind(FINDING_FALSE_HEADING))} "
            f"nível≠={len(r.of_kind(FINDING_LEVEL_MISMATCH))} "
            f"blocos_afetados={r.blocks_wrongly_reclassified}"
            + ("" if r.reliable else f"  [NÃO CONFIÁVEL: {r.gate_reason}]"),
            flush=True,
        )
    return reports


def _write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _summary(reports: list[TocValidationReport]) -> None:
    if not reports:
        print("\nNenhum documento analisado.")
        return
    reliable = [r for r in reports if r.reliable]
    no_toc = [r for r in reports if not r.toc.found]
    damaged = [r for r in reliable if r.blocks_wrongly_reclassified]
    level_bad = [r for r in reliable if r.level_mismatches()]
    level_bad_hard = [r for r in reliable if r.level_mismatches(inferred=False)]

    print("\n" + "=" * 72)
    print(f"Documentos analisados ............ {len(reports)}")
    print(f"  sem sumário utilizável ......... {len(no_toc)}")
    print(f"  com sumário confiável .......... {len(reliable)}")
    print(f"\nEntre os {len(reliable)} confiáveis:")
    print(f"  com título espúrio ............. "
          f"{sum(1 for r in reliable if r.of_kind(FINDING_FALSE_HEADING))}")
    print(f"  com QUEBRA de seção ............ {len(damaged)}   <- bibliografia/navegação "
          f"virou corpo")
    print(f"  blocos reclassificados ......... "
          f"{sum(r.blocks_wrongly_reclassified for r in damaged)}")
    print(f"  blocos com section_path poluído  "
          f"{sum(r.blocks_with_polluted_section_path for r in reliable)}"
          f"   <- entra no texto embedado se CHUNK_EMBED_SECTION_CONTEXT=true")
    print(f"  com nível divergente ........... {len(level_bad)}"
          f"  (dos quais {len(level_bad_hard)} apoiados em numeração do sumário;")
    print("                                     o restante usa o nível SUPOSTO para "
          "entradas sem número)")
    if damaged:
        print("\nDocumentos com quebra de seção (candidatos a reindexar após a correção):")
        for r in sorted(damaged, key=lambda x: -x.blocks_wrongly_reclassified):
            culprits = ", ".join(
                f"'{f.heading_text[:30]}' (pg {f.heading_page}, {f.section_kind_before}"
                f"→{f.section_kind_after}, {f.blocks_affected} blocos)"
                for f in r.findings if f.breaks_section_state
            )
            print(f"  · {r.document_id}: {culprits}")
    print("=" * 72)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Relatório sumário↔títulos (diagnóstico; não altera nada)."
    )
    ap.add_argument("--only", default="", help="analisa apenas este document_id")
    ap.add_argument("--limit", type=int, default=0, help="no máximo N documentos (0 = todos)")
    ap.add_argument("--local", default="", help="analisa um content_list_v2.json local")
    ap.add_argument("--out-dir", default=".", help="diretório dos CSVs (default: atual)")
    ap.add_argument("--no-csv", action="store_true", help="só imprime, não escreve CSV")
    args = ap.parse_args()

    if args.local:
        path = Path(args.local).resolve()
        reports = [analyze(json.loads(path.read_text(encoding="utf-8")), path.parent.name or path.stem)]
    else:
        reports = _collect_from_minio(args.only, args.limit)

    if not args.no_csv and reports:
        out_dir = Path(args.out_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        docs_csv = out_dir / "toc_report_documentos.csv"
        find_csv = out_dir / "toc_report_divergencias.csv"
        _write_csv(docs_csv, DOC_COLUMNS, [_doc_row(r) for r in reports])
        _write_csv(find_csv, FINDING_COLUMNS, [row for r in reports for row in _finding_rows(r)])
        print(f"\nCSV: {docs_csv}\n     {find_csv}")

    _summary(reports)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
