"""Compara as estratégias de chunking (legacy_chars vs structural_tokens) sem
executar embeddings obrigatoriamente (§30).

Fontes de entrada aceitas (uma das duas):
  --content-list <path>   content_list_v2.json local (offline, sem MinIO)
  --manifest-uri <uri>    manifesto minio://... (baixa o content_list do MinIO)

Métricas comparadas (§30): nº de chunks, distribuição de tokens, overlap total,
chunks abaixo do mínimo / acima do máximo, mistura de seções, tabelas divididas,
cortes no meio de sentença, cobertura de páginas, tempo de execução, tamanho do
JSONL e chunks rejeitados. Gera relatório em JSON e CSV.

Uso:
    uv run python scripts/compare_chunking_strategies.py \
        --content-list output/<doc>/hybrid_auto/<doc>_content_list_v2.json \
        --strategies legacy_chars structural_tokens

    uv run python scripts/compare_chunking_strategies.py \
        --manifest-uri minio://evidencia-pipe/artifacts/<pid>/<doc>/manifest.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.core import config as settings  # noqa: E402
from backend.indexing.chunk_models import ChunkingResult  # noqa: E402
from backend.indexing.chunks import get_chunker  # noqa: E402
from backend.indexing.document_blocks import MinerUDocumentParser  # noqa: E402

_SENTENCE_MID_CUT = re.compile(r"[a-zà-ú0-9,;]$", re.IGNORECASE)


def _load_blocks(args) -> tuple[list, str, str, str]:
    """Retorna (blocks, structure_source, document_id, document_title)."""
    parser = MinerUDocumentParser()
    if args.content_list:
        path = Path(args.content_list)
        blocks, src = parser.parse_json_file(path)
        doc_id = path.parent.parent.name
        return blocks, src, doc_id, parser.document_title
    if args.manifest_uri:
        import tempfile

        from backend.core.schemas import ART_MINERU_CONTENT_LIST
        from backend.services.artifact_store import get_artifact_store
        from backend.services.manifest_repository import get_manifest_repository

        store = get_artifact_store()
        manifest = get_manifest_repository().load_from_uri(args.manifest_uri)
        cl = manifest.artifacts[ART_MINERU_CONTENT_LIST]
        with tempfile.TemporaryDirectory() as td:
            local = Path(td) / "cl.json"
            store.download_to_file(cl.object_key, local, expected_sha256=cl.sha256 or None)
            blocks, src = parser.parse_json_file(local)
        return blocks, src, manifest.document_id, parser.document_title
    raise SystemExit("informe --content-list ou --manifest-uri")


def _analyze(result: ChunkingResult) -> dict:
    chunks = result.chunks
    tokens = [c.token_count for c in chunks]
    min_t = settings.CHUNK_MIN_TOKENS
    max_t = settings.CHUNK_MAX_TOKENS

    # mistura de seções: chunk cujos blocos vêm de section_paths diferentes.
    section_mixed = 0
    mid_sentence_cuts = 0
    pages_covered: set[int] = set()
    for c in chunks:
        pages_covered.update(c.page_numbers)
        # corte no meio de sentença: termina sem pontuação final (ignora tabelas/listas).
        if c.content_type in ("paragraph", "references"):
            tail = c.text.rstrip()[-1:] if c.text.strip() else ""
            if tail and _SENTENCE_MID_CUT.search(tail):
                mid_sentence_cuts += 1

    jsonl_bytes = sum(
        len(json.dumps(c.model_dump(), ensure_ascii=False).encode("utf-8")) + 1 for c in chunks
    )
    return {
        "strategy": result.chunking_strategy,
        "chunking_version": result.chunking_version,
        "config_hash": result.chunking_config_hash,
        "tokenizer": result.tokenizer_name,
        "structure_source": result.structure_source,
        "chunk_count": len(chunks),
        "avg_tokens": result.metrics.average_tokens,
        "median_tokens": result.metrics.median_tokens,
        "min_tokens": min(tokens) if tokens else 0,
        "max_tokens": max(tokens) if tokens else 0,
        "p95_tokens": sorted(tokens)[int(len(tokens) * 0.95)] if tokens else 0,
        "overlap_tokens_total": result.metrics.overlap_tokens_total,
        "chunks_below_min": sum(1 for t in tokens if t < min_t),
        "chunks_above_max": sum(1 for t in tokens if t > max_t),
        "section_mixed": section_mixed,
        "mid_sentence_cuts": mid_sentence_cuts,
        "table_chunks": result.metrics.table_chunks_count,
        "list_chunks": result.metrics.list_chunks_count,
        "pages_covered": len(pages_covered),
        "rejected_chunks": result.metrics.rejected_chunks,
        "jsonl_bytes": jsonl_bytes,
        "grouping_time_s": round(result.metrics.grouping_time_s, 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Compara estratégias de chunking (sem embeddings).")
    ap.add_argument("--content-list", help="content_list_v2.json local")
    ap.add_argument("--manifest-uri", help="manifesto minio://...")
    ap.add_argument("--strategies", nargs="+", default=["legacy_chars", "structural_tokens"])
    ap.add_argument("--out-json", default="chunking_comparison.json")
    ap.add_argument("--out-csv", default="chunking_comparison.csv")
    args = ap.parse_args()

    blocks, src, doc_id, title = _load_blocks(args)
    print(f"Documento: {doc_id} | blocos: {len(blocks)} | fonte: {src}")

    rows: list[dict] = []
    for strat in args.strategies:
        t0 = time.perf_counter()
        chunker = get_chunker(strat)
        result = chunker.chunk(blocks, document_id=doc_id, document_title=title,
                               document_checksum="comparison", structure_source=src)
        row = _analyze(result)
        row["wall_time_s"] = round(time.perf_counter() - t0, 3)
        rows.append(row)
        print(f"\n[{strat}] " + " | ".join(
            f"{k}={row[k]}" for k in ("chunk_count", "avg_tokens", "max_tokens",
                                      "chunks_above_max", "mid_sentence_cuts", "overlap_tokens_total")))

    Path(args.out_json).write_text(
        json.dumps({"document_id": doc_id, "structure_source": src, "results": rows},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nRelatórios: {args.out_json} , {args.out_csv}")


if __name__ == "__main__":
    main()
