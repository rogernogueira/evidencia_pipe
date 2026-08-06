"""Avaliação inicial de recuperação: legacy_chars vs structural_tokens (§31).

Compara a QUALIDADE DE BUSCA das duas estratégias sobre o MESMO conjunto de queries
rotuladas. Métricas: recall@k, precision@k, MRR, nDCG@k, taxa de chunks sem contexto
e taxa de chunks misturando seções.

Requer o modelo BGE-M3 (dense) para embeddar queries e chunks. É um harness inicial —
NÃO declara nenhuma estratégia superior sem execução real (§31).

Entrada: um arquivo de queries JSON:
    [
      {"query": "quais foram os motivos do subsídio?",
       "relevant": {"section_contains": "motiv", "text_contains": "subsídio"}}
    ]

Uso:
    uv run python scripts/evaluate_chunking_retrieval.py \
        --content-list output/<doc>/hybrid_auto/<doc>_content_list_v2.json \
        --queries queries.json --k 5
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.core import config as settings  # noqa: E402
from backend.indexing.chunks import get_chunker  # noqa: E402
from backend.indexing.document_blocks import MinerUDocumentParser  # noqa: E402


def _is_relevant(chunk, rule: dict) -> bool:
    ok = True
    if "section_contains" in rule:
        sec = " ".join(chunk.section_path).lower()
        ok = ok and rule["section_contains"].lower() in sec
    if "text_contains" in rule:
        ok = ok and rule["text_contains"].lower() in chunk.text.lower()
    return ok


def _dcg(rels: list[int]) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def _evaluate(chunks, embedder, queries: list[dict], k: int) -> dict:
    field = settings.EMBEDDING_TEXT_FIELD
    texts = [c.embedding_input(field) for c in chunks]
    dense, _ = embedder.embed_documents(texts, batch_size=settings.EMBEDDING_BATCH_SIZE)

    import numpy as np

    mat = np.array(dense, dtype="float32")
    mat /= (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)

    recalls, precisions, rrs, ndcgs = [], [], [], []
    for q in queries:
        qv, _ = embedder.embed_query(q["query"])
        qv = np.array(qv, dtype="float32")
        qv /= (np.linalg.norm(qv) + 1e-9)
        scores = mat @ qv
        order = scores.argsort()[::-1][:k]
        rel_flags = [1 if _is_relevant(chunks[i], q["relevant"]) else 0 for i in order]
        total_rel = sum(1 for c in chunks if _is_relevant(c, q["relevant"])) or 1
        hits = sum(rel_flags)
        recalls.append(hits / total_rel)
        precisions.append(hits / k)
        rr = next((1 / (r + 1) for r, f in enumerate(rel_flags) if f), 0.0)
        rrs.append(rr)
        ideal = sorted(rel_flags, reverse=True)
        ndcgs.append(_dcg(rel_flags) / (_dcg(ideal) or 1.0))

    n = len(queries) or 1
    no_context = sum(1 for c in chunks if not c.section_path) / (len(chunks) or 1)
    return {
        f"recall@{k}": round(sum(recalls) / n, 4),
        f"precision@{k}": round(sum(precisions) / n, 4),
        "mrr": round(sum(rrs) / n, 4),
        f"ndcg@{k}": round(sum(ndcgs) / n, 4),
        "chunks_without_context_rate": round(no_context, 4),
        "chunk_count": len(chunks),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Avaliação de recuperação por estratégia de chunking.")
    ap.add_argument("--content-list", required=True)
    ap.add_argument("--queries", required=True, help="JSON com queries rotuladas")
    ap.add_argument("--strategies", nargs="+", default=["legacy_chars", "structural_tokens"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out-json", default="chunking_retrieval_eval.json")
    args = ap.parse_args()

    from backend.services.embedder import BgeM3EmbedderService

    embedder = BgeM3EmbedderService()
    if not embedder.health_check():
        raise SystemExit(
            "API de embedding (vLLM) indisponível — necessária para a avaliação. "
            "Suba: docker compose up -d vllm-bge-m3 vllm-bge-m3-sparse"
        )

    queries = json.loads(Path(args.queries).read_text(encoding="utf-8"))
    parser = MinerUDocumentParser()
    path = Path(args.content_list)
    blocks, src = parser.parse_json_file(path)
    doc_id = path.parent.parent.name

    results = {}
    for strat in args.strategies:
        chunker = get_chunker(strat)
        res = chunker.chunk(blocks, document_id=doc_id, document_title=parser.document_title,
                            document_checksum="eval", structure_source=src)
        results[strat] = _evaluate(res.chunks, embedder, queries, args.k)
        print(f"[{strat}] {results[strat]}")

    Path(args.out_json).write_text(
        json.dumps({"document_id": doc_id, "k": args.k, "results": results},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nRelatório: {args.out_json}")


if __name__ == "__main__":
    main()
