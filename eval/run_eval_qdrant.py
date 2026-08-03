"""Avaliação fim-a-fim REAL contra o Qdrant ao vivo (política v2 implantada).

Diferente de `run_eval.py` (recuperação em memória, v1×v2), este consulta a collection
`evidencia_chunks` JÁ reindexada, via o mesmo `SemanticSearch` de produção (BGE-M3 +
RRF + filtro por perfil quando SEARCH_EXCLUDE_* está ligado). Mede a busca como o
usuário a vê, agora com o corpus INTEIRO (67 docs) como distratores.

Relevância: o chunk recuperado contém o `answer_span` do gold (comparação normalizada)
E pertence ao documento de origem. Métricas @k∈{1,3,5,10}: hit@k, precision@k, MRR.

Uso:  PYTHONPATH=. uv run python eval/run_eval_qdrant.py [--type hybrid] [--limit 0]
Saída: eval/results_qdrant.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path

from backend.repositories.qdrant_client import SemanticSearch

EVAL = Path("eval")
GOLD = EVAL / "goldset.jsonl"
KS = (1, 3, 5, 10)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def _relevant(result, gold_doc: str, gold_span_norm: str) -> bool:
    doc = f"{result.doc_id or ''} {result.doc_name or ''}".lower()
    if gold_doc.lower() not in doc:
        return False
    return gold_span_norm in _norm(result.snippet)


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", default="hybrid", help="hybrid | dense | sparse")
    ap.add_argument("--limit", type=int, default=0, help="avalia só as N primeiras (0=todas)")
    ap.add_argument("--profile", default="", help="perfil forçado ('' = auto pela intenção)")
    args = ap.parse_args()

    gold = [json.loads(ln) for ln in GOLD.open(encoding="utf-8")]
    if args.limit:
        gold = gold[: args.limit]
    search = SemanticSearch()
    if not await search.ensure_connected():
        raise SystemExit("Qdrant/collection indisponível — reindexe antes.")

    per_query = []
    for g in gold:
        results = await search.search(g["question"], limit=max(KS), type=args.type,
                                      profile=args.profile)
        gspan = _norm(g["answer_span"])
        rels = [_relevant(r, g["doc_id"], gspan) for r in results]
        first = next((i + 1 for i, ok in enumerate(rels) if ok), 0)
        per_query.append({"qid": g["qid"], "nct": g["nct"], "section_kind": g.get("section_kind"),
                          "rels": rels, "first": first, "n_results": len(results)})

    import numpy as np
    agg = {}
    for k in KS:
        agg[f"P@{k}"] = round(float(np.mean([sum(q["rels"][:k]) / k for q in per_query])), 4)
        agg[f"hit@{k}"] = round(float(np.mean([1.0 if any(q["rels"][:k]) else 0.0
                                               for q in per_query])), 4)
    agg["MRR@10"] = round(float(np.mean([1.0 / q["first"] if q["first"] else 0.0
                                         for q in per_query])), 4)
    agg["n"] = len(per_query)
    agg["misses"] = [q["qid"] for q in per_query if not any(q["rels"][:10])]

    out = {"type": args.type, "profile": args.profile or "auto", "live": True, "metrics": agg}
    (EVAL / "results_qdrant.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                              encoding="utf-8")
    print(f"\n===== Qdrant ao vivo (type={args.type}, profile={args.profile or 'auto'}, n={agg['n']}) =====")
    print(f"hit@1={agg['hit@1']} hit@3={agg['hit@3']} hit@5={agg['hit@5']} hit@10={agg['hit@10']} MRR={agg['MRR@10']}")
    print(f"misses@10 ({len(agg['misses'])}): {agg['misses']}")
    print(f"✅ salvo em {EVAL / 'results_qdrant.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
