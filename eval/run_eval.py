"""Avaliação fim-a-fim de recuperação: v1 (permissivo) × v2 (política corrigida).

Recupera com o BGE-M3 REAL da API vLLM (denso + esparso, fusão RRF k=60), em memória
(sem Qdrant). Requer os contêineres vllm-bge-m3 / vllm-bge-m3-sparse no ar.
Relevância por CONTEÚDO da resposta (answer_span contido no chunk) OU sobreposição de
block_id — independente das fronteiras de chunk, para comparar as formas de modo justo.
v2 aplica o filtro por perfil de recuperação (intenção da consulta), como em produção.

Métricas @k∈{1,3,5,10}: precision@k, recall@k, success@k (hit) e MRR.

Uso:  PYTHONPATH=. uv run python eval/run_eval.py
Saída: eval/results.json
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from backend.indexing.document_blocks import MinerUDocumentParser
from backend.indexing.retrieval_profile import detect_query_profile, profile_exclusions
from backend.indexing.structural_token_chunker import ChunkingConfig, StructuralTokenChunker
from backend.indexing.token_counter import get_token_counter

EVAL = Path("eval")
GOLD = EVAL / "goldset.jsonl"
DOCS = sorted(Path("output").rglob("*_content_list_v2.json"))
KS = (1, 3, 5, 10)
RRF_K = 60

PROFILE = dict(target_tokens=256, max_tokens=512, min_tokens=48, overlap_tokens=48,
               max_overlap_tokens=96, table_max_tokens=512, list_max_tokens=384,
               force_split_above_tokens=100000)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


# --------------------------------------------------------------------------
# Constrói os conjuntos de chunks das duas formas
# --------------------------------------------------------------------------

def build_chunks(form: str):
    tc = get_token_counter()
    if form == "v1":
        parser_kw = dict(normalize_text=False, reconstruct_cross_page=False)
        cfg_kw = dict(front_matter_mode="include", references_mode="separate")
    else:  # v2
        parser_kw = dict(normalize_text=True, reconstruct_cross_page=True)
        cfg_kw = dict(front_matter_mode="metadata_only", references_mode="metadata_only",
                      appendix_mode="classify", equation_mode="merge_with_context")
    out = []
    for path in DOCS:
        doc = path.parent.parent.name
        p = MinerUDocumentParser(**parser_kw)
        blocks, src = p.parse_json_file(path)
        ch = StructuralTokenChunker(
            config=ChunkingConfig(**PROFILE, chunking_version=form, **cfg_kw), token_counter=tc)
        res = ch.chunk(blocks, document_id=doc, document_title=p.document_title or doc,
                       structure_source=src)
        out.extend(res.chunks)
    return out


# --------------------------------------------------------------------------
# Embeddings (BGE-M3 real: denso + esparso)
# --------------------------------------------------------------------------

def load_model():
    """O mesmo embedder da produção (API vLLM). Falha cedo se a API estiver fora —
    avaliar com um backend diferente do indexado não diz nada."""
    from backend.services.embedder import BgeM3EmbedderService

    embedder = BgeM3EmbedderService()
    embedder.health_check(raise_on_error=True)
    return embedder


def encode(model, texts):
    dense_list, lexical = model.embed_documents(list(texts), batch_size=16)
    dense = np.asarray(dense_list, dtype=np.float32)
    dense /= (np.linalg.norm(dense, axis=1, keepdims=True) + 1e-9)   # normaliza p/ cosine
    return dense, lexical


def sparse_dot(a: dict, b: dict) -> float:
    if len(a) > len(b):
        a, b = b, a
    return float(sum(w * b.get(t, 0.0) for t, w in a.items()))


# --------------------------------------------------------------------------
# Recuperação híbrida RRF
# --------------------------------------------------------------------------

def rrf_rank(dense_scores, sparse_scores, allowed: list[int], top: int) -> list[int]:
    if not allowed:
        return []
    d_order = sorted(allowed, key=lambda i: dense_scores[i], reverse=True)
    s_order = sorted(allowed, key=lambda i: sparse_scores[i], reverse=True)
    d_rank = {idx: r for r, idx in enumerate(d_order)}
    s_rank = {idx: r for r, idx in enumerate(s_order)}
    fused = sorted(allowed, key=lambda i: 1.0 / (RRF_K + d_rank[i]) + 1.0 / (RRF_K + s_rank[i]),
                   reverse=True)
    return fused[:top]


def allowed_for_v2(chunks, profile: str) -> list[int]:
    ex = profile_exclusions(profile)
    idx = []
    for i, c in enumerate(chunks):
        if ex.require_is_reference and not c.is_reference:
            continue
        if ex.require_searchable_by_default and not c.searchable_by_default:
            continue
        if ex.exclude_is_reference and c.is_reference:
            continue
        if c.section_kind in ex.exclude_section_kinds:
            continue
        idx.append(i)
    return idx


def relevant(chunk, gold_span_norm: str, gold_doc: str, gold_blocks: set) -> bool:
    if chunk.document_id != gold_doc:
        return False
    if gold_blocks & set(chunk.block_ids):
        return True
    return gold_span_norm in _norm(chunk.text)


# --------------------------------------------------------------------------
# Avaliação
# --------------------------------------------------------------------------

def eval_form(form, chunks, model, gold, q_dense, q_lex):
    texts = [c.embedding_input("contextualized_text") for c in chunks]
    print(f"[{form}] embeddando {len(texts)} chunks…")
    d_mat, lex = encode(model, texts)
    doc_ids = [c.document_id for c in chunks]

    per_query = []
    for gi, g in enumerate(gold):
        prof = detect_query_profile(g["question"])
        allowed = list(range(len(chunks))) if form == "v1" else allowed_for_v2(chunks, prof)
        dense_scores = d_mat @ q_dense[gi]
        sparse_scores = [sparse_dot(q_lex[gi], lex[j]) for j in allowed]
        s_full = np.full(len(chunks), -1.0)
        for pos, j in enumerate(allowed):
            s_full[j] = sparse_scores[pos]
        ranked = rrf_rank(dense_scores, s_full, allowed, max(KS))

        gspan = _norm(g["answer_span"]); gdoc = g["doc_id"]; gblocks = set(g["gold_block_ids"])
        rels = [relevant(chunks[j], gspan, gdoc, gblocks) for j in ranked]
        total_rel = sum(1 for i, c in enumerate(chunks)
                        if relevant(c, gspan, gdoc, gblocks) and (form == "v1" or i in set(allowed)))
        first = next((r + 1 for r, ok in enumerate(rels) if ok), 0)
        per_query.append({"qid": g["qid"], "profile": prof, "nct": g["nct"],
                          "section_kind": g.get("section_kind"),
                          "rels": rels, "total_rel": total_rel, "first": first})
    return per_query


def aggregate(per_query):
    agg = {}
    n = len(per_query)
    for k in KS:
        p = np.mean([sum(q["rels"][:k]) / k for q in per_query])
        r = np.mean([(sum(q["rels"][:k]) / q["total_rel"]) if q["total_rel"] else 0.0
                     for q in per_query])
        hit = np.mean([1.0 if any(q["rels"][:k]) else 0.0 for q in per_query])
        agg[f"P@{k}"] = round(float(p), 4)
        agg[f"R@{k}"] = round(float(r), 4)
        agg[f"hit@{k}"] = round(float(hit), 4)
    agg["MRR@10"] = round(float(np.mean([1.0 / q["first"] if q["first"] else 0.0
                                         for q in per_query])), 4)
    agg["n"] = n
    return agg


def main():
    if not GOLD.exists():
        raise SystemExit("eval/goldset.jsonl não encontrado — rode build_goldset.py primeiro.")
    gold = [json.loads(ln) for ln in GOLD.open(encoding="utf-8")]
    print(f"gold: {len(gold)} perguntas")

    model = load_model()
    print("[query] embeddando perguntas…")
    q_dense, q_lex = encode(model, [g["question"] for g in gold])

    results = {"n": len(gold), "k": list(KS), "forms": {}, "by_profile": {}}
    pq_by_form = {}
    for form in ("v1", "v2"):
        chunks = build_chunks(form)
        print(f"[{form}] {len(chunks)} chunks no corpus")
        pq = eval_form(form, chunks, model, gold, q_dense, q_lex)
        pq_by_form[form] = pq
        results["forms"][form] = aggregate(pq)

    # quebra por perfil detectado (usa a partição da v2; mesma p/ v1)
    profiles = sorted({q["profile"] for q in pq_by_form["v1"]})
    for prof in profiles:
        results["by_profile"][prof] = {}
        for form in ("v1", "v2"):
            sub = [q for q in pq_by_form[form] if q["profile"] == prof]
            results["by_profile"][prof][form] = aggregate(sub) if sub else {}
        results["by_profile"][prof]["count"] = sum(1 for q in pq_by_form["v1"] if q["profile"] == prof)

    # quebra por section_kind da passagem-resposta (mostra o trade-off da v2).
    v1_by = {q["qid"]: q for q in pq_by_form["v1"]}
    v2_by = {q["qid"]: q for q in pq_by_form["v2"]}
    sk_stats = defaultdict(lambda: {"n": 0, "v1_hit5": 0, "v2_hit5": 0})
    delta = []
    for qid, q1 in v1_by.items():
        q2 = v2_by[qid]
        sk = q1["section_kind"] or "?"
        h1, h2 = any(q1["rels"][:5]), any(q2["rels"][:5])
        sk_stats[sk]["n"] += 1
        sk_stats[sk]["v1_hit5"] += int(h1)
        sk_stats[sk]["v2_hit5"] += int(h2)
        if h1 and not h2:
            delta.append({"qid": qid, "section_kind": sk, "nct": q1["nct"], "profile": q1["profile"]})
    results["by_section_kind"] = {k: v for k, v in sk_stats.items()}
    results["v1hit_v2miss@5"] = delta

    (EVAL / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    print("\n===== RESULTADO =====")
    for form in ("v1", "v2"):
        a = results["forms"][form]
        print(f"{form}: P@1={a['P@1']} P@5={a['P@5']} R@5={a['R@5']} R@10={a['R@10']} "
              f"hit@5={a['hit@5']} MRR={a['MRR@10']}")
    print("por perfil:")
    for prof, d in results["by_profile"].items():
        print(f"  {prof} (n={d['count']}): v1 hit@5={d['v1'].get('hit@5')} MRR={d['v1'].get('MRR@10')}"
              f" | v2 hit@5={d['v2'].get('hit@5')} MRR={d['v2'].get('MRR@10')}")
    print(f"\n✅ salvo em {EVAL / 'results.json'}")


if __name__ == "__main__":
    main()
