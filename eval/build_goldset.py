"""Gera o gold set de avaliação de recuperação (100 perguntas com passagem-resposta).

Para cada passagem elegível (parágrafo de corpo ou tabela) o LLM (DeepSeek, temp 0)
cria UMA pergunta natural, autocontida e factual, e extrai o `answer_span` como
substring EXATA da passagem. A relevância na avaliação é por CONTEÚDO da resposta
(span containment) — independente das fronteiras de chunk — para comparar v1×v2 de
forma justa.

Saída: eval/goldset.jsonl (reproduzível; re-rodar reaproveita o arquivo existente).

Uso:  PYTHONPATH=. uv run python eval/build_goldset.py [--n 100]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import re
from pathlib import Path

from openai import OpenAI

from backend.core import config as cfg
from backend.indexing.document_blocks import MinerUDocumentParser
from backend.indexing.structural_token_chunker import ChunkingConfig, StructuralTokenChunker
from backend.indexing.token_counter import get_token_counter

EVAL = Path("eval")
EVAL.mkdir(exist_ok=True)
GOLD = EVAL / "goldset.jsonl"
DOCS = sorted(Path("output").rglob("*_content_list_v2.json"))

PROFILE = dict(target_tokens=256, max_tokens=512, min_tokens=48, overlap_tokens=48,
               max_overlap_tokens=96, table_max_tokens=512, list_max_tokens=384,
               force_split_above_tokens=100000)

SYS = (
    "Você cria pares pergunta-resposta para avaliar um sistema de BUSCA em relatórios "
    "de avaliação de políticas públicas. Dada UMA passagem, gere UMA pergunta natural, "
    "específica e AUTOCONTIDA em português, cuja resposta esteja EXPLÍCITA e completa na "
    "passagem. A pergunta não pode depender de ver a passagem (não use 'segundo o texto'). "
    "Extraia answer_span como uma substring EXATA (copiada, sem alterar) da passagem que "
    "contém a resposta. Se a passagem não permitir uma pergunta factual clara e autocontida "
    "(ex.: fragmento, tabela sem rótulos, texto genérico), use answerable=false. "
    "Responda SÓ com JSON: {\"answerable\": bool, \"question\": str, \"answer_span\": str}."
)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def eligible_passages():
    """Passagens candidatas das duas coleções (corpo + tabela), espalhadas de forma
    determinística (sem aleatoriedade → reprodutível)."""
    tc = get_token_counter()
    body, tables = [], []
    for path in DOCS:
        doc = path.parent.parent.name
        parser = MinerUDocumentParser(normalize_text=True, reconstruct_cross_page=True)
        blocks, src = parser.parse_json_file(path)
        ch = StructuralTokenChunker(
            config=ChunkingConfig(**PROFILE, chunking_version="eval"), token_counter=tc)
        res = ch.chunk(blocks, document_id=doc, document_title=parser.document_title or doc,
                       structure_source=src)
        for c in res.chunks:
            if not (60 <= c.token_count <= 420):
                continue
            rec = {"doc_id": doc, "text": c.text, "block_ids": c.block_ids,
                   "section_kind": c.section_kind, "nct": c.normalized_content_type,
                   "token_count": c.token_count}
            if c.normalized_content_type == "body_paragraph":
                body.append(rec)
            elif c.normalized_content_type == "structured_table":
                tables.append(rec)
    return body, tables


def _spread(items, k):
    if k <= 0 or not items:
        return []
    if len(items) <= k:
        return items
    step = len(items) / k
    return [items[int(i * step)] for i in range(k)]


def gen_one(client, passage):
    try:
        resp = client.chat.completions.create(
            model=cfg.LLM_ENRICH_MODEL,
            messages=[{"role": "system", "content": SYS},
                      {"role": "user", "content": f"Passagem:\n\n{passage['text']}"}],
            response_format={"type": "json_object"}, temperature=0.0,
        )
        data = json.loads(resp.choices[0].message.content or "{}")
    except Exception as exc:  # noqa: BLE001
        return None
    if not data.get("answerable"):
        return None
    q = (data.get("question") or "").strip()
    ans = (data.get("answer_span") or "").strip()
    if len(q) < 12 or len(ans) < 6:
        return None
    if _norm(ans) not in _norm(passage["text"]):   # span deve ser substring exata
        return None
    return {"question": q, "answer_span": ans, "doc_id": passage["doc_id"],
            "gold_block_ids": passage["block_ids"], "section_kind": passage["section_kind"],
            "nct": passage["nct"], "passage_tokens": passage["token_count"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--table-frac", type=float, default=0.18)
    ap.add_argument("--force", action="store_true", help="regenera mesmo se já existir")
    args = ap.parse_args()

    if GOLD.exists() and not args.force:
        n = sum(1 for _ in GOLD.open())
        print(f"goldset já existe ({n} itens) em {GOLD}. Use --force para regenerar.")
        return
    if not cfg.LLM_ENRICH_API_KEY:
        raise SystemExit("Sem LLM_ENRICH_API_KEY/DEEPSEEK_API_KEY — não dá para gerar perguntas.")

    body, tables = eligible_passages()
    n_tab = int(args.n * args.table_frac)
    n_body = args.n - n_tab
    # amostra ~30% a mais para compensar passagens rejeitadas (answerable=false).
    cand = _spread(body, int(n_body * 1.3)) + _spread(tables, int(n_tab * 1.3))
    print(f"passagens elegíveis: body={len(body)} tables={len(tables)} | candidatas={len(cand)}")

    client = OpenAI(api_key=cfg.LLM_ENRICH_API_KEY, base_url=cfg.LLM_ENRICH_BASE_URL,
                    timeout=cfg.LLM_ENRICH_TIMEOUT_SECONDS or None)
    out = []
    with cf.ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(gen_one, client, p): p for p in cand}
        for i, fut in enumerate(cf.as_completed(futs), 1):
            r = fut.result()
            if r:
                out.append(r)
            if i % 10 == 0:
                print(f"  {i}/{len(cand)} processadas, {len(out)} aceitas")
            if len(out) >= args.n:
                break

    out = out[: args.n]
    for i, r in enumerate(out):
        r["qid"] = f"q{i:03d}"
    with GOLD.open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"\n✅ {len(out)} perguntas salvas em {GOLD}")
    print("  por doc:", dict(Counter(r["doc_id"] for r in out)))
    print("  por tipo:", dict(Counter(r["nct"] for r in out)))
    print("  por seção:", dict(Counter(r["section_kind"] for r in out)))


if __name__ == "__main__":
    main()
