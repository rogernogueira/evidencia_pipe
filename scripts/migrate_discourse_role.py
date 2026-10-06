#!/usr/bin/env python3
"""Migração única: copia o rótulo discursivo EXPERIMENTAL para o campo OFICIAL.

O experimento (experiments/chunk_classes/09_push_labels_to_qdrant.py) gravou os
rótulos como `exp_discourse_role*`. O pipeline agora usa o campo oficial
`discourse_role*` (gravado por backend/services/discourse_classify_service via o
estágio de classificação). Este script copia o que já existe, SEM custo de LLM:

    exp_discourse_role        -> discourse_role
    exp_discourse_role_conf   -> discourse_role_conf
    exp_discourse_role_model  -> discourse_role_model
    exp_discourse_role_source -> discourse_role_source  (+ marca "migrated:exp_discourse_role")

Faz MERGE por ponto (set_payload via batch_update_points): não toca vetores, não
reclassifica, não apaga o campo experimental. Idempotente (reexecutar sobrescreve
com os mesmos valores). Reversível com --undo (remove só os campos discourse_role*).

    uv run python scripts/migrate_discourse_role.py --dry-run
    uv run python scripts/migrate_discourse_role.py
    uv run python scripts/migrate_discourse_role.py --undo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client import models as qm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.core import config as settings  # noqa: E402

SRC_PREFIX = "exp_discourse_role"
DST_PREFIX = "discourse_role"
DST_FIELDS = [DST_PREFIX, f"{DST_PREFIX}_conf", f"{DST_PREFIX}_model", f"{DST_PREFIX}_source"]
MIGRATION_TAG = f"migrated:{SRC_PREFIX}"


def _iter_points(client: QdrantClient, collection: str, with_payload):
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection, with_payload=with_payload, with_vectors=False,
            limit=512, offset=offset,
        )
        for p in points:
            yield p
        if offset is None:
            break


def migrate(client: QdrantClient, collection: str, batch_size: int, dry_run: bool) -> int:
    """Copia exp_discourse_role* -> discourse_role* nos pontos que têm o campo origem."""
    ops: list = []
    n = 0
    for p in _iter_points(client, collection, [SRC_PREFIX, f"{SRC_PREFIX}_conf", f"{SRC_PREFIX}_model"]):
        pl = p.payload or {}
        labels = pl.get(SRC_PREFIX)
        if not labels:
            continue
        payload = {
            DST_PREFIX: labels,
            f"{DST_PREFIX}_conf": pl.get(f"{SRC_PREFIX}_conf") or {},
            f"{DST_PREFIX}_model": pl.get(f"{SRC_PREFIX}_model"),
            f"{DST_PREFIX}_source": MIGRATION_TAG,
        }
        n += 1
        if dry_run:
            continue
        ops.append(qm.SetPayloadOperation(set_payload=qm.SetPayload(payload=payload, points=[p.id])))
        if len(ops) >= batch_size:
            client.batch_update_points(collection_name=collection, update_operations=ops, wait=True)
            ops = []
    if ops:
        client.batch_update_points(collection_name=collection, update_operations=ops, wait=True)
    return n


def undo(client: QdrantClient, collection: str, batch_size: int, dry_run: bool) -> int:
    """Remove os campos discourse_role* dos pontos que os têm (não toca em exp_*)."""
    ids: list = []
    n = 0
    for p in _iter_points(client, collection, [DST_PREFIX]):
        if not (p.payload or {}).get(DST_PREFIX):
            continue
        n += 1
        if dry_run:
            continue
        ids.append(p.id)
        if len(ids) >= batch_size:
            client.delete_payload(collection_name=collection, keys=DST_FIELDS, points=ids, wait=True)
            ids = []
    if ids:
        client.delete_payload(collection_name=collection, keys=DST_FIELDS, points=ids, wait=True)
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=settings.QDRANT_URL, help="URL do Qdrant")
    ap.add_argument("--collection", default=settings.QDRANT_COLLECTION, help="coleção")
    ap.add_argument("--batch-size", type=int, default=256, help="pontos por batch_update_points")
    ap.add_argument("--dry-run", action="store_true", help="só conta, não grava")
    ap.add_argument("--undo", action="store_true", help="remove os campos discourse_role* (reverte)")
    args = ap.parse_args()

    client = QdrantClient(url=args.url, timeout=settings.QDRANT_TIMEOUT_SECONDS or None)
    action = "UNDO" if args.undo else "MIGRAÇÃO"
    prefixo = "[dry-run] " if args.dry_run else ""
    print(f"{prefixo}{action} em {args.collection} @ {args.url}")

    fn = undo if args.undo else migrate
    n = fn(client, args.collection, args.batch_size, args.dry_run)

    verbo = "seriam afetados" if args.dry_run else ("removidos" if args.undo else "migrados")
    print(f"{prefixo}OK — {n} ponto(s) {verbo}"
          + ("" if args.undo else f" ({SRC_PREFIX}* -> {DST_PREFIX}*)."))
    if args.dry_run:
        print("Nada foi gravado. Rode sem --dry-run para aplicar.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
