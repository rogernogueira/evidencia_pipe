#!/usr/bin/env python
"""Reindexa os documentos já materializados no MinIO com a config de chunking ATUAL.

Motivação: mudar os parâmetros de chunking (CHUNK_*_TOKENS no .env) altera o
`chunking_config_hash` → novos `chunk_id`. Este script aplica a nova config ao corpus
JÁ indexado SEM re-extrair o MinerU: para cada manifesto no MinIO, chama
`stage_index(force=True)`, que materializa o `content_list_v2.json` do manifesto,
re-chunka + re-embeda com a config atual, faz upsert dos NOVOS pontos e só então
remove as versões antigas (document_version diferente) do mesmo documento (§27).

Uso (a partir de /app/evidencia_pipe, no venv para ler o .env):
    uv run python scripts/reindex_from_minio.py            # reindexa tudo
    uv run python scripts/reindex_from_minio.py --dry-run   # só lista o que faria
    uv run python scripts/reindex_from_minio.py --limit 5   # primeiros 5 (teste)
    uv run python scripts/reindex_from_minio.py --only <document_id>

Observações:
  - Reusa o gpu_resource_manager (lock no Redis DB 2): é seguro rodar com o worker de
    GPU ativo — o acesso à GPU é serializado. Ainda assim, para o teste inicial
    recomenda-se --limit pequeno.
  - Processa em SÉRIE (sem paralelismo) de propósito: evita contenção de VRAM.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from uuid import UUID

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.core import config as settings
from backend.core.logger import log
from backend.core.schemas import (
    ART_MINERU_CONTENT_LIST,
    CTX_STAGE_EXTRACTED,
    STAGE_INDEXING,
    ArtifactManifest,
    PipelineContext,
)
from backend.services.artifact_store import get_artifact_store
from backend.services.manifest_repository import get_manifest_repository
from backend.services.pipeline_stages import stage_index


def _iter_manifests(store):
    """Enumera todos os manifest.json sob o prefixo de artefatos no MinIO."""
    prefix = f"{settings.MINIO_ARTIFACT_PREFIX.strip('/')}/"
    for ref in store.list_prefix(prefix):
        if ref.object_key.endswith("/manifest.json"):
            yield ref.object_key


def _load_manifest(store, key: str) -> ArtifactManifest:
    return ArtifactManifest(**store.read_json(key))


def main() -> int:
    ap = argparse.ArgumentParser(description="Reindexa o corpus do MinIO com a config de chunking atual.")
    ap.add_argument("--dry-run", action="store_true", help="apenas lista os documentos elegíveis")
    ap.add_argument("--limit", type=int, default=0, help="processa no máximo N documentos (0 = todos)")
    ap.add_argument("--only", default="", help="reindexa apenas este document_id")
    args = ap.parse_args()

    store = get_artifact_store()
    repo = get_manifest_repository()

    print(
        f"Config de chunking atual: target={settings.CHUNK_TARGET_TOKENS} "
        f"max={settings.CHUNK_MAX_TOKENS} min={settings.CHUNK_MIN_TOKENS} "
        f"overlap={settings.CHUNK_OVERLAP_TOKENS} | estratégia={settings.CHUNKING_STRATEGY}",
        flush=True,
    )

    # Coleta os manifestos elegíveis (com content_list do MinerU presente).
    eligible: list[ArtifactManifest] = []
    total_manifests = 0
    for key in _iter_manifests(store):
        total_manifests += 1
        try:
            m = _load_manifest(store, key)
        except Exception as exc:  # manifesto corrompido — pula
            print(f"  ! manifesto ilegível {key}: {exc}", flush=True)
            continue
        if args.only and m.document_id != args.only:
            continue
        if ART_MINERU_CONTENT_LIST not in m.artifacts:
            print(f"  - pulando {m.document_id} (sem content_list do MinerU)", flush=True)
            continue
        eligible.append(m)

    if args.limit and len(eligible) > args.limit:
        eligible = eligible[: args.limit]

    print(f"Manifestos: {total_manifests} | elegíveis p/ reindex: {len(eligible)}", flush=True)
    if args.dry_run:
        for m in eligible:
            print(f"  · {m.document_id}  (pipeline={m.pipeline_id}, item={m.item_uuid or '-'})", flush=True)
        return 0

    ok = 0
    failed = 0
    total_chunks = 0
    t0 = time.perf_counter()
    for i, m in enumerate(eligible, 1):
        ctx = PipelineContext(
            pipeline_id=UUID(m.pipeline_id),
            job_id=m.job_id,
            item_uuid=m.item_uuid or "",
            bitstream_uuid=m.bitstream_uuid,
            document_id=m.document_id,
            artifact_manifest_uri=repo.manifest_uri(m.pipeline_id, m.document_id),
            current_stage=CTX_STAGE_EXTRACTED,
            force=True,  # ignora o STAGE_INDEXING já concluído → re-chunka de fato
        )
        label = f"[{i}/{len(eligible)}] {m.document_id}"
        try:
            summary = stage_index(ctx)
            n = int(summary.get("chunk_count", 0))
            total_chunks += n
            ok += 1
            print(f"  ✓ {label}: {n} chunk(s)", flush=True)
        except Exception as exc:
            failed += 1
            log.exception("reindex falhou para %s", m.document_id)
            print(f"  ✗ {label}: ERRO {exc}", flush=True)

    dt = time.perf_counter() - t0
    print(
        f"\nConcluído em {dt:.1f}s — {ok} ok, {failed} falha(s), "
        f"{total_chunks} chunk(s) no total (média {total_chunks / ok:.0f}/doc)."
        if ok else f"\nConcluído em {dt:.1f}s — 0 documentos reindexados ({failed} falha(s)).",
        flush=True,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
