#!/usr/bin/env python
"""Re-EXTRAI (MinerU) e reindexa os documentos já materializados no MinIO.

Diferença para `reindex_from_minio.py`: aquele só re-chunka/re-embeda a partir do
`content_list_v2.json` JÁ guardado — não roda o MinerU. Use ESTE quando a mudança
for na EXTRAÇÃO, não no chunking. Casos típicos:

  - troca de MINERU_BACKEND (ex.: pipeline → hybrid-engine), que muda a tipagem dos
    blocos (referências viram `list_type=reference_list` em vez de parágrafo solto);
  - troca de MINERU_METHOD/MINERU_LANG;
  - upgrade de versão do MinerU no contêiner.

Para cada manifesto: `stage_mineru(force=True)` → `stage_index(force=True)`. O
`content_list_v2.json`, o markdown e as imagens são regravados no MinIO (bucket
versionado: as versões anteriores continuam recuperáveis) e o manifesto ganha uma
revisão nova.

CUSTO: roda o MinerU no contêiner para CADA documento, com lock de GPU serializando.
É ordens de magnitude mais caro que o reindex. Com hybrid-engine, some ~100s na
primeira extração de cada ciclo do contêiner (carga do VLM sob demanda).

Uso (a partir de /app/evidencia_pipe, no venv para ler o .env):
    uv run python scripts/reextract_from_minio.py --dry-run     # só lista
    uv run python scripts/reextract_from_minio.py --limit 1     # valida em um doc
    uv run python scripts/reextract_from_minio.py --only <document_id>
    uv run python scripts/reextract_from_minio.py               # acervo inteiro

Processa em SÉRIE de propósito (evita contenção de VRAM), e continua no próximo
documento quando um falha — o resumo final lista os que falharam.
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
    ART_SOURCE_PDF,
    CTX_STAGE_DOWNLOADED,
    ArtifactManifest,
    PipelineContext,
)
from backend.services.artifact_store import get_artifact_store
from backend.services.manifest_repository import get_manifest_repository
from backend.services.pipeline_stages import stage_index, stage_mineru


def _iter_manifests(store):
    """Enumera todos os manifest.json sob o prefixo de artefatos no MinIO."""
    prefix = f"{settings.MINIO_ARTIFACT_PREFIX.strip('/')}/"
    for ref in store.list_prefix(prefix):
        if ref.object_key.endswith("/manifest.json"):
            yield ref.object_key


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Re-extrai (MinerU) e reindexa o corpus do MinIO com a config atual."
    )
    ap.add_argument("--dry-run", action="store_true", help="apenas lista os documentos elegíveis")
    ap.add_argument("--limit", type=int, default=0, help="processa no máximo N documentos (0 = todos)")
    ap.add_argument("--only", default="", help="processa apenas este document_id")
    args = ap.parse_args()

    store = get_artifact_store()
    repo = get_manifest_repository()

    print(
        f"Extração: backend={settings.MINERU_BACKEND} método={settings.MINERU_METHOD} "
        f"lang={settings.MINERU_LANG} api={settings.MINERU_API_URL}",
        flush=True,
    )
    print(
        f"Chunking: target={settings.CHUNK_TARGET_TOKENS} max={settings.CHUNK_MAX_TOKENS} "
        f"min={settings.CHUNK_MIN_TOKENS} | estratégia={settings.CHUNKING_STRATEGY}",
        flush=True,
    )

    # Elegível = tem o PDF de origem (stage_mineru baixa dele; sem PDF não há o que
    # re-extrair, mesmo que o content_list antigo exista).
    eligible: list[ArtifactManifest] = []
    total = 0
    for key in _iter_manifests(store):
        total += 1
        try:
            m = ArtifactManifest(**store.read_json(key))
        except Exception as exc:
            print(f"  ! manifesto ilegível {key}: {exc}", flush=True)
            continue
        if args.only and m.document_id != args.only:
            continue
        if ART_SOURCE_PDF not in m.artifacts:
            print(f"  - pulando {m.document_id} (sem PDF de origem)", flush=True)
            continue
        eligible.append(m)

    if args.limit and len(eligible) > args.limit:
        eligible = eligible[: args.limit]

    print(f"Manifestos: {total} | elegíveis p/ re-extração: {len(eligible)}", flush=True)
    if args.dry_run:
        for m in eligible:
            print(f"  · {m.document_id}  (pipeline={m.pipeline_id})", flush=True)
        return 0

    ok = failed = total_chunks = 0
    falhas: list[tuple[str, str]] = []
    t0 = time.perf_counter()
    for i, m in enumerate(eligible, 1):
        label = f"[{i}/{len(eligible)}] {m.document_id}"
        t_doc = time.perf_counter()
        ctx = PipelineContext(
            pipeline_id=UUID(m.pipeline_id),
            job_id=m.job_id,
            item_uuid=m.item_uuid or "",
            bitstream_uuid=m.bitstream_uuid,
            document_id=m.document_id,
            artifact_manifest_uri=repo.manifest_uri(m.pipeline_id, m.document_id),
            current_stage=CTX_STAGE_DOWNLOADED,
            force=True,  # ignora os estágios já concluídos → re-extrai e re-indexa
        )
        try:
            ctx = stage_mineru(ctx)
            t_extract = time.perf_counter() - t_doc
            summary = stage_index(ctx)
            n = int(summary.get("chunk_count", 0))
            total_chunks += n
            ok += 1
            print(
                f"  ✓ {label}: {n} chunk(s) "
                f"(extração {t_extract:.0f}s, total {time.perf_counter() - t_doc:.0f}s)",
                flush=True,
            )
        except Exception as exc:
            failed += 1
            falhas.append((m.document_id, f"{type(exc).__name__}: {exc}"))
            log.exception("re-extração falhou para %s", m.document_id)
            print(f"  ✗ {label}: ERRO {exc}", flush=True)

    dt = time.perf_counter() - t0
    print(
        f"\nConcluído em {dt / 60:.1f} min — {ok} ok, {failed} falha(s), "
        f"{total_chunks} chunk(s) no total"
        + (f" (média {total_chunks / ok:.0f}/doc)." if ok else "."),
        flush=True,
    )
    if falhas:
        print("\nFalhas:")
        for doc, err in falhas:
            print(f"  · {doc}: {err[:160]}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
