"""Migração OPCIONAL dos artefatos locais existentes (output/ e data/) para o MinIO.

NÃO move nem apaga nada localmente — apenas copia para o MinIO e cria os manifestos.
É idempotente (pula documentos já migrados com source/artefatos presentes) e suporta
--dry-run. NÃO é pré-requisito para o novo pipeline funcionar.

Uso:
    uv run python scripts/migrate_artifacts_to_minio.py --dry-run
    uv run python scripts/migrate_artifacts_to_minio.py
    uv run python scripts/migrate_artifacts_to_minio.py --report migracao.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Raiz do projeto no sys.path (o script roda fora do pacote).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.core.config import OUTPUT_DIR  # noqa: E402
from backend.core.logger import log  # noqa: E402
from backend.core.schemas import (  # noqa: E402
    ART_METADATA_CANDIDATES,
    ART_MINERU_CONTENT_LIST,
    ART_MINERU_MARKDOWN,
    ART_SOURCE_PDF,
    STAGE_DOWNLOAD,
    STAGE_ENRICHMENT,
    STAGE_MINERU,
)
from backend.services import pipeline_stages as stages  # noqa: E402
from backend.services.artifact_store import get_artifact_store  # noqa: E402
from backend.services.manifest_repository import get_manifest_repository  # noqa: E402

DATA_DIR = Path("data")
CONTENT_LIST_SUFFIX = "_content_list_v2.json"


def _find_docs() -> list[dict]:
    docs = []
    for json_path in sorted(OUTPUT_DIR.rglob(f"*{CONTENT_LIST_SUFFIX}")):
        doc_dir = json_path.parent.parent
        doc_id = doc_dir.name
        md = None
        for cand in (json_path.parent / f"{doc_id}.md", doc_dir / f"{doc_id}.md"):
            if cand.exists():
                md = cand
                break
        if md is None:
            mds = sorted(doc_dir.rglob("*.md"))
            md = mds[0] if mds else None
        meta = json_path.parent / f"{doc_id}_metadata_llm.json"
        pdf = DATA_DIR / f"{doc_id}.pdf"
        docs.append({
            "doc_id": doc_id, "content_list": json_path, "markdown": md,
            "metadata": meta if meta.exists() else None,
            "source_pdf": pdf if pdf.exists() else None,
        })
    return docs


def migrate(dry_run: bool) -> list[dict]:
    store = get_artifact_store()
    repo = get_manifest_repository()
    report = []

    for d in _find_docs():
        doc_id = d["doc_id"]
        pipeline_id = stages._derive_pipeline_id("", None, doc_id)
        entry = {"doc_id": doc_id, "pipeline_id": pipeline_id, "uploaded": [], "skipped": False}

        if dry_run:
            entry["would_upload"] = [k for k in ("source_pdf", "markdown", "content_list", "metadata") if d.get(k)]
            report.append(entry)
            log.info("[dry-run] %s → pipeline_id=%s artefatos=%s", doc_id, pipeline_id, entry["would_upload"])
            continue

        repo.create(pipeline_id=pipeline_id, job_id=doc_id, document_id=doc_id)
        manifest = repo.load(pipeline_id, doc_id)

        def _put(name, path, key_parts, ctype):
            key = store.artifact_key(pipeline_id, doc_id, *key_parts)
            if manifest.is_stage_completed(STAGE_MINERU) and name in manifest.artifacts and store.exists(key):
                return None
            ref = store.put_file(key, path, content_type=ctype, name=name)
            entry["uploaded"].append(name)
            return ref

        with repo.update(pipeline_id, doc_id) as m:
            if d["source_pdf"]:
                r = _put(ART_SOURCE_PDF, d["source_pdf"], ("source", "original.pdf"), "application/pdf")
                if r:
                    m.artifacts[ART_SOURCE_PDF] = r
                    m.stage(STAGE_DOWNLOAD).status = "COMPLETED"
            r = _put(ART_MINERU_CONTENT_LIST, d["content_list"], ("mineru", "content_list_v2.json"), "application/json")
            if r:
                m.artifacts[ART_MINERU_CONTENT_LIST] = r
            if d["markdown"]:
                r = _put(ART_MINERU_MARKDOWN, d["markdown"], ("mineru", "document.md"), "text/markdown; charset=utf-8")
                if r:
                    m.artifacts[ART_MINERU_MARKDOWN] = r
            m.stage(STAGE_MINERU).status = "COMPLETED"
            if d["metadata"]:
                r = _put(ART_METADATA_CANDIDATES, d["metadata"], ("enrichment", "metadata_candidates.json"), "application/json")
                if r:
                    m.artifacts[ART_METADATA_CANDIDATES] = r
                    m.stage(STAGE_ENRICHMENT).status = "COMPLETED"

        if not entry["uploaded"]:
            entry["skipped"] = True
        report.append(entry)
        log.info("%s → %s", doc_id, "pulado (já migrado)" if entry["skipped"] else f"enviado {entry['uploaded']}")

    return report


def main():
    ap = argparse.ArgumentParser(description="Migra artefatos locais existentes para o MinIO (opcional, idempotente).")
    ap.add_argument("--dry-run", action="store_true", help="Só lista o que seria migrado.")
    ap.add_argument("--report", default="", help="Grava o relatório JSON nesse caminho.")
    args = ap.parse_args()

    report = migrate(dry_run=args.dry_run)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": args.dry_run,
        "documents": len(report),
        "entries": report,
    }
    if args.report:
        Path(args.report).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Relatório salvo em {args.report}")
    print(json.dumps({"documents": len(report), "dry_run": args.dry_run}, ensure_ascii=False))


if __name__ == "__main__":
    main()
