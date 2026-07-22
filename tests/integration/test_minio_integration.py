"""Testes de integração com MinIO REAL (§45). Pulados se o MinIO não estiver acessível.

Suba a infra antes: `docker compose up -d minio minio-init`, e configure MINIO_*
no .env (endpoint host:port). Rodar: `uv run --group test pytest tests/integration`.
"""

import os
import sys
import uuid

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.services.artifact_store import build_artifact_store  # noqa: E402
from backend.services.manifest_repository import ManifestRepository  # noqa: E402


@pytest.fixture(scope="module")
def store():
    try:
        s = build_artifact_store()
        health = s.healthcheck()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"MinIO indisponível: {exc}")
    if not health.get("reachable") or not health.get("bucket_exists"):
        pytest.skip(f"MinIO não pronto: {health}")
    return s


@pytest.fixture
def prefix():
    return f"artifacts/_it/{uuid.uuid4().hex}"


def test_upload_download_sha256(store, prefix, tmp_path):
    src = tmp_path / "f.pdf"
    src.write_bytes(b"%PDF-1.4 " + os.urandom(2048))
    ref = store.put_file(f"{prefix}/source/original.pdf", src, content_type="application/pdf")
    assert ref.size_bytes == src.stat().st_size
    dest = tmp_path / "out.pdf"
    store.download_to_file(ref.object_key, dest, expected_sha256=ref.sha256)
    assert dest.read_bytes() == src.read_bytes()
    store.delete(ref.object_key)


def test_json_and_list_prefix(store, prefix):
    for i in range(3):
        store.put_bytes(f"{prefix}/mineru/images/img_{i}.png", os.urandom(64), content_type="image/png")
    refs = store.list_prefix(f"{prefix}/mineru/images/")
    assert len(refs) == 3
    assert store.delete_prefix(f"{prefix}/mineru/images/") == 3


def test_manifest_create_update_versioning(store):
    repo = ManifestRepository(store=store)
    pid, doc = uuid.uuid4().hex, "doc-it"
    repo.create(pipeline_id=pid, job_id=doc, document_id=doc)
    with repo.update(pid, doc) as m:
        m.metrics["page_count"] = 7
        m.stage("download").status = "COMPLETED"
    reloaded = repo.load(pid, doc)
    assert reloaded.revision == 1
    assert reloaded.metrics["page_count"] == 7
    store.delete_prefix(f"{store.artifact_prefix}/{pid}/")


def test_presigned_url_expires(store, prefix):
    store.put_bytes(f"{prefix}/x.txt", b"data", content_type="text/plain")
    url = store.presigned_get_url(f"{prefix}/x.txt", ttl_seconds=60)
    assert "X-Amz-" in url and "Signature" in url
    store.delete(f"{prefix}/x.txt")
