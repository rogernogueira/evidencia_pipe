"""Testes do ManifestRepository (store MinIO fake em memória) — §44.23-25."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.core.schemas import ART_SOURCE_PDF, STAGE_DOWNLOAD, ArtifactReference  # noqa: E402
from backend.services.manifest_repository import (  # noqa: E402
    ArtifactManifestValidationError,
    ManifestRepository,
)
from tests.fakes import make_minio_store  # noqa: E402


@pytest.fixture
def repo():
    return ManifestRepository(store=make_minio_store())


def test_create_is_idempotent(repo):
    m1 = repo.create(pipeline_id="p", job_id="doc", document_id="doc")
    assert m1.revision == 0
    # criar de novo retorna o existente (não sobrescreve)
    m2 = repo.create(pipeline_id="p", job_id="doc", document_id="doc")
    assert m2.revision == m1.revision


def test_update_increments_revision_and_preserves_stages(repo):
    repo.create(pipeline_id="p", job_id="doc", document_id="doc")
    ref = ArtifactReference(name=ART_SOURCE_PDF, uri="minio://evidencia-pipe/artifacts/p/doc/source/original.pdf",
                            bucket="evidencia-pipe", object_key="artifacts/p/doc/source/original.pdf",
                            content_type="application/pdf", size_bytes=10, sha256="a" * 64)
    with repo.update("p", "doc") as m:
        m.artifacts[ART_SOURCE_PDF] = ref
        m.stage(STAGE_DOWNLOAD).status = "COMPLETED"

    reloaded = repo.load("p", "doc")
    assert reloaded.revision == 1
    assert reloaded.is_stage_completed(STAGE_DOWNLOAD)
    assert reloaded.artifacts[ART_SOURCE_PDF].sha256 == "a" * 64

    # segundo update preserva o estágio anterior e incrementa de novo
    with repo.update("p", "doc") as m:
        m.metrics["page_count"] = 42
    reloaded2 = repo.load("p", "doc")
    assert reloaded2.revision == 2
    assert reloaded2.is_stage_completed(STAGE_DOWNLOAD)  # preservado
    assert reloaded2.metrics["page_count"] == 42


def test_update_missing_manifest_raises(repo):
    with pytest.raises(ArtifactManifestValidationError):
        with repo.update("nope", "nope"):
            pass


def test_stage_helpers(repo):
    repo.create(pipeline_id="p", job_id="doc", document_id="doc")
    repo.start_stage("p", "doc", STAGE_DOWNLOAD)
    assert repo.load("p", "doc").stage(STAGE_DOWNLOAD).status == "RUNNING"
    repo.complete_stage("p", "doc", STAGE_DOWNLOAD, final=False)
    assert repo.load("p", "doc").is_stage_completed(STAGE_DOWNLOAD)
    repo.fail_stage("p", "doc", "mineru", error_type="RuntimeError", message="boom")
    m = repo.load("p", "doc")
    assert m.status == "FAILED"
    assert m.errors[0].error_type == "RuntimeError"
