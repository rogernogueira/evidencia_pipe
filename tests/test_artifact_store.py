"""Testes unitários do MinIOArtifactStore (fake client, sem MinIO real) — §44."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.services.artifact_store import (  # noqa: E402
    ArtifactIntegrityError,
    ArtifactInvalidURIError,
    ArtifactNotFoundError,
    build_uri,
    parse_uri,
    sanitize_component,
    sha256_of_bytes,
)
from tests.fakes import make_minio_store  # noqa: E402


# --- URI (§44.1-3) ---

def test_build_uri():
    assert build_uri("b", "artifacts/x/y.json") == "minio://b/artifacts/x/y.json"


def test_parse_uri_roundtrip():
    assert parse_uri("minio://b/artifacts/x/y.json") == ("b", "artifacts/x/y.json")


@pytest.mark.parametrize("uri", [
    "s3://b/k", "http://b/k", "minio://b", "minio:///k", "minio://b/",
    "minio://b/a/../../etc/passwd", "minio://b//double", "minio://user:pw@b/k",
])
def test_parse_uri_invalid(uri):
    with pytest.raises(ArtifactInvalidURIError):
        parse_uri(uri)


# --- sanitização / path traversal (§44.4-5) ---

@pytest.mark.parametrize("bad", ["", "..", ".", "../etc", "a/b", "a\\b", "x\x00y", "  ../  "])
def test_sanitize_rejects_traversal(bad):
    with pytest.raises(ArtifactInvalidURIError):
        sanitize_component(bad)


def test_sanitize_ok():
    assert sanitize_component("Relatório 2021.pdf") == "Relatório 2021.pdf"


def test_artifact_key_stays_under_prefix():
    store = make_minio_store()
    key = store.artifact_key("pid", "doc", "mineru", "content_list_v2.json")
    assert key == "artifacts/pid/doc/mineru/content_list_v2.json"
    with pytest.raises(ArtifactInvalidURIError):
        store.artifact_key("pid", "../escape", "x")


# --- sha256 (§44.14) ---

def test_sha256_bytes():
    assert sha256_of_bytes(b"abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


# --- uploads / leitura / stat (§44.6-13) ---

def test_put_bytes_and_read():
    store = make_minio_store()
    ref = store.put_bytes("artifacts/p/d/x.bin", b"hello", content_type="application/octet-stream", name="x")
    assert ref.size_bytes == 5
    assert ref.sha256 == sha256_of_bytes(b"hello")
    assert ref.etag  # ETag presente
    assert ref.version_id  # versionamento ligado no fake
    assert ref.sha256 != ref.etag  # ETag NÃO é o sha256
    assert store.read_bytes("artifacts/p/d/x.bin") == b"hello"


def test_put_text_json_roundtrip():
    store = make_minio_store()
    store.put_text("artifacts/p/d/a.txt", "olá")
    assert store.read_text("artifacts/p/d/a.txt") == "olá"
    store.put_json("artifacts/p/d/a.json", {"k": 1, "v": [1, 2]})
    assert store.read_json("artifacts/p/d/a.json") == {"k": 1, "v": [1, 2]}


def test_put_file_and_download(tmp_path):
    store = make_minio_store()
    src = tmp_path / "in.pdf"
    src.write_bytes(b"%PDF-1.4 fake")
    ref = store.put_file("artifacts/p/d/source/original.pdf", src, content_type="application/pdf")
    dest = tmp_path / "out.pdf"
    store.download_to_file("artifacts/p/d/source/original.pdf", dest, expected_sha256=ref.sha256)
    assert dest.read_bytes() == b"%PDF-1.4 fake"


def test_download_integrity_failure(tmp_path):
    store = make_minio_store()
    store.put_bytes("artifacts/p/d/x", b"data")
    with pytest.raises(ArtifactIntegrityError):
        store.download_to_file("artifacts/p/d/x", tmp_path / "o", expected_sha256="0" * 64)


def test_stat_metadata_and_read_text_max_bytes():
    store = make_minio_store()
    store.put_bytes("artifacts/p/d/m", b"1234567890", metadata={"stage": "mineru"})
    ref = store.stat("artifacts/p/d/m")
    assert ref.size_bytes == 10
    assert ref.metadata.get("stage") == "mineru"
    assert store.read_text("artifacts/p/d/m", max_bytes=4) == "1234"


def test_missing_object():
    store = make_minio_store()
    assert store.exists("artifacts/p/d/nope") is False
    with pytest.raises(ArtifactNotFoundError):
        store.read_bytes("artifacts/p/d/nope")


# --- list / delete / copy (§44.17-20) ---

def test_list_delete_prefix_and_copy():
    store = make_minio_store()
    for i in range(3):
        store.put_bytes(f"artifacts/p/d/mineru/images/img_{i}.png", b"x")
    refs = store.list_prefix("artifacts/p/d/mineru/images/")
    assert len(refs) == 3
    store.copy("artifacts/p/d/mineru/images/img_0.png", "artifacts/p/d/copy.png")
    assert store.exists("artifacts/p/d/copy.png")
    assert store.delete("artifacts/p/d/copy.png") is True
    removed = store.delete_prefix("artifacts/p/d/mineru/images/")
    assert removed == 3


def test_presigned_url_has_expiry():
    store = make_minio_store()
    store.put_bytes("artifacts/p/d/x", b"x")
    url = store.presigned_get_url("artifacts/p/d/x", ttl_seconds=900)
    assert "X-Amz-Expires=900" in url
