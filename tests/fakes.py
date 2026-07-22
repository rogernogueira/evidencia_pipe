"""Fakes para os testes unitários — MinIO em memória, sem servidor real (§44)."""

from __future__ import annotations

import io
from datetime import datetime, timezone


class _FakeStat:
    def __init__(self, size, etag, version_id, content_type, metadata):
        self.size = size
        self.etag = etag
        self.version_id = version_id
        self.content_type = content_type
        # SDK expõe metadados de usuário com prefixo x-amz-meta-.
        self.metadata = {f"x-amz-meta-{k}": v for k, v in (metadata or {}).items()}
        self.last_modified = datetime.now(timezone.utc)


class _FakeListItem:
    def __init__(self, object_name, size, etag, version_id):
        self.object_name = object_name
        self.size = size
        self.etag = etag
        self.version_id = version_id


class _FakeResponse:
    def __init__(self, data: bytes):
        self._buf = io.BytesIO(data)

    def read(self, *a):
        return self._buf.read(*a)

    def close(self):
        pass

    def release_conn(self):
        pass


class FakeMinioClient:
    """Implementação em memória das operações do SDK Minio usadas pelo store."""

    def __init__(self):
        # object_key -> dict(data, content_type, metadata, etag, version)
        self._objects: dict[str, dict] = {}
        self._buckets: set[str] = set()
        self._version_seq = 0
        self.versioning = False

    # -- bucket --
    def bucket_exists(self, bucket):
        return bucket in self._buckets

    def make_bucket(self, bucket, location=None):
        self._buckets.add(bucket)

    def get_bucket_versioning(self, bucket):
        class _V:
            status = "Enabled" if self.versioning else None
        return _V()

    def set_bucket_versioning(self, bucket, config):
        self.versioning = True

    # -- put --
    def _store(self, key, data, content_type, metadata):
        self._version_seq += 1
        etag = f"etag-{len(data)}-{self._version_seq}"
        rec = {"data": bytes(data), "content_type": content_type,
               "metadata": dict(metadata or {}), "etag": etag,
               "version": f"v{self._version_seq}" if self.versioning else None}
        self._objects[key] = rec
        return rec

    def put_object(self, bucket, key, data, length, content_type="application/octet-stream",
                   metadata=None, sse=None, part_size=0, **kw):
        raw = data.read(length) if length >= 0 else data.read()
        self._store(key, raw, content_type, metadata)

    def fput_object(self, bucket, key, file_path, content_type="application/octet-stream",
                    metadata=None, sse=None, part_size=0, **kw):
        with open(file_path, "rb") as f:
            raw = f.read()
        self._store(key, raw, content_type, metadata)

    # -- get / stat --
    def _require(self, key):
        if key not in self._objects:
            from minio.error import S3Error

            # Assinatura: S3Error(response, code, message, resource, request_id, host_id)
            raise S3Error(None, "NoSuchKey", "not found", key, "req", "host")
        return self._objects[key]

    def get_object(self, bucket, key):
        return _FakeResponse(self._require(key)["data"])

    def fget_object(self, bucket, key, file_path):
        rec = self._require(key)
        with open(file_path, "wb") as f:
            f.write(rec["data"])

    def stat_object(self, bucket, key):
        rec = self._require(key)
        return _FakeStat(len(rec["data"]), rec["etag"], rec["version"], rec["content_type"], rec["metadata"])

    def list_objects(self, bucket, prefix=None, recursive=False, **kw):
        for k, rec in sorted(self._objects.items()):
            if prefix is None or k.startswith(prefix):
                yield _FakeListItem(k, len(rec["data"]), rec["etag"], rec["version"])

    def remove_object(self, bucket, key):
        self._objects.pop(key, None)

    def remove_objects(self, bucket, delete_objects):
        for d in delete_objects:
            self._objects.pop(getattr(d, "_name", getattr(d, "name", None)), None)
        return []

    def copy_object(self, bucket, dst, source):
        src_key = source.object_name if hasattr(source, "object_name") else source._object_name
        rec = self._require(src_key)
        self._store(dst, rec["data"], rec["content_type"], rec["metadata"])

    def presigned_get_object(self, bucket, key, expires=None):
        self._require(key)
        secs = int(expires.total_seconds()) if expires else 0
        return f"http://fake-minio/{bucket}/{key}?X-Amz-Expires={secs}&X-Amz-Signature=deadbeef"


def make_minio_store(bucket="evidencia-pipe"):
    """MinIOArtifactStore com cliente fake injetado (bucket já pronto)."""
    from backend.services.artifact_store import MinIOArtifactStore

    store = MinIOArtifactStore(
        endpoint="fake:9000", access_key="k", secret_key="s", bucket=bucket,
        artifact_prefix="artifacts", bucket_versioning=True,
    )
    store._client = FakeMinioClient()
    store._client._buckets.add(bucket)
    return store
