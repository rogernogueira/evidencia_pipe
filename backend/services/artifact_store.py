"""ArtifactStore — armazenamento OFICIAL dos artefatos do pipeline (MinIO / S3).

A Celery chain transporta somente referências (URIs `minio://...`); todo conteúdo
grande (PDF, markdown, JSON MinerU, chunks, imagens, relatórios, manifesto) vive
aqui. Este módulo é o único ponto de acoplamento entre o pipeline e o SDK do MinIO:
as exceções do SDK são convertidas em exceções de domínio (ArtifactStoreError e
subclasses) e nunca vazam para os endpoints públicos.

Layout das chaves (determinístico, sob MINIO_ARTIFACT_PREFIX):

    artifacts/<pipeline_id>/<document_id>/
        manifest.json
        source/original.pdf
        mineru/{document.md, content_list_v2.json, processing_metrics.json, images/, images_manifest.json}
        enrichment/{metadata_candidates.json, raw_response.json}
        indexing/{chunks.jsonl, embedding_report.json}
        logs/processing_summary.json

Escritas seguras: staging em _tmp/<pipeline_id>/<document_id>/<uuid>/ + promoção
para a chave definitiva (ver put_file(..., promote=True) e ManifestRepository).
"""

from __future__ import annotations

import hashlib
import io
import re
import tempfile
import time
import unicodedata
from datetime import timedelta
from pathlib import Path
from typing import Any, BinaryIO, Optional, Protocol, runtime_checkable

from backend.core import config as settings
from backend.core.logger import log
from backend.core.schemas import ArtifactReference

# ---------------------------------------------------------------------------
# Exceções de domínio (§39). NUNCA propague S3Error direto para a API.
# ---------------------------------------------------------------------------


class ArtifactStoreError(Exception):
    """Erro genérico do armazenamento de artefatos."""


class ArtifactStoreUnavailable(ArtifactStoreError):
    """Backend indisponível (conectividade, autenticação, bucket ausente)."""


class ArtifactNotFoundError(ArtifactStoreError):
    """Objeto inexistente."""


class ArtifactIntegrityError(ArtifactStoreError):
    """Checksum/tamanho divergente do esperado."""


class ArtifactUploadError(ArtifactStoreError):
    """Falha ao gravar um objeto."""


class ArtifactDownloadError(ArtifactStoreError):
    """Falha ao ler/baixar um objeto."""


class ArtifactInvalidURIError(ArtifactStoreError):
    """URI `minio://` malformada, insegura ou fora do prefixo configurado."""


class ArtifactPermissionError(ArtifactStoreError):
    """Acesso negado pelo backend."""


# ---------------------------------------------------------------------------
# Constantes / helpers puros (testáveis sem MinIO).
# ---------------------------------------------------------------------------

URI_SCHEME = "minio"
_MB = 1024 * 1024
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")


def sha256_of_file(path: Path, chunk_size: int = 8 * _MB) -> tuple[str, int]:
    """Calcula SHA-256 e tamanho de um arquivo em streaming (sem carregá-lo todo)."""
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


def sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sanitize_component(name: str, *, field: str = "componente") -> str:
    """Sanitiza um único componente de object key (document_id, nome de arquivo).

    Protege contra path traversal, barras inesperadas, sequências '..', caracteres
    de controle e nomes vazios. NÃO permite separadores de caminho.
    """
    if name is None:
        raise ArtifactInvalidURIError(f"{field} vazio.")
    # Rejeita (não "limpa" silenciosamente) qualquer padrão inseguro.
    if _CONTROL_CHARS.search(str(name)):
        raise ArtifactInvalidURIError(f"{field} contém caracteres de controle: {name!r}")
    value = unicodedata.normalize("NFKC", str(name)).strip()
    if value in ("", ".", ".."):
        raise ArtifactInvalidURIError(f"{field} inválido: {name!r}")
    if "/" in value or "\\" in value:
        raise ArtifactInvalidURIError(f"{field} não pode conter separadores: {name!r}")
    return value


def build_uri(bucket: str, object_key: str) -> str:
    """minio://<bucket>/<object_key> — sem credenciais, sem host."""
    return f"{URI_SCHEME}://{bucket}/{object_key.lstrip('/')}"


def parse_uri(uri: str) -> tuple[str, str]:
    """Valida e decompõe uma URI `minio://<bucket>/<key>` em (bucket, object_key).

    Rejeita esquema diferente, ausência de bucket/key, sequências '..' e '//'.
    """
    if not isinstance(uri, str) or not uri.startswith(f"{URI_SCHEME}://"):
        raise ArtifactInvalidURIError(f"URI inválida (esperado {URI_SCHEME}://): {uri!r}")
    rest = uri[len(URI_SCHEME) + 3:]
    if "@" in rest.split("/", 1)[0]:
        raise ArtifactInvalidURIError("URI não pode conter credenciais.")
    bucket, _, object_key = rest.partition("/")
    if not bucket or not object_key:
        raise ArtifactInvalidURIError(f"URI sem bucket ou object key: {uri!r}")
    if ".." in object_key.split("/") or "//" in object_key or object_key.startswith("/"):
        raise ArtifactInvalidURIError(f"object key insegura na URI: {uri!r}")
    return bucket, object_key


# ---------------------------------------------------------------------------
# Interface (Protocol) — permite outros backends no futuro (§10).
# ---------------------------------------------------------------------------


@runtime_checkable
class ArtifactStore(Protocol):
    bucket: str

    def put_bytes(self, object_key: str, data: bytes, *, content_type: str = ..., name: str = ..., metadata: dict | None = ...) -> ArtifactReference: ...
    def put_text(self, object_key: str, text: str, *, content_type: str = ..., name: str = ..., metadata: dict | None = ...) -> ArtifactReference: ...
    def put_json(self, object_key: str, obj: Any, *, name: str = ..., metadata: dict | None = ...) -> ArtifactReference: ...
    def put_file(self, object_key: str, file_path: Path, *, content_type: str = ..., name: str = ..., metadata: dict | None = ...) -> ArtifactReference: ...
    def put_stream(self, object_key: str, stream: BinaryIO, *, content_type: str = ..., name: str = ..., metadata: dict | None = ...) -> ArtifactReference: ...

    def read_bytes(self, object_key: str) -> bytes: ...
    def read_text(self, object_key: str, *, max_bytes: int | None = ...) -> str: ...
    def read_json(self, object_key: str) -> Any: ...

    def download_to_file(self, object_key: str, dest: Path, *, expected_sha256: str | None = ...) -> Path: ...
    def open_stream(self, object_key: str) -> BinaryIO: ...

    def stat(self, object_key: str, *, name: str = ...) -> ArtifactReference: ...
    def exists(self, object_key: str) -> bool: ...
    def delete(self, object_key: str) -> bool: ...
    def delete_prefix(self, prefix: str) -> int: ...
    def list_prefix(self, prefix: str) -> list[ArtifactReference]: ...
    def copy(self, src_key: str, dst_key: str, *, name: str = ...) -> ArtifactReference: ...

    def build_uri(self, object_key: str) -> str: ...
    def parse_uri(self, uri: str) -> tuple[str, str]: ...
    def healthcheck(self) -> dict: ...


# ---------------------------------------------------------------------------
# Implementação MinIO.
# ---------------------------------------------------------------------------

# Erros S3 tipicamente transitórios (dignos de retry).
_TRANSIENT_S3_CODES = {
    "InternalError", "SlowDown", "ServiceUnavailable", "RequestTimeout",
    "RequestTimeTooSkewed", "503 SlowDown",
}


class MinIOArtifactStore:
    """Backend MinIO/S3 do ArtifactStore. Cria/reutiliza um cliente Minio, valida
    bucket/versionamento e implementa upload/download/leitura/inspeção com SHA-256
    explícito, metadados mínimos e conversão de exceções."""

    def __init__(
        self,
        *,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        secure: bool = False,
        region: str | None = None,
        artifact_prefix: str = "artifacts",
        auto_create_bucket: bool = True,
        bucket_versioning: bool = True,
        max_retries: int = 3,
        part_size_mb: int = 16,
        download_chunk_mb: int = 8,
        connect_timeout: int = 5,
        read_timeout: int = 60,
        sse_enabled: bool = False,
        sse_type: str = "SSE-S3",
    ) -> None:
        self.endpoint = endpoint
        self.bucket = bucket
        self.region = region
        self.artifact_prefix = artifact_prefix.strip("/")
        self.auto_create_bucket = auto_create_bucket
        self.bucket_versioning = bucket_versioning
        self.max_retries = max(1, max_retries)
        self.part_size = max(5 * _MB, part_size_mb * _MB)  # S3 exige >= 5 MiB
        self.download_chunk = max(_MB, download_chunk_mb * _MB)
        self._access_key = access_key
        self._secret_key = secret_key
        self._secure = secure
        self._connect_timeout = connect_timeout
        self._read_timeout = read_timeout
        self._sse = None
        self._client = None
        self._bucket_ready = False

        if sse_enabled:
            from minio.sse import SseS3, SseKMS  # type: ignore

            self._sse = SseKMS("aws/s3", {}) if str(sse_type).upper() == "SSE-KMS" else SseS3()

    # -- cliente / bucket ---------------------------------------------------

    @property
    def client(self):
        """Cliente Minio lazy (reutilizado). Fail-closed em erro de config."""
        if self._client is None:
            try:
                import urllib3
                from minio import Minio

                http = urllib3.PoolManager(
                    timeout=urllib3.Timeout(connect=self._connect_timeout, read=self._read_timeout),
                    retries=urllib3.Retry(total=0),  # retries controlados por nós (_retry)
                    maxsize=16,
                )
                self._client = Minio(
                    self.endpoint,
                    access_key=self._access_key,
                    secret_key=self._secret_key,
                    secure=self._secure,
                    region=self.region,
                    http_client=http,
                )
                # Log SEM credenciais.
                log.info("[artifact-store] MinIO endpoint=%s bucket=%s secure=%s",
                         self.endpoint, self.bucket, self._secure)
            except Exception as exc:  # pragma: no cover - erro de import/config
                raise ArtifactStoreUnavailable(f"não foi possível iniciar o cliente MinIO: {exc}") from exc
        return self._client

    def _retry(self, op: str, fn):
        """Executa `fn` com retries limitados só para falhas transitórias."""
        from minio.error import S3Error

        attempt = 0
        while True:
            attempt += 1
            try:
                return fn()
            except S3Error as exc:
                code = getattr(exc, "code", "") or ""
                if code in ("NoSuchKey", "NoSuchObject"):
                    raise ArtifactNotFoundError(f"{op}: objeto inexistente ({code}).") from exc
                if code in ("AccessDenied", "SignatureDoesNotMatch", "InvalidAccessKeyId"):
                    raise ArtifactPermissionError(f"{op}: acesso negado ({code}).") from exc
                if code in ("NoSuchBucket",):
                    raise ArtifactStoreUnavailable(f"{op}: bucket inexistente.") from exc
                transient = code in _TRANSIENT_S3_CODES
                if transient and attempt < self.max_retries:
                    self._backoff(op, attempt, exc)
                    continue
                raise ArtifactStoreError(f"{op}: erro S3 ({code}): {exc}") from exc
            except (ConnectionError, OSError, TimeoutError) as exc:
                if attempt < self.max_retries:
                    self._backoff(op, attempt, exc)
                    continue
                raise ArtifactStoreUnavailable(f"{op}: falha de rede após {attempt} tentativas: {exc}") from exc

    @staticmethod
    def _backoff(op: str, attempt: int, exc: Exception) -> None:
        delay = min(2.0, 0.2 * (2 ** (attempt - 1)))
        log.warning("[artifact-store] %s falhou (tentativa %d): %s — retry em %.2fs", op, attempt, exc, delay)
        time.sleep(delay)

    def ensure_bucket(self) -> None:
        """Confirma (e opcionalmente cria) o bucket + versionamento. Idempotente."""
        if self._bucket_ready:
            return
        try:
            exists = self._retry("bucket_exists", lambda: self.client.bucket_exists(self.bucket))
        except ArtifactStoreError:
            raise
        if not exists:
            if not self.auto_create_bucket:
                raise ArtifactStoreUnavailable(
                    f"bucket '{self.bucket}' não existe e MINIO_AUTO_CREATE_BUCKET=false."
                )
            log.info("[artifact-store] criando bucket '%s'...", self.bucket)
            self._retry("make_bucket", lambda: self.client.make_bucket(self.bucket, location=self.region))
        if self.bucket_versioning:
            self._enable_versioning()
        self._bucket_ready = True

    def _enable_versioning(self) -> None:
        try:
            from minio.commonconfig import ENABLED
            from minio.versioningconfig import VersioningConfig

            current = self.client.get_bucket_versioning(self.bucket)
            if getattr(current, "status", None) != ENABLED:
                self.client.set_bucket_versioning(self.bucket, VersioningConfig(ENABLED))
                log.info("[artifact-store] versionamento habilitado no bucket '%s'.", self.bucket)
        except Exception as exc:  # versionamento pode não ser suportado — não é fatal
            log.warning("[artifact-store] não foi possível habilitar versionamento: %s", exc)

    # -- key builders (layout determinístico) -------------------------------

    def artifact_key(self, pipeline_id: str, document_id: str, *parts: str) -> str:
        pid = sanitize_component(pipeline_id, field="pipeline_id")
        doc = sanitize_component(document_id, field="document_id")
        safe_parts = [sanitize_component(p, field="segmento") for p in parts if p != ""]
        return "/".join([self.artifact_prefix, pid, doc, *safe_parts])

    def manifest_key(self, pipeline_id: str, document_id: str) -> str:
        return self.artifact_key(pipeline_id, document_id, "manifest.json")

    def tmp_key(self, pipeline_id: str, document_id: str, token: str, *parts: str) -> str:
        pid = sanitize_component(pipeline_id, field="pipeline_id")
        doc = sanitize_component(document_id, field="document_id")
        tok = sanitize_component(token, field="token")
        safe_parts = [sanitize_component(p, field="segmento") for p in parts if p != ""]
        return "/".join(["_tmp", pid, doc, tok, *safe_parts])

    def _validate_key(self, object_key: str) -> str:
        """Garante que a key não escapa do bucket (defesa em profundidade)."""
        if not object_key or object_key.startswith("/"):
            raise ArtifactInvalidURIError(f"object key inválida: {object_key!r}")
        parts = object_key.split("/")
        if ".." in parts or "" in parts:
            raise ArtifactInvalidURIError(f"object key insegura: {object_key!r}")
        if _CONTROL_CHARS.search(object_key):
            raise ArtifactInvalidURIError("object key contém caracteres de controle.")
        return object_key

    # -- metadados / referência ---------------------------------------------

    @staticmethod
    def _s3_metadata(metadata: dict | None) -> dict:
        """Filtra metadados p/ nomes compatíveis com S3 (ASCII, sem conteúdo)."""
        out: dict[str, str] = {}
        for k, v in (metadata or {}).items():
            if v is None:
                continue
            key = re.sub(r"[^a-z0-9\-]", "-", str(k).lower())
            val = _CONTROL_CHARS.sub("", str(v))
            # ASCII-only (headers S3 não aceitam bytes não-ASCII com segurança).
            out[key] = val.encode("ascii", "ignore").decode("ascii")[:256]
        return out

    def _reference_from_stat(self, object_key: str, name: str, sha256: str = "") -> ArtifactReference:
        obj = self._retry("stat_object", lambda: self.client.stat_object(self.bucket, object_key))
        meta = {k[len("x-amz-meta-"):]: v for k, v in (obj.metadata or {}).items()
                if k.lower().startswith("x-amz-meta-")}
        return ArtifactReference(
            name=name,
            uri=self.build_uri(object_key),
            bucket=self.bucket,
            object_key=object_key,
            content_type=getattr(obj, "content_type", None) or "application/octet-stream",
            size_bytes=int(getattr(obj, "size", 0) or 0),
            sha256=sha256 or meta.get("sha256", ""),
            etag=(getattr(obj, "etag", None) or "").strip('"') or None,
            version_id=getattr(obj, "version_id", None),
            metadata=meta,
        )

    # -- uploads ------------------------------------------------------------

    def put_bytes(self, object_key, data, *, content_type="application/octet-stream", name="", metadata=None):
        self.ensure_bucket()
        object_key = self._validate_key(object_key)
        sha = sha256_of_bytes(data)
        meta = self._s3_metadata({**(metadata or {}), "sha256": sha})
        t0 = time.perf_counter()

        def _do():
            return self.client.put_object(
                self.bucket, object_key, io.BytesIO(data), length=len(data),
                content_type=content_type, metadata=meta, sse=self._sse,
            )

        try:
            self._retry("put_object", _do)
        except ArtifactStoreError as exc:
            raise ArtifactUploadError(f"falha ao gravar '{object_key}': {exc}") from exc
        ref = self._reference_from_stat(object_key, name or Path(object_key).name, sha256=sha)
        log.info("[artifact-store] upload key=%s bytes=%d sha256=%s etag=%s ver=%s %.3fs",
                 object_key, ref.size_bytes, sha[:12], ref.etag, ref.version_id, time.perf_counter() - t0)
        return ref

    def put_text(self, object_key, text, *, content_type="text/plain; charset=utf-8", name="", metadata=None):
        return self.put_bytes(object_key, text.encode("utf-8"), content_type=content_type, name=name, metadata=metadata)

    def put_json(self, object_key, obj, *, name="", metadata=None):
        import json as _json

        data = _json.dumps(obj, ensure_ascii=False, indent=2, default=str).encode("utf-8")
        return self.put_bytes(object_key, data, content_type="application/json", name=name, metadata=metadata)

    def put_file(self, object_key, file_path, *, content_type="application/octet-stream", name="", metadata=None):
        self.ensure_bucket()
        object_key = self._validate_key(object_key)
        file_path = Path(file_path)
        if not file_path.is_file():
            raise ArtifactUploadError(f"arquivo local inexistente: {file_path}")
        sha, size = sha256_of_file(file_path)
        meta = self._s3_metadata({**(metadata or {}), "sha256": sha})
        t0 = time.perf_counter()

        def _do():
            return self.client.fput_object(
                self.bucket, object_key, str(file_path),
                content_type=content_type, metadata=meta, sse=self._sse, part_size=self.part_size,
            )

        try:
            self._retry("fput_object", _do)
        except ArtifactStoreError as exc:
            raise ArtifactUploadError(f"falha ao gravar '{object_key}': {exc}") from exc
        ref = self._reference_from_stat(object_key, name or Path(object_key).name, sha256=sha)
        log.info("[artifact-store] upload-file key=%s bytes=%d sha256=%s ver=%s %.3fs",
                 object_key, size, sha[:12], ref.version_id, time.perf_counter() - t0)
        return ref

    def put_stream(self, object_key, stream, *, content_type="application/octet-stream", name="", metadata=None):
        """Grava a partir de um stream de tamanho desconhecido: faz spool em arquivo
        temporário (calculando SHA-256) e delega ao put_file — evita carregar tudo
        em memória e mantém o hash íntegro."""
        with tempfile.NamedTemporaryFile(prefix="artifact-", suffix=".part", delete=True) as tmp:
            while chunk := stream.read(self.download_chunk):
                tmp.write(chunk)
            tmp.flush()
            return self.put_file(object_key, Path(tmp.name), content_type=content_type, name=name, metadata=metadata)

    # -- leituras / downloads ----------------------------------------------

    def read_bytes(self, object_key: str) -> bytes:
        object_key = self._validate_key(object_key)
        resp = None

        def _do():
            nonlocal resp
            resp = self.client.get_object(self.bucket, object_key)
            return resp.read()

        try:
            return self._retry("get_object", _do)
        except ArtifactNotFoundError:
            raise
        except ArtifactStoreError as exc:
            raise ArtifactDownloadError(f"falha ao ler '{object_key}': {exc}") from exc
        finally:
            if resp is not None:
                resp.close()
                resp.release_conn()

    def read_text(self, object_key: str, *, max_bytes: int | None = None) -> str:
        data = self.read_bytes(object_key)
        if max_bytes is not None and len(data) > max_bytes:
            log.info("[artifact-store] read_text key=%s truncado de %d p/ %d bytes",
                     object_key, len(data), max_bytes)
            data = data[:max_bytes]
        return data.decode("utf-8", errors="replace")

    def read_json(self, object_key: str) -> Any:
        import json as _json

        return _json.loads(self.read_bytes(object_key).decode("utf-8"))

    def download_to_file(self, object_key: str, dest: Path, *, expected_sha256: str | None = None) -> Path:
        object_key = self._validate_key(object_key)
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()

        def _do():
            return self.client.fget_object(self.bucket, object_key, str(dest))

        try:
            self._retry("fget_object", _do)
        except ArtifactNotFoundError:
            raise
        except ArtifactStoreError as exc:
            raise ArtifactDownloadError(f"falha ao baixar '{object_key}': {exc}") from exc

        if expected_sha256:
            actual, _ = sha256_of_file(dest)
            if actual != expected_sha256:
                dest.unlink(missing_ok=True)
                raise ArtifactIntegrityError(
                    f"SHA-256 divergente em '{object_key}': esperado {expected_sha256[:12]}, obtido {actual[:12]}."
                )
        log.info("[artifact-store] download key=%s -> %s %.3fs", object_key, dest, time.perf_counter() - t0)
        return dest

    def open_stream(self, object_key: str) -> BinaryIO:
        """Retorna a resposta HTTP do objeto (o chamador DEVE fechar/release)."""
        object_key = self._validate_key(object_key)
        try:
            return self._retry("get_object", lambda: self.client.get_object(self.bucket, object_key))
        except ArtifactNotFoundError:
            raise
        except ArtifactStoreError as exc:
            raise ArtifactDownloadError(f"falha ao abrir '{object_key}': {exc}") from exc

    # -- inspeção / manutenção ---------------------------------------------

    def stat(self, object_key: str, *, name: str = "") -> ArtifactReference:
        object_key = self._validate_key(object_key)
        try:
            return self._reference_from_stat(object_key, name or Path(object_key).name)
        except ArtifactNotFoundError:
            raise
        except ArtifactStoreError as exc:
            raise ArtifactDownloadError(f"falha no stat de '{object_key}': {exc}") from exc

    def exists(self, object_key: str) -> bool:
        try:
            self.stat(object_key)
            return True
        except ArtifactNotFoundError:
            return False

    def delete(self, object_key: str) -> bool:
        object_key = self._validate_key(object_key)
        self._retry("remove_object", lambda: self.client.remove_object(self.bucket, object_key))
        return True

    def delete_prefix(self, prefix: str) -> int:
        from minio.deleteobjects import DeleteObject

        prefix = self._validate_key(prefix.rstrip("/") + "/") if not prefix.endswith("/") else prefix
        objs = self._retry(
            "list_objects",
            lambda: list(self.client.list_objects(self.bucket, prefix=prefix, recursive=True)),
        )
        if not objs:
            return 0
        errors = self.client.remove_objects(
            self.bucket, (DeleteObject(o.object_name) for o in objs)
        )
        n_err = 0
        for err in errors:
            n_err += 1
            log.warning("[artifact-store] erro ao excluir '%s': %s", getattr(err, "name", "?"), err)
        removed = len(objs) - n_err
        log.info("[artifact-store] delete_prefix %s -> %d objeto(s) removido(s)", prefix, removed)
        return removed

    def list_prefix(self, prefix: str) -> list[ArtifactReference]:
        objs = self._retry(
            "list_objects",
            lambda: list(self.client.list_objects(self.bucket, prefix=prefix, recursive=True)),
        )
        refs: list[ArtifactReference] = []
        for o in objs:
            refs.append(ArtifactReference(
                name=Path(o.object_name).name,
                uri=self.build_uri(o.object_name),
                bucket=self.bucket,
                object_key=o.object_name,
                size_bytes=int(o.size or 0),
                etag=(o.etag or "").strip('"') or None,
                version_id=getattr(o, "version_id", None),
            ))
        return refs

    def copy(self, src_key: str, dst_key: str, *, name: str = "") -> ArtifactReference:
        from minio.commonconfig import CopySource

        src_key = self._validate_key(src_key)
        dst_key = self._validate_key(dst_key)
        self._retry(
            "copy_object",
            lambda: self.client.copy_object(self.bucket, dst_key, CopySource(self.bucket, src_key)),
        )
        return self.stat(dst_key, name=name or Path(dst_key).name)

    # -- URIs ---------------------------------------------------------------

    def build_uri(self, object_key: str) -> str:
        return build_uri(self.bucket, object_key)

    def parse_uri(self, uri: str) -> tuple[str, str]:
        bucket, object_key = parse_uri(uri)
        if bucket != self.bucket:
            raise ArtifactInvalidURIError(
                f"URI aponta para bucket '{bucket}', esperado '{self.bucket}'."
            )
        return bucket, object_key

    # -- presigned URL / healthcheck ---------------------------------------

    def presigned_get_url(self, object_key: str, *, ttl_seconds: int) -> str:
        object_key = self._validate_key(object_key)
        if not self.exists(object_key):
            raise ArtifactNotFoundError(f"objeto inexistente para presign: {object_key}")
        return self._retry(
            "presigned_get_object",
            lambda: self.client.presigned_get_object(
                self.bucket, object_key, expires=timedelta(seconds=ttl_seconds)
            ),
        )

    def healthcheck(self) -> dict:
        """Valida conectividade, autenticação, existência do bucket e permissões de
        leitura/escrita (e exclusão, se MINIO_HEALTHCHECK_WRITE). NÃO toca artefatos
        reais: usa o prefixo reservado _healthcheck/. Nunca expõe credenciais."""
        status = {
            "reachable": False, "bucket": self.bucket, "bucket_exists": False,
            "readable": False, "writable": False,
        }
        try:
            exists = self.client.bucket_exists(self.bucket)
            status["reachable"] = True
            status["bucket_exists"] = bool(exists)
        except Exception as exc:
            log.warning("[artifact-store] healthcheck: MinIO inacessível: %s", exc)
            return status
        if not status["bucket_exists"]:
            return status
        try:
            list(self.client.list_objects(self.bucket, prefix="_healthcheck/", recursive=True))
            status["readable"] = True
        except Exception as exc:
            log.warning("[artifact-store] healthcheck: leitura falhou: %s", exc)
        if settings.MINIO_HEALTHCHECK_WRITE:
            probe_key = "_healthcheck/probe.txt"
            try:
                self.client.put_object(
                    self.bucket, probe_key, io.BytesIO(b"ok"), length=2, content_type="text/plain",
                )
                status["writable"] = True
                self.client.remove_object(self.bucket, probe_key)
            except Exception as exc:
                log.warning("[artifact-store] healthcheck: escrita falhou: %s", exc)
        return status


# ---------------------------------------------------------------------------
# Fábrica / singleton.
# ---------------------------------------------------------------------------

_store: Optional[ArtifactStore] = None


def build_artifact_store() -> ArtifactStore:
    """Constrói o store a partir das configurações do projeto (.env)."""
    backend = settings.ARTIFACT_STORE_BACKEND
    if backend != "minio":
        raise ArtifactStoreError(f"ARTIFACT_STORE_BACKEND desconhecido: {backend!r}")
    return MinIOArtifactStore(
        endpoint=settings.MINIO_ENDPOINT,
        access_key=settings.MINIO_ACCESS_KEY,
        secret_key=settings.MINIO_SECRET_KEY,
        bucket=settings.MINIO_BUCKET,
        secure=settings.MINIO_SECURE,
        region=settings.MINIO_REGION,
        artifact_prefix=settings.MINIO_ARTIFACT_PREFIX,
        auto_create_bucket=settings.MINIO_AUTO_CREATE_BUCKET,
        bucket_versioning=settings.MINIO_BUCKET_VERSIONING,
        max_retries=settings.MINIO_MAX_RETRIES,
        part_size_mb=settings.MINIO_PART_SIZE_MB,
        download_chunk_mb=settings.MINIO_DOWNLOAD_CHUNK_SIZE_MB,
        connect_timeout=settings.MINIO_CONNECT_TIMEOUT_SECONDS,
        read_timeout=settings.MINIO_READ_TIMEOUT_SECONDS,
        sse_enabled=settings.MINIO_SERVER_SIDE_ENCRYPTION_ENABLED,
        sse_type=settings.MINIO_SERVER_SIDE_ENCRYPTION_TYPE,
    )


def get_artifact_store() -> ArtifactStore:
    """Singleton lazy do store, reutilizado por API, workers e scripts."""
    global _store
    if _store is None:
        _store = build_artifact_store()
    return _store


def set_artifact_store(store: Optional[ArtifactStore]) -> None:
    """Injeção para testes (fake/memory store)."""
    global _store
    _store = store
