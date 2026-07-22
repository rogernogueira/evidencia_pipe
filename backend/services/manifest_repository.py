"""ManifestRepository — leitura/escrita segura do manifesto de artefatos no MinIO.

O manifesto é a fonte de descoberta dos artefatos de cada documento e pode ser
atualizado por retries ou processos concorrentes (worker leve, worker GPU, scripts).
Para evitar perda silenciosa de atualizações:

  - todo update roda sob um lock distribuído CURTO no Redis
    (`artifact:manifest:<pipeline_id>:<document_id>:lock`), que protege APENAS o
    ciclo read → validate → write do JSON (nunca MinerU/DeepSeek/embeddings/Qdrant);
  - a cada gravação incrementa `revision` e atualiza `updated_at`;
  - antes de gravar, revalida o ETag do objeto: se mudou desde a leitura, levanta
    ArtifactManifestConflictError (defesa extra caso o lock falhe);
  - o versionamento do bucket serve de rede de segurança/auditoria — NÃO é o único
    mecanismo de concorrência.

Se o Redis do lock estiver indisponível, o repositório degrada para "sem lock"
(loga um aviso) — mantendo o pipeline funcional em dev, com a proteção de ETag
ainda ativa.
"""

from __future__ import annotations

import threading
import uuid as _uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator, Optional

from pydantic import ValidationError

from backend.core import config as settings
from backend.core.logger import log
from backend.core.schemas import ArtifactManifest
from backend.services.artifact_store import (
    ArtifactNotFoundError,
    ArtifactStore,
    get_artifact_store,
    parse_uri,
)


class ArtifactManifestValidationError(Exception):
    """Manifesto com schema inválido."""


class ArtifactManifestConflictError(Exception):
    """O manifesto mudou entre a leitura e a gravação (ETag divergente)."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


class _RedisManifestLock:
    """Lock distribuído mínimo (SET NX PX + release por token via Lua)."""

    _RELEASE_LUA = (
        "if redis.call('get', KEYS[1]) == ARGV[1] then "
        "return redis.call('del', KEYS[1]) else return 0 end"
    )

    _client = None
    _ready = False
    _init_lock = threading.Lock()

    @classmethod
    def _get_client(cls):
        if cls._ready:
            return cls._client
        with cls._init_lock:
            if not cls._ready:
                try:
                    import redis

                    c = redis.Redis.from_url(settings.MANIFEST_LOCK_REDIS_URL)
                    c.ping()
                    cls._client = c
                    log.info("[manifest] lock via Redis em %s", settings.MANIFEST_LOCK_REDIS_URL)
                except Exception as exc:
                    cls._client = None
                    log.warning("[manifest] Redis do lock indisponível (%s) — updates SEM lock.", exc)
                cls._ready = True
        return cls._client

    @classmethod
    @contextmanager
    def acquire(cls, key: str, ttl_seconds: int) -> Iterator[bool]:
        client = cls._get_client()
        if client is None:
            yield False  # degradação graciosa
            return
        token = _uuid.uuid4().hex
        acquired = False
        # Espera curta e limitada — o lock só protege I/O de JSON pequeno.
        import time

        deadline = time.monotonic() + ttl_seconds
        while time.monotonic() < deadline:
            if client.set(key, token, nx=True, px=ttl_seconds * 1000):
                acquired = True
                break
            time.sleep(0.05)
        if not acquired:
            raise ArtifactManifestConflictError(f"não foi possível adquirir o lock do manifesto: {key}")
        try:
            yield True
        finally:
            try:
                client.eval(cls._RELEASE_LUA, 1, key, token)
            except Exception as exc:  # pragma: no cover
                log.warning("[manifest] falha ao liberar lock %s: %s", key, exc)


class ManifestRepository:
    def __init__(self, store: Optional[ArtifactStore] = None) -> None:
        self.store = store or get_artifact_store()

    # -- criação / carga ----------------------------------------------------

    def create(
        self,
        *,
        pipeline_id: str,
        job_id: str,
        document_id: str,
        item_uuid: str | None = None,
        bitstream_uuid: str | None = None,
        pipeline_version: str | None = None,
    ) -> ArtifactManifest:
        """Cria o manifesto inicial (revision=0) SE ainda não existir. Idempotente:
        se já existe, retorna o existente (suporta retry da etapa de download)."""
        key = self.store.manifest_key(pipeline_id, document_id)
        try:
            existing = self._load_key(key)
            log.info("[manifest] reutilizando manifesto existente rev=%d %s/%s",
                     existing.revision, pipeline_id, document_id)
            return existing
        except ArtifactNotFoundError:
            pass
        manifest = ArtifactManifest(
            pipeline_id=pipeline_id,
            job_id=job_id,
            document_id=document_id,
            item_uuid=item_uuid or None,
            bitstream_uuid=bitstream_uuid or None,
            pipeline_version=pipeline_version or settings.PIPELINE_VERSION,
        )
        self.store.put_json(key, manifest.model_dump(mode="json"), name="manifest")
        log.info("[manifest] criado %s/%s", pipeline_id, document_id)
        return manifest

    def _load_key(self, key: str) -> ArtifactManifest:
        raw = self.store.read_json(key)
        try:
            return ArtifactManifest.model_validate(raw)
        except ValidationError as exc:
            raise ArtifactManifestValidationError(f"manifesto inválido em '{key}': {exc}") from exc

    def load(self, pipeline_id: str, document_id: str) -> ArtifactManifest:
        return self._load_key(self.store.manifest_key(pipeline_id, document_id))

    def load_from_uri(self, uri: str) -> ArtifactManifest:
        _, key = parse_uri(uri)
        return self._load_key(key)

    def manifest_uri(self, pipeline_id: str, document_id: str) -> str:
        return self.store.build_uri(self.store.manifest_key(pipeline_id, document_id))

    # -- atualização segura -------------------------------------------------

    @contextmanager
    def update(self, pipeline_id: str, document_id: str) -> Iterator[ArtifactManifest]:
        """Contexto de atualização segura: adquire o lock, carrega o manifesto (com
        ETag), entrega-o para mutação e, ao sair, incrementa a revision e grava.

        Uso:
            with repo.update(pid, doc) as m:
                m.artifacts["source_pdf"] = ref
                m.stage("download").status = "COMPLETED"
        """
        key = self.store.manifest_key(pipeline_id, document_id)
        lock_key = f"artifact:manifest:{pipeline_id}:{document_id}:lock"
        with _RedisManifestLock.acquire(lock_key, settings.MANIFEST_LOCK_TTL_SECONDS) as locked:
            # ETag no momento da leitura (proteção adicional contra escrita concorrente).
            try:
                before = self.store.stat(key)
                manifest = self._load_key(key)
                prev_etag = before.etag
            except ArtifactNotFoundError:
                raise ArtifactManifestValidationError(
                    f"manifesto ausente para {pipeline_id}/{document_id} — crie antes de atualizar."
                )

            yield manifest

            # Revalida ETag: se mudou desde a leitura, alguém gravou no meio (só
            # possível se o lock degradou/falhou) → conflito explícito.
            if not locked and prev_etag is not None:
                current = self.store.stat(key)
                if current.etag != prev_etag:
                    raise ArtifactManifestConflictError(
                        f"manifesto {pipeline_id}/{document_id} mudou durante o update "
                        f"(etag {prev_etag} → {current.etag})."
                    )

            manifest.revision += 1
            manifest.updated_at = _now()
            ref = self.store.put_json(key, manifest.model_dump(mode="json"), name="manifest",
                                      metadata={"revision": str(manifest.revision)})
            log.info("[manifest] gravado %s/%s rev=%d etag_prev=%s etag_novo=%s",
                     pipeline_id, document_id, manifest.revision, prev_etag, ref.etag)

    # -- conveniências de estágio ------------------------------------------

    def start_stage(self, pipeline_id: str, document_id: str, stage: str, attempt: int = 1) -> None:
        with self.update(pipeline_id, document_id) as m:
            st = m.stage(stage)
            st.status = "RUNNING"
            st.attempt = max(st.attempt, attempt)
            st.started_at = _now()
            st.error = None
            m.status = "RUNNING"

    def complete_stage(self, pipeline_id: str, document_id: str, stage: str, *, final: bool = False) -> None:
        with self.update(pipeline_id, document_id) as m:
            st = m.stage(stage)
            st.status = "COMPLETED"
            st.completed_at = _now()
            if final:
                m.status = "COMPLETED"

    def fail_stage(
        self, pipeline_id: str, document_id: str, stage: str, *,
        error_type: str, message: str, attempt: int = 1, object_key: str | None = None,
    ) -> None:
        """Registra falha no manifesto SEM stack trace completo (mensagem limitada)."""
        from backend.core.schemas import ManifestError

        with self.update(pipeline_id, document_id) as m:
            st = m.stage(stage)
            st.status = "FAILED"
            st.attempt = max(st.attempt, attempt)
            st.error = message[:500]
            m.errors.append(ManifestError(
                stage=stage, error_type=error_type, message=message[:500],
                attempt=attempt, object_key=object_key,
            ))
            m.status = "FAILED"


_repo: Optional[ManifestRepository] = None


def get_manifest_repository() -> ManifestRepository:
    global _repo
    if _repo is None:
        _repo = ManifestRepository()
    return _repo
