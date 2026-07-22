"""Endpoints internos de artefatos — download administrativo via URL pré-assinada.

Os endpoints públicos NÃO devolvem o conteúdo dos artefatos (§30). Quando um
operador precisa baixar um artefato específico, este endpoint gera uma URL
pré-assinada de VALIDADE CURTA (MINIO_PRESIGNED_URL_TTL_SECONDS), somente após
autorização (header X-Internal-Token quando INTERNAL_API_TOKEN estiver definido).

A URL pré-assinada:
  - expira;
  - é gerada sob demanda (não é persistida no manifesto);
  - não é registrada integralmente nos logs (só um prefixo).

Também expõe o health check do MinIO (conectividade/bucket/leitura/escrita).
"""

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import JSONResponse

from backend.core import config as settings
from backend.core.logger import log_api
from backend.services.artifact_store import (
    ArtifactNotFoundError,
    ArtifactStoreError,
    get_artifact_store,
)
from backend.services.manifest_repository import (
    ArtifactManifestValidationError,
    get_manifest_repository,
)

router = APIRouter(prefix="/internal", tags=["internal"])


def _authorize(token: str | None) -> None:
    """Autoriza o acesso interno. Se INTERNAL_API_TOKEN estiver definido, exige o
    header X-Internal-Token igual. Se vazio (dev), libera com um aviso no log."""
    expected = settings.INTERNAL_API_TOKEN
    if not expected:
        log_api.warning("[internal] INTERNAL_API_TOKEN não definido — endpoint liberado (apenas dev).")
        return
    if token != expected:
        raise HTTPException(status_code=401, detail="não autorizado")


@router.get("/artifacts/{pipeline_id}/{document_id}/{artifact_name}/download-url")
def artifact_download_url(
    pipeline_id: str,
    document_id: str,
    artifact_name: str,
    x_internal_token: str | None = Header(default=None),
) -> JSONResponse:
    """Gera uma URL pré-assinada (curta) para baixar um artefato nomeado do manifesto."""
    _authorize(x_internal_token)
    repo = get_manifest_repository()
    store = get_artifact_store()
    try:
        manifest = repo.load(pipeline_id, document_id)
    except ArtifactNotFoundError:
        raise HTTPException(status_code=404, detail="manifesto não encontrado")
    except ArtifactManifestValidationError as e:
        raise HTTPException(status_code=500, detail=str(e))

    ref = manifest.artifacts.get(artifact_name)
    if ref is None:
        raise HTTPException(status_code=404, detail=f"artefato '{artifact_name}' não existe no manifesto")
    if ref.content_type == "application/x-directory-prefix":
        raise HTTPException(status_code=400, detail="artefato é um prefixo (diretório), não um objeto único")

    ttl = settings.MINIO_PRESIGNED_URL_TTL_SECONDS
    try:
        url = store.presigned_get_url(ref.object_key, ttl_seconds=ttl)
    except ArtifactNotFoundError:
        raise HTTPException(status_code=404, detail="objeto ausente no MinIO")
    except ArtifactStoreError as e:
        raise HTTPException(status_code=502, detail=f"falha ao gerar URL: {e}")

    # Loga só o prefixo da URL (nunca a assinatura completa).
    log_api.info("[internal] presigned %s/%s/%s ttl=%ds url=%s...",
                 pipeline_id, document_id, artifact_name, ttl, url.split("?")[0])
    return JSONResponse({
        "artifact_name": artifact_name,
        "content_type": ref.content_type,
        "size_bytes": ref.size_bytes,
        "sha256": ref.sha256,
        "expires_in_seconds": ttl,
        "url": url,
    })


@router.get("/artifacts/health")
def artifacts_health(x_internal_token: str | None = Header(default=None)) -> JSONResponse:
    """Health check do MinIO: conectividade, bucket, leitura e escrita. Sem credenciais."""
    _authorize(x_internal_token)
    try:
        status = get_artifact_store().healthcheck()
    except ArtifactStoreError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return JSONResponse({"minio": status})
