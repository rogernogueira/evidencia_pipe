"""Rotas internas de observabilidade da GPU (somente leitura).

    GET /internal/gpu/status   → manager.get_status()
    GET /internal/gpu/queue    → manager.get_queue()

Estas rotas NUNCA adquirem, liberam ou alteram o recurso — apenas consultam.

Autenticação: o projeto não possui um mecanismo de auth administrativa próprio.
Enquanto não houver, estas rotas ficam sob o prefixo `/internal` e NÃO devem ser
expostas publicamente (restrinja no proxy/rede). Como salvaguarda opcional, se a
variável de ambiente GPU_INTERNAL_TOKEN estiver definida, exige-se o header
`X-Internal-Token` correspondente.
"""

from __future__ import annotations

import os

from fastapi import APIRouter, Header, HTTPException, Query

from gpu_resource_manager import GPUBackendUnavailable

from backend.core import config as settings
from backend.services.gpu_manager import get_gpu_manager

router = APIRouter()


def _check_token(x_internal_token: str | None) -> None:
    expected = os.getenv("GPU_INTERNAL_TOKEN")
    if expected and x_internal_token != expected:
        raise HTTPException(status_code=401, detail="token interno inválido")


@router.get("/internal/gpu/status", tags=["internal-gpu"])
def gpu_status(
    resource: str | None = Query(default=None),
    x_internal_token: str | None = Header(default=None),
):
    _check_token(x_internal_token)
    res = resource or settings.GPU_RESOURCE_NAME
    try:
        return get_gpu_manager().get_status(resource=res).to_dict()
    except GPUBackendUnavailable as exc:
        raise HTTPException(status_code=503, detail=f"backend do gpu-manager indisponível: {exc}")


@router.get("/internal/gpu/queue", tags=["internal-gpu"])
def gpu_queue(
    resource: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
    x_internal_token: str | None = Header(default=None),
):
    _check_token(x_internal_token)
    res = resource or settings.GPU_RESOURCE_NAME
    try:
        manager = get_gpu_manager()
        status = manager.get_status(resource=res).to_dict()
        status["queue"] = manager.get_queue(resource=res, limit=limit)
        return status
    except GPUBackendUnavailable as exc:
        raise HTTPException(status_code=503, detail=f"backend do gpu-manager indisponível: {exc}")
