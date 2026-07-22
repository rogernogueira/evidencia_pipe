"""Rotas de busca semântica sobre a collection de chunks (`evidencia_chunks`).

    GET /api/search/semantic  → busca híbrida (RRF) / dense / sparse via Qdrant
    GET /api/search/status    → disponibilidade do mecanismo de busca
"""

import time
from typing import List

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse

from backend.api.dependencies import get_semantic_search
from backend.core.config import QDRANT_COLLECTION
from backend.core.logger import log_api
from backend.core.schemas import SearchResult
from backend.repositories.qdrant_client import SemanticSearch

router = APIRouter()


@router.get("/api/search/semantic", response_model=List[SearchResult], tags=["search"])
async def search_semantic(
    request: Request,
    q: str = Query(default="", description="Consulta de busca semântica"),
    limit: int = Query(default=10, ge=1, le=50, description="Máximo de resultados"),
    type: str = Query(default="hybrid", description="Modo: 'hybrid' (RRF dense+sparse), 'dense' ou 'sparse'"),
    doc_id: str = Query(default="", description="Filtra por doc_id (ex: 'relatorio.pdf')"),
    uuid: str = Query(default="", description="Filtra pelo UUID do item DSpace (item_uuid)"),
    semantic: SemanticSearch = Depends(get_semantic_search),
):
    """Busca semântica híbrida (bge-m3 dense + sparse → fusão RRF) via Qdrant sobre a
    collection de chunks do estágio 3 (`evidencia_chunks`)."""
    log_api.info(
        "GET /api/search/semantic?q=%r type=%r limit=%d doc_id=%r uuid=%r [client=%s]",
        q, type, limit, doc_id, uuid, request.client.host if request.client else "?",
    )
    if not await semantic.ensure_connected():
        log_api.warning(
            "Busca semântica indisponível. Verifique o Qdrant e a collection '%s'.",
            QDRANT_COLLECTION,
        )
        return JSONResponse(
            {
                "error": (
                    "Busca semântica indisponível. Verifique o Qdrant e a collection "
                    f"'{QDRANT_COLLECTION}' (python -m backend.indexing.index_chunks --reset)."
                )
            },
            status_code=503,
        )
    t0 = time.perf_counter()
    results = await semantic.search(
        q, limit=limit, doc_id=doc_id or None, uuid=uuid or None, type=type,
    )
    log_api.info(
        "GET /api/search/semantic: %d resultado(s) em %.3fs",
        len(results), time.perf_counter() - t0,
    )
    return results


@router.get("/api/search/status", tags=["search"])
async def search_status(
    semantic: SemanticSearch = Depends(get_semantic_search),
) -> JSONResponse:
    """Status do mecanismo de busca semântica (Qdrant + collection + modelo)."""
    return JSONResponse(await semantic.health())
