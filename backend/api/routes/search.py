"""Rotas de busca semântica sobre a collection de chunks (`evidencia_chunks`).

    GET /api/search/semantic   → busca híbrida (RRF) / dense / sparse via Qdrant
    GET /api/search/summarize  → AI Summary: síntese das evidências recuperadas
    GET /api/search/status     → disponibilidade do mecanismo de busca
"""

import time
from typing import List

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse

from backend.api.dependencies import get_semantic_search, get_summary_service
from backend.core.config import QDRANT_COLLECTION
from backend.core.logger import log_api
from backend.core.schemas import SearchResult, SummaryResponse
from backend.repositories.qdrant_client import SemanticSearch
from backend.services.summary_service import SummaryService

router = APIRouter()


@router.get("/api/search/semantic", response_model=List[SearchResult], tags=["search"])
async def search_semantic(
    request: Request,
    q: str = Query(default="", description="Consulta de busca semântica"),
    limit: int = Query(default=10, ge=1, le=50, description="Máximo de resultados"),
    type: str = Query(default="hybrid", description="Modo: 'hybrid' (RRF dense+sparse), 'dense' ou 'sparse'"),
    doc_id: str = Query(default="", description="Filtra por doc_id (ex: 'relatorio.pdf')"),
    uuid: str = Query(default="", description="Filtra pelo UUID do item DSpace (item_uuid)"),
    profile: str = Query(default="", description="Perfil de recuperação (§21): ''=auto | general | quantitative | methodological | bibliographic"),
    semantic: SemanticSearch = Depends(get_semantic_search),
):
    """Busca semântica híbrida (bge-m3 dense + sparse → fusão RRF) via Qdrant sobre a
    collection de chunks do estágio 3 (`evidencia_chunks`)."""
    log_api.info(
        "GET /api/search/semantic?q=%r type=%r limit=%d doc_id=%r uuid=%r profile=%r [client=%s]",
        q, type, limit, doc_id, uuid, profile, request.client.host if request.client else "?",
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
        q, limit=limit, doc_id=doc_id or None, uuid=uuid or None, type=type, profile=profile,
    )
    log_api.info(
        "GET /api/search/semantic: %d resultado(s) em %.3fs",
        len(results), time.perf_counter() - t0,
    )
    return results


@router.get("/api/search/summarize", response_model=SummaryResponse, tags=["search"])
async def search_summarize(
    request: Request,
    q: str = Query(..., min_length=1, description="Consulta para a síntese das evidências"),
    limit: int = Query(default=5, ge=1, le=20, description="Máx. de chunks recuperados"),
    type: str = Query(default="hybrid", description="Modo: 'hybrid' (RRF), 'dense' ou 'sparse'"),
    language: str = Query(default="pt-BR", description="Idioma da síntese"),
    semantic: SemanticSearch = Depends(get_semantic_search),
    summary: SummaryService = Depends(get_summary_service),
):
    """AI Summary síncrono: recupera evidências pela busca semântica e sintetiza-as
    com citações [N] validadas, sem inventar conteúdo. Segue o mesmo padrão da busca
    (público; 503 quando o mecanismo está indisponível).

    Nesta 1ª iteração não há filtros de metadados (evaluation_criteria, section,
    level) nem grupo Centralised/Decentralised — são trabalho futuro."""
    log_api.info(
        "GET /api/search/summarize?q=%r limit=%d type=%r lang=%r [client=%s]",
        q, limit, type, language, request.client.host if request.client else "?",
    )
    if not await semantic.ensure_connected():
        return JSONResponse(
            {
                "error": (
                    "Busca semântica indisponível. Verifique o Qdrant e a collection "
                    f"'{QDRANT_COLLECTION}' (python -m backend.indexing.index_chunks --reset)."
                )
            },
            status_code=503,
        )
    if not summary.llm_available():
        return JSONResponse(
            {"error": "AI Summary indisponível: LLM não configurado (LLM_ENRICH_API_KEY)."},
            status_code=503,
        )
    return await summary.summarize(q, limit=limit, type=type, language=language)


@router.get("/api/search/status", tags=["search"])
async def search_status(
    semantic: SemanticSearch = Depends(get_semantic_search),
) -> JSONResponse:
    """Status do mecanismo de busca semântica (Qdrant + collection + modelo)."""
    return JSONResponse(await semantic.health())
