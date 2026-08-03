"""Lado de LEITURA da busca semântica sobre a collection `evidencia_chunks`.

Contrapartida do indexador (backend/indexing/index_chunks.py): consulta os mesmos
vetores nomeados (`dense` 1024d COSINE + `sparse` lexical_weights) gerados pelo
BAAI/bge-m3, com fusão RRF (híbrido) ou busca isolada dense/sparse.

Reusa o embedder já carregado (BgeM3EmbedderService.embed_query) — a query passa
pelo MESMO pipeline de embedding da indexação. `normalize=False`: o indexador
(index_chunks) embeda os documentos sem fold de acentos, então a query também não
pode normalizar (simetria indexação↔busca).
"""

import time
from typing import Optional

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchAny,
    MatchValue,
    Prefetch,
    Range,
    SparseVector,
)

from backend.core.config import (
    CHUNK_CHART_MIN_CONFIDENCE,
    DENSE_MODEL,
    QDRANT_COLLECTION,
    QDRANT_TIMEOUT_SECONDS,
    QDRANT_URL,
    SEARCH_DEFAULT_PROFILE,
    SEARCH_EXCLUDE_FRONT_MATTER,
    SEARCH_EXCLUDE_LOW_CONFIDENCE_VISUAL_DATA,
    SEARCH_EXCLUDE_NAVIGATION_LISTS,
    SEARCH_EXCLUDE_REFERENCES,
)
from backend.core.logger import log, log_api
from backend.core.schemas import SearchResult
from backend.indexing.retrieval_profile import (
    PROFILE_GENERAL,
    detect_query_profile,
    profile_exclusions,
)
from backend.services.embedder import BgeM3EmbedderService


def _search_filtering_enabled() -> bool:
    """Filtro por perfil só age se alguma flag SEARCH_EXCLUDE_* estiver ligada.

    Default (todas off) → busca v1: sem filtro de perfil, comportamento inalterado.
    Requer dados indexados pela política v2 (payload com searchable_by_default etc.)."""
    return any((
        SEARCH_EXCLUDE_FRONT_MATTER,
        SEARCH_EXCLUDE_REFERENCES,
        SEARCH_EXCLUDE_NAVIGATION_LISTS,
        SEARCH_EXCLUDE_LOW_CONFIDENCE_VISUAL_DATA,
    ))


def _profile_conditions(profile: str) -> tuple[list, list]:
    """(must, must_not) do Qdrant para um perfil de recuperação (§21)."""
    ex = profile_exclusions(profile)
    must: list = []
    must_not: list = []
    if ex.require_searchable_by_default:
        must.append(FieldCondition(key="searchable_by_default", match=MatchValue(value=True)))
    if ex.require_is_reference:
        must.append(FieldCondition(key="is_reference", match=MatchValue(value=True)))
    if ex.exclude_is_reference:
        must_not.append(FieldCondition(key="is_reference", match=MatchValue(value=True)))
    if ex.exclude_section_kinds:
        must_not.append(
            FieldCondition(key="section_kind", match=MatchAny(any=list(ex.exclude_section_kinds)))
        )
    return must, must_not


class SemanticSearch:
    """Busca semântica híbrida (bge-m3 dense + sparse → RRF) via Qdrant.

    Instância única e residente durante a execução do servidor (ver
    backend/api/dependencies.py). A conexão é preguiçosa: `ensure_connected`
    tenta (re)conectar enquanto indisponível, sem gate permanente — assim o
    servidor sobe mesmo com o Qdrant fora e passa a servir busca quando ele volta.
    """

    def __init__(self):
        self._client: Optional[AsyncQdrantClient] = None
        self._embedder = BgeM3EmbedderService()
        self._available = False

    async def ensure_connected(self) -> bool:
        """Idempotente enquanto disponível; re-tenta enquanto indisponível.

        Exige o bge-m3 no cache local (require_cache=True) para não disparar um
        download bloqueante dentro da API, e a existência da collection.
        """
        if self._available:
            return True

        if not self._embedder.load_model(require_cache=True):
            log.warning("bge-m3 não está no cache local — busca semântica desabilitada.")
            self._available = False
            return False

        try:
            self._client = AsyncQdrantClient(url=QDRANT_URL, timeout=QDRANT_TIMEOUT_SECONDS or None)
            cols = await self._client.get_collections()
            names = [c.name for c in cols.collections]
            if QDRANT_COLLECTION not in names:
                log.warning(
                    "Qdrant OK mas collection '%s' não existe. "
                    "Execute: python -m backend.indexing.index_chunks --reset",
                    QDRANT_COLLECTION,
                )
                self._available = False
                return False
            self._available = True
            log.info("Qdrant conectado — collection '%s' disponível.", QDRANT_COLLECTION)
            return True
        except Exception as exc:
            log.warning("Qdrant indisponível (%s). Busca semântica desabilitada.", exc)
            self._available = False
            return False

    @property
    def available(self) -> bool:
        return self._available

    async def search(
        self,
        query: str,
        limit: int = 10,
        doc_id: Optional[str] = None,
        uuid: Optional[str] = None,
        type: str = "hybrid",
        profile: str = "",
    ) -> list[SearchResult]:
        """Consulta a collection. `type`: 'hybrid' (RRF dense+sparse), 'dense' ou 'sparse'.

        Filtros combináveis: `uuid` → payload.item_uuid; `doc_id` → payload.doc_id.
        `profile` (§21): '' = auto (detecta a intenção da consulta) | general | quantitative
        | methodological | bibliographic. O filtro de perfil só age se alguma flag
        SEARCH_EXCLUDE_* estiver ligada OU se `profile` for explicitamente informado.
        """
        if not await self.ensure_connected():
            return []
        if not query or not query.strip():
            return []

        t0 = time.perf_counter()
        try:
            dense_vector, lexical_weights = self._embedder.embed_query(query, normalize=False)
            sparse_indices = [int(k) for k in lexical_weights.keys()]
            sparse_values = [float(v) for v in lexical_weights.values()]

            conditions = []
            must_not: list = []
            if uuid and uuid.strip():
                conditions.append(FieldCondition(key="item_uuid", match=MatchValue(value=uuid.strip())))
            if doc_id and doc_id.strip():
                conditions.append(FieldCondition(key="doc_id", match=MatchValue(value=doc_id.strip())))

            # --- Filtro por perfil de recuperação (§21) ---
            explicit = (profile or "").strip().lower()
            if explicit or _search_filtering_enabled():
                resolved = explicit or (detect_query_profile(query) or SEARCH_DEFAULT_PROFILE or PROFILE_GENERAL)
                p_must, p_must_not = _profile_conditions(resolved)
                conditions.extend(p_must)
                must_not.extend(p_must_not)
                # §11: dados visuais de baixa confiança fora de QUALQUER perfil (global).
                if SEARCH_EXCLUDE_LOW_CONFIDENCE_VISUAL_DATA:
                    must_not.append(FieldCondition(
                        key="chart_data_confidence", range=Range(lt=CHUNK_CHART_MIN_CONFIDENCE)))
                log_api.info("search profile=%r (resolvido de %r)", resolved, explicit or "auto")

            query_filter = (
                Filter(must=conditions or None, must_not=must_not or None)
                if (conditions or must_not) else None
            )

            if type == "hybrid":
                results = await self._client.query_points(
                    collection_name=QDRANT_COLLECTION,
                    prefetch=[
                        Prefetch(query=dense_vector, using="dense", limit=limit * 2),
                        Prefetch(
                            query=SparseVector(indices=sparse_indices, values=sparse_values),
                            using="sparse",
                            limit=limit * 2,
                        ),
                    ],
                    query=FusionQuery(fusion=Fusion.RRF),
                    query_filter=query_filter,
                    limit=limit,
                    with_payload=True,
                )
            elif type == "sparse":
                results = await self._client.query_points(
                    collection_name=QDRANT_COLLECTION,
                    query=SparseVector(indices=sparse_indices, values=sparse_values),
                    using="sparse",
                    query_filter=query_filter,
                    limit=limit,
                    with_payload=True,
                )
            else:  # dense
                results = await self._client.query_points(
                    collection_name=QDRANT_COLLECTION,
                    query=dense_vector,
                    using="dense",
                    query_filter=query_filter,
                    limit=limit,
                    with_payload=True,
                )

            hits = [self._to_result(p) for p in results.points]
            log_api.info(
                "search semantic (%s): %d resultado(s) em %.3fs",
                type, len(hits), time.perf_counter() - t0,
            )
            return hits
        except Exception as exc:
            log_api.error("Erro na busca semântica: %s", exc)
            return []

    @staticmethod
    def _to_result(point) -> SearchResult:
        """Mapeia um ponto do Qdrant para SearchResult, com fallback payload
        legado↔estrutural (§26): section|section_title, page|page_start,
        content|text, type|content_type."""
        pl = point.payload or {}
        page = pl.get("page")
        if page is None:
            page = pl.get("page_start")
        return SearchResult(
            doc_name=pl.get("doc_name"),
            doc_id=pl.get("doc_id"),
            doc_path=pl.get("doc_path"),
            section=pl.get("section") or pl.get("section_title") or "",
            page=page,
            snippet=pl.get("content") or pl.get("text") or "",
            score=round(point.score, 4) if getattr(point, "score", None) is not None else None,
            type=pl.get("type") or pl.get("content_type") or "paragraph",
            item_uuid=pl.get("item_uuid"),
            item_handle=pl.get("item_handle"),
        )

    async def health(self) -> dict:
        """Status da busca (para /api/search/status)."""
        await self.ensure_connected()
        return {
            "semantic": self._available,
            "qdrant_url": QDRANT_URL,
            "collection": QDRANT_COLLECTION,
            "dense_model": DENSE_MODEL,
        }
