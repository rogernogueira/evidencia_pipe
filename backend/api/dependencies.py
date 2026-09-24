"""Dependências (singletons) compartilhadas pelas rotas da API.

`semantic_search` é uma instância única e residente: mantém o cliente Qdrant e a
referência ao embedder bge-m3 já carregado durante toda a execução do servidor.
"""

from backend.core.config import SUMMARY_CACHE_ENABLED
from backend.repositories.qdrant_client import SemanticSearch
from backend.services.summary_cache import SummaryCache
from backend.services.summary_service import SummaryService

# Instância global — persiste estado (cliente Qdrant, conexão) entre requisições.
semantic_search = SemanticSearch()

# AI Summary reusa o mesmo SemanticSearch residente (embedder + cliente Qdrant).
# Cache das sínteses (Redis, fallback em memória) — desligável por SUMMARY_CACHE_ENABLED.
summary_service = SummaryService(
    semantic_search, cache=SummaryCache() if SUMMARY_CACHE_ENABLED else None,
)


def get_semantic_search() -> SemanticSearch:
    return semantic_search


def get_summary_service() -> SummaryService:
    return summary_service
