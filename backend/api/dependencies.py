"""Dependências (singletons) compartilhadas pelas rotas da API.

`semantic_search` é uma instância única e residente: mantém o cliente Qdrant e a
referência ao embedder bge-m3 já carregado durante toda a execução do servidor.
"""

from backend.repositories.qdrant_client import SemanticSearch

# Instância global — persiste estado (cliente Qdrant, conexão) entre requisições.
semantic_search = SemanticSearch()


def get_semantic_search() -> SemanticSearch:
    return semantic_search
