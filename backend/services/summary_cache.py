"""summary_cache.py — cache das respostas do AI Summary (GET/POST /api/search/summarize).

A síntese custa uma ou duas chamadas ao LLM (segundos, e dinheiro) e a mesma pergunta
se repete muito na busca pública. Guardamos a SummaryResponse inteira:

  - Redis (SUMMARY_CACHE_REDIS_URL, padrão o DB do job_store), chaves `summary:resp:*`
    com TTL — compartilhado entre processos e sobrevive a restart da API;
  - fallback em memória (LRU por processo) quando o Redis não responde, como no
    job_store: a API não quebra, só perde o compartilhamento.

Invalidação por GERAÇÃO do índice: `bump_index_generation()` incrementa um contador
no Redis a cada escrita no Qdrant (chamado pela indexação, que roda nos workers
Celery ou na CLI), e a geração faz parte da chave. Reindexar um documento invalida
todas as sínteses de uma vez, sem varrer chaves — as antigas expiram pelo TTL.

A chave também leva o modelo, os prompts e as constantes da montagem das evidências:
mudar qualquer um deles em deploy não serve síntese velha.
"""

import hashlib
import json
import threading
import time
from collections import OrderedDict
from typing import Optional, Sequence

from backend.core.config import (
    LLM_SUMMARY_MODEL,
    QDRANT_COLLECTION,
    SUMMARY_CACHE_MAX_ENTRIES,
    SUMMARY_CACHE_REDIS_URL,
    SUMMARY_CACHE_TTL_SECONDS,
)
from backend.core.logger import log
from backend.core.schemas import DocumentRef, SummaryResponse

_RESP_PREFIX = "summary:resp:"
_GEN_KEY = "summary:index_gen"
# Suba quando mudar a montagem da síntese de um jeito que o fingerprint não enxerga
# (ex.: build_basic_user_message, regras de dedup).
_SCHEMA_VERSION = 1
# Redis fora do ar não pode travar a requisição: timeouts curtos, e depois de uma
# falha só se tenta reconectar passado este intervalo.
_REDIS_TIMEOUT_SECONDS = 0.5
_REDIS_RETRY_SECONDS = 30.0

_redis = None
_redis_checked_at: Optional[float] = None
_redis_lock = threading.Lock()


def _get_redis():
    """Cliente Redis lazy; None se indisponível (aciona o fallback em memória)."""
    global _redis, _redis_checked_at
    if _redis is not None:
        return _redis
    now = time.monotonic()
    if _redis_checked_at is not None and now - _redis_checked_at < _REDIS_RETRY_SECONDS:
        return None
    with _redis_lock:
        if _redis is not None:
            return _redis
        _redis_checked_at = now
        try:
            import redis  # import tardio: dependência opcional em dev sem Redis

            client = redis.Redis.from_url(
                SUMMARY_CACHE_REDIS_URL, decode_responses=True,
                socket_timeout=_REDIS_TIMEOUT_SECONDS,
                socket_connect_timeout=_REDIS_TIMEOUT_SECONDS,
            )
            client.ping()
            _redis = client
            log.info("summary_cache: usando Redis em %s", SUMMARY_CACHE_REDIS_URL)
        except Exception as exc:
            log.warning("summary_cache: Redis indisponível (%s) — cache em memória.", exc)
    return _redis


def _drop_redis(exc: Exception) -> None:
    """Descarta o cliente após uma falha; a próxima tentativa respeita o intervalo."""
    global _redis, _redis_checked_at
    log.warning("summary_cache: falha no Redis (%s) — cache em memória por ora.", exc)
    _redis = None
    _redis_checked_at = time.monotonic()


# Geração local: só é usada sem Redis, e aí só enxerga bumps do próprio processo.
_local_gen = 0


def bump_index_generation() -> None:
    """Invalida todas as sínteses em cache. Chamado a cada escrita no índice.
    Best-effort: falhar aqui não pode derrubar a indexação — o TTL cobre."""
    global _local_gen
    _local_gen += 1
    client = _get_redis()
    if client is None:
        return
    try:
        client.incr(_GEN_KEY)
    except Exception as exc:
        _drop_redis(exc)


def _index_generation() -> str:
    client = _get_redis()
    if client is not None:
        try:
            return f"r{client.get(_GEN_KEY) or 0}"
        except Exception as exc:
            _drop_redis(exc)
    return f"m{_local_gen}"


def _fingerprint() -> str:
    """Tudo que muda a síntese para a mesma pergunta além dos parâmetros da requisição."""
    from backend.services import summary_prompts, summary_service

    parts = [
        str(_SCHEMA_VERSION), LLM_SUMMARY_MODEL, QDRANT_COLLECTION,
        summary_prompts.BASIC_SUMMARY_SYSTEM_PROMPT, summary_prompts.OUTPUT_ONLY_REMINDER,
        str(summary_service.MAX_CHUNKS_PER_DOCUMENT), str(summary_service.SNIPPET_MAX_CHARS),
    ]
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]


def normalize_query(query: str) -> str:
    """Espaços colapsados: "  a   b " e "a b" são a mesma pergunta. Caixa preservada
    — o embedding (e, portanto, o retrieval) distingue maiúsculas."""
    return " ".join((query or "").split())


class SummaryCache:
    """Cache de SummaryResponse. Métodos síncronos (Redis bloqueante, com timeout
    curto): o serviço os chama via threadpool, fora do event loop."""

    def __init__(
        self,
        *,
        ttl_seconds: int = SUMMARY_CACHE_TTL_SECONDS,
        max_entries: int = SUMMARY_CACHE_MAX_ENTRIES,
    ):
        self._ttl = ttl_seconds
        self._max = max_entries
        self._mem: "OrderedDict[str, tuple[float, str]]" = OrderedDict()
        self._mem_lock = threading.Lock()
        self._fp: Optional[str] = None

    def key(
        self, query: str, *, limit: int, type: str, language: str,
        documents: Sequence[DocumentRef], roles: Optional[Sequence[str]] = None,
    ) -> str:
        """Chave da síntese. `documents` já normalizada, e na ORDEM enviada: a ordem
        define a numeração [N]. O handle entra porque é ecoado em applied_filters.
        `roles` (papéis do 1º filtro) entra ordenado — focos diferentes dão sínteses
        diferentes e não podem compartilhar entrada."""
        if self._fp is None:
            self._fp = _fingerprint()
        ident = {
            "q": normalize_query(query), "limit": limit, "type": type,
            "lang": (language or "").strip(),
            "docs": [[d.uuid, d.handle] for d in documents],
            "roles": sorted(roles) if roles else None,
            "fp": self._fp, "gen": _index_generation(),
        }
        raw = json.dumps(ident, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return _RESP_PREFIX + hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[SummaryResponse]:
        raw = None
        client = _get_redis()
        if client is not None:
            try:
                raw = client.get(key)
            except Exception as exc:
                _drop_redis(exc)
                client = None
        if client is None:
            raw = self._mem_get(key)
        if not raw:
            return None
        try:
            return SummaryResponse.model_validate_json(raw)
        except Exception as exc:  # entrada de outra versão do schema: trata como miss
            log.warning("summary_cache: entrada ilegível em %s (%s) — ignorada.", key, exc)
            return None

    def set(self, key: str, response: SummaryResponse) -> None:
        raw = response.model_dump_json()
        client = _get_redis()
        if client is not None:
            try:
                client.set(key, raw, ex=self._ttl)
                return
            except Exception as exc:
                _drop_redis(exc)
        self._mem_set(key, raw)

    def _mem_get(self, key: str) -> Optional[str]:
        with self._mem_lock:
            item = self._mem.get(key)
            if item is None:
                return None
            expires, raw = item
            if expires < time.monotonic():
                del self._mem[key]
                return None
            self._mem.move_to_end(key)
            return raw

    def _mem_set(self, key: str, raw: str) -> None:
        with self._mem_lock:
            self._mem[key] = (time.monotonic() + self._ttl, raw)
            self._mem.move_to_end(key)
            while len(self._mem) > self._max:
                self._mem.popitem(last=False)
