"""Adapter do BAAI/bge-m3 servido pela API vLLM (dense + sparse).

O modelo NÃO é mais carregado no processo do backend: a inferência acontece nos
contêineres `vllm-bge-m3` (dense) e `vllm-bge-m3-sparse` (sparse). O backend só
tokeniza, faz HTTP e monta os vetores no formato que o Qdrant espera.

Por que DOIS endpoints
----------------------
O vLLM implementa o bge-m3 como `BgeM3EmbeddingModel`, que expõe quatro pooling
tasks: `embed` (CLS + L2, o vetor denso), `token_classify` (relu(sparse_linear·h)
por token, os `lexical_weights`), `token_embed` (ColBERT) e a combinada
`embed&token_classify`. A combinada só existe na API *offline* — sobre HTTP o
servidor não registra IOProcessor para ela (v0.26.0) e o endpoint responde 500.
Cada servidor HTTP também fixa UMA task (`--pooler-config.task`). Logo: um
contêiner para o denso, outro para o esparso. O modelo é pequeno (~2,3 GB fp16),
então duas réplicas custam pouco de VRAM e ainda respondem em paralelo.

Paridade com o FlagEmbedding
----------------------------
O denso vem de `/v1/embeddings` (CLS + normalização, igual ao FlagEmbedding). O
esparso vem de `/pooling` com `task=token_classify`: o vLLM já aplica `relu` e
descarta BOS/EOS, devolvendo um peso por token restante. Aqui reconstruímos o
dicionário `{token_id: peso}` com a mesma regra do FlagEmbedding — descarta
tokens especiais e pesos <= 0, e mantém o MAIOR peso quando o token se repete.

Para o alinhamento peso↔token ser exato, tokenizamos localmente e enviamos os
IDS (não o texto) para os dois endpoints: o servidor pontua exatamente os tokens
que conhecemos, sem depender de retokenização nem de truncamento do lado dele.

Sem fallback local: se a API estiver fora, as chamadas levantam
`EmbeddingBackendError`. Falha explícita é preferível a indexar silenciosamente
com um backend diferente do usado nas queries.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import httpx
from unidecode import unidecode

from backend.core import config as settings
from backend.core.config import DENSE_MODEL
from backend.core.logger import log

# Tokens que nunca entram no vetor esparso — os mesmos quatro que o FlagEmbedding
# ignora em `_process_token_weights`. No XLM-R do bge-m3 são <s>=0, <pad>=1,
# </s>=2, <unk>=3, mas lemos do tokenizer em vez de fixar os IDs.
_UNUSED_SPECIAL_TOKENS = ("cls_token", "eos_token", "pad_token", "unk_token")


class EmbeddingBackendError(RuntimeError):
    """A API de embedding não respondeu ou respondeu fora do contrato."""


# --------------------------------------------------------------------------- #
# Tokenizer local (só tokeniza — não carrega pesos do modelo)
# --------------------------------------------------------------------------- #
_TOKENIZER: Any = None
_SPECIAL_TOKEN_IDS: frozenset = frozenset()
_TOKENIZER_LOCK = threading.Lock()


def _get_tokenizer():
    """Carrega (uma vez) o tokenizer do bge-m3. É o `tokenizer.json` do mesmo
    repositório HF do modelo — alguns MB, sem pesos, sem GPU."""
    global _TOKENIZER, _SPECIAL_TOKEN_IDS
    if _TOKENIZER is None:
        with _TOKENIZER_LOCK:
            if _TOKENIZER is None:
                from transformers import AutoTokenizer

                log.info("[embed] carregando tokenizer '%s'…", DENSE_MODEL)
                tokenizer = AutoTokenizer.from_pretrained(DENSE_MODEL, use_fast=True)
                special_map = tokenizer.special_tokens_map
                _SPECIAL_TOKEN_IDS = frozenset(
                    tokenizer.convert_tokens_to_ids(special_map[name])
                    for name in _UNUSED_SPECIAL_TOKENS
                    if name in special_map
                )
                _TOKENIZER = tokenizer
    return _TOKENIZER


def _encode_ids(texts: Sequence[str]) -> List[List[int]]:
    """Tokeniza com tokens especiais e trunca no limite do servidor."""
    tok = _get_tokenizer()
    enc = tok(
        list(texts),
        add_special_tokens=True,
        truncation=True,
        max_length=settings.EMBED_API_MAX_TOKENS,
    )
    return enc["input_ids"]


# --------------------------------------------------------------------------- #
# Clientes HTTP (um por endpoint, com pool de conexões reaproveitado)
# --------------------------------------------------------------------------- #
_CLIENTS: Dict[str, httpx.Client] = {}
_CLIENTS_LOCK = threading.Lock()


def _client(base_url: str) -> httpx.Client:
    client = _CLIENTS.get(base_url)
    if client is None:
        with _CLIENTS_LOCK:
            client = _CLIENTS.get(base_url)
            if client is None:
                client = httpx.Client(
                    base_url=base_url,
                    timeout=httpx.Timeout(settings.EMBED_API_TIMEOUT_SECONDS, connect=10.0),
                    limits=httpx.Limits(max_connections=8, max_keepalive_connections=8),
                )
                _CLIENTS[base_url] = client
    return client


def _get(base_url: str, path: str) -> dict:
    """GET sem retry (só o health check usa — a resposta rápida importa mais)."""
    try:
        response = _client(base_url).get(path)
    except httpx.HTTPError as exc:
        raise EmbeddingBackendError(f"{base_url}{path} inacessível: {exc}") from exc
    if response.status_code != 200:
        raise EmbeddingBackendError(
            f"{base_url}{path} respondeu {response.status_code}: {response.text[:200]}"
        )
    return response.json()


def _post(base_url: str, path: str, payload: dict) -> dict:
    """POST com retry em erro de transporte/5xx. 4xx não é retentado (é contrato)."""
    attempts = max(1, settings.EMBED_API_MAX_RETRIES + 1)
    last_error: Optional[Exception] = None
    for attempt in range(attempts):
        try:
            response = _client(base_url).post(path, json=payload)
        except httpx.HTTPError as exc:
            last_error = exc
            log.warning("[embed] %s%s falhou (%s), tentativa %d/%d",
                        base_url, path, type(exc).__name__, attempt + 1, attempts)
            continue
        if response.status_code == 200:
            return response.json()
        detail = response.text[:400]
        if response.status_code < 500:
            raise EmbeddingBackendError(
                f"{base_url}{path} respondeu {response.status_code}: {detail}"
            )
        last_error = EmbeddingBackendError(f"HTTP {response.status_code}: {detail}")
        log.warning("[embed] %s%s respondeu %d, tentativa %d/%d",
                    base_url, path, response.status_code, attempt + 1, attempts)
    raise EmbeddingBackendError(
        f"API de embedding indisponível em {base_url}{path}: {last_error}"
    ) from last_error


# --------------------------------------------------------------------------- #
# Chamadas às duas tasks
# --------------------------------------------------------------------------- #
def _request_dense(batch_ids: List[List[int]]) -> List[List[float]]:
    data = _post(
        settings.EMBED_API_URL,
        "/v1/embeddings",
        {"model": settings.EMBED_API_MODEL, "input": batch_ids},
    )
    try:
        items = sorted(data["data"], key=lambda d: d["index"])
        vectors = [item["embedding"] for item in items]
    except (KeyError, TypeError) as exc:
        raise EmbeddingBackendError(f"Resposta dense fora do contrato: {exc}") from exc
    if len(vectors) != len(batch_ids):
        raise EmbeddingBackendError(
            f"Resposta dense com {len(vectors)} vetores para {len(batch_ids)} entradas."
        )
    return vectors


def _request_token_weights(batch_ids: List[List[int]]) -> List[List[float]]:
    data = _post(
        settings.EMBED_API_SPARSE_URL,
        "/pooling",
        {"model": settings.EMBED_API_MODEL, "input": batch_ids, "task": "token_classify"},
    )
    try:
        items = sorted(data["data"], key=lambda d: d["index"])
        raw = [item["data"] for item in items]
    except (KeyError, TypeError) as exc:
        raise EmbeddingBackendError(f"Resposta sparse fora do contrato: {exc}") from exc
    if len(raw) != len(batch_ids):
        raise EmbeddingBackendError(
            f"Resposta sparse com {len(raw)} itens para {len(batch_ids)} entradas."
        )
    # O pooler faz squeeze(-1), mas aceitamos [[w], [w], …] por robustez.
    return [[float(w[0]) if isinstance(w, (list, tuple)) else float(w) for w in item]
            for item in raw]


def _to_lexical_weights(token_ids: List[int], weights: List[float]) -> Dict[str, float]:
    """Monta `{token_id: peso}` a partir dos pesos por token (BOS/EOS já removidos
    pelo servidor). Mesma regra do FlagEmbedding: sem tokens especiais, sem peso
    <= 0, e o maior peso vence quando o token se repete."""
    inner = token_ids[1:-1] if len(token_ids) >= 2 else []
    if len(weights) != len(inner):
        raise EmbeddingBackendError(
            f"Desalinhamento sparse: {len(weights)} pesos para {len(inner)} tokens "
            "(o servidor deve rodar com --pooler-config.task token_classify)."
        )
    result: Dict[str, float] = {}
    for token_id, weight in zip(inner, weights):
        if token_id in _SPECIAL_TOKEN_IDS or weight <= 0:
            continue
        key = str(token_id)
        if weight > result.get(key, 0.0):
            result[key] = weight
    return result


def _embed_batch(texts: Sequence[str]) -> Tuple[List[List[float]], List[Dict[str, float]]]:
    batch_ids = _encode_ids(texts)
    dense = _request_dense(batch_ids)
    token_weights = _request_token_weights(batch_ids)
    lexical = [_to_lexical_weights(ids, weights)
               for ids, weights in zip(batch_ids, token_weights)]
    return dense, lexical


# Cache LRU do embedding de query (dense + sparse), chaveado por (query, normalize).
# Permite alternar filtros sem re-embedar a mesma query — só o ANN filtrado roda.
_QUERY_CACHE: "OrderedDict[Tuple[str, bool], Tuple[List[float], Dict[str, float]]]" = OrderedDict()
_QUERY_CACHE_MAX = 256


def fold_accents(text: str) -> str:
    """Remove acentos/diacríticos (unidecode). Usado para normalizar entrada antes
    do embedding, tornando a busca lexical (sparse) robusta a acento. Aplicar de forma
    simétrica: o mesmo texto precisa ser foldado na indexação e na query."""
    return unidecode(text)


class BgeM3EmbedderService:
    """Service Adapter para o bge-m3 servido por vLLM (Dense + Sparse).

    Isola o transporte HTTP do resto da aplicação, fornecendo métodos
    padronizados para geração de embeddings de queries e documentos em batch.
    """

    def __init__(self):
        pass

    # ------------------------------------------------------------------ #
    # Saúde do backend
    # ------------------------------------------------------------------ #
    @staticmethod
    def endpoints() -> Tuple[str, str]:
        """(url do denso, url do esparso)."""
        return settings.EMBED_API_URL, settings.EMBED_API_SPARSE_URL

    def health_check(self, raise_on_error: bool = False) -> bool:
        """Verifica se os dois endpoints estão no ar e servindo o modelo esperado."""
        try:
            for base_url in self.endpoints():
                data = _get(base_url, "/v1/models")
                served = {m.get("id") for m in data.get("data", [])}
                if settings.EMBED_API_MODEL not in served:
                    raise EmbeddingBackendError(
                        f"{base_url} serve {sorted(served)}, esperado "
                        f"{settings.EMBED_API_MODEL!r}."
                    )
        except EmbeddingBackendError as exc:
            if raise_on_error:
                raise
            log.warning("[embed] backend indisponível: %s", exc)
            return False
        return True

    def is_loaded(self) -> bool:
        """Compatibilidade: 'pronto' agora significa 'API respondendo'."""
        return self.health_check(raise_on_error=False)

    # ------------------------------------------------------------------ #
    # Embeddings
    # ------------------------------------------------------------------ #
    def embed_query(self, query: str, normalize: bool = False) -> Tuple[List[float], Dict[str, float]]:
        """Gera embedding para uma query única (busca).

        Args:
            normalize: Se True, remove acentos (unidecode) antes de embeddar. Deve casar
                       com o `normalize` usado na indexação da collection consultada.

        Returns:
            Tuple[dense_vector, lexical_weights]
        """
        cache_key = (query, normalize)
        cached = _QUERY_CACHE.get(cache_key)
        if cached is not None:
            _QUERY_CACHE.move_to_end(cache_key)
            return cached

        text = fold_accents(query) if normalize else query
        dense, lexical = _embed_batch([text])
        result = (dense[0], lexical[0])

        _QUERY_CACHE[cache_key] = result
        if len(_QUERY_CACHE) > _QUERY_CACHE_MAX:
            _QUERY_CACHE.popitem(last=False)

        return result

    def embed_documents(
        self,
        texts: List[str],
        batch_size: int = 32,
        normalize: bool = False,
        *,
        task_id: Optional[str] = None,
        document_id: Optional[str] = None,
    ) -> Tuple[List[List[float]], List[Dict[str, float]]]:
        """Gera embeddings em batch para múltiplos documentos.

        Os textos são enviados à API em lotes de `batch_size`; o próprio vLLM
        agrupa e agenda a inferência. `task_id`/`document_id` só entram no log —
        não há mais lock de GPU a coordenar do lado do backend.

        Args:
            normalize: Se True, remove acentos (unidecode) antes de embeddar. Deve casar
                       com o `normalize` usado na query da collection.

        Returns:
            Tuple[lista_de_dense_vectors, lista_de_lexical_weights]
        """
        if not texts:
            return [], []

        if normalize:
            texts = [fold_accents(t) for t in texts]

        step = max(1, batch_size)
        log.info("[embed] vLLM: %d chunk(s) em lotes de %d (doc=%s)", len(texts), step, document_id)

        all_dense: List[List[float]] = []
        all_lexical: List[Dict[str, float]] = []
        for start in range(0, len(texts), step):
            dense, lexical = _embed_batch(texts[start:start + step])
            all_dense.extend(dense)
            all_lexical.extend(lexical)
        return all_dense, all_lexical

    def get_dense_dimension(self) -> int:
        """Sonda a API para descobrir a dimensão real do vetor denso."""
        dense, _ = _embed_batch(["dimension probe"])
        return len(dense[0])
