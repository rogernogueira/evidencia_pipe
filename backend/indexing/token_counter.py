"""Abstração de contagem/divisão por tokens para o chunking estrutural.

O `StructuralTokenChunker` controla o tamanho dos chunks em TOKENS do tokenizer do
modelo de embeddings (BGE-M3), não em caracteres. Este módulo isola essa dependência:

  - `TokenCounter` (Protocol): count / split / truncate;
  - `HFTokenCounter`: usa `transformers.AutoTokenizer` do modelo configurado. Carrega
    SOMENTE o tokenizer (arquivos leves já em cache), em CPU. NÃO instancia o
    `BGEM3FlagModel` nem toca a GPU — logo NÃO adquire o lock global da GPU;
  - `WhitespaceTokenCounter`: fallback determinístico (split por whitespace) usado
    quando o tokenizer HF não está disponível e `CHUNK_TOKENIZER_FALLBACK_ENABLED`.

A contagem é a operação quente do chunking (chamada por bloco/parágrafo/sentença),
por isso há um cache LRU limitado por instância.
"""

from __future__ import annotations

import re
import threading
from collections import OrderedDict
from typing import Optional, Protocol, runtime_checkable

from backend.core import config as settings
from backend.core.logger import log


class TokenizerUnavailableError(RuntimeError):
    """O tokenizer configurado não pôde ser carregado e o fallback está desabilitado."""


@runtime_checkable
class TokenCounter(Protocol):
    """Contrato mínimo de tokenização usado pelo chunker."""

    name: str

    def count(self, text: str) -> int: ...

    def split(self, text: str, max_tokens: int) -> list[str]: ...

    def truncate(self, text: str, max_tokens: int) -> str: ...


_WORD_RE = re.compile(r"\S+")


class WhitespaceTokenCounter:
    """Fallback: aproxima tokens por "palavras" (split de whitespace).

    Determinístico e sem dependências. Serve para manter o pipeline funcional quando
    o tokenizer HF não está disponível (ex.: ambiente offline sem cache)."""

    def __init__(self) -> None:
        self.name = "whitespace"

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(_WORD_RE.findall(text))

    def split(self, text: str, max_tokens: int) -> list[str]:
        """Divide por palavras em blocos de até `max_tokens`, sem cortar palavras."""
        if max_tokens <= 0:
            return [text] if text else []
        words = _WORD_RE.findall(text)
        if len(words) <= max_tokens:
            return [text] if text.strip() else []
        parts: list[str] = []
        for start in range(0, len(words), max_tokens):
            parts.append(" ".join(words[start:start + max_tokens]))
        return parts

    def truncate(self, text: str, max_tokens: int) -> str:
        words = _WORD_RE.findall(text)
        if len(words) <= max_tokens:
            return text
        return " ".join(words[:max_tokens])


class HFTokenCounter:
    """Contador baseado no `transformers.AutoTokenizer` do modelo configurado.

    Só o tokenizer é carregado (CPU); o modelo de embeddings NÃO é instanciado aqui.
    Cortes por token usam offsets de caractere (`return_offsets_mapping`) quando o
    tokenizer é "fast", para NUNCA cortar no meio de uma palavra além do necessário.
    """

    def __init__(self, tokenizer, name: str, cache_size: int = 4096) -> None:
        self._tok = tokenizer
        self.name = name
        self._cache: "OrderedDict[str, int]" = OrderedDict()
        self._cache_max = cache_size

    # -- count ------------------------------------------------------------
    def _encode_ids(self, text: str) -> list[int]:
        # add_special_tokens=False: contamos o conteúdo, não os [CLS]/[SEP].
        return self._tok.encode(text, add_special_tokens=False)

    def count(self, text: str) -> int:
        if not text:
            return 0
        cached = self._cache.get(text)
        if cached is not None:
            self._cache.move_to_end(text)
            return cached
        n = len(self._encode_ids(text))
        self._cache[text] = n
        if len(self._cache) > self._cache_max:
            self._cache.popitem(last=False)
        return n

    # -- split ------------------------------------------------------------
    def split(self, text: str, max_tokens: int) -> list[str]:
        """Divide `text` em fatias de até `max_tokens` tokens, alinhadas a offsets de
        caractere (fast tokenizer) para não cortar dentro de um token/palavra sempre
        que possível. Último recurso do chunker; usado só quando uma sentença isolada
        excede o limite."""
        if max_tokens <= 0 or not text:
            return [text] if text else []
        if self.count(text) <= max_tokens:
            return [text]

        try:
            enc = self._tok(text, add_special_tokens=False, return_offsets_mapping=True)
            offsets = enc["offset_mapping"]
        except (TypeError, KeyError, NotImplementedError):
            # Tokenizer "slow" (sem offsets): degrada para o fallback whitespace.
            return WhitespaceTokenCounter().split(text, max_tokens)

        parts: list[str] = []
        start_char = 0
        n = len(offsets)
        i = 0
        while i < n:
            end_idx = min(i + max_tokens, n)
            # offset final do último token da fatia
            end_char = offsets[end_idx - 1][1]
            piece = text[start_char:end_char].strip()
            if piece:
                parts.append(piece)
            start_char = end_char
            i = end_idx
        return parts or [text]

    # -- truncate ---------------------------------------------------------
    def truncate(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0 or not text:
            return ""
        if self.count(text) <= max_tokens:
            return text
        try:
            enc = self._tok(text, add_special_tokens=False, return_offsets_mapping=True)
            offsets = enc["offset_mapping"]
        except (TypeError, KeyError, NotImplementedError):
            return WhitespaceTokenCounter().truncate(text, max_tokens)
        if len(offsets) <= max_tokens:
            return text
        end_char = offsets[max_tokens - 1][1]
        return text[:end_char].strip()


# ---------------------------------------------------------------------------
# Fábrica / singleton (thread-safe).
# ---------------------------------------------------------------------------

_counter: Optional[TokenCounter] = None
_lock = threading.Lock()


def build_token_counter(
    model_name: Optional[str] = None,
    *,
    use_fast: Optional[bool] = None,
    fallback_enabled: Optional[bool] = None,
) -> TokenCounter:
    """Constrói o TokenCounter a partir das configs do projeto.

    Tenta carregar o `AutoTokenizer` do modelo (CPU). Se falhar e o fallback estiver
    habilitado, retorna `WhitespaceTokenCounter`; senão, levanta
    `TokenizerUnavailableError`.
    """
    model_name = model_name or settings.CHUNK_TOKENIZER_MODEL
    use_fast = settings.CHUNK_TOKENIZER_USE_FAST if use_fast is None else use_fast
    fallback_enabled = (
        settings.CHUNK_TOKENIZER_FALLBACK_ENABLED if fallback_enabled is None else fallback_enabled
    )
    try:
        # Import tardio: mantém o módulo importável sem transformers instalado.
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast)
        log.info("[chunk] tokenizer '%s' carregado (fast=%s, CPU).", model_name, use_fast)
        return HFTokenCounter(tok, name=model_name)
    except Exception as exc:
        if fallback_enabled:
            log.warning(
                "[chunk] tokenizer '%s' indisponível (%s) — usando fallback whitespace.",
                model_name, exc,
            )
            return WhitespaceTokenCounter()
        raise TokenizerUnavailableError(
            f"não foi possível carregar o tokenizer '{model_name}' e o fallback está desabilitado: {exc}"
        ) from exc


def get_token_counter() -> TokenCounter:
    """Singleton lazy do TokenCounter (reutilizado por chunker/scripts/testes)."""
    global _counter
    if _counter is None:
        with _lock:
            if _counter is None:
                _counter = build_token_counter()
    return _counter


def set_token_counter(counter: Optional[TokenCounter]) -> None:
    """Injeção para testes (contador fake/determinístico)."""
    global _counter
    _counter = counter
