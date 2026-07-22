"""Testes do TokenCounter (§32: contagem de tokens, fallback do tokenizer)."""

from backend.indexing.token_counter import (
    WhitespaceTokenCounter,
    build_token_counter,
    TokenizerUnavailableError,
)


def test_whitespace_count():
    tc = WhitespaceTokenCounter()
    assert tc.count("") == 0
    assert tc.count("uma duas tres") == 3
    assert tc.count("  espaços   múltiplos  ") == 2


def test_whitespace_split_no_word_break():
    tc = WhitespaceTokenCounter()
    parts = tc.split("a b c d e", 2)
    assert parts == ["a b", "c d", "e"]
    # nenhuma palavra é quebrada
    assert all(" " in p or p in {"a", "b", "c", "d", "e"} for p in parts)


def test_whitespace_truncate():
    tc = WhitespaceTokenCounter()
    assert tc.truncate("a b c d", 2) == "a b"
    assert tc.truncate("a b", 5) == "a b"


def test_fallback_when_model_missing():
    # modelo inexistente + fallback habilitado → WhitespaceTokenCounter
    tc = build_token_counter("modelo/inexistente-xyz", fallback_enabled=True)
    assert isinstance(tc, WhitespaceTokenCounter)


def test_fallback_disabled_raises():
    try:
        build_token_counter("modelo/inexistente-xyz", fallback_enabled=False)
    except TokenizerUnavailableError:
        return
    raise AssertionError("esperava TokenizerUnavailableError")
