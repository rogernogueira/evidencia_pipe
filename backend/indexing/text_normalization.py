"""Normalização de texto para busca/embedding (política v2, §8).

Mantêm-se DUAS versões do texto: `raw_text` (exatamente como o MinerU extraiu) e
`normalized_text` (limpo para busca). Este módulo produz o segundo.

A normalização PODE (§8): remover caracteres de controle/invisíveis, normalizar
espaços, juntar palavras hifenizadas na quebra de linha, corrigir espaço antes de
pontuação, normalizar aspas/travessões, eliminar quebras artificiais.

A normalização NÃO PODE (§8): alterar valores numéricos, datas, moedas; reescrever
conteúdo; completar partes ausentes. Por isso a correção de "espaço antes de
pontuação" só age quando há uma LETRA antes do sinal (nunca um dígito) — protegendo
números como `1 . 713` de virarem `1.713`.

Módulo puro (sem GPU, sem I/O).
"""

from __future__ import annotations

import re
import unicodedata

# Invisíveis a remover: soft hyphen, zero-width space/non-joiner/joiner, word joiner, BOM.
_INVISIBLE_RE = re.compile("[­​‌‍⁠﻿]")
# Espaços "exóticos" → espaço comum (nbsp, thin/hair/en/em/figure spaces, ideográfico).
# Preserva \n e \t (tratados à parte).
_WEIRD_SPACE_RE = re.compile("[            　]")
# Hifenização na quebra de linha: "pala-\nvra" → "palavra" (letra-hífen-quebra-minúscula).
_DEHYPHEN_RE = re.compile(r"([A-Za-zÀ-ÿ])-\n(?=[a-zà-ÿ])")
# Espaço antes de pontuação, SÓ quando precedido por LETRA (protege números, §8).
_SPACE_BEFORE_PUNCT_RE = re.compile(r"([A-Za-zÀ-ÿ])\s+([,.;:!?])")
# Aspas/travessões unicode → formas canônicas (sem remover, só uniformizar).
_QUOTE_MAP = {
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "‘": "'", "’": "'", "‚": "'", "″": '"', "′": "'",
}


def _strip_control(text: str) -> str:
    """Remove caracteres de controle (categoria Cc), preservando \\n e \\t."""
    return "".join(
        ch for ch in text
        if ch in ("\n", "\t") or unicodedata.category(ch) != "Cc"
    )


def normalize_text(text: str, *, collapse_newlines: bool = True) -> str:
    """Normaliza um texto para busca/embedding (§8), preservando números/datas/moedas.

    `collapse_newlines=True` (parágrafos): quebras internas viram espaço. Use
    `False` para blocos onde a quebra é estrutural (listas)."""
    if not text:
        return ""
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = _INVISIBLE_RE.sub("", t)
    t = _DEHYPHEN_RE.sub(r"\1", t)          # junta palavra quebrada por hífen na linha
    for src, dst in _QUOTE_MAP.items():
        t = t.replace(src, dst)
    t = _WEIRD_SPACE_RE.sub(" ", t)
    t = _strip_control(t)
    if collapse_newlines:
        t = t.replace("\n", " ")
    t = re.sub(r"[ \t]+", " ", t)           # colapsa espaços/tabs
    t = re.sub(r" *\n *", "\n", t)          # limpa espaços em torno de \n preservado
    t = _SPACE_BEFORE_PUNCT_RE.sub(r"\1\2", t)
    return t.strip()
