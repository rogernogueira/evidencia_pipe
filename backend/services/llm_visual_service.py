"""Gate de LLM para conteúdo visual (política v2, §11/§12) — OPCIONAL e DESACOPLADO.

Mesmo padrão do `llm_enrich_service`: endpoint OpenAI-compatible via LLM_ENRICH_*.
Só age quando `CHUNK_VISUAL_LLM=true` E há chave (`is_available()`), senão é no-op e a
heurística determinística de `visual_validation` decide sozinha.

  - `judge_chart(caption, context, extracted)` → (confidence, reason): juiz de coerência
    do dado extraído do gráfico (§11) — refina a heurística geográfica;
  - `describe_image(caption, content)` → descrição textual recuperável de um diagrama (§12).

NÃO altera números/valores; produz apenas metadados/descrição (§8).
"""
from __future__ import annotations

import json

from backend.core import config as cfg
from backend.core.logger import log

_client = None

_CHART_SYS = (
    "Você valida a COERÊNCIA de dados extraídos de um gráfico em relatórios de políticas "
    "públicas. Dada a legenda, o contexto e os dados extraídos (possivelmente errados pela "
    "leitura automática), decida se os dados são coerentes com a legenda. Responda SÓ JSON: "
    "{\"confidence\": number 0..1, \"reason\": string curta}. confidence baixo (<0.5) se os "
    "dados contradizem a legenda (ex.: legenda fala de estados do Brasil e os rótulos citam "
    "outros países)."
)
_IMG_SYS = (
    "Você descreve, em uma frase objetiva em português, o conteúdo recuperável de um "
    "diagrama/figura de relatório (para busca semântica). Não invente dados. Responda SÓ "
    "JSON: {\"description\": string}."
)


def is_available() -> bool:
    """True se o gate está LIGADO e há chave configurada."""
    return bool(cfg.CHUNK_VISUAL_LLM and cfg.LLM_ENRICH_API_KEY)


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI(api_key=cfg.LLM_ENRICH_API_KEY, base_url=cfg.LLM_ENRICH_BASE_URL,
                         timeout=cfg.LLM_ENRICH_TIMEOUT_SECONDS or None)
    return _client


def _call(system: str, user: str) -> dict:
    resp = _get_client().chat.completions.create(
        model=cfg.LLM_ENRICH_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format={"type": "json_object"}, temperature=0.0,
    )
    return json.loads(resp.choices[0].message.content or "{}")


def judge_chart(caption: str, context: str, extracted: str) -> tuple[float, str] | None:
    """Refina a confiança de coerência do gráfico (§11). None se indisponível/erro."""
    if not is_available():
        return None
    try:
        data = _call(_CHART_SYS, f"Legenda: {caption}\nContexto: {context}\nDados: {extracted}")
        conf = float(data.get("confidence"))
        return max(0.0, min(1.0, conf)), str(data.get("reason") or "llm")
    except Exception as exc:  # noqa: BLE001
        log.warning("llm_visual.judge_chart falhou (best-effort): %s", exc)
        return None


def describe_image(caption: str, content: str) -> str | None:
    """Descrição textual de um diagrama/figura (§12). None se indisponível/erro."""
    if not is_available():
        return None
    try:
        data = _call(_IMG_SYS, f"Legenda: {caption}\nConteúdo bruto: {content[:2000]}")
        desc = (data.get("description") or "").strip()
        return desc or None
    except Exception as exc:  # noqa: BLE001
        log.warning("llm_visual.describe_image falhou (best-effort): %s", exc)
        return None
