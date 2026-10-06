"""discourse_classify_service.py — Classificação discursiva de chunks por LLM.

Promove ao pipeline o classificador validado no experimento
(`experiments/chunk_classes/05_llm_labeler.py` + `codebook.md`): para cada chunk,
atribui um ou mais PAPÉIS DISCURSIVOS (achado, recomendacao, metodologia, …) do
codebook de 10 classes. Multirrótulo; `outro` é exclusivo.

DESACOPLADO do provedor: fala com qualquer endpoint OpenAI-compatible, reusando a
chave/endpoint do enrich (LLM_ENRICH_*). DESACOPLADO da indexação: NÃO faz parte da
chain obrigatória — roda como follow-up pós-índice (stage_classify_discourse) e grava
`discourse_role` por ponto no Qdrant via set_payload. Sem `DISCOURSE_CLASSIFY_ENABLED`
(ou sem chave) o step é pulado (is_available() → False), sem quebrar o pipeline.

Uma chamada ao LLM por chunk (tool-calling com enum das classes, reasoning off),
com concorrência limitada (DISCOURSE_CLASSIFY_CONCURRENCY). Fiel ao experimento.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional, Sequence

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from backend.core.config import (
    DISCOURSE_CLASSIFY_CONCURRENCY,
    DISCOURSE_CLASSIFY_ENABLED,
    DISCOURSE_CLASSIFY_MAX_CHARS,
    DISCOURSE_CLASSIFY_MODEL,
    LLM_ENRICH_API_KEY,
    LLM_ENRICH_BASE_URL,
    LLM_ENRICH_TIMEOUT_SECONDS,
)
from backend.core.logger import log

# Taxonomia canônica do codebook (mesma ordem de experiments/chunk_classes/heuristics.py).
CLASSES = [
    "contexto_politica", "objetivo_avaliacao", "metodologia", "achado",
    "recomendacao", "conclusao_juizo", "limitacao", "normativo_legal",
    "dado_quantitativo", "outro",
]
ALLOWED = set(CLASSES)

# Marca a origem do rótulo no payload (discourse_role_source).
SOURCE = "pipeline:discourse_classify"

_client: Optional[OpenAI] = None

SYSTEM_PROMPT = """Você é um anotador especializado em relatórios de avaliação de políticas públicas.
Classifique o TRECHO (chunk) nos papéis discursivos abaixo. É MULTIRRÓTULO: marque
todos os que se aplicam; se nenhum, use apenas "outro".

Classes:
- contexto_politica: descreve a política avaliada (objetivos, público, histórico, desenho).
- objetivo_avaliacao: o que a avaliação pretende responder; perguntas/escopo avaliativos.
- metodologia: como a avaliação foi feita (dados, fontes, desenho, técnicas, amostra).
- achado: constatação empírica sustentada por evidência produzida na avaliação.
- recomendacao: proposta de ação dirigida a um ator (verbo deôntico ou destinatário).
- conclusao_juizo: síntese/veredito sobre mérito, efetividade ou desempenho da política.
- limitacao: ressalva sobre dados, método, escopo ou generalização dos resultados.
- normativo_legal: reprodução/citação de norma ou dispositivo, sem análise própria.
- dado_quantitativo: dado numérico/tabela descritiva sem interpretação avaliativa.
- outro: administrativo, navegação, paratexto (ficha técnica, sumário, referências).

Regras: achado é constatação pontual; conclusao_juizo é síntese/veredito. Um chunk
que faz as duas coisas recebe as duas. "outro" nunca se combina com as demais.

Responda SOMENTE com JSON válido no formato:
{"labels": ["<classe>", ...], "confidence": {"<classe>": <0..1>, ...}}
Use exatamente os nomes de classe listados. `confidence` só para as classes em `labels`."""

TOOL_NAME = "rotular_chunk"

# Enum restringe os rótulos às classes válidas — a garantia mais forte disponível no
# DeepSeek (json_schema strict está indisponível lá).
TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "labels": {
            "type": "array",
            "items": {"type": "string", "enum": CLASSES},
            "description": "papéis discursivos que se aplicam ao trecho (multirrótulo)",
        },
        "confidence": {
            "type": "object",
            "additionalProperties": {"type": "number"},
            "description": "confiança 0..1 por classe marcada",
        },
    },
    "required": ["labels"],
    "additionalProperties": False,
}

_REPAIR_MSG = {
    "role": "user",
    "content": ('Sua resposta anterior não estava no formato exigido. Responda SOMENTE com '
                'JSON: {"labels": ["<classe>", ...], "confidence": {"<classe>": <0..1>}} '
                'usando exatamente as classes válidas do sistema.'),
}


class ChunkLabels(BaseModel):
    """Valida e NORMALIZA a saída do modelo. Descarta classes fora do codebook,
    trata vazio como ['outro'] e limita a confiança a [0,1]."""

    model_config = ConfigDict(extra="ignore")
    labels: list[str] = Field(default_factory=list)
    confidence: dict = Field(default_factory=dict)

    @field_validator("labels", mode="after")
    @classmethod
    def _clean_labels(cls, v: list[str]) -> list[str]:
        seen, out = set(), []
        for c in v:
            if c in ALLOWED and c not in seen:
                seen.add(c)
                out.append(c)
        # Regra do codebook: "outro" nunca se combina com as demais.
        if len(out) > 1 and "outro" in out:
            out = [c for c in out if c != "outro"]
        return out or ["outro"]

    @field_validator("confidence", mode="after")
    @classmethod
    def _clean_conf(cls, v: dict) -> dict:
        out = {}
        for k, val in (v or {}).items():
            if k in ALLOWED and isinstance(val, (int, float)):
                out[k] = max(0.0, min(1.0, float(val)))
        return out


def is_available() -> bool:
    """True se o estágio está habilitado E há chave de LLM configurada.

    ENABLED é opt-in explícito (reusa a chave do enrich, então não deve disparar
    uma chamada por chunk só porque o enrich está configurado)."""
    return bool(DISCOURSE_CLASSIFY_ENABLED and LLM_ENRICH_API_KEY)


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=LLM_ENRICH_API_KEY,
            base_url=LLM_ENRICH_BASE_URL,
            timeout=LLM_ENRICH_TIMEOUT_SECONDS or None,
        )
    return _client


def build_user_prompt(rec: dict) -> str:
    sec = rec.get("section_title") or "(sem seção)"
    ct = rec.get("content_type") or "?"
    text = (rec.get("text") or "")[:DISCOURSE_CLASSIFY_MAX_CHARS]
    return f'Seção: {sec}\nTipo de conteúdo: {ct}\n\nTRECHO:\n"""\n{text}\n"""'


def _extract_raw(resp, structured: str) -> str:
    msg = resp.choices[0].message
    if structured == "tool":
        tcs = getattr(msg, "tool_calls", None)
        if tcs:
            return tcs[0].function.arguments or "{}"
    return msg.content or "{}"


def _create(messages: list[dict], structured: str):
    """Uma chamada ao LLM. `reasoning_effort='none'` desliga o raciocínio do DeepSeek
    (tokens de raciocínio contam como saída); se o provedor recusar o parâmetro, refaz
    sem ele."""
    base: dict[str, Any] = dict(model=DISCOURSE_CLASSIFY_MODEL, messages=messages, temperature=0.0)
    if structured == "tool":
        base["tools"] = [{"type": "function", "function": {
            "name": TOOL_NAME,
            "description": "Registra os papéis discursivos do trecho.",
            "parameters": TOOL_SCHEMA,
        }}]
        base["tool_choice"] = {"type": "function", "function": {"name": TOOL_NAME}}
    else:
        base["response_format"] = {"type": "json_object"}
    try:
        return _get_client().chat.completions.create(reasoning_effort="none", **base)
    except TypeError:
        return _get_client().chat.completions.create(**base)
    except Exception as exc:  # provedor pode recusar reasoning_effort com 400
        if "reasoning" in str(exc).lower():
            return _get_client().chat.completions.create(**base)
        raise


def classify_one(rec: dict) -> dict:
    """Classifica UM chunk → {'labels': [...], 'confidence': {...}}.

    Tenta tool-calling (enum); se o parse falhar, uma tentativa de reparo; se ainda
    falhar, cai para response_format=json_object. Levanta em erro de API/parse final
    (o chamador trata como best-effort e omite o chunk)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(rec)},
    ]
    # 1) tool-calling; 2) reparo no mesmo modo; 3) fallback json_object.
    attempts = [("tool", messages), ("tool", messages + [_REPAIR_MSG]), ("json_object", messages)]
    last_exc: Optional[Exception] = None
    for structured, msgs in attempts:
        try:
            resp = _create(msgs, structured)
            m = ChunkLabels.model_validate_json(_extract_raw(resp, structured))
            return {"labels": m.labels, "confidence": m.confidence}
        except ValidationError as exc:
            last_exc = exc
            continue
    raise last_exc or RuntimeError("classificação falhou sem exceção registrada")


def classify_chunks(records: Sequence[dict]) -> dict[str, dict]:
    """Classifica vários chunks em paralelo (pool limitado por DISCOURSE_CLASSIFY_CONCURRENCY).

    `records`: itens com ao menos `point_id` e `text` (e, se houver, `section_title`,
    `content_type`). Devolve {point_id: {'labels', 'confidence'}}. Best-effort: chunks
    cuja classificação falhar são OMITIDOS (logados), nunca derrubam o lote."""
    items = [r for r in records if r.get("point_id") and (r.get("text") or "").strip()]
    if not items:
        return {}

    if not is_available():
        raise RuntimeError("DISCOURSE_CLASSIFY desabilitado ou sem chave — classificador indisponível.")

    log.info(
        "discourse classify: %d chunk(s) model=%s concorrência=%d",
        len(items), DISCOURSE_CLASSIFY_MODEL, DISCOURSE_CLASSIFY_CONCURRENCY,
    )

    def _one(rec: dict) -> tuple[str, Optional[dict]]:
        try:
            return rec["point_id"], classify_one(rec)
        except Exception as exc:  # best-effort por chunk
            log.warning("discourse classify: chunk %s falhou (%s) — omitido.", rec.get("point_id"), exc)
            return rec["point_id"], None

    out: dict[str, dict] = {}
    workers = max(1, DISCOURSE_CLASSIFY_CONCURRENCY)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for point_id, result in pool.map(_one, items):
            if result is not None:
                out[point_id] = result
    log.info("discourse classify: %d/%d chunk(s) rotulados.", len(out), len(items))
    return out
