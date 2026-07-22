"""llm_enrich_service.py — Step de enriquecimento de metadados por LLM.

DESACOPLADO do provedor: fala com qualquer endpoint OpenAI-compatible (DeepSeek,
OpenAI, etc.) configurado via LLM_ENRICH_* (ver backend/core/config.py). Recebe o
markdown extraído pelo MinerU e extrai metadados candidatos estruturados (título,
ano, instituição, resumo, ODS, achados, recomendações, etc.), com graus de
confiança.

DESACOPLADO da indexação: NÃO faz parte da chain obrigatória. Roda sob demanda no
endpoint `POST /api/files/enrich/{job_id}` ou como follow-up opcional APÓS a
indexação (propaga os metadados ao Qdrant via set_payload). Sem `LLM_ENRICH_API_KEY`
(ou o legado `DEEPSEEK_API_KEY`) o step é pulado (is_available() → False), sem
quebrar o pipeline.
"""

import json
import time
from pathlib import Path

from openai import OpenAI

from backend.core.config import (
    LLM_ENRICH_API_KEY,
    LLM_ENRICH_BASE_URL,
    LLM_ENRICH_MODEL,
    LLM_ENRICH_MAX_CHARS,
    LLM_ENRICH_PROVIDER,
    LLM_ENRICH_REVIEW_THRESHOLD,
    LLM_ENRICH_TIMEOUT_SECONDS,
    OUTPUT_DIR,
)
from backend.core.logger import log
from backend.core.schemas import LlmMetadataCandidates, DocumentMetadata

_client: OpenAI | None = None

SYSTEM_PROMPT = """\
Você é um especialista em catalogação de documentos de avaliação de políticas \
públicas. Extraia metadados do documento fornecido (em Markdown) e responda \
EXCLUSIVAMENTE com um objeto JSON válido, sem texto fora do JSON.

Regras:
- Baseie-se apenas no conteúdo do documento. Se um campo não puder ser \
determinado, use null (para texto) ou lista vazia (para listas). NÃO invente.
- Distinga `ods_candidatos` (ODS explicitamente citados no texto) de \
`ods_sugeridos_por_tema` (ODS que você infere pelo tema, mesmo sem citação).
- Os campos de confiança (`confianca_*`) são números entre 0.0 e 1.0 indicando \
sua certeza sobre aquela extração.
- `revisar` = true quando houver baixa confiança, ambiguidade ou muita inferência.
- `observacoes`: aponte limitações, campos ausentes, inferências feitas ou se o \
texto parecia truncado/ruidoso.
- Responda em português.

Chaves esperadas no JSON (use exatamente estes nomes):
titulo_candidato, ano_candidato, instituicao_candidata, tipo_documento_candidato, \
area_tematica_candidata, resumo_candidato, palavras_chave_candidatas (lista), \
abrangencia_territorial_candidata, periodo_avaliado_candidato, \
programa_politica_candidato, tipo_avaliacao_candidato, \
criterios_avaliacao_candidatos (lista), ods_candidatos (lista), \
ods_sugeridos_por_tema (lista), metodologia_candidata, \
achados_principais_candidatos (lista), recomendacoes_principais_candidatas (lista), \
confianca_titulo, confianca_ano, confianca_instituicao, confianca_resumo, \
confianca_programa_politica, confianca_ods, revisar, observacoes."""


def is_available() -> bool:
    """True se a chave da API do provedor LLM está configurada."""
    return bool(LLM_ENRICH_API_KEY)


def provider_name() -> str:
    """Nome do provedor LLM em uso (ex.: 'deepseek', 'openai')."""
    return LLM_ENRICH_PROVIDER


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=LLM_ENRICH_API_KEY,
            base_url=LLM_ENRICH_BASE_URL,
            timeout=LLM_ENRICH_TIMEOUT_SECONDS or None,
        )
    return _client


def _derive_revisar(cand: LlmMetadataCandidates) -> bool:
    """Se a LLM não decidiu `revisar`, deriva da média das confianças informadas."""
    if cand.revisar is not None:
        return cand.revisar
    confs = [
        c for c in (
            cand.confianca_titulo, cand.confianca_ano, cand.confianca_instituicao,
            cand.confianca_resumo, cand.confianca_programa_politica, cand.confianca_ods,
        ) if c is not None
    ]
    if not confs:
        return True  # sem sinal de confiança → melhor revisar
    return (sum(confs) / len(confs)) < LLM_ENRICH_REVIEW_THRESHOLD


def enrich_markdown(
    markdown: str,
    doc_id: str,
    uuid: str = "",
    arquivo_json: str = "",
    *,
    raw_sink: list[str] | None = None,
) -> DocumentMetadata:
    """Aciona a LLM sobre o markdown e devolve o metadado completo.

    Levanta RuntimeError se a chave não estiver configurada, e propaga erros da
    API/parse ao chamador (o pipeline trata como best-effort).

    Se `raw_sink` for passado, a resposta bruta (string JSON) da LLM é anexada a ele
    — usado pelo pipeline v2 para persistir `raw_response.json` no MinIO quando
    ARTIFACT_KEEP_LLM_RAW_RESPONSE=true (a resposta NÃO trafega pela chain).
    """
    if not is_available():
        raise RuntimeError("LLM_ENRICH_API_KEY não configurada — step de LLM indisponível.")

    text = markdown or ""
    truncated = len(text) > LLM_ENRICH_MAX_CHARS
    if truncated:
        text = text[:LLM_ENRICH_MAX_CHARS]

    log.info(
        "LLM enrich: doc_id=%s provider=%s model=%s chars=%d%s",
        doc_id, LLM_ENRICH_PROVIDER, LLM_ENRICH_MODEL, len(text), " (truncado)" if truncated else "",
    )
    t0 = time.perf_counter()
    resp = _get_client().chat.completions.create(
        model=LLM_ENRICH_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Documento (Markdown):\n\n{text}"},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    elapsed = round(time.perf_counter() - t0, 2)

    raw = resp.choices[0].message.content or "{}"
    if raw_sink is not None:
        raw_sink.append(raw)
    cand = LlmMetadataCandidates.model_validate_json(raw)

    revisar = _derive_revisar(cand)
    observacoes = cand.observacoes
    if truncated:
        aviso = f"[documento truncado em {LLM_ENRICH_MAX_CHARS} caracteres para a LLM]"
        observacoes = f"{observacoes} {aviso}".strip() if observacoes else aviso

    usage = getattr(resp, "usage", None)
    total_tokens = getattr(usage, "total_tokens", None) if usage else None

    meta = DocumentMetadata(
        **cand.model_dump(),
        doc_id=doc_id,
        uuid=uuid or None,
        arquivo_json=arquivo_json or None,
        llm_utilizada=LLM_ENRICH_MODEL,
        quantidade_tokens=total_tokens,
        tempo_processamento=elapsed,
    )
    meta.revisar = revisar
    meta.observacoes = observacoes
    log.info(
        "LLM enrich concluído: doc_id=%s tokens=%s tempo=%ss revisar=%s",
        doc_id, total_tokens, elapsed, revisar,
    )
    return meta


def _metadata_path(md_path: Path) -> Path:
    """JSON de metadados fica ao lado do markdown: <dir>/<doc>_metadata_llm.json."""
    return md_path.parent / f"{md_path.stem}_metadata_llm.json"


def enrich_job(job_id: str, uuid: str = "") -> DocumentMetadata:
    """Localiza o markdown do job, roda a LLM e salva o JSON em output/.

    Levanta FileNotFoundError se o markdown do job não existe.
    """
    # Import tardio evita ciclo (job_store não depende deste módulo, mas mantém coeso).
    from backend.services.job_store import find_markdown

    md_path = find_markdown(job_id)
    if md_path is None:
        raise FileNotFoundError(f"Markdown do job '{job_id}' não encontrado.")

    out_path = _metadata_path(md_path)
    arquivo_json = f"/output/{out_path.relative_to(OUTPUT_DIR).as_posix()}"

    meta = enrich_markdown(
        md_path.read_text(encoding="utf-8"),
        doc_id=job_id,
        uuid=uuid,
        arquivo_json=arquivo_json,
    )
    out_path.write_text(
        json.dumps(meta.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info("Metadados LLM salvos em %s", out_path)

    # Propaga os metadados aos pontos já indexados no Qdrant (merge via set_payload,
    # sem re-embedar). No pipeline automático o enrich roda antes da indexação, então
    # aqui não há pontos ainda (no-op) e a junção ocorre na indexação; no enrich manual
    # (doc já indexado) este passo atualiza os pontos existentes. Best-effort.
    try:
        from backend.indexing.index_chunks import sync_llm_metadata_to_qdrant

        sync_llm_metadata_to_qdrant(job_id)
    except Exception as exc:
        log.warning("Falha ao propagar metadados LLM ao Qdrant para '%s': %s", job_id, exc)

    return meta
