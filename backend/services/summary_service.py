"""summary_service.py — AI Summary síncrono sobre a busca semântica existente.

Orquestra, fora de routes/search.py, as responsabilidades do resumo:
  1. receber query e (futuramente) filtros;
  2. chamar SemanticSearch.search_points (retrieval híbrido/dense/sparse já existente)
     — uma vez, globalmente, ou uma vez POR DOCUMENTO quando o cliente envia `documents`;
  3. deduplicar (document_id + página + hash) com teto de chunks por documento;
  4. montar evidências numeradas;
  5. chamar o LLM (endpoint OpenAI-compatible, mesma config do enrich);
  6. validar as citações [N] contra as evidências enviadas;
  7. produzir os `mappings` (origem de cada evidência).

Escopo desta iteração (ver decisão): sem filtros de metadados, sem grupo
Centralised/Decentralised, apenas o resumo geral. A chamada ao LLM roda em
threadpool (run_in_threadpool) para NÃO bloquear o event loop do FastAPI — a busca
pública compartilha o mesmo processo uvicorn.
"""

import asyncio
import hashlib
import re
import time
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from fastapi.concurrency import run_in_threadpool
from openai import BadRequestError, OpenAI, UnprocessableEntityError

from backend.core.config import (
    LLM_ENRICH_API_KEY,
    LLM_ENRICH_BASE_URL,
    LLM_ENRICH_TIMEOUT_SECONDS,
    LLM_SUMMARY_DISABLE_THINKING,
    LLM_SUMMARY_MODEL,
)
from backend.core.logger import log_api
from backend.core.schemas import (
    DocumentRef,
    EvidenceMapping,
    RetrievalMetadata,
    SummaryResponse,
)
from backend.repositories.qdrant_client import SemanticSearch
from backend.services import llm_enrich_service
from backend.services.summary_prompts import (
    BASIC_SUMMARY_SYSTEM_PROMPT,
    OUTPUT_ONLY_REMINDER,
    build_basic_user_message,
)

# Diversidade documental: teto de chunks por documento nas evidências, evitando
# um top-N em que quase todos os trechos venham do mesmo PDF (§10 do plano).
# NÃO se aplica à recuperação por documento (`documents` na requisição): ali o cliente
# já escolheu os documentos, e o teto por documento é o próprio `limit` (o k) de cada
# consulta — cortar em 2 tornaria k>2 inócuo.
MAX_CHUNKS_PER_DOCUMENT = 2
# Tamanho do trecho devolvido em cada mapping (a evidência integral vai só ao LLM).
SNIPPET_MAX_CHARS = 400

_CITATION_RE = re.compile(r"\[(\d+)\]")

# --------------------------------------------------------------------------
# Saneamento do raciocínio ("thinking") que vaza no content.
#
# Modelo hybrid reasoning (DeepSeek v4, Qwen, Kimi) deveria devolver o raciocínio em
# `reasoning_content` — que este serviço simplesmente ignora, pois só lê `content`.
# Quando o provedor NÃO separa os dois, o raciocínio vem embutido no content e o que
# chega ao usuário é um `</think>` órfão seguido do rascunho, muitas vezes em chinês
# (os modelos raciocinam no idioma de treino, não no idioma pedido).
#
# Aqui o texto é saneado de forma determinística, cobrindo os três formatos que
# aparecem na prática: bloco fechado, fechamento órfão (o abridor foi consumido pelo
# template do chat) e abertura sem fechamento (resposta truncada no meio do
# raciocínio). Ver também LLM_SUMMARY_MODEL, que resolve na origem.
# --------------------------------------------------------------------------
_THINK_TAGS = "think|thinking|reasoning|thought"
_THINK_BLOCK_RE = re.compile(rf"<\s*({_THINK_TAGS})\s*>.*?<\s*/\s*\1\s*>", re.I | re.S)
_THINK_OPEN_RE = re.compile(rf"<\s*(?:{_THINK_TAGS})\s*>", re.I)
_THINK_CLOSE_RE = re.compile(rf"<\s*/\s*(?:{_THINK_TAGS})\s*>", re.I)
# Delimitadores não-XML usados por alguns modelos (Kimi, modelos com template próprio).
_THINK_MARKERS = (("◁think▷", "◁/think▷"), ("<|begin_of_thought|>", "<|end_of_thought|>"))
# Ideogramas CJK + pontuação/formas de largura total: o rastro típico do raciocínio
# vazado. Serve de SINAL (log/retry), não de censura — texto legítimo não é mutilado.
_CJK_RE = re.compile(r"[\u3000-\u303f\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff00-\uffef]")
# Acima desta fração de CJK a resposta é considerada contaminada e refeita UMA vez.
_CJK_MAX_RATIO = 0.02
_CJK_LANGUAGES = ("zh", "ja", "ko", "cmn", "yue")

# Desliga o raciocínio na ORIGEM (ver LLM_SUMMARY_DISABLE_THINKING em core/config.py):
# sem isso o rascunho vaza no content e consome o orçamento de tokens do resumo.
# `reasoning_effort: "none"` tem o mesmo efeito no provedor em uso; ficamos com o
# `thinking`, que é o parâmetro documentado da DeepSeek.
_THINKING_OFF_BODY = {"thinking": {"type": "disabled"}}

_client: Optional[OpenAI] = None
# None = ainda não se sabe se o provedor aceita _THINKING_OFF_BODY; False = recusou uma
# vez e não se tenta mais nesta vida do processo.
_thinking_param_ok: Optional[bool] = None


@dataclass
class _Evidence:
    """Evidência numerada: o que vai ao prompt e o que vira mapping na resposta."""
    index: int
    chunk_id: Optional[str]
    document_id: Optional[str]
    item_uuid: Optional[str]
    item_handle: Optional[str]
    section: str
    page: Optional[int]
    score: Optional[float]
    text: str


def _get_client() -> OpenAI:
    """Cliente OpenAI-compatible (DeepSeek etc.), configurado como o do enrich."""
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=LLM_ENRICH_API_KEY,
            base_url=LLM_ENRICH_BASE_URL,
            timeout=LLM_ENRICH_TIMEOUT_SECONDS or None,
        )
    return _client


def _fields(point) -> dict:
    """Extrai do ponto do Qdrant os campos usados na evidência, com fallback
    payload legado↔estrutural: document_id|doc_id, content|text, section|section_title,
    page|page_start."""
    pl = point.payload or {}
    page = pl.get("page")
    if page is None:
        page = pl.get("page_start")
    return {
        "chunk_id": pl.get("chunk_id"),
        "document_id": pl.get("document_id") or pl.get("doc_id"),
        "item_uuid": pl.get("item_uuid"),
        "item_handle": pl.get("item_handle"),
        "section": pl.get("section") or pl.get("section_title") or "",
        "page": page,
        "score": round(point.score, 4) if getattr(point, "score", None) is not None else None,
        "text": pl.get("content") or pl.get("text") or "",
    }


def _dedup_and_number(
    points: list, max_per_document: Optional[int] = MAX_CHUNKS_PER_DOCUMENT
) -> list[_Evidence]:
    """Deduplica por (document_id, página, hash do texto) e limita chunks por
    documento; numera as evidências restantes a partir de 1.

    `max_per_document=None` desliga o teto de diversidade — é o caso da recuperação
    por documento, em que cada documento já veio de uma consulta própria com o seu
    próprio limite. O dedup exato continua valendo nos dois casos."""
    seen: set = set()
    per_doc: dict = {}
    evidences: list[_Evidence] = []
    for p in points:
        f = _fields(p)
        text = f["text"]
        if not text.strip():
            continue
        h = hashlib.sha1(text.encode("utf-8")).hexdigest()
        key = (f["document_id"], f["page"], h)
        if key in seen:
            continue
        doc = f["document_id"]
        if max_per_document is not None and per_doc.get(doc, 0) >= max_per_document:
            continue
        seen.add(key)
        per_doc[doc] = per_doc.get(doc, 0) + 1
        evidences.append(_Evidence(index=len(evidences) + 1, **f))
    return evidences


def _unique_documents(documents: Optional[Iterable[DocumentRef]]) -> list[DocumentRef]:
    """Normaliza a lista recebida: descarta UUID em branco e repetido, preservando a
    ordem em que o cliente enviou (que é a ordem das evidências na resposta).

    Um UUID repetido produziria duas consultas idênticas e evidências duplicadas —
    o dedup por texto as removeria, mas o custo do retrieval já teria sido pago."""
    vistos: set = set()
    saida: list[DocumentRef] = []
    for d in documents or []:
        uuid = (d.uuid or "").strip()
        if not uuid or uuid in vistos:
            continue
        vistos.add(uuid)
        saida.append(DocumentRef(uuid=uuid, handle=d.handle))
    return saida


def _evidence_block(evidences: list[_Evidence]) -> str:
    """Bloco textual `[N] (section, page)\\n<texto>` enviado ao LLM."""
    parts = []
    for e in evidences:
        page = e.page if e.page is not None else "—"
        parts.append(f"[{e.index}] (section: {e.section or '—'}, page: {page})\n{e.text}")
    return "\n\n".join(parts)


def _strip_invalid_citations(summary: str, max_index: int) -> str:
    """Remove referências [N] fora de 1..max_index — nunca deixar [14] com 10
    evidências. As válidas são preservadas intactas."""
    def repl(m: "re.Match") -> str:
        n = int(m.group(1))
        return m.group(0) if 1 <= n <= max_index else ""
    return _CITATION_RE.sub(repl, summary)


def _strip_reasoning(text: str) -> tuple[str, bool]:
    """Remove o raciocínio que vazou no content. Devolve (texto, houve_remoção).

    Trata, nesta ordem: blocos fechados (`<think>…</think>` e os delimitadores não-XML),
    fechamento ÓRFÃO (fica só o `</think>` porque o abridor foi consumido pelo template
    — tudo que vem antes dele é rascunho) e abertura SEM fechamento (resposta truncada
    ainda no raciocínio — dali para frente nada é resposta)."""
    original = text
    for abre, fecha in _THINK_MARKERS:
        if abre in text and fecha in text:
            text = re.sub(re.escape(abre) + r".*?" + re.escape(fecha), "", text, flags=re.S)
        elif fecha in text:
            text = text.split(fecha)[-1]
        elif abre in text:
            text = text.split(abre)[0]

    text = _THINK_BLOCK_RE.sub("", text)

    fechamentos = list(_THINK_CLOSE_RE.finditer(text))
    if fechamentos:  # fechamento órfão: o que vem ANTES do último é rascunho
        text = text[fechamentos[-1].end():]

    abertura = _THINK_OPEN_RE.search(text)
    if abertura:  # abertura sem fechamento: dali em diante é rascunho
        text = text[: abertura.start()]

    text = text.strip()
    return text, text != original.strip()


def _cjk_ratio(text: str) -> float:
    """Fração de caracteres CJK sobre os não-brancos (0.0 quando não há texto)."""
    visiveis = sum(1 for c in text if not c.isspace())
    if not visiveis:
        return 0.0
    return len(_CJK_RE.findall(text)) / visiveis


def _contaminado(summary: str, language: str) -> bool:
    """True se a resposta ainda parece rascunho do modelo: sobrou marcador de thinking
    ou há CJK demais num idioma que não é CJK (o raciocínio vazado costuma vir em
    chinês). Idioma CJK pedido explicitamente NÃO é contaminação."""
    if _THINK_OPEN_RE.search(summary) or _THINK_CLOSE_RE.search(summary):
        return True
    if (language or "").strip().lower().startswith(_CJK_LANGUAGES):
        return False
    return _cjk_ratio(summary) > _CJK_MAX_RATIO


def _call_llm(system_prompt: str, user_message: str) -> str:
    """Chamada síncrona ao LLM — deve rodar em threadpool (não no event loop).

    Pede o raciocínio DESLIGADO (`_THINKING_OFF_BODY`). O parâmetro não é universal:
    se o provedor recusar a requisição por causa dele (400/422), a chamada é refeita
    sem o parâmetro e ele deixa de ser enviado até o processo reiniciar — melhor um
    resumo com thinking (que o saneamento cobre) do que um erro na cara do usuário.

    Lê apenas `content`: o `reasoning_content` dos modelos de raciocínio é ignorado de
    propósito (não é resposta). Quando o provedor não separa os dois, quem limpa é
    `_strip_reasoning`."""
    global _thinking_param_ok
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    desligar = LLM_SUMMARY_DISABLE_THINKING and _thinking_param_ok is not False
    extra = {"extra_body": _THINKING_OFF_BODY} if desligar else {}
    try:
        resp = _get_client().chat.completions.create(
            model=LLM_SUMMARY_MODEL, messages=messages, temperature=0.0, **extra,
        )
        if desligar:
            _thinking_param_ok = True
    except (BadRequestError, UnprocessableEntityError):
        if not desligar:
            raise
        _thinking_param_ok = False
        log_api.warning(
            "summarize: provedor recusou %r — refazendo sem o parâmetro e mantendo só "
            "o saneamento da resposta.", _THINKING_OFF_BODY,
        )
        resp = _get_client().chat.completions.create(
            model=LLM_SUMMARY_MODEL, messages=messages, temperature=0.0,
        )
    return resp.choices[0].message.content or ""


def _sanitize(raw: str, *, language: str, max_index: int) -> str:
    """Aplica ao texto cru do LLM o saneamento completo: remove o raciocínio vazado e
    as citações [N] inexistentes. Registra em log quando teve de intervir — sem isso a
    regressão volta em silêncio."""
    texto, removeu = _strip_reasoning(raw.strip())
    if removeu:
        log_api.warning(
            "summarize: raciocínio do modelo (%s) vazou no content e foi removido — "
            "considere apontar LLM_SUMMARY_MODEL para a variante sem thinking.",
            LLM_SUMMARY_MODEL,
        )
    return _strip_invalid_citations(texto, max_index)


class SummaryService:
    """Serviço do AI Summary. Reusa o SemanticSearch residente (mesmo embedder e
    cliente Qdrant da busca) e um cliente LLM OpenAI-compatible."""

    def __init__(self, semantic: SemanticSearch):
        self._semantic = semantic

    @staticmethod
    def llm_available() -> bool:
        """True se há chave de LLM configurada (mesma do enrich)."""
        return llm_enrich_service.is_available()

    async def _retrieve_per_document(
        self, query: str, documents: Sequence[DocumentRef], *, limit: int, type: str,
    ) -> list:
        """Uma recuperação INDEPENDENTE por documento: cada UUID vira uma consulta
        própria ao Qdrant (filtro item_uuid) com o seu próprio teto de `limit` chunks.

        As consultas vão em paralelo (o gargalo é a ida ao Qdrant, não a CPU), mas os
        pontos são concatenados na ORDEM em que os documentos foram enviados — assim a
        numeração [N] das evidências fica agrupada por documento e estável entre
        chamadas iguais. `search_points` já absorve os seus próprios erros devolvendo
        lista vazia; `return_exceptions` cobre o que escapar, para um documento
        problemático não derrubar a síntese dos demais."""
        resultados = await asyncio.gather(
            *(
                self._semantic.search_points(query, limit=limit, type=type, uuid=d.uuid)
                for d in documents
            ),
            return_exceptions=True,
        )
        points: list = []
        for doc, res in zip(documents, resultados):
            if isinstance(res, BaseException):
                log_api.error(
                    "summarize: retrieval do documento %s falhou (%s) — seguindo sem ele.",
                    doc.uuid, res,
                )
                continue
            if not res:
                log_api.info("summarize: documento %s sem chunks para q=%r", doc.uuid, query)
            points.extend(res)
        return points

    async def summarize(
        self,
        query: str,
        *,
        limit: int = 5,
        type: str = "hybrid",
        language: str = "pt-BR",
        documents: Optional[Sequence[DocumentRef]] = None,
    ) -> SummaryResponse:
        """Sintetiza as evidências recuperadas para `query`. Retrieval assíncrono +
        LLM em threadpool. Sem evidências → resposta com summary vazio (sem chamar o LLM).

        `documents` vazia (o padrão, e o único caso do GET): uma recuperação global de
        até `limit` chunks, com o teto de diversidade por documento — regra original,
        inalterada. `documents` preenchida: uma recuperação por UUID, cada uma trazendo
        até `limit` chunks (o k por documento), sem o teto de diversidade; a síntese é
        uma só, sobre a união das evidências."""
        docs = _unique_documents(documents)
        if docs:
            points = await self._retrieve_per_document(query, docs, limit=limit, type=type)
            # O cliente escolheu os documentos: o teto de diversidade sai de cena e cada
            # documento contribui com até os `limit` chunks que a sua consulta trouxe.
            evidences = _dedup_and_number(points, max_per_document=None)
            applied_filters = {
                "documents": [d.model_dump(exclude_none=True) for d in docs],
            }
        else:
            points = await self._semantic.search_points(query, limit=limit, type=type)
            evidences = _dedup_and_number(points)
            applied_filters = {}

        retrieval = RetrievalMetadata(
            type=type,
            fusion="rrf" if type == "hybrid" else None,
            top=limit,
            evidence_count=len(evidences),
            per_document=bool(docs),
            documents_count=len(docs),
        )
        if not evidences:
            log_api.info(
                "summarize q=%r documentos=%d: 0 evidência(s) — sem chamada ao LLM",
                query, len(docs),
            )
            return SummaryResponse(
                query=query, language=language, applied_filters=applied_filters,
                retrieval=retrieval, summary="", mappings=[],
            )

        user_message = build_basic_user_message(query, _evidence_block(evidences), language)
        t0 = time.perf_counter()
        raw = await run_in_threadpool(_call_llm, BASIC_SUMMARY_SYSTEM_PROMPT, user_message)
        summary = _sanitize(raw, language=language, max_index=len(evidences))

        # Ainda com cara de rascunho (marcador de thinking, CJK demais num idioma que
        # não é CJK) — ou vazia porque o saneamento consumiu tudo (resposta truncada
        # ainda no raciocínio)? Uma única segunda tentativa, com a instrução de formato
        # repetida no fim da mensagem — é onde o modelo obedece mais. Duas chamadas é o
        # teto: o endpoint é síncrono e o usuário está esperando.
        if not summary or _contaminado(summary, language):
            log_api.warning(
                "summarize q=%r: resposta %s — refazendo uma vez.",
                query,
                "vazia após o saneamento" if not summary
                else f"contaminada (CJK={_cjk_ratio(summary) * 100:.1f}%)",
            )
            raw = await run_in_threadpool(
                _call_llm, BASIC_SUMMARY_SYSTEM_PROMPT,
                f"{user_message}\n\n{OUTPUT_ONLY_REMINDER.format(language=language)}",
            )
            summary = _sanitize(raw, language=language, max_index=len(evidences))
            if not summary or _contaminado(summary, language):
                log_api.error(
                    "summarize q=%r: resposta segue contaminada após a 2ª tentativa "
                    "(modelo=%s) — troque LLM_SUMMARY_MODEL pela variante sem thinking.",
                    query, LLM_SUMMARY_MODEL,
                )

        log_api.info(
            "summarize q=%r documentos=%d evidências=%d em %.3fs",
            query, len(docs), len(evidences), time.perf_counter() - t0,
        )

        mappings = [
            EvidenceMapping(
                index=e.index, chunk_id=e.chunk_id, document_id=e.document_id,
                dspace_uuid=e.item_uuid, item_handle=e.item_handle,
                section=e.section, page=e.page, score=e.score,
                snippet=e.text[:SNIPPET_MAX_CHARS],
            )
            for e in evidences
        ]
        return SummaryResponse(
            query=query, language=language, applied_filters=applied_filters,
            retrieval=retrieval, summary=summary, mappings=mappings,
        )
