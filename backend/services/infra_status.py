"""Estado da infraestrutura do pipeline, numa consulta (GET /api/status).

Responde "o pipeline está de pé?" com uma linha por dependência externa — Qdrant (e
quantos chunks), MinIO (e quantos artefatos), MinerU, a API de embedding, o LLM de
enriquecimento, Redis, os workers Celery, a GPU, o DSpace — e deriva daí o que o
sistema CONSEGUE fazer agora: buscar, ingerir, extrair, indexar, enriquecer.

Relação com scripts/diagnostico.py
----------------------------------
O script é a ferramenta de bancada: roda no servidor sem venv (só biblioteca padrão),
faz round-trip real de embedding e confere o proxy da borda. Este módulo é a versão
consultável pela própria API, feita para ser chamada de fora com frequência. As duas
medem as mesmas dependências e usam os mesmos critérios; quando um veredito mudar
aqui, o script é o outro lugar a olhar.

Efeitos colaterais: nenhum, com uma exceção herdada do health check do MinIO, que
grava e remove um objeto sob `_healthcheck/` quando MINIO_HEALTHCHECK_WRITE=true
(nunca toca artefatos reais). Nada mais escreve, enfileira ou altera estado.

Custo: toda sondagem é bloqueante (HTTP, Redis, S3, broadcast do Celery) e roda numa
thread própria, então o snapshot custa o tempo da checagem mais lenta — hoje o
broadcast do Celery (~STATUS_CELERY_TIMEOUT_SECONDS) e a varredura do MinIO — e não a
soma. Duas camadas de cache por TTL evitam que uma consulta frequente vire carga:
o snapshot inteiro (STATUS_CACHE_TTL_SECONDS) e, mais longa, a contagem de artefatos
(STATUS_ARTIFACT_SCAN_TTL_SECONDS), que é a única varredura que cresce com o acervo.

Sensibilidade: cada componente separa `detail` (métricas e nomes de modelo — pode ir
para qualquer consumidor) de `internal` (URLs, versões, hostnames, PIDs, caminhos —
a topologia). Quem serializa decide se inclui `internal`; ver
backend/api/routes/status.py.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import urlsplit, urlunsplit

import httpx

from backend.core import config as settings
from backend.core.logger import log

# --------------------------------------------------------------------------- #
# Vocabulário de status
# --------------------------------------------------------------------------- #
OK = "ok"                          # funcionando
DEGRADED = "degraded"              # funciona com ressalva (ou vai morder num cenário)
DOWN = "down"                      # não funciona
NOT_CONFIGURED = "not_configured"  # opcional e desligado de propósito — não é falha
UNKNOWN = "unknown"                # a própria checagem falhou (bug/ambiente), não a infra

# Ordem de apresentação dos componentes (do núcleo da busca para as bordas).
COMPONENT_ORDER = (
    "qdrant", "embedder", "mineru", "minio", "redis", "celery",
    "gpu", "llm_enrich", "llm_visual", "dspace", "flower", "jobs", "disk",
)

# Dimensão do vetor denso do bge-m3 — o mesmo número que a collection tem de ter.
DENSE_DIM = 1024

# Token ids de "indicadores de avaliacao institucional do ensino superior" pelo
# tokenizer do BAAI/bge-m3, fixos como em scripts/diagnostico.py. Assim o round-trip
# de `probe=true` não carrega o `transformers` no processo da API só para tokenizar
# uma frase de teste (o backend não hospeda mais o modelo; ver services/embedder.py).
PROBE_TOKEN_IDS = [0, 202334, 8, 96436, 123142, 67897, 54, 48617, 14597, 2]
PROBE_INNER_TOKENS = len(PROBE_TOKEN_IDS) - 2  # o servidor descarta BOS/EOS

_PROCESS_START = time.time()


# --------------------------------------------------------------------------- #
# Estruturas
# --------------------------------------------------------------------------- #
@dataclass
class Component:
    """Veredito sobre uma dependência.

    `detail` é público (métricas, contagens, flags, nome do modelo); `internal` é
    topologia (URL, versão, host, PID, caminho) e só sai para quem se autoriza.
    `hint` diz o que quebra e/ou como corrigir — é a seta do diagnóstico.
    """

    name: str
    role: str
    critical: bool
    status: str
    message: str = ""
    hint: str = ""
    latency_ms: Optional[float] = None
    detail: dict = field(default_factory=dict)
    internal: dict = field(default_factory=dict)

    def to_dict(self, *, include_internal: bool) -> dict:
        out: dict[str, Any] = {
            "name": self.name,
            "status": self.status,
            "critical": self.critical,
            "role": self.role,
            "message": self.message,
            "latency_ms": self.latency_ms,
            "detail": self.detail,
        }
        if self.hint:
            out["hint"] = self.hint
        if include_internal:
            out["internal"] = self.internal
        return out

    @property
    def up(self) -> bool:
        """Funciona agora (inclui `degraded`: degradado ainda serve)."""
        return self.status in (OK, DEGRADED)


@dataclass
class Options:
    """O que a consulta pediu."""

    probe: bool = False      # sondagens ativas (round-trip de embedding) — mais caro
    artifacts: bool = True   # contar objetos no MinIO (varredura que cresce com o acervo)


@dataclass
class Snapshot:
    components: dict[str, Component]
    generated_at: datetime
    took_ms: float
    options: Options

    def to_dict(self, *, include_internal: bool, age_seconds: float = 0.0) -> dict:
        resumo = _resumo(self.components)
        return {
            "status": resumo["status"],
            "generated_at": self.generated_at.isoformat(),
            "age_seconds": round(age_seconds, 3),
            "took_ms": round(self.took_ms, 1),
            "detail_level": "full" if include_internal else "public",
            "probe": self.options.probe,
            "summary": resumo["summary"],
            "blocking": resumo["blocking"],
            "warnings": resumo["warnings"],
            "capabilities": capabilities(self.components),
            "components": {
                nome: c.to_dict(include_internal=include_internal)
                for nome, c in self.components.items()
            },
            "config": pipeline_config(),
            "api": api_process(include_internal=include_internal),
        }


# --------------------------------------------------------------------------- #
# Cache por TTL
# --------------------------------------------------------------------------- #
class _TtlCache:
    """Memo com validade. Uma entrada por chave; recomputa fora do prazo.

    Não protege o cálculo com lock: duas consultas simultâneas depois do prazo podem
    recomputar em paralelo (desperdício limitado, sem corrupção) — preferível a
    serializar consultas atrás de um lock enquanto a sondagem lenta corre.
    """

    def __init__(self) -> None:
        self._entries: dict[Any, tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def get(self, key: Any, ttl: float) -> tuple[Optional[Any], float]:
        """(valor, idade em segundos) — valor None quando não há entrada válida."""
        with self._lock:
            entrada = self._entries.get(key)
        if entrada is None:
            return None, 0.0
        gravado_em, valor = entrada
        idade = time.time() - gravado_em
        if ttl > 0 and idade > ttl:
            return None, 0.0
        return valor, idade

    def put(self, key: Any, valor: Any) -> None:
        with self._lock:
            self._entries[key] = (time.time(), valor)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


_snapshot_cache = _TtlCache()
_artifact_cache = _TtlCache()


def clear_caches() -> None:
    """Zera os caches (usado pelos testes e por `fresh=true`)."""
    _snapshot_cache.clear()
    _artifact_cache.clear()


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #
def _agora() -> datetime:
    return datetime.now(timezone.utc)


def sem_credenciais(url: str) -> str:
    """Remove usuário/senha da URL antes de publicá-la (`redis://user:senha@host`).

    Vale mesmo para `internal`: segredo não sai em resposta nem em log, nem para
    quem está autorizado a ver a topologia."""
    try:
        partes = urlsplit(url)
    except ValueError:
        return url
    if not partes.netloc or "@" not in partes.netloc:
        return url
    host = partes.netloc.rsplit("@", 1)[1]
    return urlunsplit((partes.scheme, host, partes.path, partes.query, partes.fragment))


def _motivo(exc: BaseException) -> str:
    """Mensagem curta de erro, sem stack e sem corpo de resposta."""
    if isinstance(exc, httpx.HTTPStatusError):
        return f"HTTP {exc.response.status_code}"
    if isinstance(exc, httpx.TimeoutException):
        return f"timeout ({type(exc).__name__})"
    if isinstance(exc, httpx.HTTPError):
        # Sem prefixo próprio: quem chama já diz o que não deu ("inalcançável — ...").
        return f"{type(exc).__name__}: {exc}"
    texto = str(exc).strip()
    return f"{type(exc).__name__}: {texto}" if texto else type(exc).__name__


def _http_get(url: str, *, timeout: Optional[float] = None,
              headers: Optional[dict] = None) -> tuple[int, Any]:
    """GET → (status, corpo). O corpo vem parseado quando é JSON, senão None.

    Distinguir "respondeu JSON" de "respondeu qualquer coisa" importa: uma borda em
    manutenção devolve HTML de erro com 200/503, e tratar isso como resposta do
    serviço seria um falso OK (o mesmo cuidado que scripts/diagnostico.py toma)."""
    with httpx.Client(timeout=timeout or settings.STATUS_HTTP_TIMEOUT_SECONDS) as client:
        resp = client.get(url, headers=headers)
    try:
        return resp.status_code, resp.json()
    except ValueError:
        return resp.status_code, None


def _http_post_json(url: str, payload: dict, *, timeout: Optional[float] = None) -> Any:
    with httpx.Client(timeout=timeout or settings.STATUS_HTTP_TIMEOUT_SECONDS) as client:
        resp = client.post(url, json=payload)
    resp.raise_for_status()
    return resp.json()


def _mb(bytes_: float) -> float:
    return round(bytes_ / (1024 * 1024), 1)


def _gb(bytes_: float) -> float:
    return round(bytes_ / (1024 ** 3), 2)


# --------------------------------------------------------------------------- #
# Qdrant — índice dos chunks
# --------------------------------------------------------------------------- #
def check_qdrant() -> Component:
    role = "índice vetorial dos chunks (busca híbrida dense+sparse)"
    url = settings.QDRANT_URL.rstrip("/")
    colecao = settings.QDRANT_COLLECTION
    comp = Component(name="qdrant", role=role, critical=True, status=UNKNOWN,
                     detail={"collection": colecao},
                     internal={"url": url})
    t0 = time.perf_counter()
    try:
        _, raiz = _http_get(f"{url}/")
    except Exception as exc:
        comp.status, comp.message = DOWN, f"servidor inalcançável — {_motivo(exc)}"
        comp.hint = "sem Qdrant não há busca nem indexação."
        return comp
    comp.internal["version"] = (raiz or {}).get("version")

    try:
        status_http, corpo = _http_get(f"{url}/collections/{colecao}")
    except Exception as exc:
        comp.status, comp.message = DOWN, f"erro ao ler a collection — {_motivo(exc)}"
        return comp
    comp.latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    if status_http == 404 or not isinstance(corpo, dict) or "result" not in corpo:
        existentes = _colecoes(url)
        comp.status = DOWN
        comp.message = f"a collection '{colecao}' não existe"
        comp.internal["collections"] = existentes
        comp.detail["collections_total"] = len(existentes)
        comp.hint = ("confira QDRANT_COLLECTION, restaure o snapshot ou reindexe "
                     "(scripts/reindex_from_minio.py).")
        return comp

    dados = corpo["result"]
    params = (dados.get("config") or {}).get("params") or {}
    densos = params.get("vectors") or {}
    esparsos = params.get("sparse_vectors") or {}
    dense_cfg = densos.get("dense") or {}
    chunks = dados.get("points_count") or 0

    comp.detail.update({
        "chunks": chunks,
        "indexed_vectors": dados.get("indexed_vectors_count"),
        "segments": dados.get("segments_count"),
        "collection_status": dados.get("status"),
        "optimizer_status": dados.get("optimizer_status"),
        "dense_vector": {"name": "dense", "size": dense_cfg.get("size"),
                         "distance": dense_cfg.get("distance"),
                         "present": bool(dense_cfg)},
        "sparse_vector": {"name": "sparse", "present": "sparse" in esparsos},
    })
    existentes = _colecoes(url)
    comp.internal["collections"] = existentes
    comp.detail["collections_total"] = len(existentes)

    # Faltando um dos vetores nomeados, a fusão RRF não roda: é falha, não ressalva.
    faltando = [n for n, presente in (("dense", bool(dense_cfg)),
                                      ("sparse", "sparse" in esparsos)) if not presente]
    if faltando:
        comp.status = DOWN
        comp.message = f"vetor(es) nomeado(s) ausente(s): {', '.join(faltando)}"
        comp.hint = "sem os dois vetores não há busca híbrida; reindexe com --reset."
        return comp

    if dense_cfg.get("size") != DENSE_DIM or str(dense_cfg.get("distance", "")).lower() != "cosine":
        comp.status = DEGRADED
        comp.message = (f"vetor denso fora do esperado (dim {dense_cfg.get('size')}, "
                        f"distância {dense_cfg.get('distance')})")
        comp.hint = f"o bge-m3 produz {DENSE_DIM}d com distância Cosine."
        return comp

    if not chunks:
        comp.status = DEGRADED
        comp.message = "collection vazia (0 chunks)"
        comp.hint = "a busca não retorna nada até indexar algum documento."
        return comp

    if str(dados.get("status", "")).lower() not in ("", "green"):
        comp.status = DEGRADED
        comp.message = f"collection em '{dados.get('status')}' ({chunks} chunks)"
        comp.hint = "otimização/indexação em curso ou com erro; a busca segue servindo."
        return comp

    comp.status = OK
    comp.message = f"{chunks} chunks indexados"
    return comp


def _colecoes(url: str) -> list[str]:
    try:
        _, corpo = _http_get(f"{url}/collections")
        return [c["name"] for c in (corpo or {}).get("result", {}).get("collections", [])]
    except Exception:
        return []


# --------------------------------------------------------------------------- #
# API de embedding (bge-m3 nos contêineres vLLM)
# --------------------------------------------------------------------------- #
def check_embedder(*, probe: bool = False) -> Component:
    from backend.services.embedder import BgeM3EmbedderService

    role = "embedding bge-m3 (denso + esparso) para indexar e consultar"
    dense_url, sparse_url = BgeM3EmbedderService.endpoints()
    modelo = settings.EMBED_API_MODEL
    mesma_instancia = dense_url == sparse_url

    comp = Component(
        name="embedder", role=role, critical=True, status=UNKNOWN,
        detail={"model_expected": modelo, "same_instance": mesma_instancia,
                "max_tokens": settings.EMBED_API_MAX_TOKENS},
        internal={"dense_url": dense_url, "sparse_url": sparse_url},
    )

    t0 = time.perf_counter()
    problemas: list[str] = []
    for rotulo, base in (("dense", dense_url), ("sparse", sparse_url)):
        alvo: dict[str, Any] = {"reachable": False, "model_served": False}
        try:
            _, corpo = _http_get(f"{base}/v1/models")
            servidos = [m.get("id") for m in (corpo or {}).get("data", [])]
            alvo["reachable"] = True
            alvo["model_served"] = modelo in servidos
            alvo["max_model_len"] = next(
                (m.get("max_model_len") for m in (corpo or {}).get("data", [])
                 if m.get("id") == modelo), None)
            comp.internal[f"{rotulo}_served"] = servidos
            if not alvo["model_served"]:
                problemas.append(f"{rotulo} serve {servidos}, esperado {modelo!r}")
        except Exception as exc:
            problemas.append(f"{rotulo} inalcançável — {_motivo(exc)}")
        try:  # versão do vLLM: informativo, some sem barulho se o endpoint não existir
            _, versao = _http_get(f"{base}/version", timeout=2.0)
            comp.internal[f"{rotulo}_version"] = (versao or {}).get("version")
        except Exception:
            pass
        comp.detail[rotulo] = alvo
        if mesma_instancia:  # um contêiner só serve as duas tasks (serviço bge-m3-cpu)
            comp.detail["sparse"] = dict(alvo)
            comp.internal["sparse_served"] = comp.internal.get("dense_served")
            comp.internal["sparse_version"] = comp.internal.get("dense_version")
            break
    comp.latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    if problemas:
        comp.status = DOWN
        comp.message = "; ".join(problemas)
        comp.hint = ("sem os dois endpoints não há indexação NEM busca: o embedder "
                     "chama denso e esparso em toda operação. Nome divergente se "
                     "ajusta em EMBED_API_MODEL (é só o rótulo servido).")
        return comp

    comp.status = OK
    comp.message = f"{modelo} servido nos dois endpoints"

    if probe:
        _probe_embedding(comp, dense_url, sparse_url, modelo)
    return comp


def _probe_embedding(comp: Component, dense_url: str, sparse_url: str, modelo: str) -> None:
    """Round-trip real nos dois endpoints (só com `probe=true`).

    É o que separa "o servidor responde /v1/models" de "o embedding funciona": mede a
    dimensão e a norma do denso (o índice usa CLS+L2, norma 1) e o alinhamento
    peso↔token do esparso, que quebra silenciosamente quando o servidor sobe com
    outra pooling task."""
    resultado: dict[str, Any] = {}
    try:
        resp = _http_post_json(f"{dense_url}/v1/embeddings",
                               {"model": modelo, "input": [PROBE_TOKEN_IDS]},
                               timeout=settings.EMBED_API_TIMEOUT_SECONDS)
        vetor = resp["data"][0]["embedding"]
        norma = sum(x * x for x in vetor) ** 0.5
        resultado["dense"] = {"dim": len(vetor), "norm": round(norma, 4)}
        if len(vetor) != DENSE_DIM:
            comp.status = DOWN
            comp.message = f"dense devolveu dim {len(vetor)}, esperado {DENSE_DIM}"
        elif abs(norma - 1.0) > 1e-3:
            comp.status = DEGRADED
            comp.message = f"dense com vetor não normalizado (norma {norma:.4f})"
            comp.hint = "o índice usa CLS+L2; norma != 1 sugere pooling diferente."
    except Exception as exc:
        comp.status = DOWN
        comp.message = f"round-trip do denso falhou — {_motivo(exc)}"
        resultado["dense"] = {"error": _motivo(exc)}

    try:
        resp = _http_post_json(f"{sparse_url}/pooling",
                               {"model": modelo, "input": [PROBE_TOKEN_IDS],
                                "task": "token_classify"},
                               timeout=settings.EMBED_API_TIMEOUT_SECONDS)
        pesos = resp["data"][0]["data"]
        resultado["sparse"] = {"weights": len(pesos), "expected": PROBE_INNER_TOKENS}
        if len(pesos) != PROBE_INNER_TOKENS:
            comp.status = DOWN
            comp.message = (f"sparse desalinhado: {len(pesos)} pesos para "
                            f"{PROBE_INNER_TOKENS} tokens internos")
            comp.hint = "o servidor precisa rodar com --pooler-config.task token_classify."
    except Exception as exc:
        comp.status = DOWN
        comp.message = f"round-trip do esparso falhou — {_motivo(exc)}"
        resultado["sparse"] = {"error": _motivo(exc)}

    comp.detail["probe"] = resultado
    if comp.status == OK:
        comp.message = f"{modelo} respondendo nos dois endpoints (round-trip conferido)"


# --------------------------------------------------------------------------- #
# MinerU — extração
# --------------------------------------------------------------------------- #
def check_mineru() -> Component:
    role = "extração de PDF (markdown + content_list_v2)"
    url = settings.MINERU_API_URL.rstrip("/")
    comp = Component(
        name="mineru", role=role, critical=True, status=UNKNOWN,
        detail={"backend": settings.MINERU_BACKEND, "method": settings.MINERU_METHOD,
                "lang": settings.MINERU_LANG},
        internal={"url": url, "timeout_seconds": settings.MINERU_TIMEOUT_SECONDS},
    )
    t0 = time.perf_counter()
    try:
        status_http, corpo = _http_get(f"{url}/health")
    except Exception as exc:
        comp.status, comp.message = DOWN, f"inalcançável — {_motivo(exc)}"
        comp.hint = ("sem MinerU não há extração de novos PDFs; a busca no que já foi "
                     "indexado continua funcionando.")
        return comp
    comp.latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    if not isinstance(corpo, dict):
        comp.status = DOWN
        comp.message = f"/health respondeu HTTP {status_http} sem JSON do MinerU"
        comp.hint = "confira MINERU_API_URL: a porta pode ser de outro serviço."
        return comp

    comp.detail.update({
        "queued_tasks": corpo.get("queued_tasks"),
        "processing_tasks": corpo.get("processing_tasks"),
        "completed_tasks": corpo.get("completed_tasks"),
        "failed_tasks": corpo.get("failed_tasks"),
        "max_concurrent_requests": corpo.get("max_concurrent_requests"),
    })
    # O protocolo do serviço é o acoplamento que já quebrou em produção (contêiner
    # 3.2.0 × cliente 3.4.4); a versão fica visível para quem for depurar.
    comp.internal.update({"version": corpo.get("version"),
                          "protocol_version": corpo.get("protocol_version")})

    saude = str(corpo.get("status", "")).lower()
    if saude in ("healthy", "ok"):
        comp.status = OK
        comp.message = (f"{settings.MINERU_BACKEND} pronto "
                        f"({corpo.get('processing_tasks', 0)} em processamento, "
                        f"{corpo.get('queued_tasks', 0)} na fila)")
    else:
        comp.status = DEGRADED
        comp.message = f"respondeu com status {corpo.get('status')!r}"
    return comp


# --------------------------------------------------------------------------- #
# MinIO — artefatos
# --------------------------------------------------------------------------- #
def check_minio(*, artifacts: bool = True) -> Component:
    role = "armazenamento oficial dos artefatos (PDF, markdown, chunks, manifesto)"
    comp = Component(name="minio", role=role, critical=True, status=UNKNOWN)

    if settings.ARTIFACT_STORE_BACKEND != "minio":
        comp.status = NOT_CONFIGURED
        comp.message = f"ARTIFACT_STORE_BACKEND={settings.ARTIFACT_STORE_BACKEND!r}"
        comp.detail["backend"] = settings.ARTIFACT_STORE_BACKEND
        return comp

    from backend.services.artifact_store import get_artifact_store

    comp.internal.update({
        "endpoint": settings.MINIO_ENDPOINT, "bucket": settings.MINIO_BUCKET,
        "prefix": settings.MINIO_ARTIFACT_PREFIX, "secure": settings.MINIO_SECURE,
        "versioning": settings.MINIO_BUCKET_VERSIONING,
    })

    t0 = time.perf_counter()
    try:
        store = get_artifact_store()
        saude = store.healthcheck()
    except Exception as exc:
        comp.status, comp.message = DOWN, f"store indisponível — {_motivo(exc)}"
        comp.hint = "sem MinIO o pipeline não grava nem lê artefatos."
        return comp
    comp.latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    comp.detail.update({
        "reachable": saude.get("reachable"), "bucket_exists": saude.get("bucket_exists"),
        "readable": saude.get("readable"), "writable": saude.get("writable"),
        "write_probe_enabled": settings.MINIO_HEALTHCHECK_WRITE,
    })

    if not saude.get("reachable"):
        comp.status, comp.message = DOWN, "inalcançável"
        comp.hint = "sem MinIO o pipeline não grava nem lê artefatos."
        return comp
    if not saude.get("bucket_exists"):
        comp.status, comp.message = DOWN, "bucket inexistente"
        comp.hint = "crie o bucket (docker compose up minio-init) ou ajuste MINIO_BUCKET."
        return comp
    if not saude.get("readable"):
        comp.status, comp.message = DOWN, "sem permissão de leitura no bucket"
        return comp
    if settings.MINIO_HEALTHCHECK_WRITE and not saude.get("writable"):
        comp.status = DOWN
        comp.message = "sem permissão de escrita no bucket"
        comp.hint = "a ingestão falha ao gravar; a busca no que já existe segue de pé."
        return comp

    comp.status = OK
    comp.message = "bucket acessível"

    if artifacts:
        contagem = contar_artefatos(store)
        comp.detail["artifacts"] = contagem
        if contagem.get("error"):
            comp.status = DEGRADED
            comp.message = f"bucket acessível, contagem falhou — {contagem['error']}"
        else:
            comp.message = (f"{contagem['objects']} artefatos "
                            f"({contagem['documents']} documentos, "
                            f"{contagem['size_gb']} GB)")
            if contagem.get("truncated"):
                comp.message += " — contagem truncada"
    return comp


def contar_artefatos(store, *, fresh: bool = False) -> dict:
    """Conta os objetos sob o prefixo de artefatos, agrupados por etapa do pipeline.

    Layout das keys (ver MinIOArtifactStore.artifact_key):
        <prefix>/<pipeline_id>/<document_id>/<etapa>/<arquivo>
        <prefix>/<pipeline_id>/<document_id>/manifest.json
    de onde saem: total de objetos e bytes, documentos (pares pipeline/documento),
    execuções (pipeline_ids) e a divisão por etapa (source/mineru/indexing/enrichment).

    É a única sondagem cujo custo cresce com o acervo — uma listagem recursiva. Por
    isso tem cache próprio, mais longo que o do snapshot (o número muda devagar), e
    um teto de objetos varridos: batendo no teto a contagem volta com
    `truncated: true` em vez de a rota travar por minutos.
    """
    chave = ("artifacts", settings.MINIO_BUCKET, settings.MINIO_ARTIFACT_PREFIX)
    ttl = 0.0 if fresh else settings.STATUS_ARTIFACT_SCAN_TTL_SECONDS
    if not fresh:
        cacheado, idade = _artifact_cache.get(chave, ttl)
        if cacheado is not None:
            return {**cacheado, "cache_age_seconds": round(idade, 1)}

    prefixo = settings.MINIO_ARTIFACT_PREFIX.strip("/")
    base = len([p for p in prefixo.split("/") if p])
    teto = settings.STATUS_ARTIFACT_SCAN_MAX_OBJECTS

    objetos = bytes_totais = manifestos = 0
    por_etapa: dict[str, int] = {}
    documentos: set[str] = set()
    execucoes: set[str] = set()
    truncado = False
    t0 = time.perf_counter()
    try:
        for object_key, tamanho in store.iter_prefix(f"{prefixo}/"):
            objetos += 1
            bytes_totais += tamanho
            partes = object_key.split("/")
            if len(partes) > base:
                execucoes.add(partes[base])
            if len(partes) > base + 1:
                documentos.add("/".join(partes[base:base + 2]))
            etapa = partes[base + 2] if len(partes) > base + 2 else "(raiz)"
            por_etapa[etapa] = por_etapa.get(etapa, 0) + 1
            if partes[-1] == "manifest.json":
                manifestos += 1
            if teto and objetos >= teto:
                truncado = True
                break
    except Exception as exc:
        log.warning("[status] contagem de artefatos falhou: %s", exc)
        return {"error": _motivo(exc)}

    contagem = {
        "objects": objetos,
        "size_bytes": bytes_totais,
        "size_gb": _gb(bytes_totais),
        "documents": len(documentos),
        "manifests": manifestos,
        "pipeline_runs": len(execucoes),
        "by_stage": dict(sorted(por_etapa.items(), key=lambda kv: -kv[1])),
        "truncated": truncado,
        "scan_ms": round((time.perf_counter() - t0) * 1000, 1),
        "scanned_at": _agora().isoformat(),
        "cache_age_seconds": 0.0,
    }
    _artifact_cache.put(chave, contagem)
    return contagem


# --------------------------------------------------------------------------- #
# Redis — broker, job_store e lock da GPU
# --------------------------------------------------------------------------- #
def check_redis() -> Component:
    role = "broker do Celery (DB 0), job_store (DB 1) e lock da GPU (DB 2)"
    alvos = {
        "broker": settings.CELERY_BROKER_URL,
        "job_store": settings.JOBSTORE_REDIS_URL,
        "gpu_manager": settings.GPU_MANAGER_REDIS_URL,
    }
    comp = Component(
        name="redis", role=role, critical=True, status=UNKNOWN,
        internal={f"{rotulo}_url": sem_credenciais(url) for rotulo, url in alvos.items()},
    )
    try:
        import redis  # dependência do celery[redis]; import tardio como no job_store
    except Exception as exc:
        comp.status, comp.message = UNKNOWN, f"cliente redis indisponível — {_motivo(exc)}"
        return comp

    t0 = time.perf_counter()
    por_alvo: dict[str, dict] = {}
    primeiro_cliente = None
    for rotulo, url in alvos.items():
        try:
            cliente = redis.Redis.from_url(url, decode_responses=True,
                                           socket_timeout=settings.STATUS_HTTP_TIMEOUT_SECONDS,
                                           socket_connect_timeout=settings.STATUS_HTTP_TIMEOUT_SECONDS)
            cliente.ping()
            por_alvo[rotulo] = {"ok": True, "keys": cliente.dbsize()}
            primeiro_cliente = primeiro_cliente or cliente
        except Exception as exc:
            por_alvo[rotulo] = {"ok": False, "error": _motivo(exc)}
    comp.latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    comp.detail["dbs"] = por_alvo

    if primeiro_cliente is not None:
        try:
            info = primeiro_cliente.info()
            comp.detail.update({
                "connected_clients": info.get("connected_clients"),
                "used_memory_mb": _mb(info.get("used_memory", 0)),
            })
            comp.internal.update({
                "version": info.get("redis_version"),
                "uptime_days": info.get("uptime_in_days"),
                "maxmemory_policy": info.get("maxmemory_policy"),
            })
        except Exception as exc:  # INFO pode estar restrito por ACL
            comp.internal["info_error"] = _motivo(exc)

        filas = _tamanho_das_filas(primeiro_cliente)
        if filas is not None:
            comp.detail["queues"] = filas
            comp.detail["backlog_total"] = sum(filas.values())

    fora = [rotulo for rotulo, r in por_alvo.items() if not r["ok"]]
    if not fora:
        comp.status = OK
        comp.message = "os três DBs respondem"
    elif fora == ["job_store"]:
        # Degradação documentada: o job_store cai para um dict por-processo, então o
        # status deixa de ser compartilhado entre a API e os workers — mas a
        # ingestão e a indexação continuam.
        comp.status = DEGRADED
        comp.message = "job_store (DB 1) fora — status de job não é compartilhado"
        comp.hint = "o job_store cai para memória por-processo; /api/files/status fica incompleto."
    else:
        comp.status = DOWN
        comp.message = f"DB(s) fora: {', '.join(fora)}"
        comp.hint = ("sem broker não há ingestão; sem o DB do gpu-manager a extração "
                     "falha (o lock é fail-closed).")
    return comp


def _tamanho_das_filas(cliente) -> Optional[dict]:
    """Backlog por fila do Celery. Os nomes saem do roteamento do celery_app (não são
    fixados aqui) — com o broker Redis, cada fila é uma lista com o nome da fila."""
    try:
        from backend.celery_app import app as celery_app

        nomes = {r["queue"] for r in (celery_app.conf.task_routes or {}).values()
                 if isinstance(r, dict) and r.get("queue")}
        nomes.add(celery_app.conf.task_default_queue)
        return {nome: cliente.llen(nome) for nome in sorted(nomes)}
    except Exception as exc:
        log.debug("[status] falha ao medir as filas: %s", exc)
        return None


# --------------------------------------------------------------------------- #
# Workers Celery
# --------------------------------------------------------------------------- #
def check_celery() -> Component:
    role = "workers que executam a chain (download/extract/llm e gpu)"
    comp = Component(name="celery", role=role, critical=True, status=UNKNOWN,
                     internal={"broker_url": sem_credenciais(settings.CELERY_BROKER_URL)})
    try:
        from backend.celery_app import app as celery_app
    except Exception as exc:
        comp.status, comp.message = UNKNOWN, f"celery_app não importável — {_motivo(exc)}"
        return comp

    esperadas = {r["queue"] for r in (celery_app.conf.task_routes or {}).values()
                 if isinstance(r, dict) and r.get("queue")}
    esperadas.add(celery_app.conf.task_default_queue)
    timeout = settings.STATUS_CELERY_TIMEOUT_SECONDS

    # Cada método do inspect é um broadcast que ESPERA o timeout inteiro (não há
    # resposta "final" para encerrar antes). Em série custaria 4×timeout; em threads
    # separadas, uma conexão do pool por chamada, custa um timeout.
    def consulta(metodo: str):
        return getattr(celery_app.control.inspect(timeout=timeout), metodo)()

    t0 = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futuros = {m: pool.submit(consulta, m)
                       for m in ("stats", "active_queues", "active", "reserved")}
            respostas = {m: f.result() or {} for m, f in futuros.items()}
    except Exception as exc:
        comp.status, comp.message = DOWN, f"broadcast falhou — {_motivo(exc)}"
        comp.hint = "sem broker o inspect não responde; ver o componente redis."
        return comp
    comp.latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    stats, filas, ativas, reservadas = (respostas["stats"], respostas["active_queues"],
                                        respostas["active"], respostas["reserved"])
    nomes = sorted(set(stats) | set(filas) | set(ativas))
    cobertas: set[str] = set()
    detalhe_workers = []
    concorrencia = tarefas_ativas = tarefas_reservadas = 0
    for nome in nomes:
        do_worker = [q.get("name") for q in (filas.get(nome) or [])]
        cobertas.update(q for q in do_worker if q)
        st = stats.get(nome) or {}
        pool_cfg = st.get("pool") or {}
        conc = pool_cfg.get("max-concurrency") or 0
        n_ativas = len(ativas.get(nome) or [])
        n_reservadas = len(reservadas.get(nome) or [])
        concorrencia += conc
        tarefas_ativas += n_ativas
        tarefas_reservadas += n_reservadas
        detalhe_workers.append({
            "name": nome, "queues": do_worker, "concurrency": conc,
            "active": n_ativas, "reserved": n_reservadas,
            "uptime_seconds": st.get("uptime"), "pid": st.get("pid"),
            "tasks_total": st.get("total"),
        })

    faltando = sorted(esperadas - cobertas)
    comp.detail.update({
        "workers_online": len(nomes),
        "concurrency_total": concorrencia,
        "active_tasks": tarefas_ativas,
        "reserved_tasks": tarefas_reservadas,
        "queues_expected": sorted(esperadas),
        "queues_covered": sorted(cobertas),
        "queues_missing": faltando,
        "inspect_timeout_seconds": timeout,
    })
    # Nomes de worker carregam o hostname, e PID/uptime identificam o processo: vão
    # para `internal`; o agregado acima basta para um painel.
    comp.internal["workers"] = detalhe_workers

    if not nomes:
        comp.status = DOWN
        comp.message = "nenhum worker respondeu"
        comp.hint = ("nada sai da fila: suba os workers (ver CELERY.md). Se eles estão "
                     "de pé, confira se falam com o MESMO broker.")
    elif faltando:
        comp.status = DEGRADED
        comp.message = f"{len(nomes)} worker(s), sem consumidor para: {', '.join(faltando)}"
        comp.hint = ("tasks roteadas para essas filas ficam paradas — a fila 'gpu' é a "
                     "da indexação; 'download'/'extract' são do worker leve.")
    else:
        comp.status = OK
        comp.message = (f"{len(nomes)} worker(s) online, {tarefas_ativas} task(s) em "
                        f"execução, todas as filas cobertas")
    return comp


# --------------------------------------------------------------------------- #
# GPU — lock compartilhado + placa
# --------------------------------------------------------------------------- #
def check_gpu() -> Component:
    role = "GPU compartilhada (mutex do gpu_resource_manager; MinerU + vLLM na mesma placa)"
    habilitado = settings.GPU_MANAGER_ENABLED
    comp = Component(
        name="gpu", role=role, critical=habilitado, status=UNKNOWN,
        detail={"manager_enabled": habilitado, "resource": settings.GPU_RESOURCE_NAME},
        internal={"redis_url": sem_credenciais(settings.GPU_MANAGER_REDIS_URL),
                  "mineru_priority": settings.MINERU_GPU_PRIORITY},
    )
    if settings.STATUS_GPU_SMI_ENABLED:
        placas = _placas()
        if placas is not None:
            comp.detail["devices"] = placas

    if not habilitado:
        comp.status = NOT_CONFIGURED
        comp.message = "GPU_MANAGER_ENABLED=false — extração roda sem o lock"
        comp.hint = "sem o mutex, MinerU e outros consumidores podem disputar VRAM."
        return comp

    t0 = time.perf_counter()
    try:
        from backend.services.gpu_manager import get_gpu_manager

        manager = get_gpu_manager()
        estado = manager.get_status(resource=settings.GPU_RESOURCE_NAME).to_dict()
        fila = manager.get_queue(resource=settings.GPU_RESOURCE_NAME, limit=10)
    except Exception as exc:
        comp.status, comp.message = DOWN, f"backend do gpu-manager indisponível — {_motivo(exc)}"
        comp.hint = ("o lock é fail-closed: sem ele a extração falha ao tentar "
                     "adquirir a GPU. Ver o DB 2 do Redis.")
        return comp
    comp.latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    dono = estado.get("owner") or {}
    comp.detail.update({
        "locked": estado.get("locked"),
        "lock_ttl_seconds": estado.get("lock_ttl_seconds"),
        "holder_service": dono.get("service") if isinstance(dono, dict) else None,
        "queue_size": estado.get("queue_size"),
    })
    comp.internal["owner"] = dono
    comp.internal["queue"] = fila
    comp.status = OK
    comp.message = ("lock livre" if not estado.get("locked")
                    else f"em uso por {dono.get('service', '?') if isinstance(dono, dict) else '?'}"
                         f" ({estado.get('queue_size', 0)} na fila)")
    return comp


def _placas() -> Optional[list[dict]]:
    """VRAM/utilização por GPU via nvidia-smi. None quando não há nvidia-smi (host sem
    GPU, contêiner sem o binário) — ausência de placa não é falha deste componente."""
    try:
        saida = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout
    except Exception:
        return None
    placas = []
    for linha in saida.strip().splitlines():
        campos = [c.strip() for c in linha.split(",")]
        if len(campos) < 6:
            continue
        try:
            total, usada = int(campos[2]), int(campos[3])
            placas.append({
                "index": int(campos[0]), "name": campos[1],
                "memory_total_mb": total, "memory_used_mb": usada,
                "memory_free_mb": total - usada,
                "utilization_percent": int(campos[4]),
                "temperature_celsius": int(campos[5]),
            })
        except ValueError:
            continue
    return placas or None


# --------------------------------------------------------------------------- #
# LLM de enriquecimento (desacoplado da indexação)
# --------------------------------------------------------------------------- #
def check_llm_enrich(*, probe: bool = True) -> Component:
    """Enriquecimento por LLM: ligado ou não, com que provedor e modelo.

    Duas chaves diferentes de "ligado": `enabled` (existe chave de API — sem ela o
    enrich é PULADO, e a indexação segue normalmente) e `auto_after_index`
    (LLM_ENRICH_AUTO, que anexa o enrich como follow-up depois de indexar). Só com as
    duas o enriquecimento acontece sem alguém chamar
    `POST /api/files/enrich/{job_id}` na mão."""
    from backend.services import llm_enrich_service as llm_enrich

    role = "enriquecimento de metadados por LLM (opcional, fora da chain obrigatória)"
    tem_chave = llm_enrich.is_available()
    auto = settings.LLM_ENRICH_AUTO
    comp = Component(
        name="llm_enrich", role=role, critical=False, status=UNKNOWN,
        detail={
            "enabled": tem_chave,
            "auto_after_index": auto,
            "active": bool(tem_chave and auto),
            "provider": llm_enrich.provider_name(),
            "model": settings.LLM_ENRICH_MODEL,
            "max_chars": settings.LLM_ENRICH_MAX_CHARS,
            "review_threshold": settings.LLM_ENRICH_REVIEW_THRESHOLD,
            "timeout_seconds": settings.LLM_ENRICH_TIMEOUT_SECONDS,
        },
        internal={"base_url": settings.LLM_ENRICH_BASE_URL},
    )

    if not tem_chave:
        comp.status = NOT_CONFIGURED
        comp.message = "desativado: sem chave de API (o enrich é pulado)"
        comp.hint = ("defina LLM_ENRICH_API_KEY para ligar. A indexação NÃO depende "
                     "disso — só os metadados enriquecidos deixam de existir.")
        return comp

    if not probe:
        comp.status = OK if auto else DEGRADED
        comp.message = (f"{comp.detail['provider']}/{settings.LLM_ENRICH_MODEL} configurado"
                        + ("" if auto else " (LLM_ENRICH_AUTO=false: só sob demanda)"))
        return comp

    # GET /models: confere chave e nome do modelo sem gastar tokens. Um modelo
    # inexistente só apareceria na primeira chamada real de enrich, tarde demais.
    t0 = time.perf_counter()
    try:
        status_http, corpo = _http_get(
            settings.LLM_ENRICH_BASE_URL.rstrip("/") + "/models",
            headers={"Authorization": f"Bearer {settings.LLM_ENRICH_API_KEY}"},
        )
    except Exception as exc:
        comp.status, comp.message = DOWN, f"provedor inalcançável — {_motivo(exc)}"
        comp.hint = "o enrich falha; a indexação continua (o enrich é desacoplado)."
        return comp
    comp.latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    if status_http in (401, 403):
        comp.status, comp.message = DOWN, f"chave rejeitada (HTTP {status_http})"
        comp.hint = "revise LLM_ENRICH_API_KEY."
        return comp
    if status_http >= 400 or not isinstance(corpo, dict):
        comp.status = DEGRADED
        comp.message = f"/models respondeu HTTP {status_http} fora do contrato OpenAI"
        comp.hint = "o provedor pode não expor /models; o enrich ainda pode funcionar."
        return comp

    servidos = [m.get("id") for m in corpo.get("data", [])]
    comp.internal["models_seen"] = servidos
    comp.detail["model_served"] = settings.LLM_ENRICH_MODEL in servidos
    if not comp.detail["model_served"]:
        comp.status = DEGRADED
        comp.message = (f"o provedor não lista {settings.LLM_ENRICH_MODEL!r} "
                        f"(serve {servidos})")
        comp.hint = "ajuste LLM_ENRICH_MODEL: toda chamada de enrich falharia."
        return comp

    comp.status = OK if auto else DEGRADED
    comp.message = (f"{comp.detail['provider']}/{settings.LLM_ENRICH_MODEL} respondendo"
                    + ("" if auto else " (LLM_ENRICH_AUTO=false: só sob demanda)"))
    if not auto:
        comp.hint = "o follow-up automático está desligado; use POST /api/files/enrich/{job_id}."
    return comp


def check_llm_visual() -> Component:
    """Gate de LLM para gráficos/imagens no chunking (§11/§12). Independente do
    enrich: usa o mesmo provedor, mas é ligado por CHUNK_VISUAL_LLM."""
    from backend.services import llm_visual_service as llm_visual

    comp = Component(
        name="llm_visual", role="validação de gráficos/imagens por LLM no chunking",
        critical=False, status=NOT_CONFIGURED,
        detail={"enabled": settings.CHUNK_VISUAL_LLM,
                "model": settings.LLM_ENRICH_MODEL,
                "chart_mode": settings.CHUNK_CHART_MODE,
                "image_mode": settings.CHUNK_IMAGE_MODE},
    )
    if not settings.CHUNK_VISUAL_LLM:
        comp.message = "desligado (CHUNK_VISUAL_LLM=false) — heurística determinística"
        return comp
    if not llm_visual.is_available():
        comp.status = DEGRADED
        comp.message = "ligado, mas sem chave de API — cai na heurística"
        comp.hint = "defina LLM_ENRICH_API_KEY ou desligue CHUNK_VISUAL_LLM."
        return comp
    comp.status = OK
    comp.message = f"ligado com {settings.LLM_ENRICH_MODEL}"
    return comp


# --------------------------------------------------------------------------- #
# DSpace, Flower, jobs, disco
# --------------------------------------------------------------------------- #
def check_dspace() -> Component:
    role = "repositório de origem dos PDFs (estágio 1)"
    url = settings.DSPACE_URL
    comp = Component(name="dspace", role=role, critical=False, status=UNKNOWN,
                     internal={"url": url})
    t0 = time.perf_counter()
    try:
        status_http, corpo = _http_get(f"{url}/server/api")
    except Exception as exc:
        comp.status, comp.message = DOWN, f"inalcançável — {_motivo(exc)}"
        comp.hint = "sem DSpace não há ingestão de novos itens; o índice segue servindo."
        return comp
    comp.latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    comp.detail["http_status"] = status_http

    # 200 com HTML é a borda em manutenção respondendo no lugar do REST — não conta
    # como DSpace de pé.
    if status_http >= 400 or not isinstance(corpo, dict):
        comp.status = DOWN
        comp.message = f"/server/api respondeu HTTP {status_http} sem JSON do REST"
        comp.hint = "sem DSpace não há ingestão de novos itens; o índice segue servindo."
        return comp
    comp.internal["dspace_version"] = corpo.get("dspaceVersion")
    comp.internal["dspace_name"] = corpo.get("dspaceName")
    comp.status = OK
    comp.message = f"REST respondendo (DSpace {corpo.get('dspaceVersion', '?')})"
    return comp


def check_flower() -> Component:
    role = "monitor do Celery (opcional)"
    url = settings.FLOWER_URL
    comp = Component(name="flower", role=role, critical=False, status=NOT_CONFIGURED,
                     internal={"url": url or None})
    if not url:
        comp.message = "FLOWER_URL não definida — monitor não checado"
        return comp
    t0 = time.perf_counter()
    try:
        status_http, corpo = _http_get(f"{url}/api/workers")
    except Exception as exc:
        comp.status, comp.message = DOWN, f"inalcançável — {_motivo(exc)}"
        comp.hint = "só a observabilidade cai; o pipeline não depende do Flower."
        return comp
    comp.latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    if status_http >= 400:
        comp.status = DEGRADED
        comp.message = f"respondeu HTTP {status_http} (API REST pode exigir autenticação)"
        return comp
    comp.status = OK
    comp.detail["workers_known"] = len(corpo) if isinstance(corpo, dict) else None
    comp.message = "monitor respondendo"
    return comp


def check_jobs() -> Component:
    """Fotografia dos índices de job (execução, sucesso, falhas) — o estado do
    pipeline, não de uma dependência."""
    from backend.services import job_store

    comp = Component(name="jobs", role="registro de status dos jobs de ingestão",
                     critical=False, status=UNKNOWN)
    t0 = time.perf_counter()
    try:
        tamanhos = job_store.index_sizes()
    except Exception as exc:
        comp.status, comp.message = UNKNOWN, f"job_store não respondeu — {_motivo(exc)}"
        return comp
    comp.latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    comp.detail.update({
        "active": tamanhos.get("active"),
        "succeeded": tamanhos.get("succeeded"),
        "failed": tamanhos.get("failed"),
        "storage": tamanhos.get("backend"),
        "ttl_seconds": settings.JOBSTORE_TTL,
    })
    if tamanhos.get("backend") != "redis":
        comp.status = DEGRADED
        comp.message = "contagem local (Redis fora): status não é compartilhado entre processos"
        comp.hint = "ver o componente redis; a API e os workers deixam de enxergar os mesmos jobs."
        return comp
    comp.status = OK
    comp.message = (f"{tamanhos.get('active')} em execução, "
                    f"{tamanhos.get('succeeded')} concluídos, "
                    f"{tamanhos.get('failed')} na fila de falhas")
    if tamanhos.get("failed"):
        comp.hint = "há falhas registradas — ver GET /api/files/failures."
    return comp


def check_disk() -> Component:
    """Espaço livre onde o pipeline escreve: o temporário das tasks (materialização
    dos artefatos) e o diretório servido em /output."""
    role = "espaço em disco do temporário das tasks e do /output"
    comp = Component(name="disk", role=role, critical=False, status=UNKNOWN)
    alvos = {"temp": Path(settings.ARTIFACT_TEMP_DIR), "output": settings.OUTPUT_DIR}
    piores: list[float] = []
    for rotulo, caminho in alvos.items():
        existente = _primeiro_existente(caminho)
        if existente is None:
            comp.detail[rotulo] = {"error": "caminho inexistente"}
            continue
        try:
            uso = shutil.disk_usage(existente)
        except Exception as exc:
            comp.detail[rotulo] = {"error": _motivo(exc)}
            continue
        livre_pct = round(uso.free * 100.0 / uso.total, 1) if uso.total else 0.0
        comp.detail[rotulo] = {"total_gb": _gb(uso.total), "used_gb": _gb(uso.used),
                               "free_gb": _gb(uso.free), "free_percent": livre_pct}
        comp.internal[f"{rotulo}_path"] = str(caminho)
        piores.append(livre_pct)

    if not piores:
        comp.status, comp.message = UNKNOWN, "nenhum dos caminhos pôde ser medido"
        return comp
    livre = min(piores)
    limite = settings.STATUS_DISK_MIN_FREE_PERCENT
    comp.detail["free_percent_min"] = livre
    comp.detail["min_free_percent_threshold"] = limite
    if livre < 1.0:
        comp.status, comp.message = DOWN, f"disco cheio ({livre}% livre)"
        comp.hint = "a extração falha ao materializar o PDF; libere espaço."
    elif livre < limite:
        comp.status, comp.message = DEGRADED, f"pouco espaço livre ({livre}%)"
        comp.hint = ("ARTIFACT_TEMP_DIR guarda o PDF e a saída do MinerU durante a "
                     "task; sem espaço a extração falha.")
    else:
        comp.status, comp.message = OK, f"{livre}% livre no volume mais apertado"
    return comp


def _primeiro_existente(caminho: Path) -> Optional[Path]:
    """O próprio caminho ou o primeiro ancestral que existe — o temporário é criado
    sob demanda, e medir o volume que o contém é o que interessa."""
    atual = caminho
    for _ in range(8):
        if atual.exists():
            return atual
        if atual.parent == atual:
            return None
        atual = atual.parent
    return None


# --------------------------------------------------------------------------- #
# Configuração e processo (contexto, não sondagem)
# --------------------------------------------------------------------------- #
def pipeline_config() -> dict:
    """Config que determina a SEMÂNTICA do índice — o que foi indexado e como é
    buscado. Vai sem restrição: são decisões de pipeline, não topologia. Serve para
    responder "os chunks no Qdrant saíram desta configuração?"."""
    return {
        "pipeline_version": settings.PIPELINE_VERSION,
        "embedding": {
            "model": settings.EMBED_API_MODEL,
            "dense_dim": DENSE_DIM,
            "max_tokens": settings.EMBED_API_MAX_TOKENS,
            "batch_size": settings.EMBEDDING_BATCH_SIZE,
            "text_field": settings.EMBEDDING_TEXT_FIELD,
        },
        "extraction": {
            "backend": settings.MINERU_BACKEND,
            "method": settings.MINERU_METHOD,
            "lang": settings.MINERU_LANG,
            "primary_source": settings.MINERU_PRIMARY_SOURCE,
            "reconstruct_cross_page_paragraphs": settings.MINERU_RECONSTRUCT_CROSS_PAGE_PARAGRAPHS,
        },
        "chunking": {
            "strategy": settings.CHUNKING_STRATEGY,
            "version": settings.CHUNKING_VERSION,
            "tokenizer": settings.CHUNK_TOKENIZER_MODEL,
            "target_tokens": settings.CHUNK_TARGET_TOKENS,
            "max_tokens": settings.CHUNK_MAX_TOKENS,
            "min_tokens": settings.CHUNK_MIN_TOKENS,
            "overlap_tokens": settings.CHUNK_OVERLAP_TOKENS,
            "normalize_text": settings.CHUNK_NORMALIZE_TEXT,
            "references_mode": settings.CHUNK_REFERENCES_MODE,
            "front_matter_mode": settings.CHUNK_FRONT_MATTER_MODE,
            "appendix_mode": settings.CHUNK_APPENDIX_MODE,
            "acronym_mode": settings.CHUNK_ACRONYM_MODE,
            "table_mode": settings.CHUNK_TABLE_MODE,
            "chart_mode": settings.CHUNK_CHART_MODE,
            "image_mode": settings.CHUNK_IMAGE_MODE,
            "equation_mode": settings.CHUNK_EQUATION_MODE,
        },
        "search": {
            "collection": settings.QDRANT_COLLECTION,
            "default_profile": settings.SEARCH_DEFAULT_PROFILE,
            "exclude_front_matter": settings.SEARCH_EXCLUDE_FRONT_MATTER,
            "exclude_references": settings.SEARCH_EXCLUDE_REFERENCES,
            "exclude_navigation_lists": settings.SEARCH_EXCLUDE_NAVIGATION_LISTS,
            "exclude_low_confidence_visual_data": settings.SEARCH_EXCLUDE_LOW_CONFIDENCE_VISUAL_DATA,
        },
    }


def api_process(*, include_internal: bool) -> dict:
    """Quem respondeu: o processo da API. Com vários uvicorn atrás de um balanceador,
    é o que diz de qual deles veio o snapshot."""
    inicio = _PROCESS_START
    try:
        import psutil

        inicio = psutil.Process(os.getpid()).create_time()
    except Exception:  # psutil ausente/sem permissão: cai para a carga do módulo
        pass
    dados: dict[str, Any] = {
        "service": "evidencia_pipe",
        "pipeline_version": settings.PIPELINE_VERSION,
        "uptime_seconds": round(time.time() - inicio, 1),
        "started_at": datetime.fromtimestamp(inicio, timezone.utc).isoformat(),
    }
    if include_internal:
        import socket

        dados.update({
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "port": os.getenv("PORT", "8020"),
            "proxy_stripped_prefix": settings.PROXY_STRIPPED_PREFIX or None,
            "proxy_guard_admin": settings.PROXY_GUARD_ADMIN,
            "internal_token_configured": bool(settings.INTERNAL_API_TOKEN),
        })
    return dados


# --------------------------------------------------------------------------- #
# Capacidades derivadas
# --------------------------------------------------------------------------- #
def capabilities(componentes: dict[str, Component]) -> dict:
    """O que o sistema consegue fazer AGORA, e quem está impedindo.

    É a leitura que interessa a quem opera: "qdrant degraded" não diz se a busca
    responde; `busca.available` diz. Cada capacidade lista os componentes que a
    bloqueiam, para o próximo passo ser óbvio."""

    def comp(nome: str) -> Optional[Component]:
        return componentes.get(nome)

    def de_pe(nome: str) -> bool:
        c = comp(nome)
        return bool(c and c.up)

    def fila_coberta(fila: str) -> bool:
        c = comp("celery")
        if not c or not c.up:
            return False
        return fila not in (c.detail.get("queues_missing") or [])

    qdrant = comp("qdrant")
    tem_chunks = bool(qdrant and (qdrant.detail.get("chunks") or 0) > 0)
    minio_escreve = bool(comp("minio") and comp("minio").up
                         and comp("minio").detail.get("writable") is not False)
    broker_ok = bool(comp("redis") and (comp("redis").detail.get("dbs", {})
                                        .get("broker", {}).get("ok")))
    gpu_ok = de_pe("gpu") or (comp("gpu") is not None
                              and comp("gpu").status == NOT_CONFIGURED)

    capacidades: dict[str, dict] = {}

    def registra(nome: str, descricao: str, requisitos: list[tuple[str, bool]]) -> None:
        bloqueios = [rotulo for rotulo, ok in requisitos if not ok]
        capacidades[nome] = {"available": not bloqueios, "blocked_by": bloqueios,
                             "description": descricao}

    registra("busca", "responder GET /api/search/semantic", [
        ("qdrant", bool(qdrant and qdrant.up)),
        ("qdrant:sem_chunks", tem_chunks),
        ("embedder", de_pe("embedder")),
    ])
    registra("ingestao", "aceitar POST /api/files/dspace/... e baixar o PDF", [
        ("redis:broker", broker_ok),
        ("celery:download", fila_coberta("download")),
        ("dspace", de_pe("dspace")),
        ("minio", minio_escreve),
    ])
    registra("extracao", "extrair markdown/blocos com o MinerU", [
        ("mineru", de_pe("mineru")),
        ("celery:extract", fila_coberta("extract")),
        ("gpu", gpu_ok),
        ("minio", minio_escreve),
    ])
    registra("indexacao", "gerar chunks/embeddings e gravar no Qdrant", [
        ("embedder", de_pe("embedder")),
        ("qdrant", bool(qdrant and qdrant.up)),
        ("celery:gpu", fila_coberta("gpu")),
        ("minio", minio_escreve),
    ])
    enrich = comp("llm_enrich")
    registra("enriquecimento_llm", "enriquecer metadados por LLM (opcional)", [
        ("llm_enrich", bool(enrich and enrich.status == OK)),
        ("celery:llm", fila_coberta("llm")),
    ])
    return capacidades


def _resumo(componentes: dict[str, Component]) -> dict:
    contagem = {OK: 0, DEGRADED: 0, DOWN: 0, NOT_CONFIGURED: 0, UNKNOWN: 0}
    bloqueando: list[str] = []
    avisos: list[str] = []
    for nome, c in componentes.items():
        contagem[c.status] = contagem.get(c.status, 0) + 1
        if c.status == DOWN and c.critical:
            bloqueando.append(nome)
        elif c.status in (DEGRADED, UNKNOWN) or c.status == DOWN:
            avisos.append(nome)

    if bloqueando:
        geral = DOWN
    elif avisos:
        geral = DEGRADED
    else:
        geral = OK
    return {
        "status": geral,
        "summary": {"total": len(componentes), **contagem},
        "blocking": bloqueando,
        "warnings": avisos,
    }


# --------------------------------------------------------------------------- #
# Coleta
# --------------------------------------------------------------------------- #
def registry(opts: Options) -> dict[str, Callable[[], Component]]:
    """Nome → checagem, na ordem de apresentação. Também é o roteador de
    GET /api/status/{component}."""
    checagens: dict[str, Callable[[], Component]] = {
        "qdrant": check_qdrant,
        "embedder": lambda: check_embedder(probe=opts.probe),
        "mineru": check_mineru,
        "minio": lambda: check_minio(artifacts=opts.artifacts),
        "redis": check_redis,
        "celery": check_celery,
        "gpu": check_gpu,
        "llm_enrich": check_llm_enrich,
        "llm_visual": check_llm_visual,
        "dspace": check_dspace,
        "flower": check_flower,
        "jobs": check_jobs,
        "disk": check_disk,
    }
    return {nome: checagens[nome] for nome in COMPONENT_ORDER}


def run_check(nome: str, opts: Optional[Options] = None) -> Component:
    """Roda uma checagem, convertendo um erro inesperado em `unknown`.

    A distinção importa: `down` é veredito sobre a infraestrutura; `unknown` é a
    sondagem que quebrou (bug, dependência ausente) e não autoriza concluir nada
    sobre o serviço."""
    checagem = registry(opts or Options())[nome]
    try:
        return checagem()
    except Exception as exc:  # pragma: no cover - rede de segurança da rota
        log.warning("[status] checagem '%s' falhou: %s", nome, exc, exc_info=True)
        return Component(name=nome, role="", critical=False, status=UNKNOWN,
                         message=f"a checagem falhou — {_motivo(exc)}")


def check_cached(nome: str, opts: Optional[Options] = None, *,
                 fresh: bool = False) -> tuple[Component, float]:
    """Uma checagem com o mesmo cache por TTL do snapshot.

    Existe para o polling dirigido de `/api/status/{component}` não sair mais caro que
    o snapshot inteiro: sem cache, um consumidor repetindo `/api/status/celery` pagaria
    uma janela de broadcast por requisição. Devolve (componente, idade em segundos)."""
    opts = opts or Options()
    chave = ("component", nome, opts.probe, opts.artifacts)
    if not fresh:
        cacheado, idade = _snapshot_cache.get(chave, settings.STATUS_CACHE_TTL_SECONDS)
        if cacheado is not None:
            return cacheado, idade
    componente = run_check(nome, opts)
    _snapshot_cache.put(chave, componente)
    return componente, 0.0


def collect(opts: Optional[Options] = None, *, fresh: bool = False) -> tuple[Snapshot, float]:
    """Snapshot completo. Devolve (snapshot, idade em segundos) — idade > 0 é resposta
    de cache, o que mantém uma consulta frequente barata (ver STATUS_CACHE_TTL_SECONDS)."""
    opts = opts or Options()
    chave = (opts.probe, opts.artifacts)
    if not fresh:
        cacheado, idade = _snapshot_cache.get(chave, settings.STATUS_CACHE_TTL_SECONDS)
        if cacheado is not None:
            return cacheado, idade

    checagens = registry(opts)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(checagens),
                            thread_name_prefix="status") as pool:
        futuros = {nome: pool.submit(run_check, nome, opts) for nome in checagens}
        # A ordem do dict é a de COMPONENT_ORDER, não a de término das threads.
        componentes = {nome: futuro.result() for nome, futuro in futuros.items()}
    snapshot = Snapshot(components=componentes, generated_at=_agora(),
                        took_ms=(time.perf_counter() - t0) * 1000, options=opts)
    _snapshot_cache.put(chave, snapshot)
    return snapshot, 0.0
