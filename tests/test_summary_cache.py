"""Cache do AI Summary (backend/services/summary_cache.py + SummaryService.summarize).

Hit não chama retrieval nem LLM; a geração do índice invalida; síntese vazia ou
contaminada não entra; requisições idênticas simultâneas dividem uma chamada ao LLM;
Redis fora do ar cai para memória. Redis = fakeredis, LLM e Qdrant = dublês.
"""

import asyncio
import os
import sys

import fakeredis
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from backend.core.schemas import DocumentRef  # noqa: E402
from backend.services import summary_cache as cache_mod  # noqa: E402
from backend.services import summary_service as svc  # noqa: E402
from tests.test_summary_per_document import LIMPO, UUID_A, UUID_B, _SemanticFake  # noqa: E402


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def redis_fake(monkeypatch):
    client = fakeredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(cache_mod, "_redis", client)
    return client


@pytest.fixture
def sem_redis(monkeypatch):
    """Redis indisponível: o cliente nunca conecta e o cache usa a memória."""
    monkeypatch.setattr(cache_mod, "_redis", None)
    monkeypatch.setattr(cache_mod, "_get_redis", lambda: None)


@pytest.fixture
def llm(monkeypatch):
    chamadas = []
    resposta = {"texto": LIMPO}

    def fake(system_prompt, user_message):
        chamadas.append(user_message)
        return resposta["texto"]

    monkeypatch.setattr(svc, "_call_llm", fake)
    fake.chamadas = chamadas
    fake.resposta = resposta
    return fake


def _service(fake=None):
    return svc.SummaryService(fake or _SemanticFake(), cache=cache_mod.SummaryCache())


@pytest.mark.anyio
async def test_segunda_chamada_igual_vem_do_cache(redis_fake, llm):
    fake = _SemanticFake()
    s = _service(fake)

    r1 = await s.summarize("cobertura")
    r2 = await s.summarize("  cobertura ")

    assert len(llm.chamadas) == 1
    assert len(fake.chamadas) == 1  # hit também pula o retrieval
    assert r1.cached is False and r2.cached is True
    assert r2.summary == r1.summary and r2.mappings == r1.mappings
    assert r2.query == "  cobertura "  # ecoa a consulta desta requisição
    assert any(k.startswith("summary:resp:") for k in redis_fake.keys())


@pytest.mark.anyio
async def test_parametros_diferentes_nao_compartilham_entrada(redis_fake, llm):
    s = _service()

    await s.summarize("cobertura")
    await s.summarize("cobertura", limit=3)
    await s.summarize("cobertura", language="en")
    await s.summarize("cobertura", documents=[DocumentRef(uuid=UUID_A)])
    # A ordem dos documentos define a numeração [N]: é outra síntese.
    await s.summarize("cobertura", documents=[DocumentRef(uuid=UUID_A), DocumentRef(uuid=UUID_B)])
    await s.summarize("cobertura", documents=[DocumentRef(uuid=UUID_B), DocumentRef(uuid=UUID_A)])

    assert len(llm.chamadas) == 6


@pytest.mark.anyio
async def test_reindexar_invalida_o_cache(redis_fake, llm):
    s = _service()

    await s.summarize("cobertura")
    cache_mod.bump_index_generation()
    r = await s.summarize("cobertura")

    assert len(llm.chamadas) == 2
    assert r.cached is False


@pytest.mark.anyio
async def test_sintese_contaminada_nao_entra_no_cache(redis_fake, llm):
    llm.resposta["texto"] = "好的，我需要分析这些证据并用葡萄牙语回答用户的问题。"
    s = _service()

    await s.summarize("cobertura")
    n = len(llm.chamadas)  # 2: a síntese e a segunda tentativa
    llm.resposta["texto"] = LIMPO
    r = await s.summarize("cobertura")

    assert len(llm.chamadas) == n + 1
    assert r.summary == LIMPO


@pytest.mark.anyio
async def test_sem_evidencias_nao_entra_no_cache(redis_fake, llm):
    fake = _SemanticFake(globais=0)
    s = _service(fake)

    await s.summarize("nada")
    await s.summarize("nada")

    # Síntese vazia não entra no cache → a 2ª chamada refaz o retrieval. Cada summarize
    # faz 2 consultas: a filtrada por achado volta vazia e dispara o fallback sem filtro.
    assert len(fake.chamadas) == 4
    assert all(c["uuid"] is None for c in fake.chamadas)
    assert not redis_fake.keys("summary:resp:*")


@pytest.mark.anyio
async def test_requisicoes_simultaneas_dividem_uma_chamada_ao_llm(redis_fake, monkeypatch):
    chamadas = []

    def lento(system_prompt, user_message):
        chamadas.append(user_message)
        import time
        time.sleep(0.2)
        return LIMPO

    monkeypatch.setattr(svc, "_call_llm", lento)
    s = _service()

    rs = await asyncio.gather(*(s.summarize("cobertura") for _ in range(5)))

    assert len(chamadas) == 1
    assert sorted(r.cached for r in rs) == [False, True, True, True, True]
    assert s._inflight == {}


@pytest.mark.anyio
async def test_sem_redis_usa_memoria(sem_redis, llm):
    s = _service()

    await s.summarize("cobertura")
    r = await s.summarize("cobertura")
    cache_mod.bump_index_generation()
    r3 = await s.summarize("cobertura")

    assert r.cached is True and r3.cached is False
    assert len(llm.chamadas) == 2


def test_memoria_respeita_ttl_e_teto(sem_redis):
    c = cache_mod.SummaryCache(ttl_seconds=60, max_entries=2)
    resp = svc.SummaryResponse(query="q", retrieval=svc.RetrievalMetadata(), summary="x")

    for k in ("a", "b", "c"):
        c.set(k, resp)
    assert c.get("a") is None  # LRU: a mais antiga saiu
    assert c.get("c") is not None

    c._mem["c"] = (0.0, c._mem["c"][1])  # expirada
    assert c.get("c") is None


@pytest.mark.anyio
async def test_sem_cache_injetado_nao_guarda_nada(redis_fake, llm):
    s = svc.SummaryService(_SemanticFake())

    await s.summarize("cobertura")
    await s.summarize("cobertura")

    assert len(llm.chamadas) == 2
    assert not redis_fake.keys("summary:resp:*")
