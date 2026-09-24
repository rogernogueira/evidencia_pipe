"""Recuperação POR DOCUMENTO do AI Summary (POST /api/search/summarize).

Com `documents` preenchida, cada UUID gera uma consulta INDEPENDENTE ao Qdrant com o
seu próprio teto de `limit` chunks (o k por documento), e o teto de diversidade
(MAX_CHUNKS_PER_DOCUMENT) sai de cena. Com `documents` vazia vale a regra original:
uma única recuperação global.

Sem Qdrant, embedder ou LLM reais: o SemanticSearch é um dublê que registra as
chamadas e `_call_llm` é monkeypatched.
"""

import os
import sys

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from backend.api.dependencies import get_semantic_search, get_summary_service  # noqa: E402
from backend.core.config import SUMMARY_MAX_DOCUMENTS  # noqa: E402
from backend.api.routes import search as search_route  # noqa: E402
from backend.core.schemas import DocumentRef  # noqa: E402
from backend.services import summary_service as svc  # noqa: E402

LIMPO = "O programa ampliou a cobertura [1][2]."

UUID_A = "fdcbf385-55de-44a4-8473-4a557996e471"
UUID_B = "6aa903f0-c708-4917-b239-306f80bb99f1"


class _Ponto:
    """Ponto cru do Qdrant, como o payload do índice o entrega."""

    def __init__(self, uuid, i):
        self.score = 0.9
        self.payload = {
            "chunk_id": f"{uuid}-c{i}", "document_id": f"{uuid}.pdf", "item_uuid": uuid,
            "item_handle": f"123456789/{i}", "section": "Resultados", "page": i,
            "content": f"Evidência {i} do documento {uuid}.",
        }


class _SemanticFake:
    """Dublê do SemanticSearch: registra cada chamada e devolve `por_uuid` chunks
    para o UUID pedido (ou `globais` chunks quando não há filtro)."""

    def __init__(self, *, por_uuid=3, globais=2, vazios=(), erros=()):
        self.por_uuid = por_uuid
        self.globais = globais
        self.vazios = set(vazios)
        self.erros = set(erros)
        self.chamadas = []

    async def ensure_connected(self):
        return True

    async def search_points(self, query, limit=5, type="hybrid", profile="", uuid=None):
        self.chamadas.append({"query": query, "limit": limit, "type": type, "uuid": uuid})
        if uuid in self.erros:
            raise RuntimeError(f"Qdrant caiu para {uuid}")
        if uuid in self.vazios:
            return []
        if uuid is None:
            return [_Ponto("global", i) for i in range(1, self.globais + 1)]
        return [_Ponto(uuid, i) for i in range(1, self.por_uuid + 1)]


@pytest.fixture
def llm(monkeypatch):
    """Substitui a chamada ao LLM e devolve a lista de mensagens enviadas."""
    enviadas = []

    def fake(system_prompt, user_message):
        enviadas.append(user_message)
        return LIMPO

    monkeypatch.setattr(svc, "_call_llm", fake)
    return enviadas


@pytest.fixture
def anyio_backend():
    return "asyncio"


# --------------------------------------------------------------------------
# SummaryService.summarize com `documents`
# --------------------------------------------------------------------------

@pytest.mark.anyio
async def test_uma_consulta_independente_por_uuid(llm):
    """Dois documentos → duas consultas, cada uma filtrada pelo seu UUID e com o
    `limit` inteiro como k (não fatiado entre os documentos)."""
    fake = _SemanticFake(por_uuid=3)

    r = await svc.SummaryService(fake).summarize(
        "cobertura", limit=3,
        documents=[DocumentRef(uuid=UUID_A, handle="123456789/150"),
                   DocumentRef(uuid=UUID_B, handle="123456789/214")],
    )

    assert [c["uuid"] for c in fake.chamadas] == [UUID_A, UUID_B]
    assert [c["limit"] for c in fake.chamadas] == [3, 3]
    assert r.retrieval.per_document is True
    assert r.retrieval.documents_count == 2
    assert r.retrieval.top == 3


@pytest.mark.anyio
async def test_k_maior_que_o_teto_de_diversidade_produz_k_evidencias_por_doc(llm):
    """O corte de MAX_CHUNKS_PER_DOCUMENT (2) não vale aqui: com k=3 cada documento
    contribui com 3 evidências. Sem isso, k>2 seria inócuo."""
    fake = _SemanticFake(por_uuid=3)

    r = await svc.SummaryService(fake).summarize(
        "cobertura", limit=3,
        documents=[DocumentRef(uuid=UUID_A), DocumentRef(uuid=UUID_B)],
    )

    assert svc.MAX_CHUNKS_PER_DOCUMENT == 2  # o teto continua existindo para a busca global
    assert r.retrieval.evidence_count == 6
    por_doc = {}
    for m in r.mappings:
        por_doc[m.dspace_uuid] = por_doc.get(m.dspace_uuid, 0) + 1
    assert por_doc == {UUID_A: 3, UUID_B: 3}


@pytest.mark.anyio
async def test_evidencias_numeradas_na_ordem_dos_documentos_enviados(llm):
    """A numeração [N] segue a ordem em que o cliente enviou os documentos, mesmo com
    as consultas indo em paralelo — respostas iguais entre chamadas iguais."""
    fake = _SemanticFake(por_uuid=2)

    r = await svc.SummaryService(fake).summarize(
        "cobertura", limit=2,
        documents=[DocumentRef(uuid=UUID_B), DocumentRef(uuid=UUID_A)],
    )

    assert [m.index for m in r.mappings] == [1, 2, 3, 4]
    assert [m.dspace_uuid for m in r.mappings] == [UUID_B, UUID_B, UUID_A, UUID_A]


@pytest.mark.anyio
async def test_applied_filters_ecoa_os_documentos_consultados(llm):
    r = await svc.SummaryService(_SemanticFake()).summarize(
        "cobertura",
        documents=[DocumentRef(uuid=UUID_A, handle="123456789/150"),
                   DocumentRef(uuid=UUID_B)],
    )

    assert r.applied_filters == {
        "documents": [
            {"uuid": UUID_A, "handle": "123456789/150"},
            {"uuid": UUID_B},
        ]
    }


@pytest.mark.anyio
async def test_uuid_repetido_nao_gera_consulta_duplicada(llm):
    fake = _SemanticFake(por_uuid=2)

    r = await svc.SummaryService(fake).summarize(
        "cobertura",
        documents=[DocumentRef(uuid=UUID_A), DocumentRef(uuid=UUID_A),
                   DocumentRef(uuid=f"  {UUID_A}  ")],
    )

    assert [c["uuid"] for c in fake.chamadas] == [UUID_A]
    assert r.retrieval.documents_count == 1


@pytest.mark.anyio
async def test_documento_sem_chunks_nao_impede_a_sintese(llm):
    """Um documento que não casa nada some das evidências; os outros seguem."""
    fake = _SemanticFake(por_uuid=2, vazios=[UUID_A])

    r = await svc.SummaryService(fake).summarize(
        "cobertura", documents=[DocumentRef(uuid=UUID_A), DocumentRef(uuid=UUID_B)],
    )

    assert {m.dspace_uuid for m in r.mappings} == {UUID_B}
    assert r.retrieval.documents_count == 2  # foi consultado, só não trouxe nada
    assert r.summary == LIMPO


@pytest.mark.anyio
async def test_falha_num_documento_nao_derruba_os_demais(llm):
    """O retrieval de um documento que estoura é registrado e ignorado — a síntese
    sai com o que os outros trouxeram, em vez de virar erro para o usuário."""
    fake = _SemanticFake(por_uuid=2, erros=[UUID_A])

    r = await svc.SummaryService(fake).summarize(
        "cobertura", documents=[DocumentRef(uuid=UUID_A), DocumentRef(uuid=UUID_B)],
    )

    assert {m.dspace_uuid for m in r.mappings} == {UUID_B}
    assert r.summary == LIMPO


@pytest.mark.anyio
async def test_sem_nenhuma_evidencia_nao_chama_o_llm(llm):
    fake = _SemanticFake(vazios=[UUID_A])

    r = await svc.SummaryService(fake).summarize(
        "cobertura", documents=[DocumentRef(uuid=UUID_A)],
    )

    assert r.summary == "" and r.mappings == []
    assert llm == []
    # Mesmo sem evidências, o contrato diz o que foi consultado.
    assert r.applied_filters == {"documents": [{"uuid": UUID_A}]}
    assert r.retrieval.per_document is True


# --------------------------------------------------------------------------
# `documents` vazia → a regra atual, intacta
# --------------------------------------------------------------------------

@pytest.mark.anyio
@pytest.mark.parametrize("documents", [None, []])
async def test_documents_vazia_mantem_a_busca_global(llm, documents):
    """Uma única consulta, sem filtro de UUID, com `limit` como total — e sem os
    campos de por-documento marcados na resposta."""
    fake = _SemanticFake(globais=2)

    r = await svc.SummaryService(fake).summarize("cobertura", limit=5, documents=documents)

    assert len(fake.chamadas) == 1
    assert fake.chamadas[0]["uuid"] is None
    assert fake.chamadas[0]["limit"] == 5
    assert r.applied_filters == {}
    assert r.retrieval.per_document is False
    assert r.retrieval.documents_count == 0


@pytest.mark.anyio
async def test_busca_global_mantem_o_teto_de_diversidade(llm):
    """Sem `documents`, o teto de 2 chunks por documento continua valendo: 4 chunks do
    MESMO documento viram 2 evidências."""
    class _MesmoDoc(_SemanticFake):
        async def search_points(self, query, limit=5, type="hybrid", profile="", uuid=None):
            self.chamadas.append({"query": query, "limit": limit, "type": type, "uuid": uuid})
            return [_Ponto("global", i) for i in range(1, 5)]

    r = await svc.SummaryService(_MesmoDoc()).summarize("cobertura", limit=4)

    assert r.retrieval.evidence_count == svc.MAX_CHUNKS_PER_DOCUMENT == 2


# --------------------------------------------------------------------------
# POST /api/search/summarize
# --------------------------------------------------------------------------

def _client(fake, *, llm_ok=True):
    """App mínimo com o router de busca e ambas as dependências sobrescritas."""
    app = FastAPI()
    app.include_router(search_route.router)
    servico = svc.SummaryService(fake)
    servico.llm_available = staticmethod(lambda: llm_ok)
    app.dependency_overrides[get_semantic_search] = lambda: fake
    app.dependency_overrides[get_summary_service] = lambda: servico
    return TestClient(app)


def test_post_com_documents_responde_a_sintese(llm):
    fake = _SemanticFake(por_uuid=2)
    resp = _client(fake).post("/api/search/summarize", json={
        "q": "cobertura",
        "limit": 2,
        "documents": [
            {"uuid": UUID_A, "handle": "123456789/150"},
            {"uuid": UUID_B, "handle": "123456789/214"},
        ],
    })

    assert resp.status_code == 200
    body = resp.json()
    assert body["summary"] == LIMPO
    assert body["retrieval"]["per_document"] is True
    assert body["retrieval"]["documents_count"] == 2
    assert len(body["mappings"]) == 4
    assert [c["uuid"] for c in fake.chamadas] == [UUID_A, UUID_B]


def test_post_sem_documents_cai_na_regra_atual(llm):
    fake = _SemanticFake(globais=2)
    resp = _client(fake).post("/api/search/summarize", json={"q": "cobertura"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["retrieval"]["per_document"] is False
    assert body["applied_filters"] == {}
    assert fake.chamadas[0]["uuid"] is None
    assert fake.chamadas[0]["limit"] == 5  # default do schema


def test_post_usa_os_defaults_do_schema(llm):
    fake = _SemanticFake()
    resp = _client(fake).post("/api/search/summarize", json={
        "q": "cobertura", "documents": [{"uuid": UUID_A}],
    })

    body = resp.json()
    assert body["language"] == "pt-BR"
    assert body["retrieval"]["type"] == "hybrid"
    assert body["retrieval"]["fusion"] == "rrf"
    assert fake.chamadas[0]["type"] == "hybrid"


@pytest.mark.parametrize("body", [
    {},                                             # q obrigatório
    {"q": ""},                                      # q vazio
    {"q": "x", "limit": 0},                         # fora da faixa
    {"q": "x", "limit": 21},
    {"q": "x", "documents": [{"handle": "123/1"}]},  # uuid obrigatório
    {"q": "x", "documents": [{"uuid": ""}]},         # uuid vazio
    {"q": "x", "documents": "nao-e-lista"},
])
def test_post_body_invalido_e_422(llm, body):
    fake = _SemanticFake()
    resp = _client(fake).post("/api/search/summarize", json=body)

    assert resp.status_code == 422
    assert fake.chamadas == []


def test_post_aceita_a_lista_no_teto(llm):
    """SUMMARY_MAX_DOCUMENTS documentos passam — o teto é inclusivo."""
    fake = _SemanticFake(por_uuid=1)
    resp = _client(fake).post("/api/search/summarize", json={
        "q": "cobertura",
        "documents": [{"uuid": f"uuid-{i}"} for i in range(SUMMARY_MAX_DOCUMENTS)],
    })

    assert resp.status_code == 200
    assert resp.json()["retrieval"]["documents_count"] == SUMMARY_MAX_DOCUMENTS
    assert len(fake.chamadas) == SUMMARY_MAX_DOCUMENTS


def test_post_acima_do_teto_e_422_sem_consultar_o_indice(llm):
    """Cada documento é uma consulta ao Qdrant numa requisição síncrona: acima do teto
    a requisição é recusada na validação, ANTES de qualquer retrieval."""
    fake = _SemanticFake()
    resp = _client(fake).post("/api/search/summarize", json={
        "q": "cobertura",
        "documents": [{"uuid": f"uuid-{i}"} for i in range(SUMMARY_MAX_DOCUMENTS + 1)],
    })

    assert resp.status_code == 422
    assert fake.chamadas == []
    assert str(SUMMARY_MAX_DOCUMENTS) in resp.text


def test_post_sem_llm_configurado_e_503(llm):
    fake = _SemanticFake()
    resp = _client(fake, llm_ok=False).post("/api/search/summarize", json={"q": "x"})

    assert resp.status_code == 503
    assert "LLM" in resp.json()["error"]
    assert fake.chamadas == []


def test_post_com_busca_indisponivel_e_503(llm):
    class _Fora(_SemanticFake):
        async def ensure_connected(self):
            return False

    fake = _Fora()
    resp = _client(fake).post("/api/search/summarize", json={"q": "x"})

    assert resp.status_code == 503
    assert fake.chamadas == []
