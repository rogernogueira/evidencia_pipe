"""Testes do status da infraestrutura (backend/services/infra_status.py e a rota).

Sem Qdrant, MinIO, Redis, Celery ou GPU reais: as sondagens de rede são substituídas
(monkeypatch em `_http_get`), o store do MinIO é um dublê com `iter_prefix`, e as
regras de agregação (status geral, capacidades, nível de detalhe) são exercitadas com
componentes montados à mão. Como em test_search_routes.py, a rota é testada num app
mínimo — só o `status.router` — para não puxar lifespan nem StaticFiles.
"""

import os
import sys

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from backend.api.routes import status as status_route  # noqa: E402
from backend.core import config as settings  # noqa: E402
from backend.services import infra_status as st  # noqa: E402

DA_BORDA = {"X-Forwarded-For": "200.130.0.2"}  # o que o mod_proxy sempre acrescenta


@pytest.fixture(autouse=True)
def sem_cache():
    """Os caches são globais ao processo — cada teste começa com eles vazios."""
    st.clear_caches()
    yield
    st.clear_caches()


def comp(nome="x", status=st.OK, *, critical=False, detail=None, internal=None):
    return st.Component(name=nome, role="papel", critical=critical, status=status,
                        detail=detail or {}, internal=internal or {})


def make_client():
    app = FastAPI()
    app.include_router(status_route.router)
    return TestClient(app)


# --------------------------------------------------------------------------
# Agregação: status geral, bloqueios e avisos
# --------------------------------------------------------------------------
def test_tudo_ok_da_status_ok():
    resumo = st._resumo({"a": comp("a", st.OK, critical=True), "b": comp("b", st.OK)})

    assert resumo["status"] == st.OK
    assert resumo["blocking"] == [] and resumo["warnings"] == []
    assert resumo["summary"]["ok"] == 2


def test_critico_fora_derruba_o_geral_e_entra_em_blocking():
    resumo = st._resumo({"qdrant": comp("qdrant", st.DOWN, critical=True),
                         "jobs": comp("jobs", st.OK)})

    assert resumo["status"] == st.DOWN
    assert resumo["blocking"] == ["qdrant"]
    assert "qdrant" not in resumo["warnings"]


def test_nao_critico_fora_apenas_degrada():
    resumo = st._resumo({"dspace": comp("dspace", st.DOWN, critical=False),
                         "qdrant": comp("qdrant", st.OK, critical=True)})

    assert resumo["status"] == st.DEGRADED
    assert resumo["blocking"] == [] and resumo["warnings"] == ["dspace"]


def test_not_configured_nao_conta_como_problema():
    resumo = st._resumo({"llm_enrich": comp("llm_enrich", st.NOT_CONFIGURED),
                         "flower": comp("flower", st.NOT_CONFIGURED)})

    assert resumo["status"] == st.OK
    assert resumo["warnings"] == []
    assert resumo["summary"]["not_configured"] == 2


def test_checagem_quebrada_vira_unknown_e_avisa():
    """`unknown` é a sondagem que falhou, não a infra: não pode virar `down`."""
    resumo = st._resumo({"redis": comp("redis", st.UNKNOWN, critical=True)})

    assert resumo["status"] == st.DEGRADED
    assert resumo["warnings"] == ["redis"] and resumo["blocking"] == []


# --------------------------------------------------------------------------
# Capacidades derivadas
# --------------------------------------------------------------------------
def componentes_saudaveis():
    return {
        "qdrant": comp("qdrant", st.OK, critical=True, detail={"chunks": 10}),
        "embedder": comp("embedder", st.OK, critical=True),
        "mineru": comp("mineru", st.OK, critical=True),
        "minio": comp("minio", st.OK, critical=True, detail={"writable": True}),
        "redis": comp("redis", st.OK, critical=True,
                      detail={"dbs": {"broker": {"ok": True}}}),
        "celery": comp("celery", st.OK, critical=True, detail={"queues_missing": []}),
        "gpu": comp("gpu", st.OK, critical=True),
        "llm_enrich": comp("llm_enrich", st.OK),
        "dspace": comp("dspace", st.OK),
    }


def test_capacidades_todas_disponiveis_com_tudo_de_pe():
    caps = st.capabilities(componentes_saudaveis())

    assert all(c["available"] for c in caps.values()), caps
    assert set(caps) == {"busca", "ingestao", "extracao", "indexacao", "enriquecimento_llm"}


def test_collection_vazia_bloqueia_a_busca_mas_nao_a_indexacao():
    componentes = componentes_saudaveis()
    componentes["qdrant"] = comp("qdrant", st.OK, critical=True, detail={"chunks": 0})

    caps = st.capabilities(componentes)

    assert caps["busca"]["available"] is False
    assert caps["busca"]["blocked_by"] == ["qdrant:sem_chunks"]
    assert caps["indexacao"]["available"] is True


def test_embedder_fora_bloqueia_busca_e_indexacao():
    componentes = componentes_saudaveis()
    componentes["embedder"] = comp("embedder", st.DOWN, critical=True)

    caps = st.capabilities(componentes)

    assert caps["busca"]["blocked_by"] == ["embedder"]
    assert caps["indexacao"]["blocked_by"] == ["embedder"]
    assert caps["extracao"]["available"] is True  # extração não embeda


def test_fila_sem_consumidor_bloqueia_a_capacidade_daquele_estagio():
    componentes = componentes_saudaveis()
    componentes["celery"] = comp("celery", st.DEGRADED, critical=True,
                                 detail={"queues_missing": ["gpu"]})

    caps = st.capabilities(componentes)

    assert caps["indexacao"]["blocked_by"] == ["celery:gpu"]
    assert caps["extracao"]["available"] is True
    assert caps["busca"]["available"] is True  # busca não passa por worker


def test_dspace_fora_bloqueia_so_a_ingestao():
    componentes = componentes_saudaveis()
    componentes["dspace"] = comp("dspace", st.DOWN)

    caps = st.capabilities(componentes)

    assert caps["ingestao"]["blocked_by"] == ["dspace"]
    assert caps["extracao"]["available"] and caps["indexacao"]["available"]


def test_componente_degradado_ainda_conta_como_de_pe():
    """`degraded` funciona com ressalva — não pode zerar a capacidade."""
    componentes = componentes_saudaveis()
    componentes["mineru"] = comp("mineru", st.DEGRADED, critical=True)

    assert st.capabilities(componentes)["extracao"]["available"] is True


def test_gpu_desligada_de_proposito_nao_bloqueia_extracao():
    """GPU_MANAGER_ENABLED=false é `not_configured`: a extração roda sem o lock."""
    componentes = componentes_saudaveis()
    componentes["gpu"] = comp("gpu", st.NOT_CONFIGURED)

    assert st.capabilities(componentes)["extracao"]["available"] is True


def test_enrich_sem_chave_indisponivel_sem_afetar_o_resto():
    componentes = componentes_saudaveis()
    componentes["llm_enrich"] = comp("llm_enrich", st.NOT_CONFIGURED)

    caps = st.capabilities(componentes)

    assert caps["enriquecimento_llm"]["available"] is False
    assert caps["indexacao"]["available"] is True


# --------------------------------------------------------------------------
# Qdrant — o veredito sobre a collection e a contagem de chunks
# --------------------------------------------------------------------------
def responde(monkeypatch, rotas: dict):
    """Substitui as chamadas HTTP: caminho (sufixo da URL) → (status, corpo).
    Um valor `Exception` é levantado, simulando servidor inalcançável."""
    def fake_get(url, **kw):
        for sufixo, resposta in rotas.items():
            if url.endswith(sufixo):
                if isinstance(resposta, BaseException):
                    raise resposta
                return resposta
        return 404, None

    monkeypatch.setattr(st, "_http_get", fake_get)


def colecao(pontos=42, *, dense=True, sparse=True, status="green", dim=1024):
    vetores = {"dense": {"size": dim, "distance": "Cosine"}} if dense else {}
    return 200, {"result": {
        "status": status, "points_count": pontos, "indexed_vectors_count": pontos,
        "segments_count": 2, "optimizer_status": "ok",
        "config": {"params": {"vectors": vetores,
                              "sparse_vectors": {"sparse": {}} if sparse else {}}},
    }}


def test_qdrant_ok_reporta_a_quantidade_de_chunks(monkeypatch):
    responde(monkeypatch, {
        "/": (200, {"version": "1.17.1"}),
        f"/collections/{settings.QDRANT_COLLECTION}": colecao(42427),
        "/collections": (200, {"result": {"collections": [{"name": settings.QDRANT_COLLECTION}]}}),
    })

    c = st.check_qdrant()

    assert c.status == st.OK
    assert c.detail["chunks"] == 42427
    assert c.detail["collection"] == settings.QDRANT_COLLECTION
    assert c.detail["sparse_vector"]["present"] is True
    assert c.internal["version"] == "1.17.1"


def test_qdrant_inalcancavel_e_down_com_dica(monkeypatch):
    responde(monkeypatch, {"/": ConnectionError("recusado")})

    c = st.check_qdrant()

    assert c.status == st.DOWN and c.critical is True
    assert "busca" in c.hint


def test_collection_inexistente_e_down_e_lista_as_que_existem(monkeypatch):
    responde(monkeypatch, {
        f"/collections/{settings.QDRANT_COLLECTION}": (404, {"status": {"error": "not found"}}),
        "/collections": (200, {"result": {"collections": [{"name": "outra"}]}}),
        "/": (200, {"version": "1.17.1"}),
    })

    c = st.check_qdrant()

    assert c.status == st.DOWN
    assert c.internal["collections"] == ["outra"]
    assert "QDRANT_COLLECTION" in c.hint


def test_collection_vazia_e_degradada_nao_down(monkeypatch):
    responde(monkeypatch, {
        "/": (200, {"version": "1.17.1"}),
        f"/collections/{settings.QDRANT_COLLECTION}": colecao(0),
        "/collections": (200, {"result": {"collections": []}}),
    })

    c = st.check_qdrant()

    assert c.status == st.DEGRADED
    assert c.detail["chunks"] == 0


def test_falta_do_vetor_esparso_e_down(monkeypatch):
    """Sem o vetor nomeado 'sparse' não há fusão RRF — é falha, não ressalva."""
    responde(monkeypatch, {
        "/": (200, {"version": "1.17.1"}),
        f"/collections/{settings.QDRANT_COLLECTION}": colecao(10, sparse=False),
        "/collections": (200, {"result": {"collections": []}}),
    })

    c = st.check_qdrant()

    assert c.status == st.DOWN
    assert "sparse" in c.message


def test_dimensao_densa_divergente_degrada(monkeypatch):
    responde(monkeypatch, {
        "/": (200, {"version": "1.17.1"}),
        f"/collections/{settings.QDRANT_COLLECTION}": colecao(10, dim=768),
        "/collections": (200, {"result": {"collections": []}}),
    })

    c = st.check_qdrant()

    assert c.status == st.DEGRADED
    assert "768" in c.message


# --------------------------------------------------------------------------
# MinerU e embedding
# --------------------------------------------------------------------------
def test_mineru_saudavel_reporta_o_backend_em_uso(monkeypatch):
    responde(monkeypatch, {"/health": (200, {
        "status": "healthy", "version": "3.4.4", "protocol_version": 2,
        "queued_tasks": 1, "processing_tasks": 2, "completed_tasks": 82, "failed_tasks": 0,
        "max_concurrent_requests": 3})})

    c = st.check_mineru()

    assert c.status == st.OK
    assert c.detail["backend"] == settings.MINERU_BACKEND
    assert c.detail["queued_tasks"] == 1
    assert c.internal["version"] == "3.4.4"


def test_mineru_porta_de_outro_servico_e_down(monkeypatch):
    """200 sem JSON do MinerU é outro serviço na porta — não conta como de pé."""
    responde(monkeypatch, {"/health": (200, None)})

    c = st.check_mineru()

    assert c.status == st.DOWN
    assert "MINERU_API_URL" in c.hint


def test_embedder_ok_quando_os_dois_endpoints_servem_o_modelo(monkeypatch):
    modelos = (200, {"data": [{"id": settings.EMBED_API_MODEL, "max_model_len": 8192}]})
    responde(monkeypatch, {"/v1/models": modelos, "/version": (200, {"version": "0.26.0"})})

    c = st.check_embedder()

    assert c.status == st.OK
    assert c.detail["dense"]["model_served"] is True
    assert c.detail["sparse"]["model_served"] is True


def test_embedder_com_nome_de_modelo_divergente_e_down(monkeypatch):
    responde(monkeypatch, {"/v1/models": (200, {"data": [{"id": "outro/modelo"}]}),
                           "/version": (200, {"version": "0.26.0"})})

    c = st.check_embedder()

    assert c.status == st.DOWN
    assert "EMBED_API_MODEL" in c.hint


def test_embedder_sem_probe_nao_faz_round_trip(monkeypatch):
    """Sem `probe=true` nenhum POST sai: o custo é só o de listar os modelos."""
    responde(monkeypatch, {"/v1/models": (200, {"data": [{"id": settings.EMBED_API_MODEL}]}),
                           "/version": (200, {"version": "0.26.0"})})

    def nao_deveria(*a, **kw):
        raise AssertionError("probe desligado não deve postar")

    monkeypatch.setattr(st, "_http_post_json", nao_deveria)

    assert st.check_embedder(probe=False).status == st.OK


def test_probe_detecta_esparso_desalinhado(monkeypatch):
    """Pesos em número diferente dos tokens internos = pooling task errada."""
    responde(monkeypatch, {"/v1/models": (200, {"data": [{"id": settings.EMBED_API_MODEL}]}),
                           "/version": (200, {"version": "0.26.0"})})

    def fake_post(url, payload, **kw):
        if url.endswith("/v1/embeddings"):
            return {"data": [{"embedding": [1.0] + [0.0] * (st.DENSE_DIM - 1)}]}
        return {"data": [{"data": [0.5]}]}  # 1 peso para 8 tokens internos

    monkeypatch.setattr(st, "_http_post_json", fake_post)

    c = st.check_embedder(probe=True)

    assert c.status == st.DOWN
    assert "token_classify" in c.hint
    assert c.detail["probe"]["dense"]["dim"] == st.DENSE_DIM


# --------------------------------------------------------------------------
# MinIO — contagem de artefatos
# --------------------------------------------------------------------------
class StoreFalso:
    """Dublê com só o que a contagem usa."""

    def __init__(self, objetos, saude=None):
        self.objetos = objetos
        self.saude = saude or {"reachable": True, "bucket_exists": True,
                               "readable": True, "writable": True}
        self.varreduras = 0

    def healthcheck(self):
        return self.saude

    def iter_prefix(self, prefix):
        self.varreduras += 1
        for chave, tamanho in self.objetos:
            if chave.startswith(prefix):
                yield chave, tamanho


def objetos_de_exemplo():
    p = settings.MINIO_ARTIFACT_PREFIX
    return [
        (f"{p}/run-1/doc-a/manifest.json", 100),
        (f"{p}/run-1/doc-a/source/original.pdf", 5000),
        (f"{p}/run-1/doc-a/mineru/document.md", 700),
        (f"{p}/run-1/doc-a/mineru/images/fig1.png", 900),
        (f"{p}/run-1/doc-a/indexing/chunks.jsonl", 300),
        (f"{p}/run-2/doc-b/manifest.json", 100),
        (f"{p}/run-2/doc-b/mineru/document.md", 400),
    ]


def test_contagem_agrupa_por_etapa_e_conta_documentos():
    store = StoreFalso(objetos_de_exemplo())

    contagem = st.contar_artefatos(store)

    assert contagem["objects"] == 7
    assert contagem["documents"] == 2          # run-1/doc-a e run-2/doc-b
    assert contagem["manifests"] == 2
    assert contagem["pipeline_runs"] == 2
    assert contagem["by_stage"]["mineru"] == 3
    assert contagem["by_stage"]["source"] == 1
    assert contagem["size_bytes"] == 7500
    assert contagem["truncated"] is False


def test_contagem_para_no_teto_e_avisa(monkeypatch):
    monkeypatch.setattr(settings, "STATUS_ARTIFACT_SCAN_MAX_OBJECTS", 3)

    contagem = st.contar_artefatos(StoreFalso(objetos_de_exemplo()))

    assert contagem["objects"] == 3
    assert contagem["truncated"] is True


def test_contagem_tem_cache_proprio_mais_longo(monkeypatch):
    monkeypatch.setattr(settings, "STATUS_ARTIFACT_SCAN_TTL_SECONDS", 300)
    store = StoreFalso(objetos_de_exemplo())

    st.contar_artefatos(store)
    segunda = st.contar_artefatos(store)

    assert store.varreduras == 1
    assert segunda["objects"] == 7
    assert st.contar_artefatos(store, fresh=True)["objects"] == 7
    assert store.varreduras == 2  # `fresh` refaz a varredura


def test_contagem_que_falha_degrada_o_minio_sem_derrubar(monkeypatch):
    class Explode(StoreFalso):
        def iter_prefix(self, prefix):
            raise OSError("bucket sumiu no meio da listagem")
            yield  # pragma: no cover

    store = Explode([])
    monkeypatch.setattr("backend.services.artifact_store.get_artifact_store", lambda: store)

    c = st.check_minio(artifacts=True)

    assert c.status == st.DEGRADED
    assert "error" in c.detail["artifacts"]


def test_minio_sem_escrita_e_down(monkeypatch):
    store = StoreFalso([], saude={"reachable": True, "bucket_exists": True,
                                  "readable": True, "writable": False})
    monkeypatch.setattr("backend.services.artifact_store.get_artifact_store", lambda: store)
    monkeypatch.setattr(settings, "MINIO_HEALTHCHECK_WRITE", True)

    c = st.check_minio(artifacts=False)

    assert c.status == st.DOWN
    assert "escrita" in c.message


def test_minio_bucket_inexistente_e_down(monkeypatch):
    store = StoreFalso([], saude={"reachable": True, "bucket_exists": False,
                                  "readable": False, "writable": False})
    monkeypatch.setattr("backend.services.artifact_store.get_artifact_store", lambda: store)

    c = st.check_minio(artifacts=False)

    assert c.status == st.DOWN and "bucket" in c.message


# --------------------------------------------------------------------------
# Enriquecimento por LLM — ligado ou não
# --------------------------------------------------------------------------
def test_enrich_sem_chave_e_not_configured(monkeypatch):
    from backend.services import llm_enrich_service as llm

    monkeypatch.setattr(llm, "LLM_ENRICH_API_KEY", "")

    c = st.check_llm_enrich()

    assert c.status == st.NOT_CONFIGURED
    assert c.detail["enabled"] is False and c.detail["active"] is False
    assert c.critical is False  # a indexação não depende do enrich


def test_enrich_com_chave_e_auto_ligado_e_ok(monkeypatch):
    from backend.services import llm_enrich_service as llm

    monkeypatch.setattr(llm, "LLM_ENRICH_API_KEY", "sk-teste")
    monkeypatch.setattr(settings, "LLM_ENRICH_API_KEY", "sk-teste")
    monkeypatch.setattr(settings, "LLM_ENRICH_AUTO", True)
    responde(monkeypatch, {"/models": (200, {"data": [{"id": settings.LLM_ENRICH_MODEL}]})})

    c = st.check_llm_enrich()

    assert c.status == st.OK
    assert c.detail["enabled"] is True
    assert c.detail["auto_after_index"] is True
    assert c.detail["active"] is True
    assert c.detail["model_served"] is True


def test_enrich_com_auto_desligado_degrada_e_explica(monkeypatch):
    """Com chave mas sem LLM_ENRICH_AUTO o enrich só roda sob demanda."""
    from backend.services import llm_enrich_service as llm

    monkeypatch.setattr(llm, "LLM_ENRICH_API_KEY", "sk-teste")
    monkeypatch.setattr(settings, "LLM_ENRICH_API_KEY", "sk-teste")
    monkeypatch.setattr(settings, "LLM_ENRICH_AUTO", False)
    responde(monkeypatch, {"/models": (200, {"data": [{"id": settings.LLM_ENRICH_MODEL}]})})

    c = st.check_llm_enrich()

    assert c.status == st.DEGRADED
    assert c.detail["active"] is False
    assert "enrich" in c.hint


def test_enrich_com_modelo_inexistente_no_provedor_degrada(monkeypatch):
    from backend.services import llm_enrich_service as llm

    monkeypatch.setattr(llm, "LLM_ENRICH_API_KEY", "sk-teste")
    monkeypatch.setattr(settings, "LLM_ENRICH_API_KEY", "sk-teste")
    responde(monkeypatch, {"/models": (200, {"data": [{"id": "outro-modelo"}]})})

    c = st.check_llm_enrich()

    assert c.status == st.DEGRADED
    assert c.detail["model_served"] is False
    assert "LLM_ENRICH_MODEL" in c.hint


def test_enrich_com_chave_rejeitada_e_down(monkeypatch):
    from backend.services import llm_enrich_service as llm

    monkeypatch.setattr(llm, "LLM_ENRICH_API_KEY", "sk-errada")
    monkeypatch.setattr(settings, "LLM_ENRICH_API_KEY", "sk-errada")
    responde(monkeypatch, {"/models": (401, {"error": "unauthorized"})})

    c = st.check_llm_enrich()

    assert c.status == st.DOWN and "rejeitada" in c.message


def test_llm_visual_desligado_e_not_configured(monkeypatch):
    monkeypatch.setattr(settings, "CHUNK_VISUAL_LLM", False)

    c = st.check_llm_visual()

    assert c.status == st.NOT_CONFIGURED and c.detail["enabled"] is False


def test_llm_visual_ligado_sem_chave_degrada(monkeypatch):
    monkeypatch.setattr(settings, "CHUNK_VISUAL_LLM", True)
    monkeypatch.setattr(settings, "LLM_ENRICH_API_KEY", "")

    c = st.check_llm_visual()

    assert c.status == st.DEGRADED and "heurística" in c.message


# --------------------------------------------------------------------------
# Jobs e utilidades
# --------------------------------------------------------------------------
def test_jobs_com_redis_fora_degrada(monkeypatch):
    from backend.services import job_store

    monkeypatch.setattr(job_store, "_get_redis", lambda: None)

    c = st.check_jobs()

    assert c.status == st.DEGRADED
    assert c.detail["storage"] == "memoria"


def test_index_sizes_conta_o_fallback_em_memoria(monkeypatch):
    from backend.services import job_store

    monkeypatch.setattr(job_store, "_get_redis", lambda: None)
    monkeypatch.setattr(job_store, "_active", {"j1": 1.0, "j2": 2.0})
    monkeypatch.setattr(job_store, "_failed", {"j3": 3.0})
    monkeypatch.setattr(job_store, "_succeeded", {})

    assert job_store.index_sizes() == {"backend": "memoria", "active": 2,
                                       "succeeded": 0, "failed": 1}


@pytest.mark.parametrize("url,esperado", [
    ("redis://user:segredo@127.0.0.1:6379/0", "redis://127.0.0.1:6379/0"),
    ("redis://:senha@host:6379/1", "redis://host:6379/1"),
    ("redis://127.0.0.1:6379/2", "redis://127.0.0.1:6379/2"),
    ("http://minio:9000", "http://minio:9000"),
])
def test_url_sai_sem_credenciais(url, esperado):
    assert st.sem_credenciais(url) == esperado


def test_checagem_que_levanta_excecao_vira_unknown(monkeypatch):
    monkeypatch.setattr(st, "check_qdrant", lambda: (_ for _ in ()).throw(RuntimeError("boom")))

    c = st.run_check("qdrant")

    assert c.status == st.UNKNOWN
    assert "boom" in c.message


def test_cache_do_snapshot_evita_remedir(monkeypatch):
    chamadas = []

    def fake_check():
        chamadas.append(1)
        return comp("qdrant", st.OK, critical=True)

    monkeypatch.setattr(st, "COMPONENT_ORDER", ("qdrant",))
    monkeypatch.setattr(st, "check_qdrant", fake_check)
    monkeypatch.setattr(settings, "STATUS_CACHE_TTL_SECONDS", 300)

    primeiro, idade1 = st.collect()
    segundo, idade2 = st.collect()

    assert len(chamadas) == 1
    assert primeiro is segundo and idade1 == 0.0 and idade2 >= 0.0

    st.collect(fresh=True)
    assert len(chamadas) == 2


# --------------------------------------------------------------------------
# A rota
# --------------------------------------------------------------------------
def instala_snapshot(monkeypatch, componentes: dict):
    """Faz a rota responder a partir de componentes montados à mão."""
    monkeypatch.setattr(st, "COMPONENT_ORDER", tuple(componentes))
    monkeypatch.setattr(st, "registry",
                        lambda opts: {n: (lambda c=c: c) for n, c in componentes.items()})


def test_rota_devolve_200_e_o_corpo_completo(monkeypatch):
    instala_snapshot(monkeypatch, componentes_saudaveis())

    resp = make_client().get("/api/status")

    assert resp.status_code == 200
    corpo = resp.json()
    assert corpo["status"] == st.OK
    assert corpo["detail_level"] == "full"
    assert set(corpo) >= {"summary", "capabilities", "components", "config", "api",
                          "blocking", "warnings", "generated_at", "took_ms"}
    assert corpo["config"]["search"]["collection"] == settings.QDRANT_COLLECTION


def test_rota_devolve_503_quando_um_critico_esta_fora(monkeypatch):
    componentes = componentes_saudaveis()
    componentes["qdrant"] = comp("qdrant", st.DOWN, critical=True)
    instala_snapshot(monkeypatch, componentes)

    resp = make_client().get("/api/status")

    assert resp.status_code == 503
    assert resp.json()["blocking"] == ["qdrant"]


def test_nao_critico_fora_continua_200(monkeypatch):
    componentes = componentes_saudaveis()
    componentes["dspace"] = comp("dspace", st.DOWN)
    instala_snapshot(monkeypatch, componentes)

    resp = make_client().get("/api/status")

    assert resp.status_code == 200
    assert resp.json()["status"] == st.DEGRADED


def test_request_da_borda_nao_recebe_a_topologia(monkeypatch):
    componentes = {"qdrant": comp("qdrant", st.OK, critical=True,
                                  detail={"chunks": 1},
                                  internal={"url": "http://192.168.105.8:6333"})}
    instala_snapshot(monkeypatch, componentes)

    resp = make_client().get("/api/status", headers=DA_BORDA)

    corpo = resp.json()
    assert corpo["detail_level"] == "public"
    assert "internal" not in corpo["components"]["qdrant"]
    assert corpo["components"]["qdrant"]["detail"]["chunks"] == 1  # métrica continua
    assert "host" not in corpo["api"] and "pid" not in corpo["api"]


def test_token_interno_libera_a_topologia_pela_borda(monkeypatch):
    componentes = {"qdrant": comp("qdrant", st.OK, critical=True,
                                  internal={"url": "http://qdrant:6333"})}
    instala_snapshot(monkeypatch, componentes)
    monkeypatch.setattr(settings, "INTERNAL_API_TOKEN", "sesamo")

    client = make_client()
    sem_token = client.get("/api/status", headers=DA_BORDA)
    com_token = client.get("/api/status", headers={**DA_BORDA, "X-Internal-Token": "sesamo"})

    assert sem_token.json()["detail_level"] == "public"
    assert com_token.json()["detail_level"] == "full"
    assert com_token.json()["components"]["qdrant"]["internal"]["url"] == "http://qdrant:6333"


def test_token_errado_nao_libera(monkeypatch):
    instala_snapshot(monkeypatch, {"qdrant": comp("qdrant", st.OK, critical=True)})
    monkeypatch.setattr(settings, "INTERNAL_API_TOKEN", "sesamo")

    resp = make_client().get("/api/status",
                             headers={**DA_BORDA, "X-Internal-Token": "chute"})

    assert resp.json()["detail_level"] == "public"


def test_probe_e_fresh_sao_ignorados_para_quem_vem_da_borda(monkeypatch):
    """Sondagem caríssima não pode ser disparada por consumidor anônimo."""
    vistos = []

    def registry_espiao(opts):
        vistos.append(opts)
        return {"qdrant": lambda: comp("qdrant", st.OK, critical=True)}

    monkeypatch.setattr(st, "COMPONENT_ORDER", ("qdrant",))
    monkeypatch.setattr(st, "registry", registry_espiao)

    corpo = make_client().get("/api/status?probe=true&fresh=true",
                              headers=DA_BORDA).json()

    assert corpo["probe"] is False
    assert vistos and vistos[-1].probe is False


def test_probe_vale_para_quem_chama_por_dentro(monkeypatch):
    vistos = []

    def registry_espiao(opts):
        vistos.append(opts)
        return {"qdrant": lambda: comp("qdrant", st.OK, critical=True)}

    monkeypatch.setattr(st, "COMPONENT_ORDER", ("qdrant",))
    monkeypatch.setattr(st, "registry", registry_espiao)

    corpo = make_client().get("/api/status?probe=true").json()

    assert corpo["probe"] is True
    assert vistos[-1].probe is True


def test_artifacts_false_chega_nas_opcoes(monkeypatch):
    vistos = []

    def registry_espiao(opts):
        vistos.append(opts)
        return {"minio": lambda: comp("minio", st.OK, critical=True)}

    monkeypatch.setattr(st, "COMPONENT_ORDER", ("minio",))
    monkeypatch.setattr(st, "registry", registry_espiao)

    make_client().get("/api/status?artifacts=false", headers=DA_BORDA)

    assert vistos[-1].artifacts is False


def test_rota_de_um_componente_mede_so_ele(monkeypatch):
    chamados = []

    def registry_espiao(opts):
        def marca(nome):
            def _check():
                chamados.append(nome)
                return comp(nome, st.OK)
            return _check
        return {"qdrant": marca("qdrant"), "minio": marca("minio")}

    monkeypatch.setattr(st, "COMPONENT_ORDER", ("qdrant", "minio"))
    monkeypatch.setattr(st, "registry", registry_espiao)

    resp = make_client().get("/api/status/minio")

    assert resp.status_code == 200
    assert chamados == ["minio"]
    assert resp.json()["name"] == "minio"


def test_componente_critico_fora_responde_503(monkeypatch):
    monkeypatch.setattr(st, "COMPONENT_ORDER", ("qdrant",))
    monkeypatch.setattr(st, "registry",
                        lambda opts: {"qdrant": lambda: comp("qdrant", st.DOWN, critical=True)})

    assert make_client().get("/api/status/qdrant").status_code == 503


def test_componente_nao_critico_fora_responde_200(monkeypatch):
    monkeypatch.setattr(st, "COMPONENT_ORDER", ("dspace",))
    monkeypatch.setattr(st, "registry",
                        lambda opts: {"dspace": lambda: comp("dspace", st.DOWN)})

    assert make_client().get("/api/status/dspace").status_code == 200


def test_componente_desconhecido_e_404():
    resp = make_client().get("/api/status/nao-existe")

    assert resp.status_code == 404
    assert "qdrant" in resp.json()["detail"]  # lista os nomes válidos


def test_rota_de_componente_tambem_respeita_o_nivel_de_detalhe(monkeypatch):
    monkeypatch.setattr(st, "COMPONENT_ORDER", ("qdrant",))
    monkeypatch.setattr(st, "registry", lambda opts: {
        "qdrant": lambda: comp("qdrant", st.OK, critical=True,
                               internal={"url": "http://interno:6333"})})

    client = make_client()

    assert "internal" not in client.get("/api/status/qdrant", headers=DA_BORDA).json()
    assert "internal" in client.get("/api/status/qdrant").json()


def test_componente_repetido_sai_do_cache(monkeypatch):
    """Repetir /api/status/celery não pode custar uma janela de broadcast por vez."""
    chamados = []

    monkeypatch.setattr(st, "COMPONENT_ORDER", ("celery",))
    monkeypatch.setattr(st, "registry", lambda opts: {
        "celery": lambda: (chamados.append(1), comp("celery", st.OK, critical=True))[1]})
    monkeypatch.setattr(settings, "STATUS_CACHE_TTL_SECONDS", 300)

    client = make_client()
    primeiro = client.get("/api/status/celery", headers=DA_BORDA).json()
    segundo = client.get("/api/status/celery", headers=DA_BORDA).json()

    assert len(chamados) == 1
    assert primeiro["age_seconds"] == 0.0 and segundo["age_seconds"] >= 0.0

    client.get("/api/status/celery?fresh=true")  # sem X-Forwarded-For: remede
    assert len(chamados) == 2


def test_fresh_de_componente_e_ignorado_pela_borda(monkeypatch):
    chamados = []

    monkeypatch.setattr(st, "COMPONENT_ORDER", ("celery",))
    monkeypatch.setattr(st, "registry", lambda opts: {
        "celery": lambda: (chamados.append(1), comp("celery", st.OK, critical=True))[1]})
    monkeypatch.setattr(settings, "STATUS_CACHE_TTL_SECONDS", 300)

    client = make_client()
    client.get("/api/status/celery", headers=DA_BORDA)
    client.get("/api/status/celery?fresh=true", headers=DA_BORDA)

    assert len(chamados) == 1
