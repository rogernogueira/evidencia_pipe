"""Testes do middleware que acomoda o proxy que remove o prefixo /api.

Ver backend/api/proxy_prefix.py e DEPLOY.md §8.1. Como em test_search_routes.py,
montamos um app mínimo com a MESMA forma de caminhos do backend real (rotas sob
/api, /health e o bloco administrativo na raiz) em vez de importar backend.main —
assim os testes não puxam lifespan, StaticFiles nem Qdrant.
"""

import os
import sys

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from fastapi import APIRouter  # noqa: E402

from backend.api.proxy_prefix import (  # noqa: E402
    StrippedPrefixMiddleware,
    caminhos_registrados,
    instala_acomodacao_de_proxy,
    segmentos_do_router,
)
from backend.core import config as settings  # noqa: E402

DA_BORDA = {"X-Forwarded-For": "200.130.0.2"}  # o que o mod_proxy sempre acrescenta


def monta_app() -> FastAPI:
    app = FastAPI()

    @app.get("/api/files/status/{job_id}")
    def status(job_id: str):
        return {"rota": "status", "job_id": job_id}

    @app.get("/api/search/semantic")
    def semantic(q: str = ""):
        return {"rota": "semantic", "q": q}

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/internal/gpu/status")
    def gpu():
        return {"rota": "gpu"}

    @app.get("/output/{nome}")
    def output(nome: str):
        return {"rota": "output", "nome": nome}

    return app


def cliente(*, guarda_admin=True) -> TestClient:
    app = monta_app()
    app.add_middleware(StrippedPrefixMiddleware, prefix="/api",
                       segmentos=segmentos_do_router(app, "/api"),
                       guarda_admin=guarda_admin)
    return TestClient(app)


# --------------------------------------------------------------------------- #
# Recolocação do prefixo
# --------------------------------------------------------------------------- #
def test_recoloca_prefixo_removido_pela_borda():
    resp = cliente().get("/files/status/abc123")
    assert resp.status_code == 200
    assert resp.json() == {"rota": "status", "job_id": "abc123"}


def test_query_string_preservada():
    resp = cliente().get("/search/semantic", params={"q": "avaliação"})
    assert resp.status_code == 200
    assert resp.json() == {"rota": "semantic", "q": "avaliação"}


def test_caminho_ja_prefixado_continua_valendo():
    """Quem fala direto com a porta 8020 (e o front do RDApp/UFT) usa /api/..."""
    resp = cliente().get("/api/files/status/abc123")
    assert resp.status_code == 200
    assert resp.json()["job_id"] == "abc123"


def test_rota_desconhecida_segue_404():
    assert cliente().get("/naoexiste").status_code == 404


def test_sem_middleware_o_caminho_sem_prefixo_da_404():
    """Garante que o teste acima mede o middleware, não o roteamento do FastAPI."""
    assert TestClient(monta_app()).get("/files/status/abc123").status_code == 404


def test_segmentos_derivados_das_rotas():
    assert segmentos_do_router(monta_app(), "/api") == {"files", "search"}


def test_segmentos_atravessam_include_router():
    """Como o backend real monta as rotas: router aninhado, não decorator no app.

    O FastAPI ≥0.139 adia o include_router — `app.routes` guarda um marcador e
    varrer só esse nível devolveria zero segmento e o middleware não seria
    instalado, silenciosamente. Este teste é o que prende esse comportamento.
    """
    interno = APIRouter()

    @interno.get("/api/files/status/{job_id}")
    def status(job_id: str):
        return {"job_id": job_id}

    @interno.get("/internal/gpu/status")
    def gpu():
        return {"rota": "gpu"}

    externo = APIRouter()
    externo.include_router(interno)

    app = FastAPI()
    app.include_router(externo)

    assert "/api/files/status/{job_id}" in set(caminhos_registrados(app))
    assert segmentos_do_router(app, "/api") == {"files"}


def test_segmento_que_existe_na_raiz_e_descartado():
    """Se algum dia existir /files na raiz, reescrever esconderia a rota real."""
    app = monta_app()

    @app.get("/files/legado")
    def legado():
        return {"rota": "legado"}

    assert "files" not in segmentos_do_router(app, "/api")


# --------------------------------------------------------------------------- #
# Guarda das rotas administrativas
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("caminho", ["/docs", "/redoc", "/openapi.json",
                                     "/internal/gpu/status", "/output/x.md"])
def test_admin_bloqueado_para_quem_vem_do_proxy(caminho):
    resp = cliente().get(caminho, headers=DA_BORDA)
    assert resp.status_code == 404
    # Corpo igual ao 404 do FastAPI: de fora, não dá para saber que a rota existe.
    assert resp.json() == {"detail": "Not Found"}


@pytest.mark.parametrize("caminho", ["/docs", "/openapi.json",
                                     "/internal/gpu/status", "/output/x.md"])
def test_admin_liberado_na_rede_interna(caminho):
    assert cliente().get(caminho).status_code == 200


def test_health_responde_mesmo_pelo_proxy():
    """O health check do proxy depende dele, e ele não revela nada."""
    resp = cliente().get("/health", headers=DA_BORDA)
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_rota_publica_nao_e_afetada_pela_guarda():
    resp = cliente().get("/files/status/abc123", headers=DA_BORDA)
    assert resp.status_code == 200


def test_guarda_desligada_libera_admin_pelo_proxy():
    resp = cliente(guarda_admin=False).get("/internal/gpu/status", headers=DA_BORDA)
    assert resp.status_code == 200


# --------------------------------------------------------------------------- #
# Instalação: o remendo é opt-in, não muda nada sem a variável
# --------------------------------------------------------------------------- #
def test_sem_variavel_nao_instala_nada(monkeypatch):
    monkeypatch.setattr(settings, "PROXY_STRIPPED_PREFIX", "")
    app = monta_app()
    instala_acomodacao_de_proxy(app)
    assert app.user_middleware == []
    assert TestClient(app).get("/files/status/abc").status_code == 404


def test_com_variavel_instala_o_middleware(monkeypatch):
    monkeypatch.setattr(settings, "PROXY_STRIPPED_PREFIX", "/api")
    monkeypatch.setattr(settings, "PROXY_GUARD_ADMIN", True)
    app = monta_app()
    instala_acomodacao_de_proxy(app)
    assert [mw.cls.__name__ for mw in app.user_middleware] == ["StrippedPrefixMiddleware"]
    assert TestClient(app).get("/files/status/abc").status_code == 200


def test_prefixo_sem_rota_correspondente_nao_instala(monkeypatch):
    """Digitar PROXY_STRIPPED_PREFIX=/v1 não pode virar reescrita silenciosa."""
    monkeypatch.setattr(settings, "PROXY_STRIPPED_PREFIX", "/v1")
    app = monta_app()
    instala_acomodacao_de_proxy(app)
    assert app.user_middleware == []
