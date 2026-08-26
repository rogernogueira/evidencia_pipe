"""Testes da autorização administrativa (backend/api/auth.py e as rotas de ingestão).

Sem DSpace real: o cliente httpx do módulo é trocado por um `MockTransport` que
responde `/api/core/sites`, `/api/authn/status` e a busca de autorização — e
registra o que foi chamado, para os testes de cache. Como em test_search_routes.py,
as rotas são exercitadas num app mínimo (só o `files.router`), com o job_store
substituído, para não puxar Redis, Celery nem a lifespan do main.
"""

import os
import sys

import httpx
import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from backend.api import auth  # noqa: E402
from backend.api.routes import files as files_route  # noqa: E402
from backend.core import config as settings  # noqa: E402

SITE = "https://dspace.example/server/api/core/sites/11111111-2222-3333-4444-555555555555"
ATALHO_EPERSON = "https://dspace.example/server/api/authn/status/eperson"
EPERSON_UUID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
TOKEN = "eyJhbGciOiJIUzI1NiJ9.token-de-teste"

# As seis rotas que passaram a exigir Bearer de administrador (método, caminho).
ROTAS_ADMIN = [
    ("GET", "/api/files/active"),
    ("GET", "/api/files/succeeded"),
    ("GET", "/api/files/failures"),
    ("GET", "/api/files/status/job-1"),
    ("GET", "/api/files/result/job-1"),
    ("POST", "/api/files/reprocess/job-1"),
]


@pytest.fixture(autouse=True)
def estado_limpo(monkeypatch):
    """Cache e self href são globais ao processo — cada teste começa do zero."""
    auth.limpar_cache()
    monkeypatch.setattr(settings, "ADMIN_AUTH_ENABLED", True)
    monkeypatch.setattr(settings, "DSPACE_SERVER_URL", "https://dspace.example/server")
    monkeypatch.setattr(settings, "ADMIN_AUTH_CACHE_TTL_SECONDS", 45.0)
    yield
    auth.limpar_cache()


class DSpaceFalso:
    """Dublê do REST do DSpace: responde as três rotas usadas na validação."""

    def __init__(self, *, autenticado=True, admin=True, status_authn=200,
                 excecao=None, sites=None):
        self.autenticado = autenticado
        self.admin = admin
        self.status_authn = status_authn
        self.excecao = excecao
        self.sites = sites
        self.chamadas: list[str] = []

    def _handler(self, request: httpx.Request) -> httpx.Response:
        caminho = request.url.path
        self.chamadas.append(caminho)
        if self.excecao is not None:
            raise self.excecao

        if caminho.endswith("/api/core/sites"):
            corpo = self.sites if self.sites is not None else {
                "_embedded": {"sites": [{"_links": {"self": {"href": SITE}}}]}
            }
            return httpx.Response(200, json=corpo)

        if caminho.endswith("/api/authn/status"):
            if self.status_authn != 200:
                return httpx.Response(self.status_authn, json={"detail": "não autorizado"})
            corpo = {
                "authenticated": self.autenticado,
                # O `_links.eperson` do /authn/status é o MESMO para todo mundo
                # (um atalho que só resolve com o token de quem perguntou) — está
                # aqui para o teste garantir que não é ele que vira identidade.
                "_links": {"eperson": {"href": ATALHO_EPERSON}},
            }
            if self.autenticado:
                corpo["_embedded"] = {"eperson": {"uuid": EPERSON_UUID,
                                                 "email": "admin@exemplo.edu"}}
            return httpx.Response(200, json=corpo)

        if caminho.endswith("/api/authz/authorizations/search/object"):
            # A UI do DSpace pergunta pelo dono do token: uri do Site + feature,
            # SEM `eperson` (ver authorization-data.service.ts).
            assert request.url.params["uri"] == SITE
            assert request.url.params["feature"] == "administratorOf"
            assert "eperson" not in request.url.params
            total = 1 if self.admin else 0
            return httpx.Response(200, json={"page": {"totalElements": total}})

        return httpx.Response(404, json={"detail": "rota não simulada"})

    def instala(self, monkeypatch):
        transporte = httpx.MockTransport(self._handler)
        monkeypatch.setattr(auth, "_client",
                            lambda: httpx.AsyncClient(transport=transporte))
        return self


def app_admin() -> FastAPI:
    """App mínimo com uma rota protegida pela dependência (sem tocar no job_store)."""
    app = FastAPI()

    @app.get("/protegida")
    async def protegida(admin: dict = Depends(auth.dspace_admin)):
        return dict(admin)

    return app


def cliente(monkeypatch, **kwargs) -> tuple[TestClient, DSpaceFalso]:
    dspace = DSpaceFalso(**kwargs).instala(monkeypatch)
    return TestClient(app_admin()), dspace


def bearer(token: str = TOKEN) -> dict:
    return {"Authorization": f"Bearer {token}"}


# --------------------------------------------------------------------------
# Os quatro veredictos: 200, 401, 403, 503
# --------------------------------------------------------------------------
def test_admin_do_dspace_passa(monkeypatch):
    c, _ = cliente(monkeypatch)

    r = c.get("/protegida", headers=bearer())

    assert r.status_code == 200
    # Identidade para o log (ver auth._identidade); a decisão é o 200 em si.
    assert r.json()["eperson_email"] == "admin@exemplo.edu"
    assert r.json()["sessao"] == auth._hash(TOKEN)[:8]
    assert ATALHO_EPERSON not in r.text  # esse link é igual para todo mundo


def test_sem_token_da_401_e_nao_403(monkeypatch):
    """401 manda o front pedir login de novo; 403 (padrão do HTTPBearer) diria à
    pessoa que a conta dela não tem permissão — mensagem errada."""
    c, dspace = cliente(monkeypatch)

    r = c.get("/protegida")

    assert r.status_code == 401
    assert r.headers.get("WWW-Authenticate") == "Bearer"
    assert dspace.chamadas == []  # nem chega a perguntar ao DSpace


def test_esquema_que_nao_e_bearer_da_401(monkeypatch):
    c, _ = cliente(monkeypatch)

    r = c.get("/protegida", headers={"Authorization": "Basic YWRtaW46c2VuaGE="})

    assert r.status_code == 401


def test_token_recusado_pelo_dspace_da_401(monkeypatch):
    c, _ = cliente(monkeypatch, status_authn=401)

    r = c.get("/protegida", headers=bearer())

    assert r.status_code == 401
    assert "expirada" in r.json()["detail"]


def test_authenticated_false_da_401(monkeypatch):
    """O DSpace responde 200 + authenticated=false para sessão que não vale mais."""
    c, _ = cliente(monkeypatch, autenticado=False)

    r = c.get("/protegida", headers=bearer())

    assert r.status_code == 401


def test_usuario_comum_da_403(monkeypatch):
    c, _ = cliente(monkeypatch, admin=False)

    r = c.get("/protegida", headers=bearer())

    assert r.status_code == 403
    assert "administrador" in r.json()["detail"]


def test_dspace_fora_do_ar_da_503_e_nunca_401(monkeypatch):
    """O ponto crítico: falha ao VALIDAR não é sessão expirada. Com 401 aqui, o
    front manda o admin relogar — e o login falharia pelo mesmo motivo."""
    c, _ = cliente(monkeypatch, excecao=httpx.ConnectTimeout("timeout"))

    r = c.get("/protegida", headers=bearer())

    assert r.status_code == 503
    assert "indisponível" in r.json()["detail"]


def test_erro_5xx_do_dspace_da_503(monkeypatch):
    def handler(request):
        return httpx.Response(502, text="<html>proxy</html>")

    monkeypatch.setattr(auth, "_client",
                        lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    c = TestClient(app_admin())

    assert c.get("/protegida", headers=bearer()).status_code == 503


def test_site_sem_self_href_da_503(monkeypatch):
    """Sem o Site não há o que perguntar sobre `administratorOf` — 503, não 403:
    devolver 403 diria que a conta não é admin, e o problema é do servidor."""
    c, _ = cliente(monkeypatch, sites={"_embedded": {"sites": []}})

    assert c.get("/protegida", headers=bearer()).status_code == 503


# --------------------------------------------------------------------------
# Cache — a aba de Ingestão recarrega a cada 5s, em até 3 listas paralelas
# --------------------------------------------------------------------------
def test_sessao_validada_e_reusada_do_cache(monkeypatch):
    c, dspace = cliente(monkeypatch)

    for _ in range(5):
        assert c.get("/protegida", headers=bearer()).status_code == 200

    # 1ª requisição: sites + authn/status + authz. As outras quatro: nada.
    assert dspace.chamadas.count("/server/api/authn/status") == 1
    assert dspace.chamadas.count("/server/api/core/sites") == 1


def test_cache_expirado_revalida(monkeypatch):
    c, dspace = cliente(monkeypatch)
    monkeypatch.setattr(settings, "ADMIN_AUTH_CACHE_TTL_SECONDS", 0.0)

    c.get("/protegida", headers=bearer())
    c.get("/protegida", headers=bearer())

    assert dspace.chamadas.count("/server/api/authn/status") == 2


def test_tokens_diferentes_nao_se_confundem(monkeypatch):
    c, dspace = cliente(monkeypatch)

    c.get("/protegida", headers=bearer("token-a"))
    c.get("/protegida", headers=bearer("token-b"))

    assert dspace.chamadas.count("/server/api/authn/status") == 2


def test_sessao_encerrada_no_dspace_vale_no_maximo_ate_o_cache_expirar(monkeypatch):
    """Logout/expiração no DSpace não derruba a sessão aqui na hora: ela vale até o
    TTL (por isso ele é curto). Vencido o TTL, a revalidação dá 401 e a entrada some."""
    dspace = DSpaceFalso().instala(monkeypatch)
    c = TestClient(app_admin())
    chave = auth._hash(TOKEN)

    assert c.get("/protegida", headers=bearer()).status_code == 200
    dspace.status_authn = 401
    assert c.get("/protegida", headers=bearer()).status_code == 200  # ainda no TTL

    auth._cache[chave] = (0.0, auth._cache[chave][1])  # simula o TTL vencido

    assert c.get("/protegida", headers=bearer()).status_code == 401
    assert chave not in auth._cache


def test_cache_nao_guarda_o_token_em_claro(monkeypatch):
    c, _ = cliente(monkeypatch)

    c.get("/protegida", headers=bearer())

    assert TOKEN not in auth._cache
    assert all(TOKEN not in chave for chave in auth._cache)
    assert list(auth._cache) == [auth._hash(TOKEN)]


def test_cache_respeita_o_teto_de_entradas(monkeypatch):
    c, _ = cliente(monkeypatch)
    monkeypatch.setattr(settings, "ADMIN_AUTH_CACHE_MAX_ENTRIES", 3)

    for i in range(10):
        c.get("/protegida", headers=bearer(f"token-{i}"))

    assert len(auth._cache) <= 3


def test_token_nunca_aparece_na_resposta_nem_no_log(monkeypatch, caplog):
    c, _ = cliente(monkeypatch, admin=False)

    with caplog.at_level("DEBUG"):
        r = c.get("/protegida", headers=bearer())

    assert TOKEN not in r.text
    assert TOKEN not in caplog.text


# --------------------------------------------------------------------------
# As rotas de verdade (backend/api/routes/files.py)
# --------------------------------------------------------------------------
@pytest.fixture
def app_files(monkeypatch):
    """App só com o files.router, com o job_store trocado por respostas fixas."""
    job = {"job_id": "job-1", "status": "concluido", "bitstream_uuid": "bs-1",
           "filename": "a.pdf", "pipeline_id": "p1", "document_id": "job-1"}
    monkeypatch.setattr(files_route, "get_job", lambda job_id: dict(job, job_id=job_id))
    monkeypatch.setattr(files_route, "list_active", lambda limit: [])
    monkeypatch.setattr(files_route, "list_succeeded", lambda limit: [])
    monkeypatch.setattr(files_route, "list_failed", lambda limit: [])
    monkeypatch.setattr(files_route, "set_status", lambda *a, **k: None)
    monkeypatch.setattr(files_route.ingest, "enqueue_chain", lambda *a, **k: None)

    app = FastAPI()
    app.include_router(files_route.router)
    return TestClient(app)


@pytest.mark.parametrize("metodo,caminho", ROTAS_ADMIN)
def test_rota_administrativa_sem_token_da_401(monkeypatch, app_files, metodo, caminho):
    DSpaceFalso().instala(monkeypatch)

    assert app_files.request(metodo, caminho).status_code == 401


@pytest.mark.parametrize("metodo,caminho", ROTAS_ADMIN)
def test_rota_administrativa_com_admin_responde(monkeypatch, app_files, metodo, caminho):
    DSpaceFalso().instala(monkeypatch)

    r = app_files.request(metodo, caminho, headers=bearer())

    assert r.status_code in (200, 202), r.text


@pytest.mark.parametrize("metodo,caminho", ROTAS_ADMIN)
def test_rota_administrativa_com_usuario_comum_da_403(monkeypatch, app_files, metodo, caminho):
    DSpaceFalso(admin=False).instala(monkeypatch)

    assert app_files.request(metodo, caminho).status_code == 401
    assert app_files.request(metodo, caminho, headers=bearer()).status_code == 403


def test_enfileiramento_continua_aberto(monkeypatch, app_files):
    """A sincronização automática (scripts/sincronizar_novos_itens.py) chama esta
    rota sem sessão DSpace — protegê-la aqui derrubaria a ingestão periódica."""
    DSpaceFalso().instala(monkeypatch)
    monkeypatch.setattr(files_route, "resolve_item_pdfs",
                        lambda uuid: [{"bitstream_uuid": "bs-1", "filename": "a.pdf",
                                       "item_handle": "123/1"}])
    monkeypatch.setattr(files_route.ingest, "enqueue_item_pdfs",
                        lambda uuid, pdfs, force: ["job-1"])

    r = app_files.post("/api/files/dspace/item/uuid-1")

    assert r.status_code == 202


def test_auth_desligada_libera_as_rotas(monkeypatch, app_files):
    """ADMIN_AUTH_ENABLED=false é a válvula de desenvolvimento — em produção fica
    ligada (o servidor avisa no log quando não está)."""
    monkeypatch.setattr(settings, "ADMIN_AUTH_ENABLED", False)
    dspace = DSpaceFalso().instala(monkeypatch)

    assert app_files.get("/api/files/active").status_code == 200
    assert dspace.chamadas == []
