"""Item ainda indisponível no DSpace: espera na fila em vez de 502.

Cobre os três pontos da mudança, sem DSpace, Redis nem Celery reais:
  - a classificação do erro (backend/services/dspace_service.item_ainda_indisponivel);
  - o endpoint POST /api/files/dspace/item/{uuid} (202 + agendamento, e o 502/422 que
    permanece para erro definitivo);
  - a task resolver_item_dspace (reagenda com backoff, desiste ao esgotar, enfileira
    as chains quando o item aparece).

O job_store é o fallback em memória (não há Redis nos testes); o que seria enfileirado
no broker é capturado por dublês de `apply_async`. Como em test_search_routes.py, a
rota roda num app mínimo — só o `files.router`.
"""

import io
import os
import sys
import urllib.error

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from backend.api.routes import files as files_route  # noqa: E402
from backend.core import config as settings  # noqa: E402
from backend.services import dspace_service, ingest_service as ingest  # noqa: E402
from backend.services import job_store  # noqa: E402

ITEM = "11111111-2222-3333-4444-555555555555"
CHAVE = f"item:{ITEM}"

PDFS = [{"bitstream_uuid": "bs-1", "filename": "relatorio.pdf", "item_handle": "123/45"}]


def http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(f"https://dspace/items/{ITEM}", code, "erro", {}, io.BytesIO(b""))


@pytest.fixture(autouse=True)
def job_store_limpo(monkeypatch):
    """Sem Redis nos testes: força o fallback em memória e zera os índices."""
    monkeypatch.setattr(job_store, "_redis", None, raising=False)
    monkeypatch.setattr(job_store, "_redis_ready", True, raising=False)
    monkeypatch.setattr(job_store, "_get_redis", lambda: None)
    for d in (job_store._jobs, job_store._failed, job_store._active, job_store._succeeded):
        d.clear()
    yield


class FilaFalsa:
    """Captura o que teria ido para o broker (chains e resoluções agendadas)."""

    def __init__(self):
        self.resolucoes = []
        self.chains = []

    def instalar(self, monkeypatch):
        from backend.tasks import resolver_item_dspace

        monkeypatch.setattr(
            resolver_item_dspace, "apply_async",
            lambda kwargs=None, countdown=None, **_: self.resolucoes.append((kwargs, countdown)),
        )
        monkeypatch.setattr(
            ingest, "enqueue_chain",
            lambda *args, **kw: self.chains.append(args),
        )
        return self


@pytest.fixture
def fila(monkeypatch):
    return FilaFalsa().instalar(monkeypatch)


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(files_route.router)
    return TestClient(app)


# --------------------------------------------------------------------------
# Classificação do erro
# --------------------------------------------------------------------------
@pytest.mark.parametrize("code", [401, 403, 404, 429, 500, 502, 503])
def test_status_transitorio_e_indisponibilidade(code):
    """401 entra aqui de propósito: o DSpace REST responde 401 (não 403) a requisição
    anônima sem permissão — inclusive em workflow/workspace, onde o item fica antes de
    ser publicado. Item em submissão, sob embargo ou retirado devolve 401 ao anônimo, e
    a ingestão é anônima; logo é "ainda não disponível", não credencial errada."""
    assert dspace_service.item_ainda_indisponivel(http_error(code)) is True


@pytest.mark.parametrize("code", [400, 405, 410])
def test_status_definitivo_nao_e_indisponibilidade(code):
    """400 = UUID malformado, 405/410 = rota/recurso que não volta. Esperar não ajuda."""
    assert dspace_service.item_ainda_indisponivel(http_error(code)) is False


def test_rede_fora_e_indisponibilidade():
    assert dspace_service.item_ainda_indisponivel(urllib.error.URLError("dns")) is True


def test_item_sem_pdf_segue_a_flag(monkeypatch):
    monkeypatch.setattr(settings, "DSPACE_ITEM_RETRY_WHEN_NO_PDF", True)
    assert dspace_service.item_ainda_indisponivel(ValueError("sem PDF")) is True
    monkeypatch.setattr(settings, "DSPACE_ITEM_RETRY_WHEN_NO_PDF", False)
    assert dspace_service.item_ainda_indisponivel(ValueError("sem PDF")) is False


def test_backoff_cresce_ate_o_teto(monkeypatch):
    monkeypatch.setattr(settings, "DSPACE_ITEM_RETRY_DELAY_SECONDS", 60)
    monkeypatch.setattr(settings, "DSPACE_ITEM_RETRY_BACKOFF", 2.0)
    monkeypatch.setattr(settings, "DSPACE_ITEM_RETRY_MAX_DELAY_SECONDS", 300)
    assert [ingest.retry_delay_seconds(n) for n in (1, 2, 3, 4, 9)] == [60, 120, 240, 300, 300]


# --------------------------------------------------------------------------
# Endpoint
# --------------------------------------------------------------------------
def test_item_indisponivel_responde_202_e_enfileira(monkeypatch, client, fila):
    monkeypatch.setattr(files_route, "resolve_item_pdfs",
                        lambda uuid: (_ for _ in ()).throw(http_error(404)))

    r = client.post(f"/api/files/dspace/item/{ITEM}")

    assert r.status_code == 202
    corpo = r.json()
    assert corpo["status"] == "aguardando_dspace"
    assert corpo["job_id"] == CHAVE
    assert corpo["jobs"] == []
    assert corpo["retry"]["attempt"] == 1 and corpo["retry"]["next_attempt"] == 2
    assert corpo["retry"]["next_retry_in_seconds"] == ingest.retry_delay_seconds(1)

    # foi agendada UMA nova tentativa, com o countdown do backoff
    kwargs, countdown = fila.resolucoes[0]
    assert kwargs == {"item_uuid": ITEM, "force": False, "attempt": 2}
    assert countdown == ingest.retry_delay_seconds(1)
    assert fila.chains == []  # nenhuma chain: ainda não há PDF

    # e o item é acompanhável como job ativo
    registro = job_store.get_job(CHAVE)
    assert registro["status"] == "na_fila" and registro["stage"] == "aguardando_dspace"
    assert [j["job_id"] for j in job_store.list_active()] == [CHAVE]


def test_repetir_o_post_nao_duplica_a_espera(monkeypatch, client, fila):
    monkeypatch.setattr(files_route, "resolve_item_pdfs",
                        lambda uuid: (_ for _ in ()).throw(http_error(404)))

    client.post(f"/api/files/dspace/item/{ITEM}")
    r = client.post(f"/api/files/dspace/item/{ITEM}")

    assert r.status_code == 202
    assert "já havia uma ingestão aguardando" in r.json()["message"]
    assert len(fila.resolucoes) == 1


def test_post_durante_a_tentativa_em_curso_tambem_nao_duplica(monkeypatch, client, fila):
    monkeypatch.setattr(files_route, "resolve_item_pdfs",
                        lambda uuid: (_ for _ in ()).throw(http_error(404)))
    client.post(f"/api/files/dspace/item/{ITEM}")
    # o worker acordou e está consultando o DSpace agora
    job_store.set_status(CHAVE, "processando", source=ingest.PENDING_SOURCE, item_uuid=ITEM)

    r = client.post(f"/api/files/dspace/item/{ITEM}")

    assert r.status_code == 202
    assert len(fila.resolucoes) == 1


def test_espera_orfa_nao_bloqueia_o_reenfileiramento(monkeypatch, client, fila):
    monkeypatch.setattr(files_route, "resolve_item_pdfs",
                        lambda uuid: (_ for _ in ()).throw(http_error(404)))
    client.post(f"/api/files/dspace/item/{ITEM}")
    # worker reiniciado: a tentativa agendada nunca rodou e o prazo ficou muito no passado
    job_store.set_status(CHAVE, "na_fila", next_retry_at="2020-01-01T00:00:00+00:00")

    r = client.post(f"/api/files/dspace/item/{ITEM}")

    assert r.status_code == 202
    assert len(fila.resolucoes) == 2


def test_force_reenfileira_a_espera(monkeypatch, client, fila):
    monkeypatch.setattr(files_route, "resolve_item_pdfs",
                        lambda uuid: (_ for _ in ()).throw(http_error(404)))

    client.post(f"/api/files/dspace/item/{ITEM}")
    client.post(f"/api/files/dspace/item/{ITEM}?force=true")

    assert len(fila.resolucoes) == 2
    assert fila.resolucoes[-1][0]["force"] is True


def test_item_em_submissao_401_vai_para_a_fila(monkeypatch, client, fila):
    """Regressão do caso real: item ainda não publicado devolve 401 ao anônimo e ia
    embora como 502 sem passar pela fila."""
    monkeypatch.setattr(files_route, "resolve_item_pdfs",
                        lambda uuid: (_ for _ in ()).throw(http_error(401)))

    r = client.post(f"/api/files/dspace/item/{ITEM}")

    assert r.status_code == 202
    assert r.json()["status"] == "aguardando_dspace"
    assert len(fila.resolucoes) == 1


def test_erro_definitivo_continua_502(monkeypatch, client, fila):
    monkeypatch.setattr(files_route, "resolve_item_pdfs",
                        lambda uuid: (_ for _ in ()).throw(http_error(400)))

    r = client.post(f"/api/files/dspace/item/{ITEM}")

    assert r.status_code == 502
    assert fila.resolucoes == []
    assert job_store.get_job(CHAVE) is None


def test_item_sem_pdf_com_a_flag_desligada_volta_a_422(monkeypatch, client, fila):
    monkeypatch.setattr(settings, "DSPACE_ITEM_RETRY_WHEN_NO_PDF", False)
    monkeypatch.setattr(files_route, "resolve_item_pdfs",
                        lambda uuid: (_ for _ in ()).throw(ValueError("Item sem PDF no ORIGINAL.")))

    r = client.post(f"/api/files/dspace/item/{ITEM}")

    assert r.status_code == 422
    assert fila.resolucoes == []


def test_item_disponivel_segue_o_caminho_de_sempre(monkeypatch, client, fila):
    monkeypatch.setattr(files_route, "resolve_item_pdfs", lambda uuid: PDFS)

    r = client.post(f"/api/files/dspace/item/{ITEM}")

    assert r.status_code == 202
    corpo = r.json()
    assert corpo["status"] == "na_fila"
    assert [j["job_id"] for j in corpo["jobs"]] == ["relatorio"]
    assert len(fila.chains) == 1
    assert fila.resolucoes == []


# --------------------------------------------------------------------------
# Task de resolução
# --------------------------------------------------------------------------
def test_task_reagenda_enquanto_o_item_nao_aparece(monkeypatch, fila):
    import backend.tasks as tasks

    monkeypatch.setattr(tasks, "resolve_item_pdfs",
                        lambda uuid: (_ for _ in ()).throw(http_error(404)))

    out = tasks.resolver_item_dspace.run(item_uuid=ITEM, force=False, attempt=3)

    assert out["status"] == "aguardando_dspace" and out["next_attempt"] == 4
    assert fila.resolucoes[0][0]["attempt"] == 4
    assert fila.resolucoes[0][1] == ingest.retry_delay_seconds(3)
    assert job_store.get_job(CHAVE)["status"] == "na_fila"
    assert job_store.list_failed() == []


def test_task_desiste_ao_esgotar_as_tentativas(monkeypatch, fila):
    import backend.tasks as tasks

    monkeypatch.setattr(settings, "DSPACE_ITEM_RETRY_MAX_ATTEMPTS", 3)
    monkeypatch.setattr(tasks, "resolve_item_pdfs",
                        lambda uuid: (_ for _ in ()).throw(http_error(404)))

    with pytest.raises(urllib.error.HTTPError):
        tasks.resolver_item_dspace.run(item_uuid=ITEM, force=False, attempt=3)

    assert fila.resolucoes == []
    registro = job_store.get_job(CHAVE)
    assert registro["status"] == "erro" and "3 tentativa(s)" in registro["error"]
    assert [j["job_id"] for j in job_store.list_failed()] == [CHAVE]


def test_task_enfileira_as_chains_quando_o_item_aparece(monkeypatch, fila):
    import backend.tasks as tasks

    monkeypatch.setattr(tasks, "resolve_item_pdfs", lambda uuid: PDFS)

    out = tasks.resolver_item_dspace.run(item_uuid=ITEM, force=False, attempt=2)

    assert out["job_ids"] == ["relatorio"]
    assert len(fila.chains) == 1
    assert fila.resolucoes == []

    registro = job_store.get_job(CHAVE)
    assert registro["status"] == "resolvido" and registro["n_pdfs"] == 1
    # o registro do item sai dos ativos e NÃO entra na lista de sucessos (que é de
    # documentos indexados); quem fica ativo é o job do PDF
    assert [j["job_id"] for j in job_store.list_active()] == ["relatorio"]
    assert job_store.list_succeeded() == []


def test_reprocess_do_item_reenfileira_a_resolucao(monkeypatch, client, fila):
    job_store.set_status(CHAVE, "erro", source=ingest.PENDING_SOURCE, item_uuid=ITEM,
                         stage=ingest.PENDING_STAGE, error="Item indisponível")
    job_store.add_failed(CHAVE)

    r = client.post(f"/api/files/reprocess/{CHAVE}")

    assert r.status_code == 202
    assert r.json()["status"] == "aguardando_dspace"
    kwargs, countdown = fila.resolucoes[0]
    assert kwargs == {"item_uuid": ITEM, "force": True, "attempt": 2}
    assert countdown == 0  # reprocesso manual tenta na hora
    assert job_store.list_failed() == []
