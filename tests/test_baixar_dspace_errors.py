"""Registro de erro do download (baixar_dspace) na fila de falhas.

Antes, toda falha HTTP do download chegava ao job como `"OSError: "`: o `retry(exc=e)`
esgotado relança o próprio HTTPError (o ramo MaxRetriesExceededError nunca rodava) e o
Celery não consegue serializá-lo, entregando ao on_failure um OSError() vazio.

Cobre, sem DSpace, Redis nem Celery reais:
  - 401/403/404/410 → erro definitivo imediato, sem retry, com mensagem clara;
  - 5xx/URLError → retry enquanto houver tentativas; esgotadas, mensagem com o código;
  - a exceção levantada sobrevive à serialização do Celery com a mensagem;
  - on_failure não apaga o `error` gravado quando a exceção chega vazia.
"""

import io
import os
import sys
import urllib.error

import pytest
from celery.utils.serialization import get_pickleable_exception

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from backend import tasks  # noqa: E402
from backend.services import job_store  # noqa: E402

BS = "75199259-0d51-401f-90fd-5f38c4b3d27e"
JOB = "job-download-1"
URL = f"https://dspace/server/api/core/bitstreams/{BS}/content"


def http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(URL, code, "erro", {}, io.BytesIO(b""))


@pytest.fixture(autouse=True)
def job_store_limpo(monkeypatch):
    """Sem Redis nos testes: força o fallback em memória e zera os índices."""
    monkeypatch.setattr(job_store, "_redis", None, raising=False)
    monkeypatch.setattr(job_store, "_redis_ready", True, raising=False)
    monkeypatch.setattr(job_store, "_get_redis", lambda: None)
    for d in (job_store._jobs, job_store._failed, job_store._active, job_store._succeeded):
        d.clear()
    yield


def download_falha_com(monkeypatch, exc):
    def boom(**_):
        raise exc

    monkeypatch.setattr(tasks.stages, "stage_download", boom)


def rodar(retries=0):
    """Executa baixar_dspace como se fosse a tentativa `retries` no worker."""
    task = tasks.baixar_dspace
    task.push_request(retries=retries)
    try:
        return task.run(bs_uuid=BS, filename=f"{BS}.pdf", job_id=JOB)
    finally:
        task.pop_request()


@pytest.fixture
def retries_agendados(monkeypatch):
    agendados = []

    class Reagendado(Exception):
        pass

    def fake_retry(exc=None, **_):
        agendados.append(exc)
        return Reagendado()

    monkeypatch.setattr(tasks.baixar_dspace, "retry", fake_retry)
    return agendados, Reagendado


@pytest.mark.parametrize("code", [401, 403, 404, 410])
def test_bitstream_indisponivel_falha_sem_retry(monkeypatch, retries_agendados, code):
    agendados, _ = retries_agendados
    download_falha_com(monkeypatch, http_error(code))

    with pytest.raises(tasks.DSpaceDownloadError) as info:
        rodar(retries=0)

    assert agendados == []
    erro = job_store.get_job(JOB)["error"]
    assert f"HTTP {code}" in erro and BS in erro and "indisponível no DSpace" in erro
    assert str(info.value) == erro


@pytest.mark.parametrize("exc", [http_error(503), urllib.error.URLError("timed out")])
def test_erro_transiente_reagenda(monkeypatch, retries_agendados, exc):
    agendados, Reagendado = retries_agendados
    download_falha_com(monkeypatch, exc)

    with pytest.raises(Reagendado):
        rodar(retries=0)

    assert agendados == [exc]
    assert job_store.get_job(JOB)["status"] == "processando"


def test_erro_transiente_esgotado_grava_mensagem(monkeypatch, retries_agendados):
    agendados, _ = retries_agendados
    download_falha_com(monkeypatch, http_error(503))

    with pytest.raises(tasks.DSpaceDownloadError):
        rodar(retries=tasks.baixar_dspace.max_retries)

    assert agendados == []
    job = job_store.get_job(JOB)
    assert job["status"] == "erro"
    assert job["error"] == f"download falhou (HTTP 503): {URL}"


def test_excecao_sobrevive_a_serializacao_do_celery():
    # O HTTPError original vira um OSError() vazio; a nossa exceção mantém a mensagem.
    assert str(get_pickleable_exception(http_error(404))) == ""

    exc = get_pickleable_exception(tasks.DSpaceDownloadError("bitstream indisponível"))
    assert str(exc) == "bitstream indisponível"


def test_on_failure_preserva_erro_gravado_quando_excecao_vem_vazia():
    job_store.set_status(JOB, "erro", stage="download", error="download falhou (HTTP 503): x")

    tasks.baixar_dspace.on_failure(OSError(), "tid", (), {"job_id": JOB}, None)

    job = job_store.get_job(JOB)
    assert job["error"] == "download falhou (HTTP 503): x"
    assert job["stage"] == "baixar_dspace"
    assert JOB in job_store._failed


def test_on_failure_grava_erro_quando_excecao_tem_mensagem():
    tasks.baixar_dspace.on_failure(
        tasks.DSpaceDownloadError("bitstream indisponível"), "tid", (), {"job_id": JOB}, None
    )

    assert job_store.get_job(JOB)["error"] == "DSpaceDownloadError: bitstream indisponível"
    assert JOB in job_store._failed
