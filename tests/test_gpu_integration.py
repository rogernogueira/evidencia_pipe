"""Testes de integração de coordenação de GPU no evidencia_pipe (§40, subconjunto
testável sem GPU/torch reais). Cobrem:

  - MinerU adquire a GPU e mantém o lock enquanto o contêiner processa (mock HTTP);
  - MinerU libera o lock quando a chamada HTTP falha;
  - MinerU libera o lock e levanta RuntimeError no timeout;
  - prioridade: script externo (10) roda antes de um consumidor comum (15); nenhum
    interrompe o MinerU;
  - falha do Redis impede a execução CUDA (fail-closed).

O MinerU não roda mais em subprocesso local: `mineru_service` fala HTTP com o
contêiner `mineru-api`. O lock continua existindo porque a GPU é a mesma — o que
mudou é quem a usa (o servidor), não a necessidade de serializar.

O bge-m3 não aparece aqui: roda nos contêineres vLLM (VRAM própria) e não disputa o
lock do gpu0. Os intervalos de uso da GPU nunca se sobrepõem.
"""

import io
import threading
import time
import zipfile
from unittest import mock

import httpx
import pytest

from gpu_resource_manager import GPUBackendUnavailable, GPUResourceManager, GPUManagerConfig


# --------------------------------------------------------------------------- #
# Dublê do cliente HTTP do mineru-api
# --------------------------------------------------------------------------- #
def _zip_bytes(stem: str) -> bytes:
    """ZIP no mesmo layout que o servidor devolve: <stem>/auto/<stem>_*.json|md."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(f"{stem}/auto/{stem}_content_list_v2.json", "[[]]")
        zf.writestr(f"{stem}/auto/{stem}.md", "# doc\n")
    return buf.getvalue()


class _Resp:
    def __init__(self, payload=None, content=b""):
        self._payload = payload
        self.content = content
        self.text = ""

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _dummy_pdf(path):
    """PDF mínimo: `_submit` abre o arquivo de verdade (o subprocesso antigo não abria)."""
    path.write_bytes(b"%PDF-1.4\n1 0 obj<</Type/Catalog>>endobj\ntrailer<</Root 1 0 R>>\n%%EOF\n")
    return path


class _FakeClient:
    """Cliente mínimo compatível com o uso que `mineru_service` faz do httpx."""

    def __init__(self, *, stem="doc", status="completed", on_post=None, on_get=None):
        self.stem = stem
        self.status = status
        self.on_post = on_post
        self.on_get = on_get

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def post(self, url, data=None, files=None):
        if self.on_post:
            self.on_post()
        return _Resp(payload={"task_id": "task-1"})

    def get(self, url):
        if self.on_get:
            self.on_get()
        if url.endswith("/result"):
            return _Resp(content=_zip_bytes(self.stem))
        return _Resp(payload={"status": self.status})


# --------------------------------------------------------------------------- #
# MinerU
# --------------------------------------------------------------------------- #
def test_mineru_holds_lock_during_http_call_and_releases(gpu_manager_fake, tmp_path):
    import backend.services.mineru_service as ms

    manager = gpu_manager_fake
    observed = {}

    def during_post():
        # enquanto o contêiner "processa", o lock deve estar retido por mineru
        st = manager.get_status(resource="gpu0")
        observed["locked_during"] = st.locked
        observed["owner_service"] = st.owner["service"] if st.owner else None
        observed["owner_document"] = st.owner.get("document_id") if st.owner else None

    pdf = _dummy_pdf(tmp_path / "doc123.pdf")
    fake = _FakeClient(stem="doc123", on_post=during_post)
    with mock.patch.object(ms, "_client", return_value=fake):
        result = ms.process_pdf(pdf, tmp_path,
                                task_id="celery-1", document_id="doc123")

    assert observed["locked_during"] is True
    assert observed["owner_service"] == "mineru"
    assert observed["owner_document"] == "doc123"
    assert result["status"] == "Sucesso"
    # o ZIP do servidor foi materializado no layout que o stage_mineru espera
    assert (tmp_path / "doc123" / "auto" / "doc123_content_list_v2.json").is_file()
    # lock liberado ao final
    assert manager.get_status(resource="gpu0").locked is False


def test_mineru_releases_lock_after_http_failure(gpu_manager_fake, tmp_path):
    """Erro de transporte propaga (como o subprocesso fazia) e solta o lock."""
    import backend.services.mineru_service as ms

    manager = gpu_manager_fake

    def boom():
        raise httpx.ConnectError("mineru-api fora do ar")

    pdf = _dummy_pdf(tmp_path / "docX.pdf")
    fake = _FakeClient(on_post=boom)
    with mock.patch.object(ms, "_client", return_value=fake):
        with pytest.raises(httpx.HTTPError):
            ms.process_pdf(pdf, tmp_path, document_id="docX")

    assert manager.get_status(resource="gpu0").locked is False


def test_mineru_task_failure_returns_error_status(gpu_manager_fake, tmp_path):
    """Falha do parse no servidor ≈ returncode != 0: NÃO levanta, vira status Erro
    (quem aborta o pipeline é o stage_mineru)."""
    import backend.services.mineru_service as ms

    manager = gpu_manager_fake
    pdf = _dummy_pdf(tmp_path / "docY.pdf")
    fake = _FakeClient(status="failed")
    with mock.patch.object(ms, "_client", return_value=fake):
        result = ms.process_pdf(pdf, tmp_path, document_id="docY")

    assert result["status"] == "Erro"
    assert manager.get_status(resource="gpu0").locked is False


def test_mineru_timeout_releases_lock_and_raises(gpu_manager_fake, tmp_path, monkeypatch):
    """MINERU_TIMEOUT_SECONDS estourado → RuntimeError e lock liberado.

    Diferente do subprocesso: a API 3.4.x não permite cancelar, então a tarefa segue
    rodando no servidor — só paramos de aguardar."""
    import backend.services.mineru_service as ms
    from backend.core import config as settings

    monkeypatch.setattr(settings, "MINERU_TIMEOUT_SECONDS", 1)
    monkeypatch.setattr(ms, "_POLL_INTERVAL_SECONDS", 0.01)
    manager = gpu_manager_fake

    # nunca sai de "pending" → o deadline é quem encerra
    pdf = _dummy_pdf(tmp_path / "slow.pdf")
    fake = _FakeClient(status="pending")
    with mock.patch.object(ms, "_client", return_value=fake):
        with pytest.raises(RuntimeError, match="timeout"):
            ms.process_pdf(pdf, tmp_path, document_id="slow")

    assert manager.get_status(resource="gpu0").locked is False


# --------------------------------------------------------------------------- #
# Prioridade / exclusão mútua entre serviços do pipeline
# --------------------------------------------------------------------------- #
def _run(manager, service, priority, hold, order, delay=0.0):
    time.sleep(delay)
    with manager.acquire(resource="gpu0", service=service, priority=priority):
        order.append(("start", service, time.time()))
        time.sleep(hold)
        order.append(("end", service, time.time()))


def test_external_priority_runs_before_indexer_and_does_not_preempt_mineru(gpu_manager_fake):
    from backend.core import config as settings

    make_peer = gpu_manager_fake._make_peer
    mineru = make_peer()
    outro = make_peer()
    ext = make_peer()
    order = []

    # MinerU (20) começa e segura
    mineru_lease = mineru.acquire_lease(resource="gpu0", service="mineru", priority=settings.MINERU_GPU_PRIORITY)
    order.append(("start", "mineru", time.time()))

    t_outro = threading.Thread(target=_run, args=(outro, "outro", 15, 0.1, order, 0.1), daemon=True)
    t_ext = threading.Thread(target=_run, args=(ext, "ext", 10, 0.1, order, 0.25), daemon=True)
    t_outro.start()
    t_ext.start()
    time.sleep(0.5)

    # não-preempção: mineru ainda é o dono
    assert mineru.get_status(resource="gpu0").owner["service"] == "mineru"
    order.append(("end", "mineru", time.time()))
    mineru_lease.release()

    t_outro.join(timeout=5)
    t_ext.join(timeout=5)

    # sem sobreposição
    iv = {}
    for k, s, ts in order:
        iv.setdefault(s, {})[k] = ts
    segs = sorted((v["start"], v["end"], s) for s, v in iv.items())
    assert all(segs[i][1] <= segs[i + 1][0] + 1e-6 for i in range(len(segs) - 1)), segs
    starts = [s for _, _, s in segs]
    assert starts[0] == "mineru"
    assert starts.index("ext") < starts.index("outro"), starts


def test_redis_failure_blocks_cuda_fail_closed():
    # manager apontando p/ um Redis inexistente → acquire levanta antes de qualquer CUDA
    cfg = GPUManagerConfig(redis_url="redis://127.0.0.1:6390/2",
                           redis_connect_timeout_seconds=0.3, redis_socket_timeout_seconds=0.3)
    mgr = GPUResourceManager(cfg)
    cuda_ran = {"flag": False}
    with pytest.raises(GPUBackendUnavailable):
        with mgr.acquire(resource="gpu0", service="mineru", priority=30):
            cuda_ran["flag"] = True
    assert cuda_ran["flag"] is False
