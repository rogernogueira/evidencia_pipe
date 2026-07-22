"""Testes de integração de coordenação de GPU no evidencia_pipe (§40, subconjunto
testável sem GPU/torch/FlagEmbedding reais). Cobrem:

  - MinerU adquire a GPU e mantém o lock durante o subprocesso (mock);
  - MinerU libera o lock após falha do subprocesso (encerrando o grupo de processos);
  - prioridade: script externo (10) roda antes do BGE (30); nenhum interrompe o MinerU;
  - ciclo de vida do BGE-M3: descarrega a VRAM quando configurado (BGEModelManager);
  - falha do Redis impede a execução CUDA (fail-closed).

Os intervalos de uso da GPU nunca se sobrepõem.
"""

import sys
import threading
import time
import types
from unittest import mock

import pytest

from gpu_resource_manager import GPUBackendUnavailable, GPUResourceManager, GPUManagerConfig


# --------------------------------------------------------------------------- #
# MinerU
# --------------------------------------------------------------------------- #
def test_mineru_holds_lock_during_subprocess_and_releases(gpu_manager_fake, tmp_path):
    import backend.services.mineru_service as ms

    manager = gpu_manager_fake
    observed = {}

    class FakeProc:
        pid = 4321
        returncode = 0

        def communicate(self, timeout=None):
            # enquanto o "subprocesso" roda, o lock deve estar retido por mineru
            st = manager.get_status(resource="gpu0")
            observed["locked_during"] = st.locked
            observed["owner_service"] = st.owner["service"] if st.owner else None
            observed["owner_document"] = st.owner.get("document_id") if st.owner else None
            return ("stdout", "stderr")

        def poll(self):
            return 0

    with mock.patch.object(ms.subprocess, "Popen", return_value=FakeProc()):
        result = ms.process_pdf(tmp_path / "doc123.pdf", tmp_path, task_id="celery-1", document_id="doc123")

    assert observed["locked_during"] is True
    assert observed["owner_service"] == "mineru"
    assert observed["owner_document"] == "doc123"
    assert result["status"] == "Sucesso"
    # lock liberado ao final
    assert manager.get_status(resource="gpu0").locked is False


def test_mineru_releases_lock_after_subprocess_failure(gpu_manager_fake, tmp_path):
    import backend.services.mineru_service as ms

    manager = gpu_manager_fake
    killed = {}

    class ExplodingProc:
        pid = 9999
        returncode = None

        def communicate(self, timeout=None):
            raise TimeoutError("subprocesso travou")

        def poll(self):
            return None

    def fake_kill_group(proc, grace_seconds=10.0):
        killed["group"] = proc.pid

    with mock.patch.object(ms.subprocess, "Popen", return_value=ExplodingProc()), \
         mock.patch.object(ms, "_terminate_process_group", side_effect=fake_kill_group):
        with pytest.raises(TimeoutError):
            ms.process_pdf(tmp_path / "docX.pdf", tmp_path, document_id="docX")

    # grupo de processos encerrado e lock liberado mesmo na falha
    assert killed.get("group") == 9999
    assert manager.get_status(resource="gpu0").locked is False


def test_mineru_timeout_kills_group_and_raises(gpu_manager_fake, tmp_path, monkeypatch):
    """MINERU_TIMEOUT_SECONDS estourado → subprocess.TimeoutExpired → grupo encerrado
    (libera a VRAM), lock liberado e RuntimeError propagado."""
    import backend.services.mineru_service as ms
    from backend.core import config as settings

    monkeypatch.setattr(settings, "MINERU_TIMEOUT_SECONDS", 1)
    manager = gpu_manager_fake
    killed = {}

    class HangingProc:
        pid = 7777
        returncode = None

        def communicate(self, timeout=None):
            raise ms.subprocess.TimeoutExpired(cmd="mineru", timeout=timeout)

        def poll(self):
            return None

    with mock.patch.object(ms.subprocess, "Popen", return_value=HangingProc()), \
         mock.patch.object(ms, "_terminate_process_group",
                           side_effect=lambda proc, grace_seconds=10.0: killed.update(group=proc.pid)):
        with pytest.raises(RuntimeError, match="timeout"):
            ms.process_pdf(tmp_path / "slow.pdf", tmp_path, document_id="slow")

    assert killed.get("group") == 7777
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


def test_external_priority_runs_before_bge_and_does_not_preempt_mineru(gpu_manager_fake):
    from backend.core import config as settings

    make_peer = gpu_manager_fake._make_peer
    mineru = make_peer()
    bge = make_peer()
    ext = make_peer()
    order = []

    # MinerU (20) começa e segura
    mineru_lease = mineru.acquire_lease(resource="gpu0", service="mineru", priority=settings.MINERU_GPU_PRIORITY)
    order.append(("start", "mineru", time.time()))

    t_bge = threading.Thread(target=_run, args=(bge, "bge-m3", settings.BGE_GPU_PRIORITY, 0.1, order, 0.1), daemon=True)
    t_ext = threading.Thread(target=_run, args=(ext, "ext", 10, 0.1, order, 0.25), daemon=True)
    t_bge.start()
    t_ext.start()
    time.sleep(0.5)

    # não-preempção: mineru ainda é o dono
    assert mineru.get_status(resource="gpu0").owner["service"] == "mineru"
    order.append(("end", "mineru", time.time()))
    mineru_lease.release()

    t_bge.join(timeout=5)
    t_ext.join(timeout=5)

    # sem sobreposição
    iv = {}
    for k, s, ts in order:
        iv.setdefault(s, {})[k] = ts
    segs = sorted((v["start"], v["end"], s) for s, v in iv.items())
    assert all(segs[i][1] <= segs[i + 1][0] + 1e-6 for i in range(len(segs) - 1)), segs
    starts = [s for _, _, s in segs]
    assert starts[0] == "mineru"
    assert starts.index("ext") < starts.index("bge-m3"), starts


def test_redis_failure_blocks_cuda_fail_closed():
    # manager apontando p/ um Redis inexistente → acquire levanta antes de qualquer CUDA
    cfg = GPUManagerConfig(redis_url="redis://127.0.0.1:6390/2",
                           redis_connect_timeout_seconds=0.3, redis_socket_timeout_seconds=0.3)
    mgr = GPUResourceManager(cfg)
    cuda_ran = {"flag": False}
    with pytest.raises(GPUBackendUnavailable):
        with mgr.acquire(resource="gpu0", service="bge-m3", priority=30):
            cuda_ran["flag"] = True
    assert cuda_ran["flag"] is False


# --------------------------------------------------------------------------- #
# Ciclo de vida do BGE-M3 na VRAM (BGEModelManager) — PyTorch/BGE mockados
# --------------------------------------------------------------------------- #
@pytest.fixture
def fake_torch_and_embedder(monkeypatch):
    """Instala um módulo torch falso e um backend.services.embedder falso com um
    'modelo' rastreável (registra to('cuda')/to('cpu') e empty_cache)."""
    calls = {"to": [], "empty_cache": 0, "synchronize": 0, "ipc_collect": 0}

    inner = types.SimpleNamespace()
    inner.to = lambda dev: calls["to"].append(dev)

    fake_model = types.SimpleNamespace(model=inner)

    # módulo embedder falso (evita importar FlagEmbedding)
    emb_mod = types.ModuleType("backend.services.embedder")
    emb_mod._SHARED_MODEL = fake_model

    class _Embedder:
        def is_loaded(self):
            return emb_mod._SHARED_MODEL is not None

        def load_model(self, require_cache=False):
            return True

    emb_mod.BgeM3EmbedderService = _Embedder
    monkeypatch.setitem(sys.modules, "backend.services.embedder", emb_mod)

    # torch falso
    torch_mod = types.ModuleType("torch")
    cuda_ns = types.SimpleNamespace(
        is_available=lambda: True,
        empty_cache=lambda: calls.__setitem__("empty_cache", calls["empty_cache"] + 1),
        ipc_collect=lambda: calls.__setitem__("ipc_collect", calls["ipc_collect"] + 1),
        synchronize=lambda: calls.__setitem__("synchronize", calls["synchronize"] + 1),
    )
    torch_mod.cuda = cuda_ns
    monkeypatch.setitem(sys.modules, "torch", torch_mod)

    return calls, emb_mod


def test_bge_unloads_vram_when_configured(fake_torch_and_embedder):
    calls, emb_mod = fake_torch_and_embedder
    from backend.services.bge_model_manager import BGEModelManager

    mm = BGEModelManager(lifecycle="lazy", unload_after_task=True)
    mm.move_to_gpu()
    assert "cuda" in calls["to"]
    assert mm.on_gpu is True

    mm.release_gpu_if_configured()
    # sincronizou, descarregou a instância (unload) e limpou o cache CUDA
    assert calls["synchronize"] >= 1
    assert calls["empty_cache"] >= 1
    assert emb_mod._SHARED_MODEL is None  # VRAM liberada (instância removida)
    assert mm.on_gpu is False


def test_bge_cpu_idle_moves_to_cpu_without_unloading(fake_torch_and_embedder):
    calls, emb_mod = fake_torch_and_embedder
    from backend.services.bge_model_manager import BGEModelManager

    mm = BGEModelManager(lifecycle="cpu_idle", unload_after_task=False)
    mm.move_to_gpu()
    mm.release_gpu_if_configured()
    assert calls["to"][-1] == "cpu"          # voltou para CPU
    assert emb_mod._SHARED_MODEL is not None  # NÃO descarregou (mantém em RAM)
