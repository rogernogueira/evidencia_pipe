"""Ciclo de vida do BGE-M3 na VRAM (específico do evidencia_pipe).

O `gpu_resource_manager` NÃO conhece PyTorch/FlagEmbedding — ele apenas arbitra o
recurso. A ocupação real da VRAM é controlada aqui: o lock Redis garante exclusão,
mas NÃO libera memória sozinho. Um lock adquirido enquanto o modelo permanece na
GPU ainda pode causar OOM para o próximo dono; por isso, antes de liberar o lease,
movemos o modelo para CPU / descarregamos e limpamos o cache CUDA.

Políticas (BGE_GPU_LIFECYCLE):
  - preload  : carrega e mantém na GPU (só quando a GPU NÃO é compartilhada).
  - lazy     : carrega sob demanda; ao terminar, sai da GPU (CPU ou unload).  [padrão]
  - cpu_idle : mantém o modelo em RAM (CPU) entre usos; sobe para GPU só sob lock.
  - unload   : descarrega totalmente a instância após cada uso.

`BGE_UNLOAD_AFTER_TASK=true` força unload após o uso, independente de lazy/cpu_idle.

Os métodos de PyTorch são best-effort e defensivos (try/except + log): em ambientes
sem CUDA/torch, o manager degrada para no-op sem quebrar o pipeline.
"""

from __future__ import annotations

import gc
import importlib
import threading
from typing import Optional

from backend.core import config as settings
from backend.core.logger import log

# Políticas válidas.
PRELOAD = "preload"
LAZY = "lazy"
CPU_IDLE = "cpu_idle"
UNLOAD = "unload"


class BGEModelManager:
    """Controla carregamento, device e limpeza de VRAM do BGE-M3.

    Reusa o modelo global do `BgeM3EmbedderService` (uma cópia por processo). A
    concorrência local do modelo é protegida por um lock de processo (o mutex
    *distribuído* é responsabilidade do gpu_resource_manager)."""

    def __init__(
        self,
        lifecycle: Optional[str] = None,
        unload_after_task: Optional[bool] = None,
    ) -> None:
        self.lifecycle = (lifecycle or settings.BGE_GPU_LIFECYCLE or LAZY).lower()
        self.unload_after_task = (
            settings.BGE_UNLOAD_AFTER_TASK if unload_after_task is None else unload_after_task
        )
        self._local_lock = threading.Lock()
        self._on_gpu = False

    # ------------------------------------------------------------------ #
    @staticmethod
    def _torch():
        import torch  # import tardio: torch não é dependência do core da lib

        return torch

    def _embedder(self):
        from backend.services.embedder import BgeM3EmbedderService

        return BgeM3EmbedderService()

    def _inner_module(self):
        """Retorna o nn.Module interno do BGEM3FlagModel para mover de device."""
        # importlib.import_module consulta sys.modules — mesma resolução usada por
        # `_embedder()` (from backend.services.embedder import ...), evitando
        # divergência entre o módulo real e um substituto injetado em testes.
        _emb = importlib.import_module("backend.services.embedder")

        model = _emb._SHARED_MODEL
        if model is None:
            return None
        # FlagEmbedding expõe o encoder em `.model`; caímos para o próprio objeto
        # se a estrutura mudar entre versões.
        return getattr(model, "model", model)

    # ------------------------------------------------------------------ #
    def is_loaded(self) -> bool:
        return self._embedder().is_loaded()

    @property
    def on_gpu(self) -> bool:
        return self._on_gpu

    def ensure_loaded(self) -> bool:
        """Garante que o modelo está instanciado (sob demanda, sem exigir cache)."""
        emb = self._embedder()
        if emb.is_loaded():
            return True
        log.info("[bge] carregando modelo sob demanda (lifecycle=%s)...", self.lifecycle)
        return emb.load_model(require_cache=False)

    def move_to_gpu(self) -> None:
        """Move o modelo para a GPU. Chamar SOMENTE dentro do lease da GPU."""
        with self._local_lock:
            if not self.ensure_loaded():
                log.warning("[bge] move_to_gpu: modelo não pôde ser carregado.")
                return
            try:
                torch = self._torch()
                if not torch.cuda.is_available():
                    log.info("[bge] CUDA indisponível — modelo permanece em CPU.")
                    return
                inner = self._inner_module()
                if inner is not None and not self._on_gpu:
                    inner.to("cuda")
                    self._on_gpu = True
                    log.info("[bge] modelo movido para a GPU.")
            except Exception as exc:  # pragma: no cover - defensivo
                log.warning("[bge] falha ao mover para GPU (seguindo): %s", exc)

    def synchronize(self) -> None:
        """Aguarda a conclusão de todo o trabalho CUDA pendente."""
        try:
            torch = self._torch()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception as exc:  # pragma: no cover
            log.debug("[bge] synchronize ignorado: %s", exc)

    def move_to_cpu(self) -> None:
        with self._local_lock:
            try:
                inner = self._inner_module()
                if inner is not None and self._on_gpu:
                    inner.to("cpu")
                    self._on_gpu = False
                    log.info("[bge] modelo movido para a CPU.")
            except Exception as exc:  # pragma: no cover
                log.warning("[bge] falha ao mover para CPU: %s", exc)
        self.empty_cache()

    def unload(self) -> None:
        """Remove a instância do modelo e libera a VRAM completamente."""
        with self._local_lock:
            _emb = importlib.import_module("backend.services.embedder")

            _emb._SHARED_MODEL = None
            self._on_gpu = False
            log.info("[bge] modelo descarregado (instância removida).")
        self.empty_cache()

    def empty_cache(self) -> None:
        """gc + esvazia o cache do alocador CUDA (não substitui o lock!)."""
        gc.collect()
        try:
            torch = self._torch()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                # ipc_collect só é útil quando há tensores compartilhados entre processos.
                try:
                    torch.cuda.ipc_collect()
                except Exception:  # pragma: no cover
                    pass
        except Exception as exc:  # pragma: no cover
            log.debug("[bge] empty_cache ignorado: %s", exc)

    def release_gpu_if_configured(self) -> None:
        """Aplica a política de ciclo de vida ANTES de liberar o lease (§31).

        Ordem recomendada pelo chamador: (1) synchronize, (2) resultados → CPU,
        (3) este método (move p/ CPU ou unload + empty_cache), (4) liberar lease."""
        self.synchronize()
        if self.unload_after_task or self.lifecycle == UNLOAD:
            self.unload()
        elif self.lifecycle in (LAZY, CPU_IDLE):
            self.move_to_cpu()
        elif self.lifecycle == PRELOAD:
            # Mantém na GPU (uso exclusivo). Só faz sentido com GPU não compartilhada.
            self.empty_cache()
        else:  # pragma: no cover - política desconhecida vira lazy
            self.move_to_cpu()


# Singleton de processo (uma cópia do modelo por processo).
_model_manager: Optional[BGEModelManager] = None
_mm_lock = threading.Lock()


def get_model_manager() -> BGEModelManager:
    global _model_manager
    if _model_manager is None:
        with _mm_lock:
            if _model_manager is None:
                _model_manager = BGEModelManager()
    return _model_manager
