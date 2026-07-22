"""Exemplo de script EXTERNO prioritário que compartilha a mesma GPU.

Qualquer script Python no mesmo ambiente (ou em outro contêiner que aponte para o
mesmo Redis DB 2 e use o mesmo GPU_RESOURCE_NAME) participa da coordenação global.

Instalação (uma vez, no ambiente do projeto):

    uv add --editable ./packages/gpu_resource_manager

Ou, em outro repositório, via wheel:

    uv build packages/gpu_resource_manager
    uv add ./dist/gpu_resource_manager-0.1.0-py3-none-any.whl

Execução:

    GPU_MANAGER_REDIS_URL=redis://127.0.0.1:6379/2 GPU_RESOURCE_NAME=gpu0 \
        python -m gpu_resource_manager.examples.external_priority_script  # (ou o caminho do arquivo)

Para participar da coordenação, o script precisa: acessar o MESMO Redis, usar o
MESMO recurso, adquirir o lease ANTES de qualquer CUDA, manter o lease durante todo
o uso e liberá-lo ao terminar. Scripts que ignorarem o gerenciador NÃO serão
controlados pela biblioteca.
"""

from __future__ import annotations

import logging
import time

from gpu_resource_manager import (
    GPUAcquisitionTimeout,
    GPUBackendUnavailable,
    GPUManagerConfig,
    GPUResourceManager,
)


def executar_inferencia_prioritaria() -> None:
    # Simula uso da GPU. Em um script real: mover modelo p/ CUDA, inferir, e
    # ANTES de sair do bloco, mover resultados p/ CPU e liberar a VRAM.
    print("  [ext] usando a GPU (inferência prioritária)...")
    time.sleep(2)
    print("  [ext] concluído.")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")

    config = GPUManagerConfig.from_env()
    manager = GPUResourceManager(config)

    try:
        with manager.acquire(
            resource=config.resource_name,
            service="script-analise-especial",
            priority=10,  # menor número = maior prioridade (acima de MinerU=20, BGE=30)
            metadata={"script": __file__, "operation": "inferencia-prioritaria"},
        ) as lease:
            print(f"  [ext] lease adquirido: request_id={lease.request_id} "
                  f"wait={lease.wait_time_seconds}s")
            executar_inferencia_prioritaria()
            lease.ensure_valid()  # confirma a posse antes do passo crítico final
        print(f"  [ext] lease liberado (held={lease.held_time_seconds}s, "
              f"metrics={lease.metrics}).")
        return 0
    except GPUAcquisitionTimeout as exc:
        print(f"  [ext] timeout aguardando a GPU: dono atual={exc.current_owner}")
        return 1
    except GPUBackendUnavailable as exc:
        print(f"  [ext] Redis do gerenciador indisponível (fail-closed): {exc}")
        return 2
    finally:
        manager.close()


if __name__ == "__main__":
    raise SystemExit(main())
