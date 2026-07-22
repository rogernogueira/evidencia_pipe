"""Aplicação Celery — orquestra o pipeline de ingestão (broker: Redis).

Desenho:
  - Cada PDF vira uma chain OBRIGATÓRIA de 3 estágios (ver backend/tasks.py):
        baixar_dspace → extrair_mineru → indexar_qdrant
    O enrich por LLM é DESACOPLADO: roda como follow-up opcional (enrich_after_index)
    APÓS a indexação, sem que o índice dependa dele.
  - Cada estágio roda numa fila própria, para separar carga:
        download / extract / llm  → worker leve (IO-bound, concorrência alta)
        gpu                        → worker dedicado (concurrency=1, dono do bge-m3)
  - Sem result backend: o status por-documento vive no job_store (Redis DB 1).
    O Flower monitora pela stream de eventos (worker_send_task_events + `-E`).

Subir os workers:
    # leve (download + mineru + llm)
    celery -A backend.celery_app worker -Q download,extract,llm -c 4 -E -n light@%h
    # GPU (embedding + upsert) — UMA cópia do modelo na VRAM
    WORKER_ROLE=gpu celery -A backend.celery_app worker -Q gpu -c 1 -E -n gpu@%h
    # monitor
    celery -A backend.celery_app flower
"""

import os
import sys
from pathlib import Path

# A raiz do projeto precisa estar no sys.path para que o pacote `backend` (incl.
# backend.indexing) seja importável. A API faz isso no main.py, mas o worker Celery
# é iniciado por `-A backend.celery_app` e não passa por lá.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from celery import Celery
from celery.signals import worker_process_init

from backend.core.config import (
    CELERY_BROKER_URL,
    CELERY_TASK_SOFT_TIME_LIMIT,
    CELERY_TASK_TIME_LIMIT,
)
from backend.core.logger import log

app = Celery("evidencia_pipe", broker=CELERY_BROKER_URL, include=["backend.tasks"])

app.conf.update(
    # Sem backend de resultado: o status é rastreado no job_store (Redis DB 1).
    task_ignore_result=True,
    # Serialização explícita — só JSON, SEM pickle (a chain trafega apenas o
    # PipelineContext leve; nenhum artefato grande passa pelo broker).
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    # Timestamps em UTC (evita o aviso de "clock drift" no Flower quando o host
    # está em fuso local).
    enable_utc=True,
    timezone="UTC",
    # Confiabilidade: só dá ack depois de processar. Combinado com o visibility_timeout
    # folgado, evita reentrega no meio de um estágio longo (MinerU/embedding).
    task_acks_late=True,
    worker_prefetch_multiplier=1,  # não "açambarca" tasks longas; distribui melhor
    broker_transport_options={"visibility_timeout": 3600},  # 1h > estágio mais lento
    # Limite de tempo por task. soft: levanta SoftTimeLimitExceeded (tratável →
    # dispara on_failure e registra a falha); hard: mata o processo do worker. O
    # hard (< visibility_timeout) evita reentrega/duplicação de uma task travada.
    task_soft_time_limit=CELERY_TASK_SOFT_TIME_LIMIT or None,
    task_time_limit=CELERY_TASK_TIME_LIMIT or None,
    # Eventos para o Flower enxergar tasks/workers em tempo real.
    worker_send_task_events=True,
    task_send_sent_event=True,
    # Roteamento por fila (uma por estágio).
    task_routes={
        "backend.tasks.baixar_dspace": {"queue": "download"},
        "backend.tasks.extrair_mineru": {"queue": "extract"},
        "backend.tasks.enrich_llm": {"queue": "llm"},
        "backend.tasks.enrich_after_index": {"queue": "llm"},
        "backend.tasks.indexar_qdrant": {"queue": "gpu"},
    },
    task_default_queue="extract",
)


@worker_process_init.connect
def _preload_embedder(**_):
    """Preload do bge-m3 — SOMENTE no worker de GPU (WORKER_ROLE=gpu) e SOMENTE
    quando explicitamente configurado.

    IMPORTANTE (GPU compartilhada com o MinerU): o `concurrency=1` da fila `gpu` só
    controla as tasks DAQUELE worker Celery — NÃO controla o subprocesso do MinerU
    nem scripts externos. Quem coordena TODOS os processos é o gpu_resource_manager
    (lock distribuído no Redis DB 2). E o ciclo de vida do modelo na VRAM é
    controlado pelo BGEModelManager (BGE_GPU_LIFECYCLE).

    Por isso, com a GPU compartilhada (GPU_SHARED_WITH_MINERU=true, o padrão) NÃO
    pré-carregamos o modelo direto na VRAM: se o bge-m3 ficasse residente, ocuparia
    memória mesmo sem inferência e poderia causar OOM quando o MinerU rodasse. O
    modelo é carregado sob demanda e movido para a GPU só sob o lock, sendo
    descarregado após a task (BGE_GPU_LIFECYCLE=lazy + BGE_UNLOAD_AFTER_TASK=true).

    O preload direto na VRAM só ocorre se BGE_GPU_LIFECYCLE=preload E a GPU não for
    compartilhada (uso dedicado) — uma escolha explícita do operador."""
    if os.getenv("WORKER_ROLE") != "gpu":
        return

    from backend.core import config as settings

    preload_to_vram = settings.BGE_GPU_LIFECYCLE == "preload" and not settings.GPU_SHARED_WITH_MINERU
    if not preload_to_vram:
        log.info(
            "[worker gpu] Preload de VRAM desativado (lifecycle=%s, shared=%s) — "
            "bge-m3 carregado sob demanda e movido à GPU sob o lock.",
            settings.BGE_GPU_LIFECYCLE, settings.GPU_SHARED_WITH_MINERU,
        )
        return

    try:
        from backend.services.bge_model_manager import get_model_manager

        mm = get_model_manager()
        from backend.services.embedder import BgeM3EmbedderService

        if BgeM3EmbedderService.is_cached_locally():
            log.info("[worker gpu] Pré-carregando bge-m3 na VRAM (lifecycle=preload, GPU dedicada)…")
            if mm.ensure_loaded():
                mm.move_to_gpu()
                log.info("[worker gpu] bge-m3 pré-carregado na VRAM.")
            else:
                log.warning("[worker gpu] bge-m3 não pôde ser carregado — fallback sob demanda.")
        else:
            log.warning("[worker gpu] bge-m3 não está no cache local — carga sob demanda no 1º job.")
    except Exception as exc:  # pragma: no cover
        log.warning("[worker gpu] Falha ao pré-carregar o embedder: %s", exc)
