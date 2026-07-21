"""Aplicação Celery — orquestra o pipeline de ingestão (broker: Redis).

Desenho:
  - Cada PDF vira uma chain de 4 estágios (ver backend/tasks.py):
        baixar_dspace → extrair_mineru → enrich_llm → indexar_qdrant
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

from backend.core.config import CELERY_BROKER_URL
from backend.core.logger import log

app = Celery("evidencia_pipe", broker=CELERY_BROKER_URL, include=["backend.tasks"])

app.conf.update(
    # Sem backend de resultado: o status é rastreado no job_store (Redis DB 1).
    task_ignore_result=True,
    # Serialização explícita (args são strings/dicts simples).
    task_serializer="json",
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
    # Eventos para o Flower enxergar tasks/workers em tempo real.
    worker_send_task_events=True,
    task_send_sent_event=True,
    # Roteamento por fila (uma por estágio).
    task_routes={
        "backend.tasks.baixar_dspace": {"queue": "download"},
        "backend.tasks.extrair_mineru": {"queue": "extract"},
        "backend.tasks.enrich_llm": {"queue": "llm"},
        "backend.tasks.indexar_qdrant": {"queue": "gpu"},
    },
    task_default_queue="extract",
)


@worker_process_init.connect
def _preload_embedder(**_):
    """Pré-carrega o bge-m3 na VRAM — SOMENTE no worker de GPU (WORKER_ROLE=gpu).

    Workers Celery são processos separados do FastAPI e não herdam o modelo
    pré-carregado no lifespan da API. Se todos os workers carregassem o modelo,
    haveria N cópias na VRAM → OOM. Por isso só o worker de GPU (concurrency=1)
    faz o preload; nos demais o embedder nem é tocado. Best-effort: sem cache
    local, o modelo é carregado sob demanda no 1º job de GPU."""
    if os.getenv("WORKER_ROLE") != "gpu":
        return
    try:
        from backend.services.embedder import BgeM3EmbedderService

        embedder = BgeM3EmbedderService()
        if embedder.is_cached_locally():
            log.info("[worker gpu] Pré-carregando bge-m3 na VRAM…")
            ok = embedder.load_model(require_cache=True)
            log.info("[worker gpu] bge-m3: %s", "carregado" if ok else "fallback sob demanda")
        else:
            log.warning("[worker gpu] bge-m3 não está no cache local — carga sob demanda no 1º job.")
    except Exception as exc:  # pragma: no cover
        log.warning("[worker gpu] Falha ao pré-carregar o embedder: %s", exc)
