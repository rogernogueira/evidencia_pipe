# Orquestração com Celery + Redis

O pipeline de ingestão roda como uma **chain** Celery por PDF:

```
baixar_dspace → extrair_mineru → enrich_llm → indexar_qdrant
   (download)      (extract)        (llm)          (gpu)
```

Cada estágio tem uma fila própria. A API só **enfileira** (responde `202` na hora);
os **workers** executam. O status por-documento vive no `job_store` (Redis DB 1) e é
consultado em `GET /api/files/status/{job_id}`. O **Flower** monitora a infra.

## 1. Subir a infra (Redis + Flower)

```bash
docker compose up -d          # redis :6379, flower :5555
```

Flower: http://localhost:5555

## 2. Subir a API (produtor)

```bash
uv run python backend/main.py     # :8020 — só enfileira, não processa
```

## 3. Subir os workers (consumidores)

Rodam no venv do host (GPU + modelos locais). **Dois** workers:

```bash
# Worker leve — download, MinerU e LLM (IO-bound, pode ter concorrência)
uv run celery -A backend.celery_app worker -Q download,extract,llm -c 4 -E -n light@%h

# Worker de GPU — embedding + upsert no Qdrant.
# concurrency=1 e WORKER_ROLE=gpu para carregar UMA cópia do bge-m3 na VRAM.
WORKER_ROLE=gpu uv run celery -A backend.celery_app worker -Q gpu -c 1 -E -n gpu@%h
```

> ⚠️ Nunca rode a fila `gpu` com `-c > 1`: cada processo carregaria o bge-m3 → OOM de VRAM.
> O `-E` habilita eventos (necessário para o Flower ver as tasks).

## Configuração (env)

| Var | Default | Papel |
|-----|---------|-------|
| `REDIS_URL` | `redis://127.0.0.1:6379` | base do broker e do job_store |
| `CELERY_BROKER_URL` | `${REDIS_URL}/0` | broker |
| `JOBSTORE_REDIS_URL` | `${REDIS_URL}/1` | status dos jobs |
| `JOBSTORE_TTL` | `604800` (7d) | expiração dos registros de job |

## Semântica de erro

- **download / MinerU** falham → job vira `erro`, a chain para (não indexa).
- **enrich (LLM)** falha → best-effort: loga, marca `llm_error` e **segue** para indexação.
- **index** falha → job fica `concluido` com `index_error` (o markdown já é válido).
- Estado inicial `na_fila` = enfileirado, ainda não pego por um worker.

Erros de download **não** são mais síncronos no endpoint de item (responde `202`);
acompanhe em `/status` ou no Flower. O endpoint de bitstream avulso mantém o `502`
síncrono (baixa antes de enfileirar).
