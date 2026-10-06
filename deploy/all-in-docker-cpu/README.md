# Esboço: stack tudo-em-Docker, CPU-only

Variante **self-contained e sem GPU** do `evidencia_pipe`: um `docker compose up`
sobe a infra, os serviços de extração/embedding **e** a aplicação (API + workers).
Contrasta com o `docker-compose.yml` da raiz, que é híbrido — infra em contêiner,
API/workers no host (systemd/venv) e Qdrant externo (ver `DEPLOY.md`).

> **Status: ESBOÇO.** Pensado para validar o fluxo fim-a-fim sem placa, não para
> throughput. Em CPU o MinerU é ordens de magnitude mais lento que em GPU.

## Hoje (híbrido) × alvo (tudo em Docker)

Na máquina de produção atual **nem tudo roda em contêiner**: a infra e a inferência
estão em Docker, mas a **API FastAPI e os 2 workers Celery rodam no host**, no venv
`.venv`, orquestrados por `systemd` (`evidencia-api`, `evidencia-worker-light`,
`evidencia-worker-gpu`). O Qdrant é um contêiner **externo/legado**, fora do compose.

| Componente | Hoje (produção) | Alvo (este diretório) |
|---|---|---|
| API + workers | host — venv + systemd | **contêiner** (`api`, `worker-light`, `worker-index`) |
| Redis / MinIO / Flower | contêiner | contêiner |
| Qdrant | contêiner externo legado | **contêiner no stack** (`qdrant_data`) |
| MinerU / BGE | contêiner local (GPU) | CPU local (`cpu`) **ou** remoto (`remote`) |

A diferença central deste esboço é **tirar a app do host** e pôr tudo em Docker —
sem `.venv`/systemd no destino. Os dados já processados vêm por restore (ver
`host-prep/backup-restore.md`), não por reprocessamento.

## O que sobe

| Serviço | Papel | Observação |
|---|---|---|
| `redis` | broker Celery + job_store + gpu_manager | igual à raiz |
| `flower` | monitor Celery | `:5555` |
| `minio` + `minio-init` | storage S3 dos artefatos | `:9000/:9001` |
| `qdrant` | persistência vetorial | **agora no stack** (na raiz é externo) |
| `mineru-cpu` | extração (estágio 2), backend `pipeline` | imagem nova `deploy/mineru-cpu` |
| `bge-m3-cpu` | embedding denso+esparso num contêiner | reusa `deploy/bge-m3-cpu`, sem `network_mode: host` |
| `api` | FastAPI | `:8020` |
| `worker-light` | filas `download,extract,llm` | `-c 2` |
| `worker-index` | fila `gpu` (embedding+Qdrant) | `-c 1` |

As três últimas compartilham a imagem `evidencia_pipe_app:local` (`Dockerfile` daqui).

## Como rodar

```bash
cd deploy/all-in-docker-cpu
docker compose -f docker-compose.cpu.yml build
docker compose -f docker-compose.cpu.yml up -d
docker compose -f docker-compose.cpu.yml ps
```

Precisa do `.env` na raiz do repo (copie de `.env.example`). O `env.cpu` deste
diretório é carregado **depois** e só sobrescreve os endereços `127.0.0.1` →
nome de serviço, mais os backends CPU.

## Decisões / pendências do esboço

1. **MinerU não tinha caminho CPU.** A imagem de produção parte de `vllm/vllm-openai`
   (CUDA). Criei `deploy/mineru-cpu/Dockerfile` com torch CPU + `mineru[core]` e só
   os modelos do `pipeline` (sem VLM). Validar a flag de device (`MINERU_DEVICE_MODE`)
   contra a versão fixada do MinerU.
2. **`gpu_resource_manager` fica ocioso.** O mutex de GPU coordena host×contêiner no
   modelo híbrido; sem GPU aqui ele não protege nada. Mantido para não mexer no
   código do worker — dá para desativar depois.
3. **Imagem do app carrega libs CUDA à toa.** `pyproject.toml` fixa `onnxruntime-gpu`
   e `fastembed-gpu`. Instalam em CPU mas pesam ~2,5 GB. Para enxugar, trocar pelas
   variantes CPU na imagem (ver comentário no `Dockerfile`) — idealmente via um
   extra/grupo opcional no `pyproject` em vez de `uv pip install` solto.
4. **Concorrência conservadora.** `-c 2`/`-c 1`: em CPU, extrações MinerU paralelas
   competem pelo mesmo host. Ajustar aos núcleos disponíveis.
5. **Versão do Qdrant:** `qdrant/qdrant:v1.17.1`, alinhada à origem de produção
   (`192.168.105.8`, confirmada 1.17.1). Necessária para restaurar snapshot/dir bruto
   sem erro de versão — ver `host-prep/backup-restore.md`.

Nada aqui altera o `docker-compose.yml` da raiz nem o código do backend.
