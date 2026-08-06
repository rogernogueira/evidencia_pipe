# Deploy — systemd (auto-start no boot)

Units systemd para o `evidencia_pipe` subir **automaticamente no boot**. O pipeline tem
duas camadas: a **infra em Docker** (redis, flower, minio, mineru) e os **processos no
host** (API + 2 workers Celery, no venv por causa da GPU/modelos locais). Os contêineres
já têm `restart: unless-stopped`; estas units cobrem os processos do host e amarram tudo
num alvo único.

## Units

| Arquivo | Papel |
|---|---|
| `evidencia-compose.service` | oneshot: `docker compose up -d redis flower minio minio-init mineru-pipeline vllm-bge-m3 vllm-bge-m3-sparse` (`ExecStop` = `docker compose stop`) |
| `evidencia-api.service` | API FastAPI — `uv run python backend/main.py` (:8181, via `PORT` no `.env`) |
| `evidencia-worker-light.service` | worker leve — filas `download,extract,llm`, `-c 4` |
| `evidencia-worker-gpu.service` | worker GPU — fila `gpu`, `-c 1`, `WORKER_ROLE=gpu` (sonda a API de embedding na subida) |
| `evidencia.target` | alvo agregador dos 4 — sobe/derruba tudo de uma vez |

A API e os workers têm `Requires=`/`After=evidencia-compose.service` (dependem de
Redis/MinIO) e `Restart=on-failure`. O `evidencia-compose` roda `After=docker.service`.

## Instalar

```bash
sudo ./install.sh              # copia, daemon-reload, enable e start
sudo ./install.sh --no-start   # instala e habilita, sem iniciar agora
```

## Operar

```bash
systemctl status evidencia.target      # visão geral do stack
systemctl restart evidencia.target      # reinicia infra + API + workers
systemctl stop evidencia-worker-gpu     # parar um serviço isolado
journalctl -u evidencia-api -f          # logs ao vivo da API
journalctl -u evidencia-worker-gpu -f   # logs do worker de GPU
```

## Remover

```bash
sudo ./uninstall.sh   # para, desabilita e apaga as units (não toca em Docker/dados)
```

## Pressupostos (ajuste os units se o host divergir)

- `WorkingDirectory=/app/evidencia_pipe` em todos os serviços do host.
- `uv` em `/root/.local/bin/uv` (host roda como `root`); o `.env` do projeto é carregado
  sozinho (`load_dotenv` em `backend/core/config.py`).
- GPU única → sobe **`mineru-pipeline` (:8012)**. Para usar o VLM (`mineru`, :8011),
  edite o `ExecStart` de `evidencia-compose.service`.
- **API de embedding**: `vllm-bge-m3` (:8000, denso) e `vllm-bge-m3-sparse` (:8001,
  esparso) sobem junto com a infra. Sem eles a indexação e a busca falham — não há
  fallback local. São dois porque o servidor HTTP do vLLM fixa uma pooling task por
  instância (ver `backend/services/embedder.py`).
- `%%h` nos `-n light@%%h`/`gpu@%%h`: o systemd desescapa para `%h`, que o Celery expande
  como hostname (nome do worker).
