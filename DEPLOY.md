# Deploy — `evidencia_pipe`

Passo a passo de uma instalação do zero até o pipeline processando um documento.

O sistema tem **três camadas**, e é isso que explica o formato do deploy:

| Camada | Onde roda | O quê |
|---|---|---|
| Infra | Docker (`docker-compose.yml`) | Redis, MinIO, Flower, MinerU, os dois vLLM do bge-m3 |
| Processos | Host, no venv (`systemd`) | API FastAPI + 2 workers Celery |
| Externo | Fora deste repositório | **Qdrant** e o **DSpace** |

Os processos do host ficam fora do Docker de propósito: eles falam com a GPU e leem
o `.env` do diretório do projeto. Quem sobe tudo junto no boot é o `evidencia.target`.

---

## 0. Pré-requisitos

**Hardware.** GPU NVIDIA com VRAM suficiente para o MinerU **mais** os dois contêineres
de embedding. Na máquina de referência (RTX A6000, 48 GB) o orçamento é: bge-m3
`2 × 0,12 × 48 GB ≈ 11,5 GB` reservados no startup + MinerU (`pipeline` ~6,5 GB;
`hybrid-engine` precisa de ~24 GB livres, senão **aborta**). Numa GPU única, rode o
serviço `mineru-pipeline` **ou** o `mineru`, nunca os dois.

**Software.**

```bash
docker --version                 # + plugin compose v2
nvidia-ctk --version             # NVIDIA Container Toolkit
uv --version                     # gerenciador de deps (Python 3.13+)
python3 --version
```

O compose usa **CDI** para a GPU nos serviços vLLM (`devices: nvidia.com/gpu=all`).
Confirme que o device existe antes de continuar:

```bash
grep -c 'name: all' /var/run/cdi/nvidia.yaml     # deve ser >= 1
docker run --rm --device nvidia.com/gpu=all ubuntu:22.04 ls /dev/nvidiactl
```

Se o CDI não estiver configurado: `sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml`.

**Disco.** ~55 GB só de imagens (MinerU ~30 GB, bge-m3 ~21 GB), mais os artefatos no
MinIO. Comece com 100 GB livres com folga.

---

## 1. Código e dependências

```bash
git clone git@github.com:rogernogueira/evidencia_pipe.git /app/evidencia_pipe
cd /app/evidencia_pipe
uv sync
```

O `uv sync` instala também a lib do workspace `packages/gpu_resource_manager` em modo
editável. Para rodar os testes: `uv sync --group test`.

O venv do host é **leve**: nem `mineru` nem `torch` entram nele. A extração roda no
contêiner (via HTTP) e o embedding também — o host só precisa do **tokenizer** do
bge-m3, que o `transformers` baixa sozinho no primeiro uso (~17 MB, exige rede uma
vez). Para não pagar isso no primeiro job, pré-aqueça o cache:

```bash
uv run python -c "
from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('BAAI/bge-m3')
print('tokenizer em cache')"
```

> Os units systemd assumem `/app/evidencia_pipe` e `uv` em `/root/.local/bin/uv`.
> Em outro caminho, ajuste o `WorkingDirectory`/`ExecStart` em `deploy/systemd/`.

---

## 2. Configuração (`.env`)

```bash
cp .env.example .env
```

O que **precisa** ser revisado antes de subir:

| Variável | Observação |
|---|---|
| `QDRANT_URL` | aponta para o Qdrant do passo 3. **Não** existe serviço Qdrant neste compose |
| `MINERU_API_URL` | `:8012` para o serviço `mineru-pipeline`, `:8011` para o `mineru` (VLM) |
| `MINERU_BACKEND` | casado com o serviço acima: `pipeline` ou `hybrid-engine` |
| `PORT` | porta da API (o default do código é 8020; a instalação de referência usa 8181) |
| `LLM_ENRICH_API_KEY` | **opcional** — sem ela o enrich é pulado e a indexação segue |
| `MINIO_ROOT_USER` / `MINIO_ROOT_PASSWORD` | troque de `minioadmin` em produção |
| `GPU_MANAGER_REDIS_URL` | dentro do compose use `redis://redis:6379/2`; **nos processos do host, `redis://127.0.0.1:6379/2`** |

Os defaults de embedding (`EMBED_API_URL=:8000`, `EMBED_API_SPARSE_URL=:8001`) servem
quando tudo roda no mesmo host — que é o caso dos workers no venv.

O `.env` **não** é versionado e é lido sozinho por `backend/core/config.py`
(`load_dotenv`), inclusive pelos units systemd via `WorkingDirectory`.

---

## 3. Qdrant (dependência externa)

O Qdrant **não está** no `docker-compose.yml` deste repositório — ele é compartilhado
com o projeto legado. Na máquina de referência roda como o contêiner `mineru_qdrant`,
a partir de `/app/minerU/docker-compose.yml`, com o storage em
`/app/minerU/qdrant_storage`.

Numa instalação nova, suba um por conta e aponte o `QDRANT_URL`:

```bash
docker run -d --name qdrant --restart unless-stopped \
  -p 6333:6333 -p 6334:6334 \
  -v /srv/qdrant_storage:/qdrant/storage \
  qdrant/qdrant:latest

curl -s http://127.0.0.1:6333/collections | head -c 200
```

A **collection é criada automaticamente** na primeira indexação, com a dimensão densa
sondada da própria API de embedding — não crie à mão.

---

## 4. Build das imagens

Duas imagens são construídas localmente. Cada uma **assa os modelos dentro da imagem**,
então o build é demorado (dezenas de minutos na primeira vez) e o runtime não depende
de download nem do cache do host.

```bash
docker compose build mineru-pipeline      # evidencia_mineru:local (~30 GB)
docker compose build vllm-bge-m3          # evidencia_bge_m3:local (~21 GB)
```

O `vllm-bge-m3-sparse` reaproveita a mesma imagem do `vllm-bge-m3` — não precisa
construir de novo.

Conferência rápida de que os pesos do bge-m3 entraram (sem `sparse_linear.pt` **não há
vetor esparso**, e a busca híbrida perde a perna lexical em silêncio):

```bash
docker run --rm --entrypoint /bin/bash evidencia_bge_m3:local \
  -c 'ls /root/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/*/ | grep -E "sparse_linear|pytorch_model"'
```

---

## 5. Subir a infra

```bash
docker compose up -d redis flower minio minio-init mineru-pipeline \
                     vllm-bge-m3 vllm-bge-m3-sparse

docker compose ps
```

Os dois vLLM levam **1–3 minutos** para ficar `healthy` (carga do modelo +
compilação do `torch.compile`); o healthcheck tem `start_period: 300s` por isso.
O `minio-init` roda uma vez, cria o bucket privado e sai — status `exited (0)` é o
esperado.

---

## 6. Verificação da infra

```bash
# Embedding — os dois têm de responder e servir BAAI/bge-m3
for p in 8000 8001; do curl -sf http://127.0.0.1:$p/health && \
  curl -s http://127.0.0.1:$p/v1/models | head -c 120; echo; done

curl -sf http://127.0.0.1:8012/docs > /dev/null && echo "MinerU OK"
curl -sf http://127.0.0.1:9000/minio/health/live && echo "MinIO OK"
docker exec evidencia_redis redis-cli ping
curl -s "$(grep ^QDRANT_URL .env | cut -d= -f2)/collections" | head -c 120
```

Teste funcional do embedding, que é o que a indexação e a busca realmente usam:

```bash
uv run python -c "
from backend.services.embedder import BgeM3EmbedderService
e = BgeM3EmbedderService()
d, s = e.embed_query('teste de embedding')
print('denso:', len(d), 'dims | esparso:', len(s), 'tokens')"
```

Esperado: `denso: 1024 dims | esparso: N tokens` com **N > 0**. Se o esparso vier
vazio, veja *Problemas conhecidos*.

---

## 7. API e workers (systemd)

```bash
cd deploy/systemd
sudo ./install.sh          # copia as units, daemon-reload, enable e start
systemctl --no-pager --plain list-units 'evidencia*'
```

Sobem quatro serviços sob o `evidencia.target`: `evidencia-compose` (oneshot que
garante a infra), `evidencia-api`, `evidencia-worker-light` e `evidencia-worker-gpu`.

```bash
curl -sf http://127.0.0.1:8181/health          # ajuste à sua PORT
curl -s  http://127.0.0.1:8181/api/search/status
```

`{"semantic": true, ...}` confirma que a API alcançou o Qdrant **e** a API de embedding.

> Na subida, a API e o worker podem logar `API de embedding indisponível` se os vLLM
> ainda estiverem compilando. É esperado e se resolve sozinho: a sonda é informativa e
> a reconexão é preguiçosa, sem precisar reiniciar nada.

Detalhes de operação das units: [`deploy/systemd/README.md`](deploy/systemd/README.md).

---

## 8. Teste fim a fim

```bash
curl -X POST http://127.0.0.1:8181/api/files/dspace/item/<ITEM_UUID>
```

Responde `202` com um job por PDF. Acompanhe:

```bash
curl -s http://127.0.0.1:8181/api/files/status/<JOB_ID>
journalctl -u evidencia-worker-gpu -f
```

Estados: `na_fila` → `processando` (download/mineru/llm/index) → `concluido` | `erro`.
Com o job concluído, a busca precisa devolver resultado:

```bash
curl -s "http://127.0.0.1:8181/api/search/semantic?q=avaliação&limit=3"
```

O parâmetro é **`q`**, não `query` — com o nome errado a API devolve `[]` sem erro.

Fila de falhas: `GET /api/files/failures`; reprocessar: `POST /api/files/reprocess/{job_id}`.

---

## 9. Operação

```bash
systemctl status evidencia.target       # visão geral
systemctl restart evidencia.target      # reinicia infra + API + workers
journalctl -u evidencia-api -f
docker compose logs -f vllm-bge-m3
```

⚠️ `restart evidencia.target` executa `docker compose stop`, que **derruba toda a
infra** (Redis, MinIO, MinerU). Para recarregar só as units sem downtime da infra,
use `systemctl daemon-reload` e reinicie os serviços do host individualmente.

**Atualizar o código:**

```bash
git pull && uv sync
systemctl restart evidencia-api evidencia-worker-light evidencia-worker-gpu
```

**Atualizar uma imagem** (ex.: nova versão do vLLM ou do MinerU):

```bash
docker compose build vllm-bge-m3
docker compose up -d vllm-bge-m3 vllm-bge-m3-sparse
```

---

## Problemas conhecidos

**`BgeM3EmbeddingModel has no vLLM implementation`.** A imagem base do vLLM é antiga.
Essa arquitetura só existe a partir da **0.26**; a tag `latest` pode estar meses atrás.
A versão está fixada no `ARG VLLM_VERSION` de `deploy/vllm-bge-m3/Dockerfile`.

**Busca híbrida sem a perna lexical / esparso vazio.** Quase sempre é o
`--hf-overrides` não tendo chegado íntegro ao vLLM — sem ele o bge-m3 sobe como XLM-R
puro, **sem erro nenhum**. O YAML com bloco `>` tokeniza como shell e come as aspas do
JSON, por isso os argumentos JSON no compose vão entre aspas simples. Confira o valor
que chega de verdade:

```bash
docker compose config | grep -A2 hf-overrides
docker logs vllm-bge-m3 | grep -i "non-default args"
```

**`/v1/embeddings` respondendo 501 "model does not support Embeddings API".** Falta o
`--pooler-config.task` explícito: sem ele o vLLM cai na task combinada
`embed&token_classify`, que não tem handler HTTP. Cada instância fixa **uma** task —
é justamente por isso que existem dois contêineres.

**MinerU aborta por VRAM.** `MINERU_BACKEND=hybrid-engine` exige ~24 GB livres. Ou use
`pipeline`, ou reduza o `EMBED_GPU_UTIL`, ou não rode `mineru` e `mineru-pipeline` juntos.

**Indexação ou busca falhando com `EmbeddingBackendError`.** É o comportamento
projetado: **não há fallback local**. Sem os contêineres vLLM não há embedding —
o erro traz a URL que falhou.

---

## Referências

- [`README.md`](README.md) — visão geral, estágios do pipeline e chunking
- [`CELERY.md`](CELERY.md) — filas, workers e coordenação de GPU
- [`deploy/systemd/README.md`](deploy/systemd/README.md) — units e pressupostos do host
- [`backend/services/embedder.py`](backend/services/embedder.py) — por que são dois contêineres vLLM
