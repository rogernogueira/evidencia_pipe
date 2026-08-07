# Deploy — `evidencia_pipe`

Passo a passo de uma instalação do zero até o pipeline processando um documento.

O sistema tem **três camadas**, e é isso que explica o formato do deploy:

| Camada | Onde roda | O quê |
|---|---|---|
| Infra | Docker (`docker-compose.yml`) | Redis, MinIO, Flower, MinerU, os dois vLLM do bge-m3 |
| Processos | Host, no venv (`systemd`) | API FastAPI + 2 workers Celery |
| Externo | Fora deste repositório | **Qdrant** e o **DSpace** |

Os processos do host leem o `.env` do diretório do projeto. Quem sobe tudo junto no
boot é o `evidencia.target`.

**Existem dois cenários de instalação.** Escolha antes de continuar:

| | Cenário A — host com GPU | Cenário B — host **sem** GPU |
|---|---|---|
| MinerU | contêiner local | remoto, via `MINERU_API_URL` |
| Embedding | 2 contêineres vLLM locais | remoto, via `EMBED_API_*` |
| Serviços locais | tudo | só Redis, MinIO e Flower |
| Mudança de código | nenhuma | **nenhuma** |

O backend é 100% desacoplado por HTTP e o venv do host **não tem `torch` nem
`mineru`** — por isso o cenário B funciona sem adaptação. Os passos 1, 2, 7, 8 e 9
valem para os dois; os passos 3–6 são do cenário A. O **[passo 10](#10-cenário-b--servidor-sem-gpu)**
cobre o B.

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

## 10. Cenário B — servidor sem GPU

Aqui o servidor roda **só** a API, os workers e a infra leve; a extração e o embedding
ficam num host com GPU. Nada de código muda — é tudo `.env` e quais serviços subir.

Exemplo com o host de GPU em `172.16.115.60`, MinerU na `8004` e embedding na `8003`:

```dotenv
MINERU_API_URL=http://172.16.115.60:8004
MINERU_BACKEND=pipeline            # tem de casar com o que roda LÁ

EMBED_API_URL=http://172.16.115.60:8003          # instância da task `embed`
EMBED_API_SPARSE_URL=http://172.16.115.60:????   # instância da task `token_classify`

# Sem GPU local não há o que arbitrar: o lock envolveria uma chamada HTTP a uma
# GPU que não é deste host, serializando o pipeline à toa.
GPU_MANAGER_ENABLED=false
```

> ⚠️ **Confirme quantas instâncias de embedding existem no host remoto.** O backend
> precisa de **duas**: uma servindo a task `embed` (vetor denso) e outra servindo
> `token_classify` (vetor esparso). Um único vLLM atende **uma** task só — se a `8003`
> for a única, metade do contrato não existe e a busca híbrida fica sem a perna
> lexical. O motivo está em [`backend/services/embedder.py`](backend/services/embedder.py).

Rode **no host da GPU** para descobrir qual task a `8003` serve. Não use `/v1/models`
para isso — ele responde `200` nas duas, só lista o modelo. Quem discrimina é o par
abaixo (códigos verificados nas duas instâncias):

```bash
P=8003
curl -s -o /dev/null -w "v1/embeddings:%{http_code} " -X POST http://127.0.0.1:$P/v1/embeddings \
  -H 'Content-Type: application/json' -d '{"model":"BAAI/bge-m3","input":["teste"]}'
curl -s -o /dev/null -w "pooling:%{http_code}\n" -X POST http://127.0.0.1:$P/pooling \
  -H 'Content-Type: application/json' \
  -d '{"model":"BAAI/bge-m3","input":["teste"],"task":"token_classify"}'
```

| Resposta | Instância | O que falta |
|---|---|---|
| `v1/embeddings:200  pooling:400` | **densa** (`embed`) | subir a esparsa com `--pooler-config '{"task":"token_classify"}'` noutra porta |
| `v1/embeddings:501  pooling:200` | **esparsa** (`token_classify`) | subir a densa com `--pooler-config '{"task":"embed"}'` |
| `v1/embeddings:501  pooling:500` | nenhuma das duas | o servidor caiu na task combinada; falta o `--pooler-config.task` explícito |

Se o `/pooling` responder `200` mas o vetor esparso vier vazio na prática, o
`--hf-overrides` não chegou íntegro — veja *Problemas conhecidos*.

Se faltar uma instância, o [passo 4](#4-build-das-imagens) e o
[`docker-compose.yml`](docker-compose.yml) deste repositório servem de referência —
suba a que falta **no host da GPU**, numa porta livre, e aponte o `.env`.

**Serviços a subir no host sem GPU** — só a infra leve:

```bash
docker compose up -d redis flower minio minio-init
```

Nada de `mineru*` nem `vllm*`: exigem GPU e falhariam na subida.

**No systemd**, encurte a lista de serviços sem editar o unit versionado:

```bash
sudo systemctl edit evidencia-compose
```

```ini
[Service]
Environment="COMPOSE_SERVICES=redis flower minio minio-init"
```

```bash
sudo systemctl daemon-reload && sudo systemctl restart evidencia-compose
```

**Rede.** O host sem GPU precisa alcançar, além do Qdrant e do DSpace, as portas de
MinerU e embedding no host da GPU. Confirme antes de instalar — de dentro do servidor:

```bash
for hp in 172.16.115.60:8004 172.16.115.60:8003; do
  timeout 5 bash -c "cat < /dev/null > /dev/tcp/${hp/:/\/}" \
    && echo "$hp alcançável" || echo "$hp INALCANÇÁVEL"
done
```

Do outro lado, os contêineres vLLM sobem com `network_mode: host`, então escutam em
`0.0.0.0` — exponha as portas só para a rede interna.

**Tudo em CPU, sem host de GPU nenhum?** É outra conversa, não coberta aqui. O MinerU
até roda (`MINERU_DEVICE_MODE=cpu` + `MINERU_BACKEND=pipeline`, bem mais lento), mas o
embedding não tem saída pronta: **não existe imagem CPU oficial do `vllm/vllm-openai`**
— seria preciso construir o vLLM do fonte (`docker/Dockerfile.cpu`) ou reintroduzir o
caminho local com FlagEmbedding em CPU, removido na migração para a API.

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
