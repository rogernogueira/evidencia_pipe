# Esboço: stack tudo-em-Docker, com GPU local

Variante **self-contained com GPU** do `evidencia_pipe`: um `docker compose up` sobe
infra + inferência de GPU + Qdrant **e** a aplicação (API + workers). É o modelo
híbrido de hoje levado ao fim — a app sai do venv/systemd do host e vira contêiner.

> **Status: ESBOÇO.** Os serviços de GPU são o `docker-compose.yml` da raiz
> adaptados para a rede do compose (sem `network_mode: host`), para os contêineres
> da app os acharem por nome. Os flags delicados do vLLM (`--hf-overrides`,
> `--pooler-config`) e o `ipc: host` foram preservados.

## Pré-requisitos do host

GPU NVIDIA + `nvidia-container-toolkit` + **CDI** gerado:

```bash
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
grep -c 'name: all' /var/run/cdi/nvidia.yaml   # >= 1
```

As 3 inferências usam `devices: nvidia.com/gpu=all` (CDI, não `deploy.resources` —
ver o porquê nos comentários do compose da raiz). A app e os workers **não** usam GPU.

## O que sobe

| Serviço | Papel | GPU |
|---|---|---|
| `redis` `flower` `minio`(+init) `qdrant` | infra + índice vetorial | — |
| `mineru-pipeline` | extração (estágio 2), backend pipeline | ✅ |
| `vllm-bge-m3` / `-sparse` | embedding denso / esparso | ✅ |
| `api` | FastAPI (:8020) | — |
| `worker-light` | filas `download,extract,llm` (`-c 4`) | — |
| `worker-index` | fila `gpu` = embedding + Qdrant (`-c 1`) | — |

## Como rodar

```bash
cd deploy/all-in-docker-gpu
docker compose -f docker-compose.gpu.yml build      # reusa evidencia_mineru:local / evidencia_bge_m3:local se já existirem
docker compose -f docker-compose.gpu.yml up -d
docker compose -f docker-compose.gpu.yml ps
```

Precisa do `.env` na raiz. O `env.gpu` é carregado depois e só ajusta endereços
(nomes de serviço) + `GPU_MANAGER_*`.

## Orçamento de VRAM (GPU única)

bge-m3 reserva `2 × EMBED_GPU_UTIL × VRAM` no startup (`0,12` → ~11,5 GB numa de 48 GB).
Mais o MinerU:

| MinerU | VRAM | Serviço / `MINERU_BACKEND` |
|---|---|---|
| **pipeline** (default) | ~6,5 GB | `mineru-pipeline` / `pipeline` |
| VLM / hybrid | precisa ~24 GB livres | criar serviço `mineru` / `hybrid-engine` |

Numa GPU única rode **um** MinerU. O `GPU_MANAGER_ENABLED=true` serializa os estágios
de GPU dos workers (extract × index) para não estourar a VRAM — é por isso que ele
fica ligado aqui (ao contrário da variante `remote`).

## Diferença para as outras variantes

| | `cpu` | `gpu` (aqui) | `remote` |
|---|---|---|---|
| MinerU/BGE | CPU local | **GPU local** | outra máquina |
| GPU manager | irrelevante | **on** | off |
| Hardware | qualquer | GPU NVIDIA | sem GPU local |

Dados já processados entram por restore (`../all-in-docker-cpu/host-prep/backup-restore.md`),
não por reprocessamento.
