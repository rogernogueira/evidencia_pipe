# Orquestração com Celery + Redis

O pipeline de ingestão roda como uma **chain** Celery **obrigatória** por PDF:

```
baixar_dspace → extrair_mineru → indexar_qdrant
   (download)      (extract)         (gpu)
```

O **enriquecimento por LLM é DESACOPLADO** — não faz parte da chain obrigatória.
Quando há provedor configurado e `LLM_ENRICH_AUTO=true`, um follow-up opcional
(`enrich_after_index`, fila `llm`) é anexado via `link` **após** a indexação: ele
gera `metadata_candidates.json` e propaga os metadados ao Qdrant (`set_payload`, sem
re-embedar). Assim a indexação **nunca espera nem depende do LLM**. O enrich também
pode ser disparado sob demanda em `POST /api/files/enrich/{job_id}`.

Cada estágio tem uma fila própria. A API só **enfileira** (responde `202` na hora);
os **workers** executam. O status por-documento vive no `job_store` (Redis DB 1) e é
consultado em `GET /api/files/status/{job_id}`. O **Flower** monitora a infra.

## Transporte da chain: só referências, nunca conteúdo

A chain transporta **apenas um `PipelineContext` leve** (identificadores + a URI do
manifesto). Os artefatos (PDF, markdown, JSON MinerU, chunks, imagens, relatórios)
vivem no **MinIO** e são descobertos pelo `manifest.json` de cada documento. Cada
task: abre o manifesto → baixa só o necessário para um tempdir → processa → grava os
novos artefatos no MinIO → atualiza o manifesto (sob lock curto no Redis DB 1) →
**valida o payload de saída** e retorna o contexto.

```
baixar_dspace     → {pipeline_id, job_id, item_uuid, document_id, artifact_manifest_uri, current_stage:"downloaded", ...}
extrair_mineru    → idem, current_stage:"extracted"
indexar_qdrant    → {pipeline_id, document_id, status, chunk_count, indexed_count, artifact_manifest_uri, qdrant_collection}
enrich_after_index → (follow-up opcional) reconstrói o contexto do manifesto, enriquece e propaga ao Qdrant; retorna o summary intacto
```

Antes de retornar, cada task chama `validate_chain_payload_size` (limite
`CELERY_CHAIN_MAX_PAYLOAD_BYTES=16384`, `CELERY_CHAIN_ENFORCE_LIGHTWEIGHT_CONTEXT=true`):
rejeita payloads acima do limite (`PipelinePayloadTooLargeError`), **chaves proibidas**
(`markdown`, `chunks`, `embeddings`, `mineru_json`, `pdf_bytes`, `raw_response`, …) e
valores binários/não-serializáveis. Serialização Celery: **só JSON, sem pickle**.

**Retries** reabrem o mesmo `PipelineContext` e reutilizam artefatos já válidos no
MinIO (SHA-256 + estágio `COMPLETED` no manifesto); `?force=true` reprocessa. Ver
[`pipeline_stages.py`](backend/services/pipeline_stages.py),
[`artifact_store.py`](backend/services/artifact_store.py) e
[`manifest_repository.py`](backend/services/manifest_repository.py).

## 1. Subir a infra (Redis + Flower + MinIO)

```bash
docker compose up -d          # redis :6379, flower :5555, minio :9000/:9001 (+ bucket privado)
```

Flower: http://localhost:5555 · Console MinIO (admin): http://localhost:9001

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

## Coordenação da GPU (gpu_resource_manager)

`concurrency=1` na fila `gpu` controla **apenas** as tasks daquele worker Celery.
Ele **não** controla o subprocesso do MinerU (outro processo) nem scripts externos
que também usam CUDA. Quem coordena **todos** os processos é o
[`gpu_resource_manager`](packages/gpu_resource_manager/) via lock distribuído no
**Redis DB 2** (`GPU_MANAGER_REDIS_URL`). E a ocupação da VRAM (mesmo sem inferência)
é controlada pela política de ciclo de vida do modelo (`BGE_GPU_LIFECYCLE`), não pelo
lock — o lock Redis não libera VRAM sozinho.

Camadas de controle:

| Camada | O que controla |
|--------|----------------|
| `concurrency=1` (fila `gpu`) | Uma task de GPU por vez **naquele** worker |
| `worker_prefetch_multiplier=1` | Não "açambarca" tasks longas (já configurado) |
| `gpu_resource_manager` (Redis DB 2) | Exclusão entre **todos** os processos (MinerU, BGE-M3, scripts externos) |
| `BGE_GPU_LIFECYCLE` + `BGEModelManager` | Ocupação real da VRAM (descarga após a task) |

Preload na VRAM: com a GPU compartilhada (`GPU_SHARED_WITH_MINERU=true`, padrão) o
worker **não** pré-carrega o bge-m3 na VRAM — o modelo é carregado sob demanda e
movido para a GPU só sob o lock, sendo descarregado após a task. O preload direto na
VRAM só ocorre com `BGE_GPU_LIFECYCLE=preload` **e** GPU não compartilhada.

Chunking fora do lock (estágio `indexar_qdrant`): o parsing do `content_list_v2.json`,
a **tokenização** (só o tokenizer do BGE-M3 em CPU — nunca o modelo completo na GPU),
o chunking estrutural (`StructuralTokenChunker`), os filtros de qualidade e a escrita
de `chunks.jsonl`/`chunking_report.json` acontecem **antes** de adquirir o lock. O
lock da GPU é adquirido **apenas** ao redor do embedding (dense+sparse); o build do
payload e o upsert no Qdrant ocorrem depois, já com os vetores na CPU. Além disso, os
chunks antigos de um documento só são removidos **após** o upsert dos novos ser
confirmado (uma falha no novo chunking não apaga os chunks ativos anteriores).

Observabilidade: `GET /internal/gpu/status`, `GET /internal/gpu/queue`, ou
`gpu-manager status|queue|health|cancel`.

## Configuração (env)

| Var | Default | Papel |
|-----|---------|-------|
| `REDIS_URL` | `redis://127.0.0.1:6379` | base do broker e do job_store |
| `CELERY_BROKER_URL` | `${REDIS_URL}/0` | broker |
| `JOBSTORE_REDIS_URL` | `${REDIS_URL}/1` | status dos jobs (+ lock do manifesto) |
| `JOBSTORE_TTL` | `604800` (7d) | expiração dos registros de job |
| `MINIO_ENDPOINT` | `127.0.0.1:9000` | store de artefatos (host); no compose use `minio:9000` |
| `MINIO_BUCKET` | `evidencia-pipe` | bucket privado dos artefatos |
| `CELERY_CHAIN_MAX_PAYLOAD_BYTES` | `16384` | teto do payload da chain |

## Semântica de erro

- **download / MinerU** falham → job vira `erro`, a chain para (não indexa).
- **index** falha → job fica `concluido` com `index_error` (o markdown já é válido).
- **enrich (LLM)** — fora da chain: totalmente best-effort. Roda **após** a indexação
  e **nunca** altera o status já `concluido` do job; falhas são só logadas.
- Estado inicial `na_fila` = enfileirado, ainda não pego por um worker.

Erros de download **não** são mais síncronos no endpoint de item (responde `202`);
acompanhe em `/status` ou no Flower. O endpoint de bitstream avulso mantém o `502`
síncrono (baixa antes de enfileirar).

## Robustez: retries, timeouts e fila de falhas

**Retries.** Só o **download** faz retry no nível da task (`max_retries=3`,
`default_retry_delay=15s`, apenas para `HTTPError`/`URLError`). `mineru`/`index` **não**
retentam. No nível do broker, `task_acks_late=True` + `worker_prefetch_multiplier=1` fazem
o Redis **reentregar** a task se o worker morrer antes do ack.

**Timeouts.**

| Onde | Var | Default |
|---|---|---|
| Task Celery (soft → `SoftTimeLimitExceeded`) | `CELERY_TASK_SOFT_TIME_LIMIT` | `3000` (50 min) |
| Task Celery (hard → mata o worker) | `CELERY_TASK_TIME_LIMIT` | `3300` (55 min) |
| Subprocesso MinerU (`communicate(timeout=…)`) | `MINERU_TIMEOUT_SECONDS` | `2400` (40 min) |
| Cliente LLM (OpenAI-compat) | `LLM_ENRICH_TIMEOUT_SECONDS` | `120` |
| Cliente Qdrant | `QDRANT_TIMEOUT_SECONDS` | `60` |

Ordem intencional: `MINERU_TIMEOUT (40m) < soft (50m) < hard (55m) < visibility_timeout
(60m)`. Assim o MinerU travado estoura primeiro e falha limpo (encerra o grupo de
processos, libera a VRAM); o soft limit levanta uma exceção tratável (dispara `on_failure`
e registra a falha) antes do hard limit; e o hard limit (< `visibility_timeout`) evita que
o broker reentregue a task a outro worker enquanto a original ainda roda (duplicação).

**Fila de falhas (não é um DLQ de broker).** Jobs que falham num estágio (ou concluem com
`index_error`) entram num índice leve no Redis (`jobs:failed`, sorted set). Consulta e
reprocessamento:

- `GET /api/files/failures?limit=100` — lista os jobs na fila (mais recentes primeiro).
- `POST /api/files/reprocess/{job_id}?force=true` — re-enfileira a chain reusando a origem
  (bitstream/item) do `job_store`. `force=false` reaproveita etapas concluídas
  (idempotência por manifesto/SHA — ex.: reindexar sem re-extrair).

Re-enfileirar um job (ingestão ou reprocess) o **remove** da fila; indexar com sucesso
também. Entradas cujo registro de job expirou (TTL) são **podadas** ao listar.
