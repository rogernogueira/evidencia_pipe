# evidencia_pipe

Fork independente do pipeline de ingestão DSpace do `minerU`. Reproduz o endpoint
de **ingestão por item-UUID** como endpoint principal, rodando a cadeia **obrigatória**
de 3 estágios por PDF, em background (FastAPI `BackgroundTasks`):

1. **DSpace** — resolve os bitstreams PDF do bundle `ORIGINAL` do item (REST API) e baixa cada PDF.
2. **MinerU** — extração → markdown + `<doc>_content_list_v2.json`.
3. **Qdrant** — chunking estrutural por tokens + embeddings bge-m3 (dense/sparse) + upsert (grava `item_uuid`/`item_handle` no payload de cada chunk).

O **enriquecimento de metadados por LLM é DESACOPLADO** e **opcional** (provedor
OpenAI-compatible, configurável via `LLM_ENRICH_*`): não faz parte da chain
obrigatória. Roda como follow-up **após** a indexação (ou sob demanda em
`POST /api/files/enrich/{job_id}`) e propaga os metadados ao Qdrant via `set_payload`,
sem re-embedar → `metadata_candidates.json`. Sem provedor configurado, é pulado e a
indexação segue normalmente — a indexação nunca depende do LLM.

É um fork **self-contained**: os serviços foram copiados do `minerU`, não há
import em runtime da árvore original.

## Armazenamento de artefatos (MinIO) — pipeline v2

Todo artefato do pipeline (PDF fonte, markdown, `content_list_v2.json`, imagens,
`chunks.jsonl`, relatórios) é persistido no **MinIO** (S3-compatível), que é a
**fonte oficial** compartilhada entre API, worker leve, worker GPU e scripts. A
**Celery chain transporta apenas um `PipelineContext` leve** (identificadores + a
URI do manifesto, ≤ `CELERY_CHAIN_MAX_PAYLOAD_BYTES`); nenhum conteúdo grande passa
pelo broker Redis. Cada PDF tem um **manifesto próprio** que descreve seus artefatos:

```
minio://evidencia-pipe/artifacts/<pipeline_id>/<document_id>/
    manifest.json
    source/original.pdf
    mineru/{document.md, content_list_v2.json, processing_metrics.json, images/, images_manifest.json}
    enrichment/{metadata_candidates.json, raw_response.json}
    indexing/{chunks.jsonl, chunking_report.json, embedding_report.json}
```

## Chunking estrutural por tokens (estágio 3)

O chunking usa o **`StructuralTokenChunker`** (padrão, `CHUNKING_STRATEGY=structural_tokens`):
o limite principal é medido em **tokens do BGE-M3** (não em caracteres). O
`content_list_v2.json` do MinerU é a fonte prioritária; o markdown é fallback
(`structure_source=mineru_json|markdown|plain_text`). O documento é normalizado em
uma sequência de `DocumentBlock` (título/heading, parágrafo, lista, tabela, legenda,
fórmula, nota, referência) preservando ordem, página e hierarquia de seções
(`section_path`).

Formação: **seção → bloco → parágrafo → sentença → tokens** (corte direto por token
só como último recurso). Tabelas e listas são unidades estruturais (tabelas grandes
dividem por linhas repetindo o cabeçalho; listas por itens). O **overlap é
estrutural** (parágrafo/sentenças completos, não caracteres) e nunca cruza seções,
tabelas ou referências. Cada chunk recebe `chunk_id` **determinístico**
(SHA-256 de documento + checksum + versão + hash da config + índice + hash do texto),
metadados de rastreabilidade (páginas, `section_path`, `block_ids`) e um
`contextualized_text` (`Documento:`/`Seção:` + texto) — controlado por
`EMBEDDING_TEXT_FIELD`. Os chunks são gravados em `chunks.jsonl` **antes** do
embedding e um `chunking_report.json` resume as métricas.

Parsing, tokenização, chunking, filtros e persistência rodam em **CPU, fora do lock
da GPU** (só o embedding adquire o lock). Nenhum chunk trafega pela Celery chain.

- Estratégia legada por caracteres continua disponível: `CHUNKING_STRATEGY=legacy_chars`
  (`LegacyCharacterChunker`, envelopa o `MinerUChunker`) — para comparação A/B.
- Configuração completa em [`.env.example`](.env.example) (`CHUNK_*`,
  `CHUNK_TOKENIZER_MODEL`, `CHUNK_TARGET/MAX/MIN_TOKENS`, `CHUNK_OVERLAP_TOKENS`, …),
  validada em [`config.validate_chunking_config`](backend/core/config.py).

Comparar estratégias sobre um documento (sem embeddings):

```bash
uv run python scripts/compare_chunking_strategies.py \
    --content-list output/<doc>/hybrid_auto/<doc>_content_list_v2.json \
    --strategies legacy_chars structural_tokens
```

Avaliar recuperação (requer BGE-M3 + queries rotuladas):

```bash
uv run python scripts/evaluate_chunking_retrieval.py \
    --content-list output/<doc>/hybrid_auto/<doc>_content_list_v2.json \
    --queries queries.json --k 5
```

Implementação: [`MinIOArtifactStore`](backend/services/artifact_store.py) +
[`ManifestRepository`](backend/services/manifest_repository.py) (update sob lock
curto + `revision` + versionamento do bucket). A barreira anti-regressão
([`validate_chain_payload_size`](backend/core/schemas.py)) rejeita payloads grandes
ou com chaves proibidas (`markdown`, `chunks`, `embeddings`, `pdf_bytes`, …).

## Endpoints

| Método | Rota | Descrição |
|--------|------|-----------|
| `POST` | `/api/files/dspace/item/{uuid}` | **Principal** — ingere todos os PDFs de um item DSpace. `?force=true` reprocessa. |
| `POST` | `/api/files/dspace/{uuid}` | Ingere um bitstream (PDF) específico (download roda no worker). |
| `GET`  | `/api/files/status/{job_id}` | Status resumido do job (sem artefatos). |
| `GET`  | `/api/files/result/{job_id}` | **Resumo** do resultado (contagens + `artifact_id`) — não devolve o conteúdo. |
| `POST` | `/api/files/enrich/{job_id}` | Roda o enrich por LLM (desacoplado) sobre um job (lê o markdown do MinIO); se já indexado, propaga ao Qdrant. |
| `GET`  | `/api/files/active` | Lista os **jobs em execução** (`na_fila`/`processando`): IDs + resumo (estágio, arquivo, `updated_at`), mais recentes primeiro. |
| `GET`  | `/api/files/succeeded` | Lista os **últimos jobs bem sucedidos** (concluídos sem `index_error`): IDs + resumo (chunks, indexados, `artifact_id`), mais recentes primeiro. |
| `GET`  | `/api/files/failures` | Lista os jobs na **fila de falhas** (erro num estágio ou `index_error`), mais recentes primeiro. |
| `POST` | `/api/files/reprocess/{job_id}` | Re-enfileira a chain de um job que falhou (reusa a origem; `?force=true` reprocessa do zero). |
| `GET`  | `/internal/artifacts/{pipeline_id}/{document_id}/{artifact_name}/download-url` | URL pré-assinada curta (autorizada via `X-Internal-Token`). |
| `GET`  | `/internal/artifacts/health` | Health check do MinIO (bucket/leitura/escrita). |
| `GET`  | `/internal/gpu/status` | Status do recurso de GPU (dono, TTL, fila). Não expor publicamente. |
| `GET`  | `/internal/gpu/queue` | Fila de solicitações de GPU (prioridade efetiva). |

> O download de conteúdo é feito **apenas** via URL pré-assinada (curta, gerada sob
> demanda, nunca persistida no manifesto/logs). O bucket é **privado** (sem acesso
> anônimo).

## Como rodar

```bash
cp .env.example .env   # ajuste DSPACE_URL, QDRANT_URL, MINERU_API_URL, LLM_ENRICH_API_KEY, MINIO_*
uv sync
docker compose up -d redis minio minio-init   # Redis + MinIO (bucket privado + versionamento)
docker compose up -d mineru-pipeline          # MinerU pipeline-only (:8012, ~6,5 GB VRAM)
uv run python backend/main.py                  # sobe em http://127.0.0.1:8020
```

### MinerU: dois serviços (escolha por VRAM × qualidade)

O `docker-compose.yml` define **dois** serviços MinerU (imagem única `evidencia_mineru:local`, de `tmp/Dockerfile`, base vLLM). Ambos exigem **`--gpus all`** (com device único o vLLM falha com `Device string must not be empty`). Numa GPU única, rode **um ou outro** — juntos estouram a VRAM.

| Serviço | Porta | VLM | VRAM | Quando usar |
|---------|-------|-----|------|-------------|
| `mineru-pipeline` | 8012 | não (`--enable-vlm-preload false`) | ~6,5 GB | **Padrão** — PDFs digitais/relatórios; mesma qualidade de tabela/texto, ~25% mais rápido, e libera a GPU p/ o bge-m3 rodar em paralelo |
| `mineru` | 8011 | sim (vLLM residente) | ~36-44 GB | Layouts difíceis (manuscrito, tabelas irregulares, fórmulas, scans ruins) |

O backend é escolhido por env, casado com o serviço: **pipeline-only** → `MINERU_API_URL=http://127.0.0.1:8012` + `MINERU_BACKEND=pipeline` (default do `.env.example`); **VLM** → `:8011` + `MINERU_BACKEND=hybrid-auto-engine`. O cliente MinerU no venv precisa da MESMA versão do servidor (protocolo) — hoje `mineru[all]~=3.4.4`.

> Workers/scripts no **host** usam `MINIO_ENDPOINT=127.0.0.1:9000`; dentro do compose
> o endpoint é `minio:9000`. Em produção: credenciais próprias (não `minioadmin`),
> `MINIO_SECURE=true` (TLS) e um usuário do pipeline com política de menor privilégio.
> Migração opcional dos artefatos locais existentes:
> `uv run python scripts/migrate_artifacts_to_minio.py --dry-run`.

## Exemplo

```bash
curl -X POST http://127.0.0.1:8020/api/files/dspace/item/<ITEM_UUID>
```

Resposta `202` com a lista de jobs criados (um por PDF). Acompanhe cada um em
`/api/files/status/{job_id}` e recupere o markdown em `/api/files/result/{job_id}`.

## Requisitos externos

- **MinIO** (docker-compose) acessível em `MINIO_ENDPOINT` — armazenamento oficial dos artefatos.
- **MinerU API** acessível em `MINERU_API_URL` (usa GPU).
- **Qdrant** acessível em `QDRANT_URL`.
- Modelo **BAAI/bge-m3** no cache HuggingFace (senão é baixado no 1º job).
- `LLM_ENRICH_API_KEY` (ou o legado `DEEPSEEK_API_KEY`) para o enrich por LLM — **opcional e desacoplado**; sem ela o enrich é pulado e a indexação segue normalmente.
- **Redis** (docker-compose): DB 0 = broker Celery, DB 1 = job_store (+ lock do
  manifesto), DB 2 = coordenação da GPU (`gpu_resource_manager`).

## Coordenação da GPU compartilhada

A **mesma GPU física** é usada pelo subprocesso do MinerU (estágio 2) e pelo BGE-M3
(estágio 3), além de eventuais scripts externos. Sem coordenação, execuções
simultâneas causam CUDA OOM, queda de workers e jobs inconsistentes. O
`concurrency=1` da fila `gpu` do Celery só controla **aquele** worker — não o
subprocesso do MinerU nem scripts externos.

A biblioteca independente [`gpu_resource_manager`](packages/gpu_resource_manager/)
(Redis DB 2) arbitra o recurso: **exclusão mútua distribuída**, **fila global com
prioridade + FIFO + aging**, **lease com TTL/heartbeat** e recuperação por TTL após
`kill -9`. É **não preemptiva** (não mata CUDA em execução) e **fail-closed** (sem
Redis, ninguém usa a GPU).

- **MinerU** adquire o recurso (prioridade `MINERU_GPU_PRIORITY=20`) só ao redor do
  subprocesso; download/DSpace/CSV ficam fora do lock.
- **BGE-M3** adquire (prioridade `BGE_GPU_PRIORITY=30`) só ao redor da inferência;
  chunking, filtros e upsert no Qdrant ficam fora do lock. O
  [`BGEModelManager`](backend/services/bge_model_manager.py) descarrega a VRAM após
  a task (`BGE_GPU_LIFECYCLE=lazy`, `BGE_UNLOAD_AFTER_TASK=true`) — o lock Redis não
  libera VRAM sozinho.
- **Scripts externos** usam a mesma lib/Redis/recurso (ver
  [exemplo](packages/gpu_resource_manager/examples/external_priority_script.py) e o
  README da biblioteca). CLI: `gpu-manager status|queue|health|cancel`.

Ligue/desligue com `GPU_MANAGER_ENABLED` (ver `.env.example`).

## Testes

```bash
uv run --group test pytest packages/gpu_resource_manager/tests/unit -q   # lib (fakeredis)
uv run --group test pytest tests -q                                      # store/manifesto/context/etapas (MinIO fake)
# integração com MinIO real (docker compose up -d minio minio-init):
uv run --group test pytest tests/integration -q
# processos reais / Redis real (docker-compose up -d):
GPU_MANAGER_REDIS_URL=redis://127.0.0.1:6379/2 \
  uv run --group test pytest packages/gpu_resource_manager/tests/integration -q
```
