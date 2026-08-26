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

Avaliar recuperação (requer a API de embedding no ar + queries rotuladas):

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
| `POST` | `/api/files/dspace/item/{uuid}` | **Principal** — ingere todos os PDFs de um item DSpace. `?force=true` reprocessa. Item ainda indisponível no DSpace **não** dá 502: a ingestão fica na fila e o item é reconsultado com backoff (ver [CELERY.md](CELERY.md)). |
| `POST` | `/api/files/dspace/{uuid}` | Ingere um bitstream (PDF) específico (download roda no worker). |
| `GET`  | `/api/files/status/{job_id}` 🔒 | Status resumido do job (sem artefatos). |
| `GET`  | `/api/files/result/{job_id}` 🔒 | **Resumo** do resultado (contagens + `artifact_id`) — não devolve o conteúdo. |
| `POST` | `/api/files/enrich/{job_id}` | Roda o enrich por LLM (desacoplado) sobre um job (lê o markdown do MinIO); se já indexado, propaga ao Qdrant. |
| `GET`  | `/api/files/active` 🔒 | Lista os **jobs em execução** (`na_fila`/`processando`): IDs + resumo (estágio, arquivo, `updated_at`), mais recentes primeiro. |
| `GET`  | `/api/files/succeeded` 🔒 | Lista os **últimos jobs bem sucedidos** (concluídos sem `index_error`): IDs + resumo (chunks, indexados, `artifact_id`), mais recentes primeiro. |
| `GET`  | `/api/files/failures` 🔒 | Lista os jobs na **fila de falhas** (erro num estágio ou `index_error`), mais recentes primeiro. |
| `POST` | `/api/files/reprocess/{job_id}` 🔒 | Re-enfileira a chain de um job que falhou (reusa a origem; `?force=true` reprocessa do zero). Em `item:{uuid}`, reinicia a espera pelo item no DSpace. |
| `GET`  | `/api/status` 🔒 | **Estado de toda a infraestrutura** + o que o sistema consegue fazer agora. Ver abaixo. |
| `GET`  | `/api/status/{component}` 🔒 | Um componente só (`qdrant`, `minio`, `celery`, …), com o mesmo cache. |
| `GET`  | `/internal/artifacts/{pipeline_id}/{document_id}/{artifact_name}/download-url` | URL pré-assinada curta (autorizada via `X-Internal-Token`). |
| `GET`  | `/internal/artifacts/health` | Health check do MinIO (bucket/leitura/escrita). |
| `GET`  | `/internal/gpu/status` | Status do recurso de GPU (dono, TTL, fila). Não expor publicamente. |
| `GET`  | `/internal/gpu/queue` | Fila de solicitações de GPU (prioridade efetiva). |

> 🔒 = exige `Authorization: Bearer <token do DSpace>` de um **administrador do
> repositório**. Quem autoriza é o próprio DSpace: a API valida o token em
> `/api/authn/status` e pergunta em `/api/authz/authorizations/search/object`
> (`feature=administratorOf`) se o dono é admin — é o mesmo token e a mesma checagem
> que o dspace-angular já usa, então não há login separado nem chave compartilhada.
> Sem token → **401**; token válido de quem não é admin → **403**; DSpace fora do ar
> ou lento na validação → **503** (nunca 401: isso mandaria o operador relogar à toa).
> Ver [backend/api/auth.py](backend/api/auth.py) e as variáveis `ADMIN_AUTH_*` no
> [.env.example](.env.example). O DSpace que valida os tokens não precisa ser o mesmo
> de onde vêm os PDFs — `DSPACE_SERVER_URL` é independente de `DSPACE_URL`.
> As rotas de **enfileiramento** (e o `enrich`) seguem
> abertas — quem as chama é a sincronização automática, que roda no host e não tem
> sessão DSpace. O destino delas não é o Bearer: é sair do acesso externo e ficar
> restritas à rede local (borda/firewall). Ver [DEPLOY.md §8.2](DEPLOY.md).

> O download de conteúdo é feito **apenas** via URL pré-assinada (curta, gerada sob
> demanda, nunca persistida no manifesto/logs). O bucket é **privado** (sem acesso
> anônimo).

Na instalação do IBICT a API é publicada por um proxy reverso na borda:
**`https://devrdapp.ibict.br/api/...`** (mesmos caminhos da tabela). Só as rotas
`/api/*` são publicadas — `/health`, `/docs`, `/output` e `/internal/*` seguem
restritos à rede interna. Onde a borda remover o prefixo `/api`, o backend recoloca
via `PROXY_STRIPPED_PREFIX` e barra a raiz administrativa. Detalhes, conferência e
o remendo: [DEPLOY.md §8.1](DEPLOY.md).

## Status da infraestrutura

`GET /api/status` responde numa consulta se o pipeline está de pé, com uma linha por
dependência — e o que ela mede de concreto:

| Componente | O que sai |
|------------|-----------|
| `qdrant` | collection configurada, **quantidade de chunks**, vetores `dense`/`sparse`, dimensão/distância, estado dos segmentos |
| `embedder` | os dois endpoints vLLM do bge-m3, **modelo servido** × `EMBED_API_MODEL`, `max_model_len`; com `?probe=true`, round-trip real (dimensão, norma, alinhamento do esparso) |
| `mineru` | `/health` do serviço, **backend/método/idioma em uso**, tarefas na fila/processando/concluídas/falhas |
| `minio` | bucket, leitura e escrita, **quantidade de artefatos** e bytes, documentos, execuções e divisão por etapa (`source`/`mineru`/`indexing`/`enrichment`) |
| `redis` | os três DBs (broker, job_store, lock da GPU), **backlog por fila** do Celery, clientes e memória |
| `celery` | **workers online**, concorrência somada, tasks ativas/reservadas, filas cobertas e **filas sem consumidor** |
| `gpu` | lock do `gpu_resource_manager` (dono e fila) e VRAM/utilização da placa |
| `llm_enrich` | **ligado ou não** (`enabled` = tem chave, `auto_after_index` = roda sozinho após indexar, `active` = os dois), provedor, **modelo em uso** e se o provedor realmente serve esse modelo |
| `llm_visual` | gate de LLM do chunking de gráficos/imagens (`CHUNK_VISUAL_LLM`) |
| `dspace` | `/server/api` respondendo JSON do REST (origem dos PDFs) |
| `flower`, `jobs`, `disk` | monitor (se `FLOWER_URL`), índices de job (em execução/concluídos/falhas) e espaço livre onde o pipeline escreve |

Além do estado por componente, a resposta traz `capabilities` — **o que o sistema
consegue fazer agora** (`busca`, `ingestao`, `extracao`, `indexacao`,
`enriquecimento_llm`), cada uma com os componentes que a bloqueiam — e `config`, a
configuração que define a semântica do índice (chunking, embedding, filtros de busca).

A rota **exige Bearer de administrador do DSpace**, inclusive de dentro do host: ela
descreve a infraestrutura toda. Como obter o `$TOKEN` (o login do DSpace exige token
CSRF) está em [DEPLOY.md §8.2](DEPLOY.md); sem token, o que responde é o `/health`.

```bash
A="Authorization: Bearer $TOKEN"
curl -s -H "$A" http://127.0.0.1:8020/api/status | jq '{status, capabilities, blocking}'
curl -s -H "$A" http://127.0.0.1:8020/api/status/celery          # só os workers
curl -s -H "$A" 'http://127.0.0.1:8020/api/status?probe=true'    # + round-trip de embedding
curl -s -H "$A" 'http://127.0.0.1:8020/api/status?artifacts=false'  # pula a contagem no MinIO
```

**Código HTTP**: 200 com o pipeline de pé (mesmo degradado), **503** quando um
componente crítico está fora — dá para usar direto em monitoramento. O corpo é o mesmo
JSON nos dois casos.

**Nível de detalhe**: métricas (contagens, flags, nome de modelo) vão para qualquer
consumidor; a topologia (`internal`: URLs, versões, hostnames, PIDs, caminhos) só para
quem chama pela rede interna — sem `X-Forwarded-For`, a mesma convenção que barra a
raiz administrativa — ou apresenta `X-Internal-Token` igual a `INTERNAL_API_TOKEN`.
`detail_level` na resposta diz qual nível veio. As opções caras (`fresh`, `probe`)
também são restritas ao nível completo: pela borda, a resposta sai do cache
(`STATUS_CACHE_TTL_SECONDS`), então consultar de segundo em segundo não vira carga.

Para diagnóstico **de bancada** — no servidor, sem venv, incluindo a conferência do
proxy da borda — o complemento é `python3 scripts/diagnostico.py --public-url ...`,
que mede as mesmas dependências com os mesmos critérios.

## Como rodar

Atalho para desenvolvimento. Para uma instalação do zero (build das imagens, Qdrant,
systemd, verificação e troubleshooting), siga o [`DEPLOY.md`](DEPLOY.md).

```bash
cp .env.example .env   # ajuste DSPACE_URL, QDRANT_URL, MINERU_API_URL, LLM_ENRICH_API_KEY, MINIO_*
uv sync
docker compose up -d                                 # infra leve: Redis + MinIO + Flower
docker compose up -d mineru-pipeline                 # MinerU pipeline-only (:8012, ~6,5 GB VRAM)
docker compose up -d vllm-bge-m3 vllm-bge-m3-sparse  # embedding (:8000 denso, :8001 esparso)
uv run python backend/main.py                        # sobe em http://127.0.0.1:8020
```

### MinerU: dois serviços (escolha por VRAM × qualidade)

> Os serviços que exigem GPU (`mineru`, `mineru-pipeline`, `vllm-bge-m3`,
> `vllm-bge-m3-sparse`) estão atrás do profile **`gpu`**: um `docker compose up -d` sem
> argumentos sobe só a infra leve. Nomeá-los explicitamente (como acima) ativa o profile
> sozinho; para subir tudo de uma vez, `docker compose --profile gpu up -d`.

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

Se o item **ainda não estiver disponível** no DSpace (em submissão/workflow, embargo,
DSpace fora do ar), a resposta continua `202` — agora com `status="aguardando_dspace"`
e `jobs: []`: a ingestão fica na fila e o item é reconsultado com backoff até os PDFs
aparecerem. Acompanhe a espera em `/api/files/status/item:<ITEM_UUID>`. Detalhes e
variáveis em [CELERY.md](CELERY.md).

## Requisitos externos

- **MinIO** (docker-compose) acessível em `MINIO_ENDPOINT` — armazenamento oficial dos artefatos.
- **MinerU API** acessível em `MINERU_API_URL` (usa GPU).
- **Qdrant** acessível em `QDRANT_URL`.
- **API de embedding** (docker-compose): `vllm-bge-m3` (denso, `EMBED_API_URL`) e
  `vllm-bge-m3-sparse` (esparso, `EMBED_API_SPARSE_URL`), ambos da imagem
  `evidencia_bge_m3:local` (`deploy/vllm-bge-m3/Dockerfile`), com os **pesos assados
  dentro** e `HF_HUB_OFFLINE=1` — como os serviços MinerU, a versão fica isolada na
  imagem e o runtime não depende do cache do host nem da rede. O backend não carrega
  mais o modelo — só o tokenizer do **BAAI/bge-m3** (cache HuggingFace), usado no
  chunking por tokens e no alinhamento dos `lexical_weights`. Sem a API não há
  indexação nem busca.
- `LLM_ENRICH_API_KEY` (ou o legado `DEEPSEEK_API_KEY`) para o enrich por LLM — **opcional e desacoplado**; sem ela o enrich é pulado e a indexação segue normalmente.
- **Redis** (docker-compose): DB 0 = broker Celery, DB 1 = job_store (+ lock do
  manifesto), DB 2 = coordenação da GPU (`gpu_resource_manager`).

## Coordenação da GPU compartilhada

A **mesma GPU física** é usada pelo subprocesso do MinerU (estágio 2), pelos
contêineres vLLM do bge-m3 (estágio 3) e por eventuais scripts externos. Sem coordenação, execuções
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
- **BGE-M3** ficou **fora** do lock: roda nos contêineres vLLM, cada um com um teto de
  VRAM reservado no startup (`--gpu-memory-utilization`), então não há disputa a
  arbitrar. O worker só faz HTTP (ver [`embedder.py`](backend/services/embedder.py)) —
  chunking, filtros e upsert no Qdrant seguem locais.
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
