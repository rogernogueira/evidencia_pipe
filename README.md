# evidencia_pipe

Fork independente do pipeline de ingestão DSpace do `minerU`. Reproduz o endpoint
de **ingestão por item-UUID** como endpoint principal, rodando a cadeia completa
de 4 estágios por PDF, em background (FastAPI `BackgroundTasks`):

1. **DSpace** — resolve os bitstreams PDF do bundle `ORIGINAL` do item (REST API) e baixa cada PDF.
2. **MinerU** — extração → markdown + `<doc>_content_list_v2.json`.
3. **Qdrant** — chunking + embeddings bge-m3 (dense/sparse) + upsert (grava `item_uuid`/`item_handle` no payload de cada chunk).
4. **LLM (DeepSeek)** — enriquecimento de metadados → `<doc>_metadata_llm.json`.

É um fork **self-contained**: os serviços foram copiados do `minerU`, não há
import em runtime da árvore original.

## Endpoints

| Método | Rota | Descrição |
|--------|------|-----------|
| `POST` | `/api/files/dspace/item/{uuid}` | **Principal** — ingere todos os PDFs de um item DSpace. |
| `POST` | `/api/files/dspace/{uuid}` | Ingere um bitstream (PDF) específico. |
| `GET`  | `/api/files/status/{job_id}` | Status do job. |
| `GET`  | `/api/files/result/{job_id}` | Markdown extraído (text/markdown). |
| `POST` | `/api/files/enrich/{job_id}` | Roda só o estágio 4 (LLM) sobre um job. |

## Como rodar

```bash
cp .env.example .env   # ajuste DSPACE_URL, QDRANT_URL, MINERU_API_URL, DEEPSEEK_API_KEY
uv sync
uv run python backend/main.py   # sobe em http://127.0.0.1:8020
```

## Exemplo

```bash
curl -X POST http://127.0.0.1:8020/api/files/dspace/item/<ITEM_UUID>
```

Resposta `202` com a lista de jobs criados (um por PDF). Acompanhe cada um em
`/api/files/status/{job_id}` e recupere o markdown em `/api/files/result/{job_id}`.

## Requisitos externos

- **MinerU API** acessível em `MINERU_API_URL` (usa GPU).
- **Qdrant** acessível em `QDRANT_URL`.
- Modelo **BAAI/bge-m3** no cache HuggingFace (senão é baixado no 1º job).
- `DEEPSEEK_API_KEY` para o estágio 4 (opcional; sem ela o estágio é pulado).
