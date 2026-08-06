# MinerU 4.0.0a5 — serviço de avaliação (porta 8013)

Imagem **paralela** à de produção. Não substitui nada: existe para comparar a
extração da 4.0 contra a 3.4.4 antes de qualquer migração.

| | Produção | Avaliação |
|---|---|---|
| Versão | 3.4.4 (última estável, 10/07/2026) | 4.0.0a5 (pré-lançamento, 30/07/2026) |
| Imagem | `evidencia_mineru:local` (`tmp/Dockerfile`) | `evidencia_mineru:v4` (este diretório) |
| Contêiner | `evidencia_mineru_pipeline` | `evidencia_mineru_v4` |
| Porta | 8012 (e 8011 para o VLM) | 8013 |
| Sobe no boot? | sim (`evidencia-compose.service`) | **não** (profile + `restart: no`) |
| Usado pelo pipeline? | sim (`MINERU_API_URL` no `.env`) | não |

## Operação

```bash
docker compose --profile mineru-v4 build mineru-v4     # ~30 GB, dezenas de minutos
docker compose --profile mineru-v4 up -d mineru-v4
curl -f http://127.0.0.1:8013/docs                      # health
docker compose --profile mineru-v4 down mineru-v4       # remove o contêiner
```

Outra versão sem editar arquivo:

```bash
docker compose --profile mineru-v4 build \
  --build-arg MINERU_VERSION=4.0.0a6 mineru-v4
```

O build **falha de propósito** se a versão instalada divergir da pedida — uma imagem
que mente sobre sua versão é pior que build quebrado.

## O pareamento cliente↔servidor

A extração roda a CLI do MinerU **no host** (`uv run mineru --api-url …`, ver
`backend/services/mineru_service.py:139`), que é um cliente HTTP do `mineru-api`. O
venv do host tem 3.4.4 (`pyproject.toml`), então **não aponte o pipeline para a
8013**: cliente 3.4.4 contra servidor 4.0.0a5 pode divergir de protocolo, que é
exatamente o modo de falha do contêiner legado `mineru-api` (3.2.0), desativado em
06/08/2026.

Para exercitar a 4.0, use um cliente 4.0 em venv separado:

```bash
uv venv /tmp/mineru4 && \
  VIRTUAL_ENV=/tmp/mineru4 uv pip install --prerelease=allow 'mineru[core]==4.0.0a5'
VIRTUAL_ENV=/tmp/mineru4 uv run mineru \
  -p <arquivo.pdf> -o /tmp/saida-v4 --api-url http://127.0.0.1:8013 -b pipeline
```

## Resultado da avaliação (06/08/2026)

Comparação feita nas páginas 83–88 do `exames-educ-basica_relatorio-de-avaliacao`.

**A 4.0.0a5 NÃO corrige o falso título.** Produz exatamente o mesmo bloco:

```json
{"type": "title", "content": {"title_content": [{"type": "text", "content": "EVEX?"}],
 "level": 2}, "bbox": [114, 361, 169, 378]}
```

E mantém `Referências bibliográficas` também em `level: 2` — ou seja, a causa raiz do
desempilhamento em `document_blocks.py:536` continua idêntica. Rodando o parser do
projeto sobre a saída da 4.0, o dano se reproduz: os blocos após o `EVEX?` perdem o
`section_kind` de bibliografia.

Conclusão: **o validador de sumário (`toc_validator`) é necessário nas duas versões.**
Atualizar o MinerU não substitui a correção.

### O que a 4.0 melhora

Blocos de referência ganham tipo próprio — `sub_type: "ref_text"` (63 dos 86 blocos do
trecho). Isso torna a detecção de bibliografia independente da máquina de estados de
seção: mesmo com o `EVEX?` quebrando o estado, os blocos se auto-identificam. Não é
perfeito — a página 84, inteira de referências, veio como `text` — mas reduziria o
dano dos 14 chunks mal classificados.

### Ressalva metodológica

A 4.0 rodou no backend `hybrid` (default); produção usa `-b pipeline`. Não é
comparação like-for-like. O que é conclusivo é que o falso título aparece nas duas.

## Compatibilidade de formato: mudou o nome, não o schema

| 3.4.4 | 4.0.0a5 |
|---|---|
| `content_list_v2.json` | **`structured_content`** (mesmo schema — lista de páginas, `content.title_content[]`, `list_items[].item_content[]`) |
| — | `content_list` (formato ANTIGO, plano: `{type, text, text_level}`) — **incompatível** com o parser |

Verificado: `MinerUDocumentParser.parse_json()` consome o `structured_content` da 4.0
sem alteração (36 blocos, `structure_source=mineru_json`). Já o `content_list` da 4.0
é o formato v1 e seria descartado pelo parser silenciosamente — atenção ao pedir o
formato certo.

O `--format` do `mineru-kit parse` só expõe `markdown|middle_json|zip`; para obter
`structured_content` é preciso usar a API (`POST /v1/parse/jobs` com
`"output_formats": ["structured_content"]`).

## Consumo de VRAM

Após processar, o serviço segura **~25 GB** (`VLLM::EngineCore`). Numa GPU única isso
compete com produção — pare o contêiner ao terminar a avaliação:

```bash
docker stop evidencia_mineru_v4
```

## O que comparar

A pergunta que motivou esta imagem: **a 4.0 corrige os erros de layout que poluem o
chunking?** O diagnóstico da 3.4.4 está em `scripts/toc_heading_report.py`
(sumário × títulos detectados). Rode-o sobre a saída da 4.0 e compare:

```bash
uv run python scripts/toc_heading_report.py \
  --local /tmp/saida-v4/<doc>/auto/content_list_v2.json --no-csv
```

Casos concretos de referência, todos falsos títulos que o MinerU 3.4.4 produziu:

| Documento | Página | Fragmento | Dano |
|---|---|---|---|
| `exames-educ-basica_relatorio-de-avaliacao` | 85 | `EVEX?` | desempilhou "Referências bibliográficas"; 10 blocos viraram corpo |
| `relatorio_avaliacao-cmas-2020-pmcmv` | 65 | `Modelo Lógico` | idem, 10 blocos |
| `relatorio_avaliacao-cmas-2020-molestias-graves` | 46 | `A<sub>p</sub>ênd ice A` | OCR quebrou "Apêndice A"; 3 blocos |

No acervo inteiro (67 documentos) a 3.4.4 produziu 23 blocos reclassificados e 2.812
com `section_path` poluído. É esse número que a 4.0 precisa melhorar para justificar
a migração.

## Antes de migrar, verifique o schema

`backend/indexing/document_blocks.py` parseia o `content_list_v2.json` bloco a bloco
(`title`/`paragraph`/`list`/`table`/`chart`/`index`…). Um major bump pode mudar esse
formato — e todo o chunking v2 depende dele. Compare a estrutura antes de trocar o
`MINERU_API_URL`, e lembre que migrar exige subir também o pin do `pyproject.toml`
e reindexar o acervo (`scripts/reindex_from_minio.py`).
