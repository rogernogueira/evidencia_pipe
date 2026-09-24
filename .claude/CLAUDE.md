# evidencia_pipe — notas para o Claude

## `experiments/` — sandbox fora do projeto

- `experiments/` é um projeto uv **independente**: venv própria (`experiments/.venv`)
  e lock próprio (`experiments/uv.lock`). Não compartilha a `.venv` nem o `uv.lock`
  da raiz.
- **Não é versionado** (está no `.gitignore`). Nada do backend, dos testes ou dos
  scripts pode importar ou depender de `experiments/`.
- Está em `[tool.uv.workspace].exclude` no `pyproject.toml` da raiz. Não o adicione
  aos `members` — `uv init`/`uv add` dentro dele podem tentar fazer isso; se
  acontecer, reverta.
- Para rodar algo lá: `cd experiments && uv run ...` (ou `uv run --project experiments ...`).
  Nunca `uv add` na raiz para satisfazer um experimento.
- Para usar o código do projeto num experimento, instale-o como dependência
  editável do experimento (`cd experiments && uv add --editable ..`), em vez de
  mexer em `sys.path`.
- Quando um experimento virar algo a manter, promova-o para o repo (`scripts/`,
  `eval/` ou `backend/`), com as dependências declaradas no `pyproject.toml` da raiz.
