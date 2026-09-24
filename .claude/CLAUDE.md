# evidencia_pipe — notas para o Claude

## `experiments/` — sandbox fora do projeto

### Isolamento
- `experiments/` é um projeto uv **independente**: venv própria (`experiments/.venv`)
  e lock próprio (`experiments/uv.lock`). Não compartilha a `.venv` nem o `uv.lock`
  da raiz.
- Nada do backend, dos testes ou dos scripts pode importar ou depender de
  `experiments/` — num clone limpo a pasta não existe.

### uv
- Está em `[tool.uv.workspace].exclude` no `pyproject.toml` da raiz. Não o adicione
  aos `members` — `uv init`/`uv add` dentro dele podem tentar fazer isso; se
  acontecer, reverta.
- Para rodar algo lá: `cd experiments && uv run ...` (ou `uv run --project experiments ...`).
- Dependência de experimento vai no `experiments/pyproject.toml`
  (`cd experiments && uv add ...`). Nunca `uv add` na raiz para satisfazer um experimento.
- Para usar o código do projeto num experimento, instale-o como dependência
  editável do experimento (`cd experiments && uv add --editable ..`), em vez de
  mexer em `sys.path`.

### Git — dois repositórios separados
- O repo principal **ignora** `experiments/` (`.gitignore`). Nunca force a pasta
  nele (`git add -f`) nem remova a entrada do `.gitignore`.
- `experiments/` tem **repositório git próprio** (`experiments/.git`, branch `main`).
  Commits de experimento são feitos lá dentro (`git -C experiments ...`), separados
  dos commits do projeto — nunca misture os dois num mesmo pedido de commit sem
  deixar claro qual repo recebe o quê.
- Esse repo **não tem remoto** configurado: o conteúdo existe só nesta máquina.
  Não crie nem faça push para remoto sem o usuário pedir.
- `experiments/.gitignore` ignora `.venv/`, `__pycache__/`, `*.py[oc]` e `.env`.
  Dados grandes ou saídas de rodada também não devem ir para esse repo.

### Promoção
- Quando um experimento virar algo a manter, promova-o para o repo principal
  (`scripts/`, `eval/` ou `backend/`), com as dependências declaradas no
  `pyproject.toml` da raiz e testes em `tests/`.
