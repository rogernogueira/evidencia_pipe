# Imagem ÚNICA da aplicação (API FastAPI + workers Celery) — a mesma imagem roda
# os três serviços `api`, `worker-light` e `worker-index`, só muda o `command`.
#
# ESBOÇO CPU-only: no modelo híbrido atual (DEPLOY.md) estes processos rodam no
# venv do host via systemd; aqui eles viram contêineres. O backend é 100%
# desacoplado por HTTP (MinerU, embedding, Qdrant, MinIO), então não precisa de
# GPU nem de torch/mineru — só do cliente HTTP e do tokenizer do bge-m3.
FROM python:3.13-slim

# uv (mesmo gerenciador do host). Copiado da imagem oficial — sem pip install.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

WORKDIR /app

# curl p/ os healthchecks do compose; sem toolchain de build (as wheels bastam).
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# 1) Dependências primeiro (camada cacheável). Precisa do membro de workspace
#    `packages/` e do README (pyproject.readme) para resolver o projeto raiz.
COPY pyproject.toml uv.lock README.md ./
COPY packages ./packages
RUN uv sync --frozen --no-dev --no-install-project

# 2) Código do projeto e sync final (instala o projeto raiz em editável).
COPY backend ./backend
COPY scripts ./scripts
RUN uv sync --frozen --no-dev

# ATENÇÃO (ver README deste diretório): pyproject fixa `onnxruntime-gpu` e
# `fastembed-gpu`. Instalam em CPU, mas carregam ~2,5 GB de libs CUDA inúteis.
# Para uma imagem CPU enxuta, troque pelas variantes CPU depois do sync:
#   RUN uv pip install onnxruntime fastembed

# API na 8020 por padrão (backend/main.py lê PORT). Workers sobrescrevem o command.
EXPOSE 8020
CMD ["python", "backend/main.py"]
