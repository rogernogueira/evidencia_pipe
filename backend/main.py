import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.core.config import BASE_DIR, OUTPUT_DIR
from backend.core.logger import log
from backend.api.router import api_router

APP_PORT = int(os.getenv("PORT", "8020"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("=== evidencia_pipe iniciando ===")
    log.info("BASE_DIR: %s", BASE_DIR)
    log.info("OUTPUT_DIR: %s  (existe=%s)", OUTPUT_DIR, OUTPUT_DIR.exists())

    # Pré-carrega o embedder bge-m3 (estágio 3) para a primeira requisição não pagar
    # o custo do load. Guardado por try/except: sem GPU/cache o servidor sobe mesmo
    # assim e o modelo é carregado sob demanda (require_cache=False no worker).
    try:
        from backend.services.embedder import BgeM3EmbedderService

        embedder = BgeM3EmbedderService()
        if embedder.is_cached_locally():
            log.info("Pré-carregando modelo de embedding bge-m3…")
            ok = embedder.load_model(require_cache=True)
            log.info("bge-m3: %s", "carregado" if ok else "não carregado (fallback sob demanda)")
        else:
            log.warning("bge-m3 não está no cache local — será carregado sob demanda no 1º job.")
    except Exception as exc:  # pragma: no cover - preload é best-effort
        log.warning("Falha ao pré-carregar o embedder (seguindo sem preload): %s", exc)

    log.info("=== Servidor pronto (porta %d) ===", APP_PORT)
    yield
    log.info("=== evidencia_pipe encerrando ===")


app = FastAPI(title="evidencia_pipe — DSpace ingestion pipeline", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["health"])
async def health():
    return {"status": "ok", "service": "evidencia_pipe"}


app.include_router(api_router)

# Serve os artefatos gerados (markdown, imagens, JSONs) sob /output.
OUTPUT_STATIC = OUTPUT_DIR if OUTPUT_DIR.exists() else BASE_DIR
app.mount("/output", StaticFiles(directory=str(OUTPUT_STATIC)), name="output")


if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=APP_PORT, reload=False)
