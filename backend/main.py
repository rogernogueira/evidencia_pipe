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

    # Sonda a API de embedding (estágio 3) — o bge-m3 roda nos contêineres vLLM, não
    # neste processo. O servidor sobe mesmo com a API fora: quem depende dela (busca
    # e indexação) falha explicitamente na chamada, com a URL no erro.
    try:
        from backend.services.embedder import BgeM3EmbedderService

        embedder = BgeM3EmbedderService()
        dense_url, sparse_url = embedder.endpoints()
        if embedder.health_check():
            log.info("API de embedding OK (dense=%s, sparse=%s)", dense_url, sparse_url)
        else:
            log.warning("API de embedding indisponível (dense=%s, sparse=%s) — busca e "
                        "indexação vão falhar até os contêineres vLLM subirem.",
                        dense_url, sparse_url)
    except Exception as exc:  # pragma: no cover - a sonda é best-effort
        log.warning("Falha ao sondar a API de embedding (seguindo): %s", exc)

    log.info("=== Servidor pronto (porta %d) ===", APP_PORT)
    yield
    log.info("=== evidencia_pipe encerrando ===")


app = FastAPI(title="evidencia_pipe — DSpace ingestion pipeline", lifespan=lifespan)

# CORS com credenciais: o front (rdapp.comais.uft.edu.br) chama a API com
# withCredentials=true. Com credenciais, o navegador PROÍBE Access-Control-Allow-Origin=*;
# é preciso ecoar a origem específica + Allow-Credentials. Por isso listamos as origens
# conhecidas (+ CORS_ALLOW_ORIGINS no .env) e um regex para os subdomínios institucionais.
cors_allow_origins = [
    "http://192.168.105.8",
    "http://192.168.105.8:8181",
    "https://rdapp.comais.uft.edu.br",
    "https://api.rdapp.comais.uft.edu.br",
    "http://api.rdapp.comais.uft.edu.br",
    "http://localhost:4000",
    "http://localhost",
    "http://127.0.0.1:4000",
    "http://172.16.24.74:4000",
    "https://devrdapp.ibict.br",
]

extra_origins = os.getenv("CORS_ALLOW_ORIGINS", "")
if extra_origins.strip():
    cors_allow_origins.extend([o.strip() for o in extra_origins.split(",") if o.strip()])

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_allow_origins,
    allow_origin_regex=r"https?://([a-z0-9-]+\.)*(comais\.uft\.edu\.br|ibict\.br)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["health"])
async def health():
    return {"status": "ok", "service": "evidencia_pipe — DSpace ingestion pipeline"}


app.include_router(api_router)

# Serve os artefatos gerados (markdown, imagens, JSONs) sob /output.
OUTPUT_STATIC = OUTPUT_DIR if OUTPUT_DIR.exists() else BASE_DIR
app.mount("/output", StaticFiles(directory=str(OUTPUT_STATIC)), name="output")


if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=APP_PORT, reload=False)
