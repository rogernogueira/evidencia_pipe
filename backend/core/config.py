import os
from pathlib import Path

from dotenv import load_dotenv

# O backend fica em /app/evidencia_pipe/backend/core/config.py
# Então o BASE_DIR aponta para /app/evidencia_pipe (raiz do projeto).
BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = BASE_DIR / "output"

# Carrega o .env da raiz do projeto antes de ler as variáveis abaixo.
# (python-dotenv é dependência declarada; sem isto o .env é ignorado.)
load_dotenv(BASE_DIR / ".env")

# Qdrant — persistência vetorial dos chunks (dense + sparse) do estágio 3.
# Collection separada da do minerU para não misturar os corpora.
QDRANT_URL = os.getenv("QDRANT_URL", "http://192.168.105.8:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "evidencia_chunks")

# Modelo de embedding (bge-m3: dense 1024d + sparse lexical_weights).
DENSE_MODEL = "BAAI/bge-m3"
CACHE_DIR = "/root/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181/"

# API do MinerU rodando via Docker (container mineru-api, profile "api") — estágio 2.
MINERU_API_URL = os.getenv("MINERU_API_URL", "http://127.0.0.1:8010")

# Repositório DSpace (RDApp) — estágio 1. Bitstreams são baixados de
# {DSPACE_URL}/server/api/core/bitstreams/{uuid}/content
DSPACE_URL = os.getenv("DSPACE_URL", "https://rdapp.comais.uft.edu.br")

# Redis — broker do Celery (orquestração do pipeline) e backing do job_store
# (status compartilhado entre o processo da API e os workers). DB 0 = broker,
# DB 1 = job_store, para não misturar as chaves.
REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379")
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", f"{REDIS_URL}/0")
JOBSTORE_REDIS_URL = os.getenv("JOBSTORE_REDIS_URL", f"{REDIS_URL}/1")
# TTL (segundos) dos registros de job no Redis. Padrão: 7 dias.
JOBSTORE_TTL = int(os.getenv("JOBSTORE_TTL", str(7 * 24 * 3600)))

# DeepSeek — API oficial (OpenAI-compatible). Usada pelo estágio 4 de
# enriquecimento de metadados por LLM (llm_enrich_service). Sem DEEPSEEK_API_KEY
# o step é pulado.
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
# Teto de caracteres do markdown enviado à LLM (controle de custo/contexto).
# Documentos grandes são truncados; a truncagem é sinalizada em `observacoes`.
DEEPSEEK_MAX_CHARS = int(os.getenv("DEEPSEEK_MAX_CHARS", "220000"))
# Abaixo deste grau de confiança médio, o metadado é marcado para revisão humana.
DEEPSEEK_REVIEW_THRESHOLD = float(os.getenv("DEEPSEEK_REVIEW_THRESHOLD", "0.6"))
