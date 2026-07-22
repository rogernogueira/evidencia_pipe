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

# --------------------------------------------------------------------------
# GPUResourceManager — coordenação da GPU compartilhada (MinerU + BGE-M3 + scripts
# externos). Biblioteca independente (packages/gpu_resource_manager); aqui só lemos
# as variáveis do MESMO .env (sem criar mecanismo paralelo). O manager usa um Redis
# EXCLUSIVO (DB 2 por padrão) — NÃO reutiliza o broker (DB 0) nem o job_store (DB 1).
# --------------------------------------------------------------------------
GPU_MANAGER_ENABLED = os.getenv("GPU_MANAGER_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
GPU_MANAGER_REDIS_URL = os.getenv("GPU_MANAGER_REDIS_URL", f"{REDIS_URL}/2")
GPU_RESOURCE_NAME = os.getenv("GPU_RESOURCE_NAME", "gpu0")

# Prioridades dos consumidores internos (menor número = maior prioridade).
# A biblioteca NÃO conhece "mineru"/"bge-m3"; os nomes/prioridades vivem aqui.
MINERU_GPU_PRIORITY = int(os.getenv("MINERU_GPU_PRIORITY", "20"))
BGE_GPU_PRIORITY = int(os.getenv("BGE_GPU_PRIORITY", "30"))

# Ciclo de vida do BGE-M3 na VRAM (ver backend/services/bge_model_manager.py).
#   preload | lazy | cpu_idle | unload   — padrão p/ GPU compartilhada: lazy.
BGE_GPU_LIFECYCLE = os.getenv("BGE_GPU_LIFECYCLE", "lazy").strip().lower()
BGE_UNLOAD_AFTER_TASK = os.getenv("BGE_UNLOAD_AFTER_TASK", "true").strip().lower() in {"1", "true", "yes", "on"}
GPU_SHARED_WITH_MINERU = os.getenv("GPU_SHARED_WITH_MINERU", "true").strip().lower() in {"1", "true", "yes", "on"}

# Enriquecimento de metadados por LLM (llm_enrich_service) — DESACOPLADO do
# provedor e da indexação. Qualquer endpoint OpenAI-compatible serve; o provedor
# é apenas um conjunto de defaults (base_url/model). Sem chave, o enrich é pulado
# e a indexação segue normalmente — o enrich NÃO faz parte da chain obrigatória.
#
# A config genérica é LLM_ENRICH_*; os nomes legados DEEPSEEK_* continuam sendo
# lidos como fallback (compatibilidade com .env existentes).
LLM_ENRICH_PROVIDER = os.getenv("LLM_ENRICH_PROVIDER", "deepseek").strip().lower()

# Defaults por provedor (só base_url e model; a chave vem sempre do ambiente).
_LLM_ENRICH_PROVIDER_DEFAULTS = {
    "deepseek": {"base_url": "https://api.deepseek.com", "model": "deepseek-v4-flash"},
    "openai": {"base_url": "https://api.openai.com/v1", "model": "gpt-4o-mini"},
}
_llm_enrich_defaults = _LLM_ENRICH_PROVIDER_DEFAULTS.get(
    LLM_ENRICH_PROVIDER, _LLM_ENRICH_PROVIDER_DEFAULTS["deepseek"]
)

LLM_ENRICH_API_KEY = os.getenv("LLM_ENRICH_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
LLM_ENRICH_BASE_URL = (
    os.getenv("LLM_ENRICH_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL")
    or _llm_enrich_defaults["base_url"]
)
LLM_ENRICH_MODEL = (
    os.getenv("LLM_ENRICH_MODEL") or os.getenv("DEEPSEEK_MODEL")
    or _llm_enrich_defaults["model"]
)
# Teto de caracteres do markdown enviado à LLM (controle de custo/contexto).
# Documentos grandes são truncados; a truncagem é sinalizada em `observacoes`.
LLM_ENRICH_MAX_CHARS = int(os.getenv("LLM_ENRICH_MAX_CHARS") or os.getenv("DEEPSEEK_MAX_CHARS", "220000"))
# Abaixo deste grau de confiança médio, o metadado é marcado para revisão humana.
LLM_ENRICH_REVIEW_THRESHOLD = float(
    os.getenv("LLM_ENRICH_REVIEW_THRESHOLD") or os.getenv("DEEPSEEK_REVIEW_THRESHOLD", "0.6")
)
# Dispara o enrich automaticamente como follow-up APÓS a indexação (fora da chain
# obrigatória); no-op sem chave. Default true: mantém o enriquecimento automático,
# agora desacoplado da indexação.
LLM_ENRICH_AUTO = (os.getenv("LLM_ENRICH_AUTO", "true").strip().lower() in {"1", "true", "yes", "on"})

# Aliases legados (compatibilidade com imports/código existente).
DEEPSEEK_API_KEY = LLM_ENRICH_API_KEY
DEEPSEEK_BASE_URL = LLM_ENRICH_BASE_URL
DEEPSEEK_MODEL = LLM_ENRICH_MODEL
DEEPSEEK_MAX_CHARS = LLM_ENRICH_MAX_CHARS
DEEPSEEK_REVIEW_THRESHOLD = LLM_ENRICH_REVIEW_THRESHOLD


# --------------------------------------------------------------------------
# Armazenamento de artefatos (MinIO / S3-compatível) — fonte OFICIAL dos
# artefatos do pipeline (PDF, markdown, JSON MinerU, chunks, relatórios,
# manifesto). A Celery chain transporta SOMENTE referências (URIs minio://),
# nunca o conteúdo. Ver backend/services/artifact_store.py.
# --------------------------------------------------------------------------
def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


ARTIFACT_STORE_BACKEND = os.getenv("ARTIFACT_STORE_BACKEND", "minio").strip().lower()

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "127.0.0.1:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_SECURE = _as_bool(os.getenv("MINIO_SECURE"), False)
MINIO_REGION = os.getenv("MINIO_REGION", "us-east-1")

MINIO_BUCKET = os.getenv("MINIO_BUCKET", "evidencia-pipe")
# Prefixo raiz dos artefatos dentro do bucket. Toda object key produzida pelo
# store é validada para permanecer sob este prefixo (defesa contra path traversal).
MINIO_ARTIFACT_PREFIX = os.getenv("MINIO_ARTIFACT_PREFIX", "artifacts").strip("/")

MINIO_AUTO_CREATE_BUCKET = _as_bool(os.getenv("MINIO_AUTO_CREATE_BUCKET"), True)
MINIO_BUCKET_VERSIONING = _as_bool(os.getenv("MINIO_BUCKET_VERSIONING"), True)

MINIO_CONNECT_TIMEOUT_SECONDS = int(os.getenv("MINIO_CONNECT_TIMEOUT_SECONDS", "5"))
MINIO_READ_TIMEOUT_SECONDS = int(os.getenv("MINIO_READ_TIMEOUT_SECONDS", "60"))
MINIO_MAX_RETRIES = int(os.getenv("MINIO_MAX_RETRIES", "3"))

MINIO_MULTIPART_THRESHOLD_MB = int(os.getenv("MINIO_MULTIPART_THRESHOLD_MB", "64"))
MINIO_PART_SIZE_MB = int(os.getenv("MINIO_PART_SIZE_MB", "16"))

MINIO_PRESIGNED_URL_TTL_SECONDS = int(os.getenv("MINIO_PRESIGNED_URL_TTL_SECONDS", "900"))

MINIO_SERVER_SIDE_ENCRYPTION_ENABLED = _as_bool(os.getenv("MINIO_SERVER_SIDE_ENCRYPTION_ENABLED"), False)
MINIO_SERVER_SIDE_ENCRYPTION_TYPE = os.getenv("MINIO_SERVER_SIDE_ENCRYPTION_TYPE", "SSE-S3")

# Health check: quando True, grava e remove um objeto pequeno sob _healthcheck/
# para provar permissão de escrita/exclusão (nunca toca artefatos reais).
MINIO_HEALTHCHECK_WRITE = _as_bool(os.getenv("MINIO_HEALTHCHECK_WRITE"), True)

# Streaming (evita carregar artefatos grandes em memória).
MINIO_DOWNLOAD_CHUNK_SIZE_MB = int(os.getenv("MINIO_DOWNLOAD_CHUNK_SIZE_MB", "8"))
MINIO_UPLOAD_CHUNK_SIZE_MB = int(os.getenv("MINIO_UPLOAD_CHUNK_SIZE_MB", "16"))

# Diretório de trabalho temporário (materialização local de artefatos durante uma
# task). Sempre limpo ao final; ver ARTIFACT_KEEP_LOCAL_TEMP_ON_FAILURE.
ARTIFACT_TEMP_DIR = os.getenv("ARTIFACT_TEMP_DIR", "/tmp/evidencia-pipe")
ARTIFACT_RETENTION_DAYS = int(os.getenv("ARTIFACT_RETENTION_DAYS", "30"))
ARTIFACT_TEMP_RETENTION_HOURS = int(os.getenv("ARTIFACT_TEMP_RETENTION_HOURS", "24"))

ARTIFACT_KEEP_FAILED_JOBS = _as_bool(os.getenv("ARTIFACT_KEEP_FAILED_JOBS"), True)
ARTIFACT_KEEP_SOURCE_PDF = _as_bool(os.getenv("ARTIFACT_KEEP_SOURCE_PDF"), True)
ARTIFACT_KEEP_MINERU_OUTPUT = _as_bool(os.getenv("ARTIFACT_KEEP_MINERU_OUTPUT"), True)
ARTIFACT_KEEP_CHUNKS = _as_bool(os.getenv("ARTIFACT_KEEP_CHUNKS"), True)
# Depuração: preserva o diretório temporário local quando uma task falha. O caminho
# NUNCA é retornado pela API pública (só aparece nos logs internos).
ARTIFACT_KEEP_LOCAL_TEMP_ON_FAILURE = _as_bool(os.getenv("ARTIFACT_KEEP_LOCAL_TEMP_ON_FAILURE"), False)

# Enriquecimento / LLM.
LLM_MAX_INPUT_ARTIFACT_BYTES = int(os.getenv("LLM_MAX_INPUT_ARTIFACT_BYTES", str(10 * 1024 * 1024)))
ARTIFACT_KEEP_LLM_RAW_RESPONSE = _as_bool(os.getenv("ARTIFACT_KEEP_LLM_RAW_RESPONSE"), False)

# Chunks / embeddings / batches.
CHUNK_FILE_FORMAT = os.getenv("CHUNK_FILE_FORMAT", "jsonl").strip().lower()
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
QDRANT_UPSERT_BATCH_SIZE = int(os.getenv("QDRANT_UPSERT_BATCH_SIZE", "100"))

# --------------------------------------------------------------------------
# Chunking estrutural baseado em tokens (StructuralTokenChunker).
#
# A estratégia padrão passa a ser `structural_tokens` (limites em TOKENS do
# tokenizer do BGE-M3, preservando seções/páginas/tabelas/listas). O chunker
# legado (`legacy_chars`, por caracteres) permanece disponível para comparação.
# Toda a tokenização/chunking roda em CPU, FORA do lock global da GPU.
# --------------------------------------------------------------------------
CHUNKING_STRATEGY = os.getenv("CHUNKING_STRATEGY", "structural_tokens").strip().lower()
CHUNKING_VERSION = os.getenv("CHUNKING_VERSION", "structural-token-v1").strip()

# Tokenizer usado para CONTAR tokens (nunca carrega o modelo BGE-M3 completo na
# GPU — só o tokenizer, em CPU). Fallback whitespace quando indisponível.
CHUNK_TOKENIZER_MODEL = os.getenv("CHUNK_TOKENIZER_MODEL", "BAAI/bge-m3").strip()
CHUNK_TOKENIZER_USE_FAST = _as_bool(os.getenv("CHUNK_TOKENIZER_USE_FAST"), True)
CHUNK_TOKENIZER_FALLBACK_ENABLED = _as_bool(os.getenv("CHUNK_TOKENIZER_FALLBACK_ENABLED"), True)

# Alvos/limites de tamanho (tokens).
CHUNK_TARGET_TOKENS = int(os.getenv("CHUNK_TARGET_TOKENS", "512"))
CHUNK_MAX_TOKENS = int(os.getenv("CHUNK_MAX_TOKENS", "768"))
CHUNK_MIN_TOKENS = int(os.getenv("CHUNK_MIN_TOKENS", "80"))

CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "64"))
CHUNK_MAX_OVERLAP_TOKENS = int(os.getenv("CHUNK_MAX_OVERLAP_TOKENS", "128"))

CHUNK_TABLE_MAX_TOKENS = int(os.getenv("CHUNK_TABLE_MAX_TOKENS", "1024"))
CHUNK_LIST_MAX_TOKENS = int(os.getenv("CHUNK_LIST_MAX_TOKENS", "768"))
CHUNK_FORCE_SPLIT_ABOVE_TOKENS = int(os.getenv("CHUNK_FORCE_SPLIT_ABOVE_TOKENS", "1536"))

# Contextualização do texto embeddado (título do documento + caminho da seção).
CHUNK_EMBED_DOCUMENT_TITLE = _as_bool(os.getenv("CHUNK_EMBED_DOCUMENT_TITLE"), True)
CHUNK_EMBED_SECTION_CONTEXT = _as_bool(os.getenv("CHUNK_EMBED_SECTION_CONTEXT"), True)

# Limpeza estrutural.
CHUNK_REMOVE_REPEATED_HEADERS = _as_bool(os.getenv("CHUNK_REMOVE_REPEATED_HEADERS"), True)
CHUNK_REMOVE_REPEATED_FOOTERS = _as_bool(os.getenv("CHUNK_REMOVE_REPEATED_FOOTERS"), True)
CHUNK_KEEP_FOOTNOTES = _as_bool(os.getenv("CHUNK_KEEP_FOOTNOTES"), True)

# Referências bibliográficas: include | separate | exclude.
CHUNK_REFERENCES_MODE = os.getenv("CHUNK_REFERENCES_MODE", "separate").strip().lower()

# Campo textual efetivamente enviado ao embedder: text | contextualized_text.
EMBEDDING_TEXT_FIELD = os.getenv("EMBEDDING_TEXT_FIELD", "contextualized_text").strip()


def validate_chunking_config() -> None:
    """Valida as invariantes de tamanho do chunking (§9). Falha cedo (na carga do
    módulo) com mensagem clara — evita gerar chunks silenciosamente inválidos."""
    checks = [
        (CHUNK_MIN_TOKENS < CHUNK_TARGET_TOKENS,
         f"CHUNK_MIN_TOKENS ({CHUNK_MIN_TOKENS}) deve ser < CHUNK_TARGET_TOKENS ({CHUNK_TARGET_TOKENS})"),
        (CHUNK_TARGET_TOKENS <= CHUNK_MAX_TOKENS,
         f"CHUNK_TARGET_TOKENS ({CHUNK_TARGET_TOKENS}) deve ser <= CHUNK_MAX_TOKENS ({CHUNK_MAX_TOKENS})"),
        (CHUNK_OVERLAP_TOKENS < CHUNK_TARGET_TOKENS,
         f"CHUNK_OVERLAP_TOKENS ({CHUNK_OVERLAP_TOKENS}) deve ser < CHUNK_TARGET_TOKENS ({CHUNK_TARGET_TOKENS})"),
        (CHUNK_MAX_OVERLAP_TOKENS < CHUNK_MAX_TOKENS,
         f"CHUNK_MAX_OVERLAP_TOKENS ({CHUNK_MAX_OVERLAP_TOKENS}) deve ser < CHUNK_MAX_TOKENS ({CHUNK_MAX_TOKENS})"),
        (CHUNK_FORCE_SPLIT_ABOVE_TOKENS >= CHUNK_MAX_TOKENS,
         f"CHUNK_FORCE_SPLIT_ABOVE_TOKENS ({CHUNK_FORCE_SPLIT_ABOVE_TOKENS}) deve ser >= CHUNK_MAX_TOKENS ({CHUNK_MAX_TOKENS})"),
        (CHUNKING_STRATEGY in {"structural_tokens", "legacy_chars"},
         f"CHUNKING_STRATEGY inválido: {CHUNKING_STRATEGY!r} (use 'structural_tokens' ou 'legacy_chars')"),
        (CHUNK_REFERENCES_MODE in {"include", "separate", "exclude"},
         f"CHUNK_REFERENCES_MODE inválido: {CHUNK_REFERENCES_MODE!r}"),
        (EMBEDDING_TEXT_FIELD in {"text", "contextualized_text"},
         f"EMBEDDING_TEXT_FIELD inválido: {EMBEDDING_TEXT_FIELD!r}"),
    ]
    problems = [msg for ok, msg in checks if not ok]
    if problems:
        raise ValueError("Configuração de chunking inválida: " + "; ".join(problems))


validate_chunking_config()

# Limites de caracteres do chunker LEGADO (preservados temporariamente, §9).
LEGACY_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
LEGACY_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "250"))
LEGACY_MIN_CHUNK_CHARS = int(os.getenv("MIN_CHUNK_CHARS", "300"))

# Barreira anti-regressão do payload da Celery chain (ver core/schemas.py).
CELERY_CHAIN_MAX_PAYLOAD_BYTES = int(os.getenv("CELERY_CHAIN_MAX_PAYLOAD_BYTES", "16384"))
CELERY_CHAIN_ENFORCE_LIGHTWEIGHT_CONTEXT = _as_bool(
    os.getenv("CELERY_CHAIN_ENFORCE_LIGHTWEIGHT_CONTEXT"), True
)

# --------------------------------------------------------------------------
# Timeouts / limites de execução (robustez do pipeline).
#
# Ordenação intencional dos limites (do mais curto ao mais longo):
#   MINERU_TIMEOUT (40min) < soft task limit (50min) < hard task limit (55min)
#     < broker visibility_timeout (60min).
# Assim: o MinerU trava → estoura o timeout do subprocesso e falha limpo ANTES do
# soft limit; o soft limit levanta SoftTimeLimitExceeded (tratável, dispara
# on_failure e registra a falha) ANTES do hard limit matar o processo; e o hard
# limit (< visibility_timeout) evita que o broker reentregue a task a outro worker
# enquanto a original ainda roda (que causaria processamento duplicado).
# --------------------------------------------------------------------------
# Limite de tempo por task Celery (segundos). 0 = sem limite.
CELERY_TASK_SOFT_TIME_LIMIT = int(os.getenv("CELERY_TASK_SOFT_TIME_LIMIT", "3000"))  # 50 min
CELERY_TASK_TIME_LIMIT = int(os.getenv("CELERY_TASK_TIME_LIMIT", "3300"))            # 55 min
# Timeout do subprocesso do MinerU (segundos). Ao estourar, o grupo de processos é
# encerrado (libera a VRAM) e a task falha. 0 = sem timeout.
MINERU_TIMEOUT_SECONDS = int(os.getenv("MINERU_TIMEOUT_SECONDS", "2400"))            # 40 min
# Timeout dos clientes externos (segundos).
LLM_ENRICH_TIMEOUT_SECONDS = int(os.getenv("LLM_ENRICH_TIMEOUT_SECONDS", "120"))
QDRANT_TIMEOUT_SECONDS = int(os.getenv("QDRANT_TIMEOUT_SECONDS", "60"))

# Versão lógica do pipeline (usada no manifesto e na idempotência de reprocessos).
PIPELINE_VERSION = os.getenv("PIPELINE_VERSION", "2.0")

# Lock distribuído do manifesto (curto; só protege read-validate-write do JSON).
# Reusa o Redis do job_store (DB 1) por padrão — sem criar mecanismo paralelo.
MANIFEST_LOCK_REDIS_URL = os.getenv("MANIFEST_LOCK_REDIS_URL", JOBSTORE_REDIS_URL)
MANIFEST_LOCK_TTL_SECONDS = int(os.getenv("MANIFEST_LOCK_TTL_SECONDS", "30"))

# Token opcional para os endpoints internos (presigned URL). Se vazio, o endpoint
# fica liberado apenas em dev — em produção defina um valor forte.
INTERNAL_API_TOKEN = os.getenv("INTERNAL_API_TOKEN", "")
