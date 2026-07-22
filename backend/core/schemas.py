import json
from datetime import datetime, timezone
from typing import Any, Mapping, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

class ChunkMetadata(BaseModel):
    """Metadados que acompanham cada chunk semântico extraído pelo MinerU."""
    page: int = Field(..., description="Número da página onde o texto foi extraído")
    section: str = Field(..., description="A qual seção o bloco de texto pertence")
    doc_id: str = Field(..., description="Identificador único do documento (ex: nome do arquivo ou slug)")
    doc_name_md: Optional[str] = Field(None, description="Nome do arquivo markdown original, se disponível")
    type: str = Field(..., description="Tipo do bloco: 'paragraph', 'table', 'paragraph_part', etc.")
    part: Optional[int] = Field(None, description="Se for dividido ('paragraph_part'), o índice da parte. 1-indexed.")
    quality_score: float = Field(1.0, description="Score composto de qualidade do chunk (0.0=péssimo, 1.0=excelente)")
    token_count: int = Field(0, description="Contagem aproximada de tokens no conteúdo do chunk")
    ocr_noise_score: float = Field(0.0, description="Score de ruído OCR (0.0=limpo, 1.0=muito ruidoso)")
    is_heading_only: bool = Field(False, description="True se o chunk contém apenas texto de heading sem corpo")

class DocumentChunk(BaseModel):
    """Estrutura completa de um chunk extraído antes de ir para o banco vetorial."""
    content: str = Field(..., description="Texto limpo e pronto para a geração de embedding")
    metadata: ChunkMetadata = Field(..., description="Atributos associados para o payload do Qdrant")


class SearchResult(BaseModel):
    """Um resultado (chunk) da busca semântica sobre a collection `evidencia_chunks`.

    Deriva do payload gravado pelo indexador (backend/indexing/index_chunks.py):
    campos com fallback para tolerar tanto o payload legado quanto o estrutural
    (§26): section|section_title, page|page_start, content|text, type|content_type.
    """
    doc_name: Optional[str] = Field(None, description="Nome do arquivo markdown do documento")
    doc_id: Optional[str] = Field(None, description="Identificador do documento (ex: 'relatorio.pdf')")
    doc_path: Optional[str] = Field(None, description="Caminho web do markdown (sob /output)")
    section: str = Field("", description="Seção a que o chunk pertence")
    page: Optional[int] = Field(None, description="Página de origem do chunk")
    snippet: str = Field("", description="Trecho de texto do chunk (conteúdo casado)")
    score: Optional[float] = Field(None, description="Score de relevância (RRF/dense/sparse do Qdrant)")
    type: Optional[str] = Field("paragraph", description="Tipo do bloco: paragraph, table, ...")
    item_uuid: Optional[str] = Field(None, description="UUID do item DSpace de origem, se houver")
    item_handle: Optional[str] = Field(None, description="Handle do item DSpace de origem, se houver")


class LlmMetadataCandidates(BaseModel):
    """Metadados candidatos extraídos do markdown por uma LLM (DeepSeek).

    Só os campos de CONTEÚDO e de CONFIANÇA — os campos operacionais (doc_id,
    uuid, arquivo_json, llm_utilizada, quantidade_tokens, tempo_processamento)
    são preenchidos pelo serviço, não pela LLM. Todos opcionais/com default para
    a resposta ser robusta a campos ausentes. `coerce_numbers_to_str` tolera a LLM
    devolver números onde se espera texto (ex.: ano_candidato=2021).
    """

    model_config = ConfigDict(coerce_numbers_to_str=True)

    titulo_candidato: Optional[str] = Field(None, description="Título principal/oficial do documento")
    ano_candidato: Optional[str] = Field(None, description="Ano de publicação/emissão/referência")
    instituicao_candidata: Optional[str] = Field(None, description="Instituição/órgão responsável")
    tipo_documento_candidato: Optional[str] = Field(None, description="Tipo do documento (relatório, parecer, nota técnica, ...)")
    area_tematica_candidata: Optional[str] = Field(None, description="Área temática predominante")

    resumo_candidato: Optional[str] = Field(None, description="Resumo sintético (objetivo, objeto, metodologia, achados, conclusões)")

    palavras_chave_candidatas: list[str] = Field(default_factory=list, description="Palavras-chave/termos relevantes")
    abrangencia_territorial_candidata: Optional[str] = Field(None, description="Abrangência geográfica (Brasil, estado, município, ...)")
    periodo_avaliado_candidato: Optional[str] = Field(None, description="Período temporal analisado")
    programa_politica_candidato: Optional[str] = Field(None, description="Programa/política pública analisada")
    tipo_avaliacao_candidato: Optional[str] = Field(None, description="Tipo de avaliação (ex ante, ex post, impacto, processo, ...)")
    criterios_avaliacao_candidatos: list[str] = Field(default_factory=list, description="Critérios (eficácia, eficiência, efetividade, ...)")
    ods_candidatos: list[str] = Field(default_factory=list, description="ODS explicitamente mencionados no documento")
    ods_sugeridos_por_tema: list[str] = Field(default_factory=list, description="ODS sugeridos pela LLM a partir do tema")
    metodologia_candidata: Optional[str] = Field(None, description="Metodologia utilizada (abordagem, dados, técnicas)")
    achados_principais_candidatos: list[str] = Field(default_factory=list, description="Principais achados/constatações")
    recomendacoes_principais_candidatas: list[str] = Field(default_factory=list, description="Principais recomendações")

    confianca_titulo: Optional[float] = Field(None, ge=0, le=1, description="Confiança na extração do título [0,1]")
    confianca_ano: Optional[float] = Field(None, ge=0, le=1, description="Confiança no ano [0,1]")
    confianca_instituicao: Optional[float] = Field(None, ge=0, le=1, description="Confiança na instituição [0,1]")
    confianca_resumo: Optional[float] = Field(None, ge=0, le=1, description="Confiança no resumo [0,1]")
    confianca_programa_politica: Optional[float] = Field(None, ge=0, le=1, description="Confiança no programa/política [0,1]")
    confianca_ods: Optional[float] = Field(None, ge=0, le=1, description="Confiança nos ODS [0,1]")

    revisar: Optional[bool] = Field(None, description="Metadado precisa de revisão humana (baixa confiança/ambiguidade)")
    observacoes: Optional[str] = Field(None, description="Observações: limitações, campos ausentes, inferências, truncagem")


class DocumentMetadata(LlmMetadataCandidates):
    """Metadado completo do documento: candidatos da LLM + campos operacionais.

    É o objeto salvo em `output/<doc>/<doc>_metadata_llm.json` e devolvido pelo
    endpoint de enriquecimento.
    """

    doc_id: str = Field(..., description="Identificador interno do documento (stem do arquivo)")
    uuid: Optional[str] = Field(None, description="UUID do item no DSpace")
    arquivo_json: Optional[str] = Field(None, description="Caminho do JSON gerado (/output/...)")

    llm_utilizada: Optional[str] = Field(None, description="Modelo LLM usado")
    quantidade_tokens: Optional[int] = Field(None, description="Total de tokens (prompt+completion) da chamada")
    tempo_processamento: Optional[float] = Field(None, description="Tempo da geração dos metadados (s)")


# ==========================================================================
# Pipeline v2 — contexto leve + referências tipadas + manifesto de artefatos.
#
# A Celery chain transporta SOMENTE o PipelineContext (pequeno, JSON-serializável).
# Todo conteúdo (PDF, markdown, JSON MinerU, chunks, imagens, relatórios) vive no
# MinIO e é descoberto pelo manifesto (artifact_manifest_uri).
# ==========================================================================

# Nomes canônicos dos estágios registrados no manifesto (`manifest.stages`).
STAGE_DOWNLOAD = "download"
STAGE_MINERU = "mineru"
STAGE_ENRICHMENT = "enrichment"
STAGE_INDEXING = "indexing"

# Rótulos de progresso no PipelineContext.current_stage.
CTX_STAGE_QUEUED = "queued"
CTX_STAGE_DOWNLOADED = "downloaded"
CTX_STAGE_EXTRACTED = "extracted"
CTX_STAGE_ENRICHED = "enriched"
CTX_STAGE_INDEXED = "indexed"

# Nomes lógicos dos artefatos (chaves em `manifest.artifacts`).
ART_SOURCE_PDF = "source_pdf"
ART_MINERU_MARKDOWN = "mineru_markdown"
ART_MINERU_CONTENT_LIST = "mineru_content_list"
ART_MINERU_METRICS = "mineru_metrics"
ART_MINERU_IMAGES = "mineru_images"
ART_MINERU_IMAGES_MANIFEST = "mineru_images_manifest"
ART_METADATA_CANDIDATES = "metadata_candidates"
ART_LLM_RAW_RESPONSE = "llm_raw_response"
ART_CHUNKS = "chunks"
ART_CHUNKING_REPORT = "chunking_report"
ART_EMBEDDING_REPORT = "embedding_report"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ArtifactReference(BaseModel):
    """Referência tipada a um objeto no MinIO. NÃO contém o conteúdo do objeto.

    `etag` NÃO deve ser tratado como SHA-256: uploads multipart produzem ETags
    que não representam o hash íntegro do conteúdo. O `sha256` é sempre calculado
    explicitamente pelo pipeline.

    Para "diretórios" lógicos (prefixo de imagens), use `object_count`/
    `total_size_bytes` e content_type `application/x-directory-prefix`; nesse caso
    `sha256`/`size_bytes` podem ficar vazios.
    """

    name: str
    uri: str
    bucket: str
    object_key: str
    content_type: str = "application/octet-stream"
    size_bytes: int = 0
    sha256: str = ""
    etag: Optional[str] = None
    version_id: Optional[str] = None
    created_at: datetime = Field(default_factory=_utcnow)
    metadata: dict[str, str] = Field(default_factory=dict)
    # Somente para referências de prefixo (ex.: imagens).
    object_count: Optional[int] = None
    total_size_bytes: Optional[int] = None


class StageState(BaseModel):
    """Estado de um estágio no manifesto."""

    status: str = "PENDING"  # PENDING | RUNNING | COMPLETED | FAILED
    attempt: int = 0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None


class ManifestError(BaseModel):
    stage: str
    error_type: str
    message: str
    attempt: int = 1
    object_key: Optional[str] = None
    at: datetime = Field(default_factory=_utcnow)


class ArtifactManifest(BaseModel):
    """Manifesto por-documento — fonte de descoberta dos artefatos. Persistido em
    minio://<bucket>/<prefix>/<pipeline_id>/<document_id>/manifest.json."""

    schema_version: str = "1.0"
    revision: int = 0
    pipeline_id: str
    job_id: str
    item_uuid: Optional[str] = None
    item_handle: Optional[str] = None
    bitstream_uuid: Optional[str] = None
    document_id: str
    pipeline_version: str = "2.0"
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    status: str = "RUNNING"  # RUNNING | COMPLETED | FAILED

    artifacts: dict[str, ArtifactReference] = Field(default_factory=dict)
    stages: dict[str, StageState] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    errors: list[ManifestError] = Field(default_factory=list)

    def stage(self, name: str) -> StageState:
        st = self.stages.get(name)
        if st is None:
            st = StageState()
            self.stages[name] = st
        return st

    def is_stage_completed(self, name: str) -> bool:
        st = self.stages.get(name)
        return bool(st and st.status == "COMPLETED")


class PipelineContext(BaseModel):
    """Contexto leve transportado pela Celery chain — apenas identificadores e a
    URI do manifesto. Validado na entrada e na saída de cada task."""

    model_config = ConfigDict(extra="forbid")

    pipeline_id: UUID
    job_id: str
    item_uuid: str = ""
    bitstream_uuid: Optional[str] = None
    document_id: str
    artifact_manifest_uri: str
    current_stage: str
    pipeline_version: str = "2.0"
    warnings_count: int = 0
    force: bool = False  # reprocessamento forçado (ignora artefatos existentes)

    def to_message(self) -> dict[str, Any]:
        """Serialização JSON-safe para trafegar na chain (UUID → str)."""
        return json.loads(self.model_dump_json())


# --------------------------------------------------------------------------
# Barreira anti-regressão do payload da chain (§26/§27).
# --------------------------------------------------------------------------

class PipelinePayloadTooLargeError(ValueError):
    """O contexto retornado por uma task excedeu CELERY_CHAIN_MAX_PAYLOAD_BYTES."""


class ForbiddenChainPayloadError(ValueError):
    """O contexto retornado por uma task contém uma chave/estrutura proibida
    (markdown, chunks, embeddings, bytes, etc.)."""


# Chaves cujo NOME denuncia transporte de conteúdo grande pela chain.
FORBIDDEN_CONTEXT_KEYS = frozenset({
    "markdown", "markdown_content", "full_text", "content_list", "mineru_json",
    "chunks", "embeddings", "dense_vectors", "sparse_vectors", "qdrant_points",
    "images", "binary_data", "raw_response", "pdf_bytes", "file_content",
})


def _looks_like_binary(value: Any) -> bool:
    return isinstance(value, (bytes, bytearray, memoryview))


def validate_chain_payload_size(
    payload: Mapping[str, Any],
    max_bytes: int,
    *,
    enforce_lightweight: bool = True,
) -> int:
    """Valida o payload de saída de uma task ANTES de retorná-lo à chain.

    1. rejeita chaves proibidas (nome) e valores binários/não-serializáveis;
    2. serializa em JSON e mede em bytes;
    3. rejeita acima do limite.

    Retorna o tamanho serializado (bytes). Não registra o conteúdo, apenas o
    tamanho e o motivo — para não vazar dado sensível nos logs.
    """
    if not isinstance(payload, Mapping):
        raise ForbiddenChainPayloadError(
            f"payload da chain deve ser um mapeamento, veio {type(payload).__name__}."
        )

    if enforce_lightweight:
        for key, value in payload.items():
            if key.lower() in FORBIDDEN_CONTEXT_KEYS:
                raise ForbiddenChainPayloadError(
                    f"chave proibida no payload da chain: '{key}'."
                )
            if _looks_like_binary(value):
                raise ForbiddenChainPayloadError(
                    f"valor binário proibido no payload da chain (chave '{key}')."
                )

    try:
        # SEM default=str: valores não-serializáveis (sets, objetos, DataFrames,
        # arrays, tensores) devem FALHAR aqui, não serem silenciosamente convertidos.
        serialized = json.dumps(payload, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise ForbiddenChainPayloadError(
            f"payload da chain não é serializável em JSON: {exc}"
        ) from exc

    size = len(serialized.encode("utf-8"))
    if size > max_bytes:
        raise PipelinePayloadTooLargeError(
            f"payload da chain tem {size} bytes (> limite {max_bytes})."
        )
    return size
