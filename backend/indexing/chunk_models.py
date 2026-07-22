"""Modelos de dados e contratos do chunking estrutural.

Contém:
  - exceções específicas do chunking (§28);
  - `DocumentBlock`: unidade estrutural normalizada do documento (§6);
  - `StructuralChunk`: chunk de saída, rico em metadados de rastreabilidade (§18)
    — NUNCA contém embeddings;
  - `ChunkingMetrics` / `ChunkingResult`: métricas e resultado agregado (§23/§29);
  - `DocumentChunker` (Protocol): interface comum legado/estrutural (§21);
  - helpers de hash da configuração e de id determinístico (§20).

Todos os modelos são Pydantic (tipados, serializáveis em JSON/JSONL).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Optional, Protocol, runtime_checkable

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Exceções (§28)
# ---------------------------------------------------------------------------

class ChunkingError(Exception):
    """Erro genérico do chunking."""


class DocumentStructureParseError(ChunkingError):
    """Falha ao parsear a estrutura do documento MinerU (JSON/Markdown)."""


class ChunkSizeValidationError(ChunkingError):
    """Um chunk textual comum ultrapassou o limite absoluto configurado."""


class OversizedBlockError(ChunkingError):
    """Um bloco isolado (ex.: sentença única) excede o limite mesmo após split."""


class ChunkPersistenceError(ChunkingError):
    """Falha ao serializar/persistir os chunks (JSONL/relatório)."""


# ---------------------------------------------------------------------------
# Tipos de bloco estruturais (§6)
# ---------------------------------------------------------------------------

BLOCK_TITLE = "title"
BLOCK_HEADING = "heading"
BLOCK_PARAGRAPH = "paragraph"
BLOCK_LIST_ITEM = "list_item"
BLOCK_LIST = "list"
BLOCK_TABLE = "table"
BLOCK_TABLE_CAPTION = "table_caption"
BLOCK_FIGURE_CAPTION = "figure_caption"
BLOCK_FORMULA = "formula"
BLOCK_QUOTE = "quote"
BLOCK_FOOTNOTE = "footnote"
BLOCK_HEADER = "header"
BLOCK_FOOTER = "footer"
BLOCK_PAGE_NUMBER = "page_number"
BLOCK_REFERENCE = "reference"
BLOCK_UNKNOWN = "unknown"

# Blocos que compõem o corpo textual chunkável (na ordem do documento).
BODY_BLOCK_TYPES = frozenset({
    BLOCK_TITLE, BLOCK_HEADING, BLOCK_PARAGRAPH, BLOCK_LIST, BLOCK_LIST_ITEM,
    BLOCK_TABLE, BLOCK_FIGURE_CAPTION, BLOCK_FORMULA, BLOCK_QUOTE,
    BLOCK_FOOTNOTE, BLOCK_REFERENCE,
})


class DocumentBlock(BaseModel):
    """Bloco estrutural normalizado (independe da fonte MinerU JSON/Markdown)."""

    block_id: str
    block_type: str
    text: str
    order_index: int

    page_number: Optional[int] = None
    heading_level: Optional[int] = None
    section_path: list[str] = Field(default_factory=list)

    bbox: Optional[list[float]] = None
    source_reference: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class StructuralChunk(BaseModel):
    """Chunk de saída — rastreável e SEM embeddings (§18)."""

    chunk_id: str
    document_id: str
    item_uuid: str = ""
    bitstream_uuid: Optional[str] = None

    chunk_index: int
    text: str
    contextualized_text: Optional[str] = None

    token_count: int
    embedding_token_count: Optional[int] = None
    character_count: int

    content_type: str
    block_ids: list[str] = Field(default_factory=list)

    document_title: Optional[str] = None
    section_title: Optional[str] = None
    section_path: list[str] = Field(default_factory=list)
    heading_level: Optional[int] = None

    page_start: Optional[int] = None
    page_end: Optional[int] = None
    page_numbers: list[int] = Field(default_factory=list)

    overlap_token_count: int = 0
    overlap_source_chunk_id: Optional[str] = None

    split_reason: Optional[str] = None
    split_method: Optional[str] = None

    source_artifact_uri: str = ""
    document_checksum: str = ""

    chunking_strategy: str = ""
    chunking_version: str = ""
    chunking_config_hash: str = ""
    tokenizer_name: str = ""

    metadata: dict[str, Any] = Field(default_factory=dict)

    def embedding_input(self, field: str) -> str:
        """Texto a embeddar conforme EMBEDDING_TEXT_FIELD, com fallback para `text`."""
        if field == "contextualized_text" and self.contextualized_text:
            return self.contextualized_text
        return self.text


class ChunkingMetrics(BaseModel):
    """Métricas agregadas do chunking (compõem o chunking_report.json, §23)."""

    document_block_count: int = 0
    chunk_count: int = 0

    average_tokens: float = 0.0
    median_tokens: float = 0.0
    min_tokens: int = 0
    max_tokens: int = 0

    overlap_tokens_total: int = 0

    tables_count: int = 0
    table_chunks_count: int = 0
    list_chunks_count: int = 0

    sentence_split_count: int = 0
    token_fallback_split_count: int = 0

    removed_header_blocks: int = 0
    removed_footer_blocks: int = 0
    removed_page_number_blocks: int = 0
    rejected_chunks: int = 0

    blocks_by_type: dict[str, int] = Field(default_factory=dict)
    rejected_by_reason: dict[str, int] = Field(default_factory=dict)

    parse_time_s: float = 0.0
    tokenize_time_s: float = 0.0
    grouping_time_s: float = 0.0

    structure_source: str = ""
    warnings: list[str] = Field(default_factory=list)


class ChunkingResult(BaseModel):
    """Resultado do chunking: chunks + métricas + identificação da config."""

    document_id: str
    chunks: list[StructuralChunk] = Field(default_factory=list)
    metrics: ChunkingMetrics = Field(default_factory=ChunkingMetrics)

    chunking_strategy: str = ""
    chunking_version: str = ""
    chunking_config_hash: str = ""
    tokenizer_name: str = ""
    structure_source: str = ""

    def report(self) -> dict[str, Any]:
        """Monta o dict do chunking_report.json (§23)."""
        m = self.metrics
        return {
            "chunking_strategy": self.chunking_strategy,
            "chunking_version": self.chunking_version,
            "chunking_config_hash": self.chunking_config_hash,
            "tokenizer_name": self.tokenizer_name,
            "document_block_count": m.document_block_count,
            "chunk_count": m.chunk_count,
            "average_tokens": m.average_tokens,
            "median_tokens": m.median_tokens,
            "min_tokens": m.min_tokens,
            "max_tokens": m.max_tokens,
            "overlap_tokens_total": m.overlap_tokens_total,
            "tables_count": m.tables_count,
            "table_chunks_count": m.table_chunks_count,
            "list_chunks_count": m.list_chunks_count,
            "sentence_split_count": m.sentence_split_count,
            "token_fallback_split_count": m.token_fallback_split_count,
            "removed_header_blocks": m.removed_header_blocks,
            "removed_footer_blocks": m.removed_footer_blocks,
            "removed_page_number_blocks": m.removed_page_number_blocks,
            "rejected_chunks": m.rejected_chunks,
            "blocks_by_type": m.blocks_by_type,
            "rejected_by_reason": m.rejected_by_reason,
            "parse_time_s": round(m.parse_time_s, 3),
            "tokenize_time_s": round(m.tokenize_time_s, 3),
            "grouping_time_s": round(m.grouping_time_s, 3),
            "structure_source": self.structure_source,
            "warnings": m.warnings,
        }


# ---------------------------------------------------------------------------
# Interface comum (§21)
# ---------------------------------------------------------------------------

@runtime_checkable
class DocumentChunker(Protocol):
    """Contrato comum entre LegacyCharacterChunker e StructuralTokenChunker."""

    strategy: str

    def chunk(
        self,
        blocks: list[DocumentBlock],
        *,
        document_id: str,
        document_title: Optional[str] = None,
        document_checksum: str = "",
        item_uuid: str = "",
        bitstream_uuid: Optional[str] = None,
        source_artifact_uri: str = "",
        structure_source: str = "",
    ) -> ChunkingResult: ...


# ---------------------------------------------------------------------------
# Hash da config + id determinístico (§20)
# ---------------------------------------------------------------------------

def compute_config_hash(config: dict[str, Any]) -> str:
    """SHA-256 curto (16 hex) de um dict de configuração — estável e ordenado."""
    payload = json.dumps(config, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def normalize_text_for_hash(text: str) -> str:
    """Normaliza whitespace para tornar o hash do texto robusto a reflows triviais."""
    return " ".join(text.split())


def make_chunk_id(
    *,
    document_id: str,
    bitstream_uuid: Optional[str],
    document_checksum: str,
    chunking_version: str,
    chunking_config_hash: str,
    chunk_index: int,
    normalized_text: str,
) -> str:
    """chunk_id determinístico (§20): estável quando documento e config não mudam."""
    text_hash = hashlib.sha256(normalize_text_for_hash(normalized_text).encode("utf-8")).hexdigest()
    seed = "|".join([
        document_id,
        bitstream_uuid or "",
        document_checksum or "",
        chunking_version or "",
        chunking_config_hash or "",
        str(chunk_index),
        text_hash,
    ])
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()
