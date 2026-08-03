"""StructuralTokenChunker — chunking estrutural controlado por TOKENS (§10-§19).

Ordem de formação: seção → bloco → parágrafo → sentença → tokens (§10). Só usa
corte direto por tokens como último recurso, quando uma única sentença excede o
limite. Preserva seções (`section_path`), páginas, tabelas e listas como unidades,
associa legendas, aplica overlap ESTRUTURAL (unidades completas, não caracteres) e
produz `StructuralChunk` determinístico e rastreável — SEM embeddings.

Também expõe `LegacyCharacterChunker`, um adaptador que envelopa o `MinerUChunker`
por caracteres na mesma interface `DocumentChunker` (§21), para comparação A/B.

Nada aqui usa GPU: só o `TokenCounter` (CPU) e regex. É seguro rodar FORA do lock
global da GPU (§24).
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from statistics import median
from typing import Any, Callable, Optional

from backend.core import config as settings
from backend.indexing.chunk_models import (
    BLOCK_FIGURE_CAPTION,
    BLOCK_FORMULA,
    BLOCK_HEADING,
    BLOCK_LIST,
    BLOCK_REFERENCE,
    BLOCK_TABLE,
    FRONT_MATTER_SECTION_KINDS,
    NAVIGATION_SECTION_KINDS,
    ChunkingMetrics,
    ChunkingResult,
    ChunkSizeValidationError,
    DocumentBlock,
    StructuralChunk,
    compute_config_hash,
    make_chunk_id,
)
from backend.indexing.chunk_quality_filters import FilterOutcome
from backend.indexing.cross_page_reconstruction import block_pages, block_printed_pages
from backend.indexing.retrieval_profile import classify_retrieval
from backend.indexing.token_counter import TokenCounter, get_token_counter

# Seções pré-textuais/navegação puladas do chunking quando front_matter_mode=metadata_only.
_FRONT_MATTER_SKIP_KINDS = FRONT_MATTER_SECTION_KINDS | NAVIGATION_SECTION_KINDS

# Divisão em sentenças tolerante ao português (mantém a pontuação final).
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+(?=[A-ZÀ-Ú0-9\"'(\[])")
_MD_SEPARATOR_RE = re.compile(r"^\s*\|?[\s\-:|]+\|?\s*$")


@dataclass
class ChunkingConfig:
    """Snapshot imutável da configuração de chunking (base do config_hash)."""

    target_tokens: int = settings.CHUNK_TARGET_TOKENS
    max_tokens: int = settings.CHUNK_MAX_TOKENS
    min_tokens: int = settings.CHUNK_MIN_TOKENS
    overlap_tokens: int = settings.CHUNK_OVERLAP_TOKENS
    max_overlap_tokens: int = settings.CHUNK_MAX_OVERLAP_TOKENS
    table_max_tokens: int = settings.CHUNK_TABLE_MAX_TOKENS
    list_max_tokens: int = settings.CHUNK_LIST_MAX_TOKENS
    force_split_above_tokens: int = settings.CHUNK_FORCE_SPLIT_ABOVE_TOKENS
    embed_document_title: bool = settings.CHUNK_EMBED_DOCUMENT_TITLE
    embed_section_context: bool = settings.CHUNK_EMBED_SECTION_CONTEXT
    references_mode: str = settings.CHUNK_REFERENCES_MODE
    front_matter_mode: str = settings.CHUNK_FRONT_MATTER_MODE
    appendix_mode: str = settings.CHUNK_APPENDIX_MODE
    equation_mode: str = settings.CHUNK_EQUATION_MODE
    equation_min_confidence: float = settings.CHUNK_EQUATION_MIN_CONFIDENCE
    chunking_version: str = settings.CHUNKING_VERSION

    def hashable(self) -> dict[str, Any]:
        return {
            "target_tokens": self.target_tokens,
            "max_tokens": self.max_tokens,
            "min_tokens": self.min_tokens,
            "overlap_tokens": self.overlap_tokens,
            "max_overlap_tokens": self.max_overlap_tokens,
            "table_max_tokens": self.table_max_tokens,
            "list_max_tokens": self.list_max_tokens,
            "force_split_above_tokens": self.force_split_above_tokens,
            "embed_document_title": self.embed_document_title,
            "embed_section_context": self.embed_section_context,
            "references_mode": self.references_mode,
            "front_matter_mode": self.front_matter_mode,
            "appendix_mode": self.appendix_mode,
            "equation_mode": self.equation_mode,
            "equation_min_confidence": self.equation_min_confidence,
            "chunking_version": self.chunking_version,
        }


# Resultado leve de avaliação de qualidade (§22) — desacoplado dos filtros.
FilterFn = Callable[[str, str, int, list[str]], FilterOutcome]


@dataclass
class _Draft:
    """Rascunho de um chunk antes de virar StructuralChunk (facilita overlap/merge)."""

    text: str
    content_type: str
    blocks: list[DocumentBlock]
    section_path: list[str]
    heading_level: Optional[int]
    split_reason: Optional[str] = None
    split_method: Optional[str] = None
    overlap_token_count: int = 0
    overlap_block_ids: list[str] = field(default_factory=list)
    extra_meta: dict[str, Any] = field(default_factory=dict)
    allow_overlap: bool = True


class StructuralTokenChunker:
    strategy = "structural_tokens"

    def __init__(
        self,
        *,
        config: Optional[ChunkingConfig] = None,
        token_counter: Optional[TokenCounter] = None,
        filter_fn: Optional[FilterFn] = None,
    ) -> None:
        self.cfg = config or ChunkingConfig()
        self.tokens = token_counter or get_token_counter()
        self.config_hash = compute_config_hash(self.cfg.hashable())
        if filter_fn is not None:
            self._filter = filter_fn
        else:
            from backend.indexing.chunk_quality_filters import evaluate_structural_chunk

            _min = self.cfg.min_tokens

            def _default_filter(text, content_type, token_count, section_path):
                return evaluate_structural_chunk(
                    text, content_type, token_count, section_path, min_tokens=_min
                )

            self._filter = _default_filter
        self._metrics = ChunkingMetrics()

    # -- helpers de token ---------------------------------------------------
    def _count(self, text: str) -> int:
        return self.tokens.count(text)

    # -- fronteiras de estado do bloco (§5/§6/§15) --------------------------
    def _should_chunk(self, b: DocumentBlock) -> bool:
        """Decide se um bloco entra no chunking. Default v1: só exclui furniture.

        v2 (via config): pula pré-textuais/navegação (front_matter_mode=metadata_only)
        e referências (references_mode exclude|metadata_only)."""
        if not b.chunkable:
            return False
        if (self.cfg.front_matter_mode == "metadata_only"
                and b.section_kind in _FRONT_MATTER_SKIP_KINDS):
            return False
        if b.block_type == BLOCK_REFERENCE and self.cfg.references_mode in ("exclude", "metadata_only"):
            return False
        # §9: equação inline suspeita (ordinal/símbolo mal lido) fora do embedding.
        if (b.block_type == BLOCK_FORMULA and self.cfg.equation_mode == "merge_with_context"
                and b.equation_confidence is not None
                and b.equation_confidence < self.cfg.equation_min_confidence):
            return False
        return True

    # ------------------------------------------------------------------ API
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
    ) -> ChunkingResult:
        self._metrics = ChunkingMetrics(structure_source=structure_source)
        self._metrics.document_block_count = len(blocks)
        for b in blocks:
            self._metrics.blocks_by_type[b.block_type] = (
                self._metrics.blocks_by_type.get(b.block_type, 0) + 1
            )
        self._metrics.tables_count = self._metrics.blocks_by_type.get(BLOCK_TABLE, 0)

        t0 = time.perf_counter()
        drafts = self._build_drafts(blocks)
        self._metrics.grouping_time_s = time.perf_counter() - t0

        # Materializa StructuralChunks (aplica filtros + ids + overlap metadata).
        chunks: list[StructuralChunk] = []
        prev_id: Optional[str] = None
        for draft in drafts:
            approved, reason, quality = self._apply_filters(draft)
            if not approved:
                self._metrics.rejected_chunks += 1
                self._metrics.rejected_by_reason[reason] = (
                    self._metrics.rejected_by_reason.get(reason, 0) + 1
                )
                continue
            chunk = self._materialize(
                draft, index=len(chunks), prev_id=prev_id,
                document_id=document_id, document_title=document_title,
                document_checksum=document_checksum, item_uuid=item_uuid,
                bitstream_uuid=bitstream_uuid, source_artifact_uri=source_artifact_uri,
                quality=quality,
            )
            chunks.append(chunk)
            prev_id = chunk.chunk_id

        self._finalize_metrics(chunks)
        return ChunkingResult(
            document_id=document_id,
            chunks=chunks,
            metrics=self._metrics,
            chunking_strategy=self.strategy,
            chunking_version=self.cfg.chunking_version,
            chunking_config_hash=self.config_hash,
            tokenizer_name=self.tokens.name,
            structure_source=structure_source,
        )

    # -- formação de rascunhos ---------------------------------------------
    def _build_drafts(self, blocks: list[DocumentBlock]) -> list[_Draft]:
        drafts: list[_Draft] = []
        # Grupo de parágrafos pendentes da seção corrente.
        pending: list[DocumentBlock] = []
        pending_path: list[str] = []
        pending_level: Optional[int] = None

        def flush():
            nonlocal pending, pending_path, pending_level
            if pending:
                drafts.extend(self._chunks_from_text_group(pending, pending_path, pending_level))
                pending = []

        for b in blocks:
            if b.block_type == BLOCK_HEADING:
                # Mudança de seção: fecha o grupo anterior (sem overlap entre seções).
                flush()
                pending_path = b.section_path
                pending_level = b.heading_level
                continue

            # v2: pula blocos não-chunkáveis (furniture §6) e pré-textuais/navegação
            # quando front_matter_mode=metadata_only (§5). Refs tratadas abaixo.
            if not self._should_chunk(b):
                continue

            # Blocos standalone (não se misturam ao grupo de parágrafos).
            if b.block_type == BLOCK_TABLE:
                flush()
                drafts.extend(self._chunks_from_table(b))
                continue
            if b.block_type == BLOCK_LIST:
                flush()
                drafts.extend(self._chunks_from_list(b))
                continue
            if b.block_type == BLOCK_FIGURE_CAPTION:
                flush()
                drafts.append(self._chunk_from_figure(b))
                continue
            if b.block_type == BLOCK_REFERENCE and self.cfg.references_mode == "separate":
                flush()
                drafts.extend(self._chunks_from_references([b]))
                continue

            # Parágrafo/quote/formula/footnote/reference(include): entram no grupo.
            if pending and b.section_path != pending_path:
                flush()
            if not pending:
                pending_path = b.section_path
                pending_level = b.heading_level or pending_level
            pending.append(b)

        flush()
        return drafts

    # -- grupo de parágrafos (seção → bloco → parágrafo → sentença) --------
    def _chunks_from_text_group(
        self, group: list[DocumentBlock], section_path: list[str], level: Optional[int],
    ) -> list[_Draft]:
        cfg = self.cfg
        content_type = "references" if group[0].block_type == BLOCK_REFERENCE else "paragraph"
        drafts: list[_Draft] = []
        cur_blocks: list[DocumentBlock] = []
        cur_text = ""
        cur_tokens = 0

        def close():
            nonlocal cur_blocks, cur_text, cur_tokens
            if cur_text.strip():
                drafts.append(_Draft(
                    text=cur_text.strip(), content_type=content_type,
                    blocks=list(cur_blocks), section_path=section_path, heading_level=level,
                    allow_overlap=(content_type != "references"),
                ))
            cur_blocks, cur_text, cur_tokens = [], "", 0

        for b in group:
            btoks = self._count(b.text)
            # Bloco isolado maior que o máximo → divide por sentença/token.
            if btoks > cfg.max_tokens:
                close()
                drafts.extend(self._split_oversized_block(b, section_path, level, content_type))
                continue
            # Fecha o chunk atual se a adição estourar o máximo.
            if cur_blocks and (cur_tokens + btoks) > cfg.max_tokens:
                close()
            cur_blocks.append(b)
            cur_text = f"{cur_text}\n\n{b.text}".strip() if cur_text else b.text
            cur_tokens += btoks
            # Alcançou o alvo → fecha (permite crescer só até o máximo).
            if cur_tokens >= cfg.target_tokens:
                close()
        close()

        drafts = self._merge_small(drafts, section_path)
        return self._apply_structural_overlap(drafts)

    def _split_oversized_block(
        self, b: DocumentBlock, section_path: list[str], level: Optional[int], content_type: str,
    ) -> list[_Draft]:
        """Parágrafo grande: sentença → agrupa até o alvo → token fallback (§11)."""
        cfg = self.cfg
        sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(b.text) if s.strip()]
        drafts: list[_Draft] = []
        cur: list[str] = []
        cur_tokens = 0
        used_token_fallback = False

        def close(reason: str, method: str):
            nonlocal cur, cur_tokens
            if cur:
                drafts.append(_Draft(
                    text=" ".join(cur).strip(), content_type=content_type, blocks=[b],
                    section_path=section_path, heading_level=level,
                    split_reason=reason, split_method=method,
                ))
                self._metrics.sentence_split_count += 1
            cur, cur_tokens = [], 0

        for s in sentences:
            stoks = self._count(s)
            if stoks > cfg.max_tokens:
                # Sentença única maior que o limite → ÚLTIMO recurso: corte por tokens.
                close("oversized_paragraph", "sentence")
                for piece in self.tokens.split(s, cfg.max_tokens):
                    drafts.append(_Draft(
                        text=piece, content_type=content_type, blocks=[b],
                        section_path=section_path, heading_level=level,
                        split_reason="oversized_paragraph", split_method="token_fallback",
                    ))
                    used_token_fallback = True
                continue
            if cur and (cur_tokens + stoks) > cfg.target_tokens:
                close("oversized_paragraph", "sentence")
            cur.append(s)
            cur_tokens += stoks
        close("oversized_paragraph", "sentence")
        if used_token_fallback:
            self._metrics.token_fallback_split_count += 1
        # Overlap por sentenças entre as partes do mesmo parágrafo.
        return self._apply_structural_overlap(drafts)

    # -- merge de chunks abaixo do mínimo ----------------------------------
    def _merge_small(self, drafts: list[_Draft], section_path: list[str]) -> list[_Draft]:
        cfg = self.cfg
        if len(drafts) < 2:
            return drafts
        out: list[_Draft] = []
        for d in drafts:
            if out and self._count(d.text) < cfg.min_tokens:
                prev = out[-1]
                merged_text = f"{prev.text}\n\n{d.text}"
                if self._count(merged_text) <= cfg.max_tokens:
                    prev.text = merged_text
                    prev.blocks.extend(d.blocks)
                    continue
            out.append(d)
        return out

    # -- overlap estrutural (§17) ------------------------------------------
    def _apply_structural_overlap(self, drafts: list[_Draft]) -> list[_Draft]:
        cfg = self.cfg
        if cfg.overlap_tokens <= 0 or len(drafts) < 2:
            return drafts
        for i in range(1, len(drafts)):
            cur = drafts[i]
            prev = drafts[i - 1]
            if not cur.allow_overlap or not prev.allow_overlap:
                continue
            # Não sobrepõe através de mudança clara de seção.
            if cur.section_path != prev.section_path:
                continue
            overlap = self._tail_overlap(prev.text)
            if not overlap:
                continue
            otoks = self._count(overlap)
            # Não deixa o chunk estourar o máximo por causa do overlap.
            if self._count(cur.text) + otoks > cfg.max_tokens:
                continue
            cur.text = f"{overlap}\n\n{cur.text}"
            cur.overlap_token_count = otoks
            cur.overlap_block_ids = [b.block_id for b in prev.blocks]
        return drafts

    def _tail_overlap(self, text: str) -> str:
        """Extrai o "rabo" do chunk anterior como unidades completas: último
        parágrafo; se maior que o teto, últimas sentenças até overlap_tokens."""
        cfg = self.cfg
        paragraphs = [p for p in text.split("\n\n") if p.strip()]
        candidate = paragraphs[-1].strip() if paragraphs else text.strip()
        if self._count(candidate) <= cfg.max_overlap_tokens and self._count(candidate) >= 1:
            if self._count(candidate) <= cfg.overlap_tokens * 2:
                return candidate
        # Reduz por sentenças a partir do fim até caber em overlap_tokens.
        sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(candidate) if s.strip()]
        acc: list[str] = []
        total = 0
        for s in reversed(sentences):
            stoks = self._count(s)
            if total + stoks > cfg.overlap_tokens and acc:
                break
            acc.insert(0, s)
            total += stoks
            if total >= cfg.overlap_tokens:
                break
        return " ".join(acc).strip()

    # -- tabelas (§12) ------------------------------------------------------
    def _chunks_from_table(self, b: DocumentBlock) -> list[_Draft]:
        cfg = self.cfg
        caption = b.metadata.get("table_caption")
        table_md = b.metadata.get("table_markdown") or ""
        full_tokens = self._count(b.text)
        table_id = b.block_id.replace("block", "table")

        base_meta = {
            "table_id": table_id,
            "table_title": caption,
            "image_uri": b.metadata.get("image_uri"),
        }

        # Tabela pequena → uma unidade só.
        if full_tokens <= cfg.table_max_tokens or not table_md:
            self._metrics.table_chunks_count += 1
            return [_Draft(
                text=b.text, content_type="table", blocks=[b],
                section_path=b.section_path, heading_level=b.heading_level,
                allow_overlap=False,
                extra_meta={**base_meta, "row_start": 0, "row_end": self._table_rows(table_md),
                            "is_table_continuation": False},
            )]

        # Tabela grande → divide por grupos de linhas, repetindo o cabeçalho (§12).
        return self._split_table_by_rows(b, table_md, caption, base_meta)

    @staticmethod
    def _table_rows(table_md: str) -> int:
        return sum(1 for ln in table_md.splitlines() if "|" in ln and not _MD_SEPARATOR_RE.match(ln))

    def _split_table_by_rows(
        self, b: DocumentBlock, table_md: str, caption: Optional[str], base_meta: dict,
    ) -> list[_Draft]:
        cfg = self.cfg
        lines = [ln for ln in table_md.splitlines() if ln.strip()]
        header_lines: list[str] = []
        data_lines: list[str] = []
        seen_sep = False
        for ln in lines:
            if _MD_SEPARATOR_RE.match(ln):
                header_lines.append(ln)
                seen_sep = True
                continue
            if not seen_sep and not header_lines:
                header_lines.append(ln)  # primeira linha = cabeçalho
            elif not seen_sep:
                data_lines.append(ln)
            else:
                data_lines.append(ln)
        if not header_lines:
            header_lines = lines[:1]
            data_lines = lines[1:]

        header_text = "\n".join(header_lines)
        cap_prefix = f"{caption}\n" if caption else ""
        drafts: list[_Draft] = []
        row_start = 0
        cur_rows: list[str] = []

        def close(is_cont: bool):
            nonlocal cur_rows, row_start
            if not cur_rows:
                return
            body = "\n".join(cur_rows)
            text = f"{cap_prefix}{header_text}\n{body}".strip()
            row_end = row_start + len(cur_rows)
            drafts.append(_Draft(
                text=text, content_type="table", blocks=[b],
                section_path=b.section_path, heading_level=b.heading_level, allow_overlap=False,
                extra_meta={**base_meta, "row_start": row_start, "row_end": row_end,
                            "is_table_continuation": is_cont},
            ))
            self._metrics.table_chunks_count += 1
            row_start = row_end
            cur_rows = []

        header_tokens = self._count(f"{cap_prefix}{header_text}")
        for row in data_lines:
            prospective = header_tokens + self._count("\n".join(cur_rows + [row]))
            if cur_rows and prospective > cfg.table_max_tokens:
                close(is_cont=len(drafts) > 0)
            cur_rows.append(row)
        close(is_cont=len(drafts) > 0)
        return drafts or [_Draft(
            text=b.text, content_type="table", blocks=[b], section_path=b.section_path,
            heading_level=b.heading_level, allow_overlap=False, extra_meta=base_meta,
        )]

    # -- listas (§13) -------------------------------------------------------
    def _chunks_from_list(self, b: DocumentBlock) -> list[_Draft]:
        cfg = self.cfg
        meta = b.metadata or {}
        list_type = meta.get("list_type", "unordered")
        items = [ln for ln in b.text.splitlines() if ln.strip()]
        base_meta = {"list_type": list_type}
        self._metrics.list_chunks_count += 1

        if self._count(b.text) <= cfg.list_max_tokens:
            return [_Draft(
                text=b.text, content_type="list", blocks=[b], section_path=b.section_path,
                heading_level=b.heading_level, allow_overlap=False,
                extra_meta={**base_meta, "list_start_index": 1, "list_end_index": len(items),
                            "is_list_continuation": False},
            )]

        drafts: list[_Draft] = []
        cur: list[str] = []
        start_idx = 1
        for idx, item in enumerate(items, 1):
            if cur and self._count("\n".join(cur + [item])) > cfg.list_max_tokens:
                drafts.append(_Draft(
                    text="\n".join(cur), content_type="list", blocks=[b],
                    section_path=b.section_path, heading_level=b.heading_level, allow_overlap=False,
                    extra_meta={**base_meta, "list_start_index": start_idx,
                                "list_end_index": idx - 1, "is_list_continuation": len(drafts) > 0},
                ))
                self._metrics.list_chunks_count += 1
                cur = []
                start_idx = idx
            cur.append(item)
        if cur:
            drafts.append(_Draft(
                text="\n".join(cur), content_type="list", blocks=[b],
                section_path=b.section_path, heading_level=b.heading_level, allow_overlap=False,
                extra_meta={**base_meta, "list_start_index": start_idx,
                            "list_end_index": len(items), "is_list_continuation": len(drafts) > 0},
            ))
        return drafts

    # -- figuras (§14) ------------------------------------------------------
    def _chunk_from_figure(self, b: DocumentBlock) -> _Draft:
        fig_id = b.block_id.replace("block", "figure")
        return _Draft(
            text=b.text, content_type="figure_caption", blocks=[b],
            section_path=b.section_path, heading_level=b.heading_level, allow_overlap=False,
            extra_meta={
                "figure_id": fig_id,
                "figure_uri": b.metadata.get("image_uri"),
            },
        )

    # -- referências (§16) --------------------------------------------------
    def _chunks_from_references(self, blocks: list[DocumentBlock]) -> list[_Draft]:
        drafts = self._chunks_from_text_group(blocks, blocks[0].section_path, blocks[0].heading_level)
        for d in drafts:
            d.content_type = "references"
            d.allow_overlap = False
            d.extra_meta["is_reference_section"] = True
        return drafts

    # -- materialização + filtros ------------------------------------------
    def _apply_filters(self, draft: _Draft) -> tuple[bool, str, dict]:
        token_count = self._count(draft.text)
        outcome = self._filter(draft.text, draft.content_type, token_count, draft.section_path)
        return outcome.approved, outcome.reason, outcome.quality

    @staticmethod
    def _looks_complete(text: str) -> bool:
        """Heurística leve de completude semântica (§22): termina em pontuação final."""
        t = (text or "").rstrip()
        return bool(t) and t[-1] in ".!?…”\")"

    def _contextualize(self, text: str, document_title: Optional[str], section_path: list[str]) -> Optional[str]:
        cfg = self.cfg
        parts: list[str] = []
        if cfg.embed_document_title and document_title:
            parts.append(f"Documento: {document_title}")
        if cfg.embed_section_context and section_path:
            parts.append(f"Seção: {' > '.join(section_path)}")
        if not parts:
            return None
        return "\n\n".join(parts + [text])

    def _materialize(
        self, draft: _Draft, *, index: int, prev_id: Optional[str],
        document_id: str, document_title: Optional[str], document_checksum: str,
        item_uuid: str, bitstream_uuid: Optional[str], source_artifact_uri: str,
        quality: dict,
    ) -> StructuralChunk:
        cfg = self.cfg
        token_count = self._count(draft.text)
        # Barreira dura: nenhum chunk textual comum pode ultrapassar o limite absoluto.
        if draft.content_type not in ("table",) and token_count > cfg.force_split_above_tokens:
            raise ChunkSizeValidationError(
                f"chunk {index} ({draft.content_type}) tem {token_count} tokens "
                f"(> force_split_above_tokens={cfg.force_split_above_tokens})."
            )
        section_title = draft.section_path[-1] if draft.section_path else None
        contextualized = self._contextualize(draft.text, document_title, draft.section_path)
        embedding_token_count = self._count(contextualized) if contextualized else token_count

        # Inclui páginas de reconstrução cross-page (§7) além da página do bloco.
        pages = sorted({p for b in draft.blocks for p in block_pages(b)})
        printed_pages = sorted({p for b in draft.blocks for p in block_printed_pages(b)})

        # --- Classificação de recuperação (§21/§22) ---
        section_kind = next((b.section_kind for b in draft.blocks if b.section_kind), None)
        origin = draft.extra_meta.get("origin") or (
            draft.blocks[0].metadata.get("origin") if draft.blocks else None
        )
        rc = classify_retrieval(draft.content_type, section_kind, origin=origin)
        cross_page_merged = any(getattr(b, "cross_page_merged", False) for b in draft.blocks)
        quality_score = quality.get("quality_score") if quality else None
        semantic_completeness = (
            self._looks_complete(draft.text)
            if draft.content_type in ("paragraph", "references") else None
        )

        chunk_id = make_chunk_id(
            document_id=document_id, bitstream_uuid=bitstream_uuid,
            document_checksum=document_checksum, chunking_version=cfg.chunking_version,
            chunking_config_hash=self.config_hash, chunk_index=index, normalized_text=draft.text,
        )
        meta = {k: v for k, v in draft.extra_meta.items() if v is not None}
        if quality:
            meta["quality"] = quality
        return StructuralChunk(
            chunk_id=chunk_id, document_id=document_id, item_uuid=item_uuid,
            bitstream_uuid=bitstream_uuid, chunk_index=index, text=draft.text,
            contextualized_text=contextualized, token_count=token_count,
            embedding_token_count=embedding_token_count, character_count=len(draft.text),
            content_type=draft.content_type, block_ids=[b.block_id for b in draft.blocks],
            document_title=document_title, section_title=section_title,
            section_path=draft.section_path, heading_level=draft.heading_level,
            page_start=pages[0] if pages else None, page_end=pages[-1] if pages else None,
            page_numbers=pages,
            printed_page_start=printed_pages[0] if printed_pages else None,
            printed_page_end=printed_pages[-1] if printed_pages else None,
            printed_page_numbers=printed_pages,
            section_kind=section_kind,
            normalized_content_type=rc.normalized_content_type,
            retrieval_profile=rc.retrieval_profile,
            searchable_by_default=rc.searchable_by_default,
            ranking_weight=rc.ranking_weight,
            is_table=rc.is_table, is_chart=rc.is_chart,
            is_reference=rc.is_reference, is_appendix=rc.is_appendix,
            quality_score=quality_score,
            semantic_completeness=semantic_completeness,
            cross_page_merged=cross_page_merged,
            overlap_token_count=draft.overlap_token_count,
            overlap_source_chunk_id=prev_id if draft.overlap_token_count else None,
            split_reason=draft.split_reason, split_method=draft.split_method,
            source_artifact_uri=source_artifact_uri, document_checksum=document_checksum,
            chunking_strategy=self.strategy, chunking_version=cfg.chunking_version,
            chunking_config_hash=self.config_hash, tokenizer_name=self.tokens.name,
            metadata={**meta, "overlap_block_ids": draft.overlap_block_ids} if draft.overlap_block_ids else meta,
        )

    def _finalize_metrics(self, chunks: list[StructuralChunk]) -> None:
        m = self._metrics
        m.chunk_count = len(chunks)
        if chunks:
            toks = [c.token_count for c in chunks]
            m.average_tokens = round(sum(toks) / len(toks), 2)
            m.median_tokens = float(median(toks))
            m.min_tokens = min(toks)
            m.max_tokens = max(toks)
            m.overlap_tokens_total = sum(c.overlap_token_count for c in chunks)


# ---------------------------------------------------------------------------
# Adaptador do chunker legado à interface comum (§21).
# ---------------------------------------------------------------------------

class LegacyCharacterChunker:
    """Envelopa o `MinerUChunker` (por caracteres) na interface `DocumentChunker`.

    Reconstrói o content_list_v2 mínimo a partir dos `DocumentBlock` (para reaproveitar
    o chunker antigo) e converte a saída em `StructuralChunk` — permitindo comparação
    A/B pelo mesmo caminho de persistência/embedding."""

    strategy = "legacy_chars"

    def __init__(self, *, token_counter: Optional[TokenCounter] = None) -> None:
        from backend.indexing.chunks import MinerUChunker

        self._impl = MinerUChunker(
            max_chunk_chars=settings.LEGACY_CHUNK_SIZE,
            overlap_chars=settings.LEGACY_CHUNK_OVERLAP,
            min_chunk_chars=settings.LEGACY_MIN_CHUNK_CHARS,
        )
        self.tokens = token_counter or get_token_counter()
        self.config_hash = compute_config_hash({
            "max_chunk_chars": settings.LEGACY_CHUNK_SIZE,
            "overlap_chars": settings.LEGACY_CHUNK_OVERLAP,
            "min_chunk_chars": settings.LEGACY_MIN_CHUNK_CHARS,
            "version": "legacy-chars-v1",
        })

    def chunk(
        self, blocks: list[DocumentBlock], *, document_id: str,
        document_title: Optional[str] = None, document_checksum: str = "",
        item_uuid: str = "", bitstream_uuid: Optional[str] = None,
        source_artifact_uri: str = "", structure_source: str = "",
    ) -> ChunkingResult:
        content_list = _blocks_to_content_list(blocks)
        legacy_chunks = self._impl.process(content_list, doc_id=document_id)
        version = "legacy-chars-v1"
        chunks: list[StructuralChunk] = []
        for i, lc in enumerate(legacy_chunks):
            token_count = self.tokens.count(lc.content)
            page = lc.metadata.page
            chunk_id = make_chunk_id(
                document_id=document_id, bitstream_uuid=bitstream_uuid,
                document_checksum=document_checksum, chunking_version=version,
                chunking_config_hash=self.config_hash, chunk_index=i, normalized_text=lc.content,
            )
            chunks.append(StructuralChunk(
                chunk_id=chunk_id, document_id=document_id, item_uuid=item_uuid,
                bitstream_uuid=bitstream_uuid, chunk_index=i, text=lc.content,
                contextualized_text=None, token_count=token_count,
                embedding_token_count=token_count, character_count=len(lc.content),
                content_type=lc.metadata.type,
                block_ids=[], document_title=document_title,
                section_title=lc.metadata.section, section_path=[lc.metadata.section],
                page_start=page, page_end=page, page_numbers=[page] if page else [],
                source_artifact_uri=source_artifact_uri, document_checksum=document_checksum,
                chunking_strategy=self.strategy, chunking_version=version,
                chunking_config_hash=self.config_hash, tokenizer_name=self.tokens.name,
                metadata={"quality_score": lc.metadata.quality_score},
            ))
        metrics = ChunkingMetrics(structure_source=structure_source, chunk_count=len(chunks),
                                  document_block_count=len(blocks))
        if chunks:
            toks = [c.token_count for c in chunks]
            metrics.average_tokens = round(sum(toks) / len(toks), 2)
            metrics.median_tokens = float(median(toks))
            metrics.min_tokens = min(toks)
            metrics.max_tokens = max(toks)
        return ChunkingResult(
            document_id=document_id, chunks=chunks, metrics=metrics,
            chunking_strategy=self.strategy, chunking_version=version,
            chunking_config_hash=self.config_hash, tokenizer_name=self.tokens.name,
            structure_source=structure_source,
        )


def _blocks_to_content_list(blocks: list[DocumentBlock]) -> list[list[dict]]:
    """Reconstrói um content_list_v2 mínimo (por página) a partir dos DocumentBlocks,
    para o chunker legado (que consome o formato MinerU)."""
    pages: dict[int, list[dict]] = {}
    for b in blocks:
        page = b.page_number or 1
        pages.setdefault(page, [])
        if b.block_type == BLOCK_HEADING:
            pages[page].append({"type": "title",
                                "content": {"title_content": [{"type": "text", "content": b.text}],
                                            "level": b.heading_level or 1}})
        elif b.block_type == BLOCK_TABLE:
            pages[page].append({"type": "table",
                                "content": {"html": b.metadata.get("table_markdown", ""),
                                            "table_caption": [{"type": "text", "content": b.metadata.get("table_caption") or ""}]}})
        else:
            pages[page].append({"type": "paragraph",
                                "content": {"paragraph_content": [{"type": "text", "content": b.text}]}})
    return [pages[p] for p in sorted(pages)]
