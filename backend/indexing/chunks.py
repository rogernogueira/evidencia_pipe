import json
import logging
import re
import sys
from pathlib import Path
from typing import Optional

from markdownify import markdownify as md_convert

from backend.core.schemas import DocumentChunk, ChunkMetadata
from backend.indexing.chunk_quality_filters import apply_all_filters, ChunkQualityVerdict

log = logging.getLogger("mineru.chunker")


class MinerUChunker:
    def __init__(
        self,
        max_chunk_chars: int = 1200,
        overlap_chars: int = 250,
        min_chunk_chars: int = 300,
        min_chunk_tokens: int = 10,
        ocr_noise_threshold: float = 0.65,
        enable_llm_repair: bool = False,
        llm_model: str = "gemma4:latest",
    ):
        self.max_chunk_chars = max_chunk_chars
        self.overlap_chars = overlap_chars
        self.min_chunk_chars = min_chunk_chars
        self.min_chunk_tokens = min_chunk_tokens
        self.ocr_noise_threshold = ocr_noise_threshold
        self.enable_llm_repair = enable_llm_repair
        self.llm_model = llm_model
        self.ignore_types = {"page_header", "page_number", "footnote", "discarded_blocks"}
        self._rejected_count = 0
        self._rejection_reasons: dict[str, int] = {}
        self._repaired_count = 0

    def _clean_text(self, text: str) -> str:
        """Remove OCR noise: letras maiúsculas isoladas com espaços (ex: 'I N P C' → 'INPC').

        Não afeta palavras legítimas em caixa alta (ex: 'GESTÃO ESCOLAR' permanece intacto).
        """
        # Colapsa apenas sequências de letras ISOLADAS separadas por espaço:
        # "I N P C"  → "INPC"   (cada "palavra" tem 1 char)
        # "ESTRATÉGIAS PEDAGÓGICAS" → permanece (palavras com 2+ chars)
        text = re.sub(
            r"(?<!\w)([A-ZÀ-Ú])\s(?=[A-ZÀ-Ú](?:\s|$))",
            r"\1",
            text,
        )
        return re.sub(r"\s+", " ", text).strip()

    def _split_with_overlap(self, text: str) -> list[str]:
        """Divide texto longo em sub-chunks por sentença.

        O overlap garante continuidade semântica entre chunks consecutivos.
        """
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks: list[str] = []
        current_content = ""

        for sentence in sentences:
            fits = len(current_content) + len(sentence) <= self.max_chunk_chars
            if fits:
                current_content += (" " + sentence if current_content else sentence)
            else:
                if current_content:
                    chunks.append(current_content.strip())

                # Overlap: pega os últimos N chars e encontra uma palavra limpa
                overlap_text = current_content[-self.overlap_chars:]
                first_space = overlap_text.find(" ")
                # Evita truncar demais quando o primeiro espaço aparece cedo demais.
                if 0 < first_space < len(overlap_text) // 2:
                    overlap_text = overlap_text[first_space:].strip()

                current_content = (overlap_text + " " + sentence).strip()

        if current_content:
            chunks.append(current_content.strip())

        return chunks

    def _extract_content(self, block: dict) -> str:
        """Extrai texto limpo de um bloco MinerU content_list_v2."""
        b_type = block.get("type")
        content_data = block.get("content", {})

        if b_type == "paragraph":
            parts = content_data.get("paragraph_content", [])
            return "".join(p.get("content", "") for p in parts)

        if b_type == "title":
            parts = content_data.get("title_content", [])
            return "".join(p.get("content", "") for p in parts)

        if b_type == "table":
            html = content_data.get("html", "")
            caption = "".join(
                c.get("content", "") for c in content_data.get("table_caption", [])
            )
            # Converte HTML para Markdown preservando estrutura sem ruído de tags
            table_md = md_convert(html, heading_style="ATX").strip()
            return f"TABELA: {caption}\n{table_md}" if caption else f"TABELA:\n{table_md}"

        return ""

    def _apply_quality_filter(self, text: str, block_type: str) -> ChunkQualityVerdict:
        """Aplica filtros de qualidade ao texto bruto (ANTES da junção de seção)."""
        return apply_all_filters(
            raw_text=text,
            block_type=block_type,
            min_chars=self.min_chunk_chars,
            min_tokens=self.min_chunk_tokens,
            ocr_noise_threshold=self.ocr_noise_threshold,
        )

    def _track_rejection(self, reason: str, doc_id: str, text_preview: str) -> None:
        """Registra rejeição para métricas e log."""
        self._rejected_count += 1
        self._rejection_reasons[reason] = self._rejection_reasons.get(reason, 0) + 1
        log.debug(
            "Chunk rejeitado [%s] doc='%s': '%s...'",
            reason, doc_id, text_preview[:80],
        )

    @property
    def rejection_summary(self) -> dict:
        """Retorna resumo de rejeições para diagnóstico."""
        return {
            "total_rejected": self._rejected_count,
            "by_reason": dict(self._rejection_reasons),
            "total_repaired": self._repaired_count,
        }

    def _maybe_repair(self, text: str, qm) -> str:
        """Reparo de OCR via LLM local (Ollama) foi removido neste fork —
        fora do escopo do pipeline de ingestão. Retorna o texto inalterado.

        Os parâmetros `enable_llm_repair`/`llm_model` são mantidos no construtor
        apenas por compatibilidade de assinatura com `index_chunks.index_all`.
        """
        return text

    def process(self, content_list_v2: list, doc_id: str = "") -> list[DocumentChunk]:
        """Processa o content_list_v2 e retorna chunks semânticos anotados.

        Filtros de qualidade são aplicados ao texto bruto ANTES da junção
        do header de seção, evitando que chunks ruins sejam indexados.

        Args:
            content_list_v2: Lista de páginas, cada uma com lista de blocos.
            doc_id: Identificador do documento (ex: nome do arquivo MD).

        Returns:
            Lista de objetos DocumentChunk tipados (apenas os aprovados).
        """
        final_chunks: list[DocumentChunk] = []
        current_section = "Introdução"
        self._rejected_count = 0
        self._rejection_reasons = {}
        self._repaired_count = 0

        for page_idx, page_blocks in enumerate(content_list_v2):
            page_num = page_idx + 1

            for block in page_blocks:
                b_type = block.get("type")
                if b_type in self.ignore_types:
                    continue

                raw_content = self._extract_content(block)
                if not raw_content:
                    continue

                # Títulos atualizam a seção global mas não viram chunks
                if b_type == "title":
                    current_section = self._clean_text(raw_content)
                    continue

                clean_content = self._clean_text(raw_content)

                # ── FILTRO DE QUALIDADE (antes da junção da seção) ──
                verdict = self._apply_quality_filter(clean_content, b_type or "paragraph")
                if not verdict.approved:
                    self._track_rejection(verdict.reason, doc_id, clean_content)
                    continue

                qm = verdict.quality_metrics

                # Tabelas: preservadas íntegras (não fazem split)
                if b_type == "table":
                    repaired_content = self._maybe_repair(raw_content, qm)
                    final_chunks.append(DocumentChunk(
                        content=repaired_content,
                        metadata=ChunkMetadata(
                            page=page_num, section=current_section,
                            doc_id=doc_id + ".pdf", doc_name_md=doc_id, type="table",
                            quality_score=qm.quality_score,
                            token_count=qm.token_count,
                            ocr_noise_score=qm.ocr_noise_score,
                            is_heading_only=qm.is_heading_only,
                        ),
                    ))
                    continue

                # Parágrafos longos: split com overlap + header injetado
                if len(clean_content) > self.max_chunk_chars:
                    sub_chunks = self._split_with_overlap(clean_content)
                    for i, sub in enumerate(sub_chunks):
                        sub_verdict = self._apply_quality_filter(sub, "paragraph_part")
                        if not sub_verdict.approved:
                            self._track_rejection(sub_verdict.reason, doc_id, sub)
                            continue
                        sq = sub_verdict.quality_metrics
                        repaired_sub = self._maybe_repair(sub, sq)
                        final_chunks.append(DocumentChunk(
                            content=repaired_sub,
                            metadata=ChunkMetadata(
                                page=page_num, section=current_section,
                                doc_id=doc_id + ".pdf", doc_name_md=doc_id,
                                type="paragraph_part", part=i + 1,
                                quality_score=sq.quality_score,
                                token_count=sq.token_count,
                                ocr_noise_score=sq.ocr_noise_score,
                                is_heading_only=sq.is_heading_only,
                            ),
                        ))
                else:
                    # Parágrafo normal
                    repaired_content = self._maybe_repair(clean_content, qm)
                    final_chunks.append(DocumentChunk(
                        content=repaired_content,
                        metadata=ChunkMetadata(
                            page=page_num, section=current_section,
                            doc_id=doc_id + ".pdf", doc_name_md=doc_id, type="paragraph",
                            quality_score=qm.quality_score,
                            token_count=qm.token_count,
                            ocr_noise_score=qm.ocr_noise_score,
                            is_heading_only=qm.is_heading_only,
                        ),
                    ))

        if self._rejected_count > 0:
            log.info(
                "Filtros de qualidade [%s]: %d chunk(s) rejeitado(s) — %s",
                doc_id, self._rejected_count, self._rejection_reasons,
            )

        return final_chunks


# ---------------------------------------------------------------------------
# Ponto de entrada / camada de compatibilidade (§5/§21).
#
# `chunks.py` permanece o módulo importado pelo pipeline. `get_chunker()` seleciona
# a estratégia (structural_tokens por padrão, legacy_chars para comparação) e devolve
# um objeto que implementa a interface comum `DocumentChunker` (backend.indexing.
# chunk_models). O `MinerUChunker` acima continua disponível como implementação legada.
# ---------------------------------------------------------------------------

def get_chunker(strategy: Optional[str] = None):
    """Fábrica do chunker ativo. `strategy` sobrepõe CHUNKING_STRATEGY.

    Retorna um `DocumentChunker` (structural_tokens | legacy_chars). Importações
    tardias evitam carregar o tokenizer/transformers quando este módulo é usado só
    pelo caminho legado (MinerUChunker.process)."""
    from backend.core import config as settings

    strategy = (strategy or settings.CHUNKING_STRATEGY).strip().lower()
    if strategy == "legacy_chars":
        from backend.indexing.structural_token_chunker import LegacyCharacterChunker

        return LegacyCharacterChunker()
    if strategy == "structural_tokens":
        from backend.indexing.structural_token_chunker import StructuralTokenChunker

        return StructuralTokenChunker()
    raise ValueError(f"CHUNKING_STRATEGY desconhecida: {strategy!r}")


# ---------------------------------------------------------------------------
# Uso standalone: python -m backend.indexing.chunks <path_to_content_list_v2.json>
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    default_path = (
        "output/despesas-educacao-irpf-relatorio-de-avaliacao/hybrid_auto"
        "/despesas-educacao-irpf-relatorio-de-avaliacao_content_list_v2.json"
    )
    json_path = Path(sys.argv[1] if len(sys.argv) > 1 else default_path)

    if not json_path.exists():
        print(f"Arquivo não encontrado: {json_path}")
        print("Uso: python -m backend.indexing.chunks <caminho_para_content_list_v2.json>")
        sys.exit(1)

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    chunker = MinerUChunker()
    chunks = chunker.process(data, doc_id=json_path.stem)

    print(f"Gerados {len(chunks)} chunks semânticos de '{json_path.name}'")
    print(f"  Tipos: { {c.metadata.type for c in chunks} }")
    print(f"  Seções: {len({c.metadata.section for c in chunks})} únicas")
