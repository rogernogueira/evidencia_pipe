"""Normalização do documento MinerU em uma sequência ordenada de `DocumentBlock`.

Prioridade de parsing (§7):
    1. content_list_v2.json  → structure_source="mineru_json"
    2. Markdown estruturado  → structure_source="markdown"
    3. texto simples         → structure_source="plain_text"

O JSON do MinerU (verificado em amostras reais deste repositório) é uma LISTA de
páginas; cada página é uma lista de blocos `{type, content, bbox, sub_type?}`:

    title             content.title_content[], content.level
    paragraph         content.paragraph_content[]
    list              content.list_type, content.list_items[].item_content[]
    table             content.html, content.table_caption[], content.image_source.path
    chart             content.content (md/texto), content.chart_caption[], image_source
    image             content.image_source.path, content.content
    equation_interline content.math_content, content.math_type
    page_header/page_footer/page_number/page_footnote  content.<type>_content[]

Inline: itens `{type:"text"|"equation_inline", content}`.

Este módulo NÃO usa GPU. Cabeçalhos/rodapés repetidos e números de página isolados
são detectados e removidos de forma rastreável (métricas), conforme §15.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

from backend.indexing.chunk_models import (
    BLOCK_FIGURE_CAPTION,
    BLOCK_FOOTER,
    BLOCK_FOOTNOTE,
    BLOCK_FORMULA,
    BLOCK_HEADER,
    BLOCK_HEADING,
    BLOCK_LIST,
    BLOCK_PAGE_NUMBER,
    BLOCK_PARAGRAPH,
    BLOCK_REFERENCE,
    BLOCK_TABLE,
    BLOCK_UNKNOWN,
    SECTION_ACRONYM_LIST,
    SECTION_BIBLIOGRAPHY,
    DocumentBlock,
    DocumentStructureParseError,
)
from backend.indexing.cross_page_reconstruction import reconstruct_cross_page
from backend.indexing.section_classifier import SectionStateMachine, parse_acronym_line
from backend.indexing.text_normalization import normalize_text as _normalize
from backend.indexing.visual_validation import (
    chart_data_confidence,
    classify_image,
    image_is_indexable,
    mermaid_to_text,
)

# Header/footer é considerado "repetido" quando o MESMO texto normalizado aparece
# em pelo menos este número de páginas (ou fração das páginas, o que for menor).
_REPEAT_MIN_PAGES = 3
_REPEAT_MIN_FRACTION = 0.5

# Tipos cuja estrutura NÃO deve ser normalizada como prosa (§8): a quebra/marcação
# é significativa (listas, tabelas em markdown, legendas de figura, fórmula LaTeX).
_NO_NORMALIZE_TYPES = frozenset({BLOCK_LIST, BLOCK_TABLE, BLOCK_FIGURE_CAPTION, BLOCK_FORMULA})

# Operadores/comandos LaTeX que indicam uma equação "de verdade" (§9).
_MATH_OPERATORS_RE = re.compile(
    r"[=<>+×·]|\\(frac|sum|int|prod|sqrt|underbrace|overbrace|times|cdot|le|ge|neq|approx|"
    r"partial|nabla|alpha|beta|lambda|sigma|mu|log|ln|exp|min|max)\b"
)
# Padrão de ordinal/símbolo mal lido como equação (§9): "3^{o}", "5 \textdegree", "_{3^0}".
_ORDINAL_LIKE_RE = re.compile(r"^\s*[_^]?\s*\{?\s*\d{1,3}\s*[\^_]?\s*\{?\s*[oaºª°0]\s*\}?\s*\}?\s*$")


def _norm_ws(text: str) -> str:
    return " ".join((text or "").split())


def _inline_to_text(items: Any) -> str:
    """Concatena uma lista de itens inline `{type, content}` em texto plano.

    equation_inline é preservado como `$latex$` (o BGE-M3 lida com isso melhor do
    que descartar); text é usado como está."""
    if not isinstance(items, list):
        return ""
    parts: list[str] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        content = it.get("content", "")
        if not isinstance(content, str):
            continue
        if it.get("type") == "equation_inline":
            content = content.strip()
            parts.append(f"${content}$" if content else "")
        else:
            parts.append(content)
    return "".join(parts).strip()


class MinerUDocumentParser:
    """Converte o content_list_v2.json (ou markdown) em `list[DocumentBlock]`.

    Mantém a ordem original, a página, a hierarquia de títulos (`section_path`),
    a origem do bloco (`source_reference`) e as coordenadas (`bbox`) quando houver.
    """

    def __init__(
        self,
        *,
        remove_repeated_headers: bool = True,
        remove_repeated_footers: bool = True,
        keep_footnotes: bool = True,
        images_base_uri: str = "",
        normalize_text: bool = False,
        reconstruct_cross_page: bool = False,
        chart_mode: str = "raw",
        image_mode: str = "always",
        chart_min_confidence: float = 0.8,
        visual_llm: bool = False,
    ) -> None:
        self.remove_repeated_headers = remove_repeated_headers
        self.remove_repeated_footers = remove_repeated_footers
        self.keep_footnotes = keep_footnotes
        self.images_base_uri = images_base_uri.rstrip("/")
        # Política v2: normalização de texto (§8) e reconstrução cross-page (§7).
        self.normalize_text = normalize_text
        self.reconstruct_cross_page = reconstruct_cross_page
        # Política v2: validação de gráfico (§11) e imagem (§12).
        self.chart_mode = chart_mode
        self.image_mode = image_mode
        self.chart_min_confidence = chart_min_confidence
        self.visual_llm = visual_llm
        # Métricas preenchidas durante o parse (lidas pelo chunker).
        self.removed_header_blocks = 0
        self.removed_footer_blocks = 0
        self.removed_page_number_blocks = 0
        self.cross_page_merges = 0
        self.document_title: Optional[str] = None
        self.warnings: list[str] = []
        # Dicionário de siglas (§5) — expansão de consulta; NÃO gera chunks.
        self.acronyms: dict[str, str] = {}

    # -- API pública --------------------------------------------------------

    def parse_json_file(self, path: Path) -> tuple[list[DocumentBlock], str]:
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise DocumentStructureParseError(f"content_list inválido em '{path}': {exc}") from exc
        return self.parse_json(data)

    def parse_json(self, data: Any) -> tuple[list[DocumentBlock], str]:
        """Parseia a estrutura MinerU. Retorna (blocos, structure_source)."""
        if not isinstance(data, list):
            raise DocumentStructureParseError("content_list MinerU deve ser uma lista.")
        # Normaliza para lista-de-páginas: se vier lista plana de blocos, trata como 1 página.
        pages = data if (data and isinstance(data[0], list)) else [data]

        raw = self._flatten(pages)
        raw = self._drop_repeated_furniture(raw)
        blocks = self._assign_sections(raw)
        if self.reconstruct_cross_page:
            blocks, self.cross_page_merges = reconstruct_cross_page(blocks)
        return blocks, "mineru_json"

    def parse_markdown(self, markdown: str) -> tuple[list[DocumentBlock], str]:
        """Fallback (§7): parseia markdown estruturado (headings/parágrafos/listas/tabelas)."""
        raw = self._parse_markdown_raw(markdown)
        blocks = self._assign_sections(raw)
        return blocks, "markdown"

    def parse_plain_text(self, text: str) -> tuple[list[DocumentBlock], str]:
        """Último recurso (§7): parágrafos separados por linha em branco."""
        raw: list[dict] = []
        for i, para in enumerate(re.split(r"\n\s*\n", text or "")):
            para = para.strip()
            if para:
                raw.append({"type": BLOCK_PARAGRAPH, "text": para, "page": None,
                            "level": None, "source": "plain_text"})
        blocks = self._assign_sections(raw)
        return blocks, "plain_text"

    # -- extração de blocos MinerU -----------------------------------------

    def _flatten(self, pages: list) -> list[dict]:
        """Achata páginas → blocos "crus" (dicts intermediários), preservando ordem/página.

        Também captura o número de página IMPRESSO (bloco `page_number`) por página (§2)
        e o associa a todos os blocos daquela página."""
        out: list[dict] = []
        printed_by_page: dict[int, int] = {}
        for page_idx, page_blocks in enumerate(pages):
            page_num = page_idx + 1
            if not isinstance(page_blocks, list):
                continue
            for block in page_blocks:
                if not isinstance(block, dict):
                    continue
                # Captura o nº impresso antes de qualquer descarte (§2).
                if block.get("type") == "page_number":
                    printed = self._printed_page_number(block)
                    if printed is not None:
                        printed_by_page[page_num] = printed
                parsed = self._parse_block(block, page_num, page_idx)
                if parsed is not None:
                    out.append(parsed)
        # Associa o nº impresso a cada bloco da página.
        for b in out:
            b["printed_page_number"] = printed_by_page.get(b.get("page"))
        return out

    @staticmethod
    def _printed_page_number(block: dict) -> Optional[int]:
        """Extrai o número impresso de um bloco page_number (§2). Apenas arábico."""
        text = _inline_to_text(((block.get("content") or {}).get("page_number_content", [])))
        m = re.search(r"\d+", text or "")
        return int(m.group()) if m else None

    def _parse_block(self, block: dict, page_num: int, page_index: int) -> Optional[dict]:
        b_type = block.get("type")
        content = block.get("content")
        bbox = block.get("bbox")
        base = {"page": page_num, "page_index": page_index, "bbox": bbox,
                "level": None, "source": f"mineru:{b_type}"}

        if b_type == "title":
            text = _inline_to_text((content or {}).get("title_content", []))
            level = (content or {}).get("level")
            if not text:
                return None
            if self.document_title is None:
                self.document_title = text
            return {**base, "type": BLOCK_HEADING, "text": text,
                    "level": int(level) if isinstance(level, int) else 1}

        if b_type == "paragraph":
            text = _inline_to_text((content or {}).get("paragraph_content", []))
            return {**base, "type": BLOCK_PARAGRAPH, "text": text} if text else None

        if b_type == "list":
            return self._parse_list(content, base)

        if b_type == "table":
            return self._parse_table(content, base)

        if b_type == "chart":
            return self._parse_chart(content, base)

        if b_type == "image":
            return self._parse_image(content, base)

        if b_type == "equation_interline":
            math = (content or {}).get("math_content", "")
            math = math.strip() if isinstance(math, str) else ""
            if not math:
                return None
            return {**base, "type": BLOCK_FORMULA, "text": f"$$ {math} $$",
                    "meta": {"equation_confidence": self._equation_confidence(math)}}

        if b_type == "page_header":
            text = _inline_to_text((content or {}).get("page_header_content", []))
            return {**base, "type": BLOCK_HEADER, "text": text} if text else None

        if b_type == "page_footer":
            text = _inline_to_text((content or {}).get("page_footer_content", []))
            return {**base, "type": BLOCK_FOOTER, "text": text} if text else None

        if b_type == "page_number":
            text = _inline_to_text((content or {}).get("page_number_content", []))
            return {**base, "type": BLOCK_PAGE_NUMBER, "text": text} if text else None

        if b_type == "page_footnote":
            text = _inline_to_text((content or {}).get("page_footnote_content", []))
            return {**base, "type": BLOCK_FOOTNOTE, "text": text} if text else None

        # Tipo desconhecido: tenta extrair algum texto plano para não perder conteúdo.
        text = self._best_effort_text(content)
        return {**base, "type": BLOCK_UNKNOWN, "text": text} if text else None

    def _parse_list(self, content: Any, base: dict) -> Optional[dict]:
        content = content or {}
        items = content.get("list_items", [])
        list_type = content.get("list_type", "text_list")
        ordered = "order" in str(list_type).lower() or list_type == "ordered_list"
        lines: list[str] = []
        for i, item in enumerate(items, 1):
            if not isinstance(item, dict):
                continue
            item_text = _inline_to_text(item.get("item_content", []))
            if not item_text:
                continue
            marker = f"{i}." if ordered else "-"
            lines.append(f"{marker} {item_text}")
        if not lines:
            return None
        return {
            **base, "type": BLOCK_LIST, "text": "\n".join(lines),
            "meta": {
                "list_type": "ordered" if ordered else "unordered",
                "list_start_index": 1,
                "list_end_index": len(lines),
                "item_count": len(lines),
            },
        }

    def _parse_table(self, content: Any, base: dict) -> Optional[dict]:
        content = content or {}
        html = content.get("html", "")
        caption = _inline_to_text(content.get("table_caption", []))
        footnote = _inline_to_text(content.get("table_footnote", []))
        table_md = self._html_table_to_markdown(html)
        parts = []
        if caption:
            parts.append(caption)
        if table_md:
            parts.append(table_md)
        if footnote:
            parts.append(f"Fonte: {footnote}")
        text = "\n".join(parts).strip()
        if not text:
            return None
        img_path = ((content.get("image_source") or {}).get("path") or "")
        return {
            **base, "type": BLOCK_TABLE, "text": text,
            "meta": {
                "table_caption": caption or None,
                "table_type": content.get("table_type"),
                "table_markdown": table_md,
                "table_html": html or None,          # §10.1: fonte do quality score
                "table_footnote": footnote or None,
                "image_uri": self._image_uri(img_path),
            },
        }

    def _parse_chart(self, content: Any, base: dict) -> Optional[dict]:
        content = content or {}
        caption = _inline_to_text(content.get("chart_caption", []))
        footnote = _inline_to_text(content.get("chart_footnote", []))
        body = content.get("content", "")
        body = body.strip() if isinstance(body, str) else ""
        looks_tabular = "|" in body and "---" in body
        img_path = ((content.get("image_source") or {}).get("path") or "")

        # §11: confiança do dado extraído vs legenda/contexto. No modo caption_context,
        # dado abaixo do limiar NÃO entra no embedding — só legenda + fonte.
        confidence, reason = chart_data_confidence(caption, footnote, body)
        # Gate de LLM opcional (§11): refina a confiança. No-op se desligado/sem chave.
        if self.visual_llm and body:
            try:
                from backend.services.llm_visual_service import judge_chart
                verdict = judge_chart(caption, footnote, body)
                if verdict is not None:
                    confidence, reason = verdict[0], f"llm:{verdict[1]}"
            except Exception:  # noqa: BLE001
                pass
        trust_data = self.chart_mode != "caption_context" or confidence >= self.chart_min_confidence

        meta = {
            "table_caption": caption or None,
            "figure_caption": caption or None,
            "image_uri": self._image_uri(img_path),
            "origin": "chart",
            "chart_data_confidence": confidence,
            "chart_confidence_reason": reason,
        }
        if looks_tabular and trust_data:
            meta["table_markdown"] = body
            parts = [p for p in (caption, body, (f"Fonte: {footnote}" if footnote else "")) if p]
            return {**base, "type": BLOCK_TABLE, "text": "\n".join(parts).strip(), "meta": meta}
        # Gráfico não confiável (ou não tabular): embedda só legenda + fonte (§11).
        parts = [p for p in (caption, (f"Fonte: {footnote}" if footnote else "")) if p]
        text = "\n".join(parts).strip()
        if not text:
            return None
        return {**base, "type": BLOCK_FIGURE_CAPTION, "text": text, "meta": meta}

    def _parse_image(self, content: Any, base: dict) -> Optional[dict]:
        content = content or {}
        raw_content = content.get("content", "")
        raw_content = raw_content.strip() if isinstance(raw_content, str) else ""
        caption = _inline_to_text(content.get("image_caption", []))
        footnote = _inline_to_text(content.get("image_footnote", []))
        img_path = ((content.get("image_source") or {}).get("path") or "")

        # §12: classifica a imagem e converte diagrama Mermaid em descrição textual.
        kind = classify_image(caption, raw_content, img_path)
        if kind in ("flowchart", "diagram") and "mermaid" in raw_content.lower():
            description = mermaid_to_text(raw_content)
        elif kind in ("flowchart", "diagram"):
            description = mermaid_to_text(raw_content) or raw_content
        else:
            description = raw_content

        # Texto embeddável (§12/§20): legenda + descrição (nunca URI/código bruto).
        parts = [p for p in (caption, description, (f"Fonte: {footnote}" if footnote else "")) if p]
        text = "\n".join(parts).strip() or (caption or Path(img_path).name if img_path else "")
        if not text:
            return None

        meta = {"image_uri": self._image_uri(img_path), "origin": "image",
                "image_kind": kind, "figure_caption": caption or None}
        out = {**base, "type": BLOCK_FIGURE_CAPTION, "text": text, "meta": meta}
        # §12: no modo conditional, imagem sem legenda/conteúdo útil (logo, decorativa)
        # é preservada mas NÃO chunkável.
        if self.image_mode == "conditional" and not image_is_indexable(kind, caption, description):
            out["chunkable"] = False
        return out

    def _image_uri(self, path: str) -> Optional[str]:
        if not path:
            return None
        if self.images_base_uri:
            return f"{self.images_base_uri}/{Path(path).name}"
        return path

    @staticmethod
    def _equation_confidence(latex: str) -> float:
        """Confiança de que o LaTeX é uma equação real, não um ordinal/símbolo (§9).

        Baixa (<0.4) quando: parece ordinal, é muito curto ou não tem operadores
        matemáticos. Alta (~0.9) quando há operadores/comandos LaTeX relevantes."""
        s = (latex or "").strip()
        if not s:
            return 0.0
        # ordinal/símbolo mal reconhecido (ex.: "3^{o}", "5 \textdegree").
        if _ORDINAL_LIKE_RE.match(s) or "textdegree" in s:
            return 0.25
        n_ops = len(_MATH_OPERATORS_RE.findall(s))
        if n_ops >= 1 and len(s) >= 6:
            return 0.9
        if len(re.sub(r"[\s{}\\]", "", s)) < 5:  # pouquíssimo conteúdo matemático
            return 0.3
        return 0.6

    @staticmethod
    def _best_effort_text(content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, dict):
            for key in ("content", "text"):
                v = content.get(key)
                if isinstance(v, str) and v.strip():
                    return v.strip()
            # tenta qualquer *_content lista de inline
            for k, v in content.items():
                if k.endswith("_content") and isinstance(v, list):
                    t = _inline_to_text(v)
                    if t:
                        return t
        return ""

    @staticmethod
    def _html_table_to_markdown(html: str) -> str:
        if not html:
            return ""
        try:
            from markdownify import markdownify as md_convert

            return md_convert(html, heading_style="ATX").strip()
        except Exception:
            # Sem markdownify (não deveria acontecer — é dependência): remove tags.
            return _norm_ws(re.sub(r"<[^>]+>", " ", html))

    # -- remoção de header/footer/page_number repetidos (§15) --------------

    def _drop_repeated_furniture(self, raw: list[dict]) -> list[dict]:
        total_pages = len({b["page"] for b in raw if b.get("page")}) or 1
        threshold = min(_REPEAT_MIN_PAGES, max(2, int(total_pages * _REPEAT_MIN_FRACTION)))

        def repeated_keys(block_type: str) -> set[str]:
            # conta em quantas PÁGINAS distintas cada texto normalizado aparece
            per_key_pages: dict[str, set[int]] = {}
            for b in raw:
                if b["type"] == block_type:
                    key = _norm_ws(b["text"]).lower()
                    per_key_pages.setdefault(key, set()).add(b.get("page") or 0)
            return {k for k, pages in per_key_pages.items() if len(pages) >= threshold}

        rep_headers = repeated_keys(BLOCK_HEADER) if self.remove_repeated_headers else set()
        rep_footers = repeated_keys(BLOCK_FOOTER) if self.remove_repeated_footers else set()

        kept: list[dict] = []
        for b in raw:
            t = b["type"]
            key = _norm_ws(b["text"]).lower()
            # Números de página isolados são sempre ruído estrutural (§15).
            if t == BLOCK_PAGE_NUMBER:
                self.removed_page_number_blocks += 1
                continue
            # Cabeçalhos/rodapés RECORRENTES: removidos (rastreado). Os não recorrentes
            # são mantidos como blocos-furniture (o chunker só consome BODY_BLOCK_TYPES,
            # então não poluem os chunks, mas permanecem auditáveis no parse).
            if t == BLOCK_HEADER and key in rep_headers:
                self.removed_header_blocks += 1
                continue
            if t == BLOCK_FOOTER and key in rep_footers:
                self.removed_footer_blocks += 1
                continue
            if t == BLOCK_FOOTNOTE and not self.keep_footnotes:
                continue
            kept.append(b)
        return kept

    # -- atribuição de seção + materialização em DocumentBlock -------------

    def _assign_sections(self, raw: list[dict]) -> list[DocumentBlock]:
        """Atribui `section_path` (hierarquia por numeração, §4) e `section_kind`
        (máquina de estados, §4) a cada bloco. Também alimenta o dicionário de siglas
        (§5) e converte parágrafos/listas em referências dentro da bibliografia (§15)."""
        blocks: list[DocumentBlock] = []
        stack: list[tuple[int, str]] = []  # (heading_level, title)
        sm = SectionStateMachine()

        for order_index, b in enumerate(raw):
            b_type = b["type"]
            text = (b.get("text") or "").strip()
            if not text:
                continue

            if b_type == BLOCK_HEADING:
                level, kind = sm.feed_heading(text, b.get("level"))
                # atualiza a pilha hierárquica pelo nível efetivo (numeração > MinerU)
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, text))
                section_path = [t for _, t in stack]
                blocks.append(self._mk_block(order_index, BLOCK_HEADING, text, b,
                                             section_path, kind, heading_level=level))
                continue

            kind = sm.current
            effective_type = b_type
            if kind == SECTION_BIBLIOGRAPHY and b_type in (BLOCK_PARAGRAPH, BLOCK_LIST, BLOCK_FOOTNOTE):
                effective_type = BLOCK_REFERENCE

            # Dicionário de siglas (§5): coleta pares de listas/parágrafos da seção.
            if kind == SECTION_ACRONYM_LIST and b_type in (BLOCK_LIST, BLOCK_PARAGRAPH):
                self._collect_acronyms(text)

            section_path = [t for _, t in stack]
            blocks.append(self._mk_block(order_index, effective_type, text, b, section_path, kind))
        return blocks

    def _collect_acronyms(self, text: str) -> None:
        for line in text.splitlines():
            pair = parse_acronym_line(line)
            if pair:
                sigla, expansao = pair
                self.acronyms.setdefault(sigla, expansao)

    def _mk_block(
        self, order_index: int, block_type: str, text: str, raw: dict,
        section_path: list[str], section_kind: Optional[str] = None,
        *, heading_level: Optional[int] = None,
    ) -> DocumentBlock:
        # §6: cabeçalho/rodapé/nº de página são preservados, mas não chunkáveis.
        is_furniture = block_type in (BLOCK_HEADER, BLOCK_FOOTER, BLOCK_PAGE_NUMBER)
        meta = raw.get("meta", {}) or {}
        # Override de chunkabilidade (§12: imagem logo/decorativa em modo conditional).
        chunkable = raw.get("chunkable", not is_furniture)
        # §8: raw_text = extraído; text = normalizado (só para prosa; lista/tabela/figura/
        # fórmula mantêm a estrutura). Preserva números/datas/moedas.
        normalized = text
        if self.normalize_text and block_type not in _NO_NORMALIZE_TYPES:
            normalized = _normalize(text) or text
        return DocumentBlock(
            block_id=f"block-{order_index:04d}",
            block_type=block_type,
            text=normalized,
            raw_text=text,
            equation_confidence=meta.get("equation_confidence"),
            chart_data_confidence=meta.get("chart_data_confidence"),
            order_index=order_index,
            page_number=raw.get("page"),
            page_index=raw.get("page_index"),
            printed_page_number=raw.get("printed_page_number"),
            heading_level=heading_level,
            section_path=section_path,
            section_kind=section_kind,
            normalized_type="page_furniture" if is_furniture else None,
            preserve=True,
            chunkable=chunkable,
            embeddable=chunkable,
            indexable=chunkable,
            bbox=[float(x) for x in raw["bbox"]] if isinstance(raw.get("bbox"), list) else None,
            source_reference=raw.get("source"),
            metadata=meta,
        )

    # -- markdown fallback --------------------------------------------------

    def _parse_markdown_raw(self, markdown: str) -> list[dict]:
        raw: list[dict] = []
        lines = (markdown or "").splitlines()
        i = 0
        n = len(lines)
        para: list[str] = []
        list_buf: list[str] = []

        def flush_para():
            if para:
                text = _norm_ws(" ".join(para))
                if text:
                    raw.append({"type": BLOCK_PARAGRAPH, "text": text, "page": None,
                                "level": None, "source": "markdown", "bbox": None})
                para.clear()

        def flush_list():
            if list_buf:
                raw.append({"type": BLOCK_LIST, "text": "\n".join(list_buf), "page": None,
                            "level": None, "source": "markdown", "bbox": None,
                            "meta": {"list_type": "unordered", "list_start_index": 1,
                                     "list_end_index": len(list_buf), "item_count": len(list_buf)}})
                list_buf.clear()

        while i < n:
            line = lines[i].rstrip()
            stripped = line.strip()
            heading = re.match(r"^(#{1,6})\s+(.*)$", stripped)
            table_row = "|" in stripped and stripped.count("|") >= 2
            list_item = re.match(r"^([-*+]|\d+[.)])\s+(.+)$", stripped)

            if heading:
                flush_para()
                flush_list()
                level = len(heading.group(1))
                raw.append({"type": BLOCK_HEADING, "text": heading.group(2).strip(),
                            "page": None, "level": level, "source": "markdown", "bbox": None})
                i += 1
                continue

            if table_row:
                flush_para()
                flush_list()
                tbl: list[str] = []
                while i < n and "|" in lines[i]:
                    tbl.append(lines[i].strip())
                    i += 1
                raw.append({"type": BLOCK_TABLE, "text": "\n".join(tbl), "page": None,
                            "level": None, "source": "markdown", "bbox": None,
                            "meta": {"table_markdown": "\n".join(tbl)}})
                continue

            if list_item:
                flush_para()
                list_buf.append(f"- {list_item.group(2).strip()}")
                i += 1
                continue

            if not stripped:
                flush_para()
                flush_list()
                i += 1
                continue

            flush_list()
            para.append(stripped)
            i += 1

        flush_para()
        flush_list()
        # define document_title = primeiro heading, se houver
        if self.document_title is None:
            for b in raw:
                if b["type"] == BLOCK_HEADING:
                    self.document_title = b["text"]
                    break
        return raw
