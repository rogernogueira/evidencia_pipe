"""Validação de conteúdo visual: tabelas (§10), gráficos (§11), imagens (§12).

Funções PURAS e determinísticas (sem GPU, sem rede) usadas pelo parser/chunker:

  - `chart_data_confidence(caption, footnote, extracted)` — coerência do dado do gráfico
    (o caso UF-Brasil × países estrangeiros é o mais crítico, §11);
  - `table_quality_score(...)` — score §10.1 (HTML/cabeçalho/legenda/consistência/contexto);
  - `classify_image(...)` — decorative/logo/diagram/flowchart/figure (§12);
  - `mermaid_to_text(...)` — diagrama Mermaid → descrição textual recuperável (§12);
  - `table_row_representation(...)` — representação "por linhas" da tabela (§10.2).

O gate por LLM (§11/§12) é opcional e vive em `llm_visual_service` (desacoplado).
"""

from __future__ import annotations

import re

# --- Geografia (§11): estados/regiões do Brasil × países estrangeiros -------
_BR_STATES = (
    "acre", "alagoas", "amapá", "amazonas", "bahia", "ceará", "distrito federal",
    "espírito santo", "goiás", "maranhão", "mato grosso", "mato grosso do sul",
    "minas gerais", "pará", "paraíba", "paraná", "pernambuco", "piauí",
    "rio de janeiro", "rio grande do norte", "rio grande do sul", "rondônia",
    "roraima", "santa catarina", "são paulo", "sergipe", "tocantins",
)
_BR_GEO_HINTS = re.compile(
    r"unidade\s+federativa|\bUF\b|por\s+estado|por\s+regi[ãa]o|brasil|brasileir"
    r"|nordeste|sudeste|centro-oeste", re.IGNORECASE)
_BR_STATE_RE = re.compile("|".join(re.escape(s) for s in _BR_STATES), re.IGNORECASE)
_FOREIGN = (
    "colômbia", "colombia", "argentina", "chile", "peru", "venezuela", "bolívia",
    "bolivia", "equador", "uruguai", "paraguai", "méxico", "mexico", "estados unidos",
    "espanha", "portugal", "frança", "alemanha", "itália",
)
_FOREIGN_RE = re.compile(r"\b(" + "|".join(re.escape(f) for f in _FOREIGN) + r")\b", re.IGNORECASE)


def _caption_is_brazilian_geo(caption: str, footnote: str) -> bool:
    ctx = f"{caption} {footnote}"
    return bool(_BR_GEO_HINTS.search(ctx) or _BR_STATE_RE.search(ctx))


def chart_data_confidence(caption: str, footnote: str, extracted: str) -> tuple[float, str]:
    """Confiança de que o dado extraído do gráfico é coerente com a legenda (§11).

    Ordem de confiança: legenda > contexto > dados extraídos. O caso crítico: legenda
    fala de UF do Brasil e os rótulos extraídos citam países estrangeiros → 0.0.
    """
    caption = caption or ""
    extracted = extracted or ""
    # Caso crítico: incoerência geográfica.
    if _caption_is_brazilian_geo(caption, footnote):
        foreign = set(m.group(0).lower() for m in _FOREIGN_RE.finditer(extracted))
        if len(foreign) >= 2 and not _BR_STATE_RE.search(extracted):
            return 0.0, "geo_incoherent_foreign_countries"
    if not extracted.strip():
        return 0.5, "no_extracted_data"
    if not caption.strip():
        return 0.5, "no_caption"
    # Sobreposição léxica mínima entre legenda e dado extraído (sinal fraco de coerência).
    cap_tokens = set(re.findall(r"\w{4,}", caption.lower()))
    ext_tokens = set(re.findall(r"\w{4,}", extracted.lower()))
    if cap_tokens and ext_tokens:
        overlap = len(cap_tokens & ext_tokens) / len(cap_tokens)
        if overlap < 0.05 and _FOREIGN_RE.search(extracted):
            return 0.3, "low_overlap_with_caption"
    return 0.85, "coherent"


# --- Tabelas (§10.1) -------------------------------------------------------

_TR_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.IGNORECASE | re.DOTALL)
_CELL_RE = re.compile(r"<t[hd][^>]*>(.*?)</t[hd]>", re.IGNORECASE | re.DOTALL)


def _parse_html_rows(html: str) -> list[list[str]]:
    rows = []
    for tr in _TR_RE.findall(html or ""):
        cells = [re.sub(r"<[^>]+>", " ", c).strip() for c in _CELL_RE.findall(tr)]
        if cells:
            rows.append(cells)
    return rows


def _parse_md_rows(markdown: str) -> list[list[str]]:
    rows = []
    for ln in (markdown or "").splitlines():
        if "|" not in ln:
            continue
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if all(re.fullmatch(r"[\s\-:]*", c) for c in cells):   # linha separadora
            continue
        if cells:
            rows.append(cells)
    return rows


def table_quality_score(
    html: str, caption: str, footnote: str, section_kind: str | None = None,
    markdown: str = "",
) -> tuple[float, dict]:
    """Score de indexabilidade de uma tabela (§10.1), em [0,1].

    0.25 HTML válido + 0.20 cabeçalho + 0.15 legenda + 0.15 consistência de colunas
    + 0.15 contexto de seção + 0.10 fonte/nota explicativa. Usa o markdown como fonte
    de estrutura quando não há HTML (ex.: tabela vinda de um gráfico coerente)."""
    rows = _parse_html_rows(html or "")
    if len(rows) < 2 and markdown:
        rows = _parse_md_rows(markdown)
    ncols = max((len(r) for r in rows), default=0)
    html_valid = 1.0 if (len(rows) >= 2 and ncols >= 2) else 0.0
    has_header = 1.0 if (rows and any(not re.fullmatch(r"[\d.,%\s\-]*", c) for c in rows[0])) else 0.0
    has_caption = 1.0 if (caption or "").strip() else 0.0
    if rows:
        counts = [len(r) for r in rows]
        col_consistency = sum(1 for c in counts if c == ncols) / len(counts)
    else:
        col_consistency = 0.0
    # Seção informativa (corpo/apêndice) dá contexto; navegação/front-matter não.
    if section_kind is None:
        section_ctx = 0.5
    elif section_kind in ("body", "analytical_appendix", "administrative_appendix"):
        section_ctx = 1.0
    else:
        section_ctx = 0.0
    explanatory = 1.0 if (footnote or "").strip() else 0.0
    score = (0.25 * html_valid + 0.20 * has_header + 0.15 * has_caption
             + 0.15 * col_consistency + 0.15 * section_ctx + 0.10 * explanatory)
    return round(score, 3), {
        "html_valid": html_valid, "has_header": has_header, "has_caption": has_caption,
        "col_consistency": round(col_consistency, 2), "section_context": section_ctx,
        "explanatory": explanatory, "rows": len(rows), "cols": ncols,
    }


def table_row_representation(markdown: str, caption: str, max_rows: int = 40) -> str:
    """Representação "por linhas" de uma tabela markdown (§10.2), melhor para buscas
    factuais ("qual foi o subsídio em 2019?"). Cada linha vira "Cab: valor. …"."""
    lines = [ln for ln in (markdown or "").splitlines() if ln.strip() and "|" in ln]
    rows = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in lines]
    rows = [r for r in rows if not all(re.fullmatch(r"[\s\-:]*", c) for c in r)]  # tira separador
    if len(rows) < 2:
        return ""
    header = rows[0]
    out = []
    if caption:
        out.append(caption.strip())
    for r in rows[1:max_rows + 1]:
        pairs = [f"{header[i]}: {v}" for i, v in enumerate(r) if i < len(header) and v]
        if pairs:
            out.append(". ".join(pairs) + ".")
    return "\n".join(out).strip()


# --- Imagens/diagramas (§12) -----------------------------------------------

_MERMAID_LABEL_RE = re.compile(r'[\[({]"?([^"\]}\)|]{2,80}?)"?[\])}]')
_MERMAID_EDGE_RE = re.compile(r"-->|---|==>|-\.->")


def classify_image(caption: str, content: str, path: str) -> str:
    """Classifica a imagem (§12): flowchart | diagram | logo | figure | decorative."""
    c = (content or "").strip()
    low_all = f"{caption} {path}".lower()
    if c.startswith("```mermaid") or re.search(r"\b(graph|flowchart)\b", c[:40], re.IGNORECASE):
        return "flowchart" if _MERMAID_EDGE_RE.search(c) else "diagram"
    if re.search(r"\b(logotipo|logomarca|bras[ãa]o)\b|logo", low_all):
        return "logo"
    if not (caption or "").strip() and len(c) < 8:
        return "decorative"
    return "figure"


def mermaid_to_text(content: str) -> str:
    """Converte um diagrama Mermaid em descrição textual recuperável (§12).

    Extrai os rótulos dos nós e sinaliza que é um fluxo — evita embeddar código bruto."""
    if not content:
        return ""
    labels = []
    for m in _MERMAID_LABEL_RE.finditer(content):
        lbl = re.sub(r"\s+", " ", m.group(1).replace("\\n", " ")).strip()
        if lbl and lbl not in labels and not re.fullmatch(r"[A-Za-z]\d*", lbl):
            labels.append(lbl)
    if not labels:
        return ""
    return "Diagrama de fluxo relacionando: " + "; ".join(labels[:24]) + "."


def image_is_indexable(kind: str, caption: str, description: str) -> bool:
    """Uma imagem só é indexável com legenda + conteúdo recuperável, e não sendo
    logotipo/decorativa (§12)."""
    if kind in ("logo", "decorative"):
        return False
    has_caption = bool((caption or "").strip())
    has_desc = bool((description or "").strip())
    return has_caption and has_desc
