"""Filtros de qualidade de chunks — aplicados ANTES da junção da seção ao chunk.

Cada filtro recebe o texto bruto (sem header de seção) e o tipo do bloco,
retornando um ChunkQualityVerdict que decide se o chunk deve ser mantido.

Os filtros são composicionais: podem ser combinados via apply_all_filters().
O payload do Qdrant é enriquecido com métricas de qualidade para análise.

Uso:
    from backend.indexing.chunk_quality_filters import apply_all_filters

    verdict = apply_all_filters(raw_text, block_type="paragraph")
    if not verdict.approved:
        continue  # descarta chunk ruim
    # usa verdict.quality_metrics para enriquecer o payload
"""

import re
import unicodedata
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Resultado de um filtro
# ---------------------------------------------------------------------------

@dataclass
class QualityMetrics:
    """Métricas de qualidade anexadas ao payload do Qdrant."""
    quality_score: float = 1.0
    token_count: int = 0
    ocr_noise_score: float = 0.0
    is_heading_only: bool = False
    char_count: int = 0
    rejection_reason: str = ""
    strange_words: list[str] = field(default_factory=list)
    broken_words: list[str] = field(default_factory=list)
    needs_llm_repair: bool = False


@dataclass
class ChunkQualityVerdict:
    """Resultado da avaliação de qualidade de um chunk."""
    approved: bool = True
    reason: str = ""
    quality_metrics: QualityMetrics = field(default_factory=QualityMetrics)


# ---------------------------------------------------------------------------
# Constantes configuráveis
# ---------------------------------------------------------------------------

MIN_CHARS_DEFAULT = 300
SEMANTIC_COMPLETE_MIN_CHARS = 80
MIN_TOKENS_DEFAULT = 10
URL_PATTERN = re.compile(
    r"https?://[^\s<>\"']+|www\.[^\s<>\"']+",
    re.IGNORECASE,
)
OCR_NOISE_PATTERNS = [
    re.compile(r"[^\w\s,.;:!?()\-–—/\[\]{}@#$%&*+=<>°ºª€£¥]"),  # caracteres estranhos
    re.compile(r"(\w)\s(\w)\s(\w)\s(\w)"),                         # letras isoladas com espaços (OCR)
    re.compile(r"[|]{2,}"),                                        # pipes repetidos
    re.compile(r"_{3,}"),                                          # underlines repetidos
    re.compile(r"\.{4,}"),                                         # pontos excessivos
    re.compile(r"(\d\s){5,}"),                                     # dígitos espaçados (tabela OCR ruim)
]
HEADING_ONLY_MAX_CHARS = 120
TABLE_FRAGMENT_MIN_ROWS = 2
SENTENCE_END_PATTERN = re.compile(r"[.!?;:]\s*$")


# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------

def _count_tokens(text: str) -> int:
    """Contagem aproximada de tokens (split por espaços e pontuação)."""
    return len(re.findall(r"\S+", text))


def _calc_ocr_noise_score(text: str) -> float:
    """Score de ruído OCR entre 0.0 (limpo) e 1.0 (muito ruidoso).

    Avalia a proporção de caracteres anômalos e padrões de OCR defeituoso.
    """
    if not text:
        return 0.0

    total_chars = len(text)
    noise_char_count = 0

    for pattern in OCR_NOISE_PATTERNS:
        matches = pattern.findall(text)
        noise_char_count += sum(len(m) if isinstance(m, str) else len(m[0]) for m in matches)

    raw_ratio = noise_char_count / total_chars

    alpha_chars = sum(1 for c in text if c.isalpha())
    alpha_ratio = alpha_chars / total_chars if total_chars > 0 else 0.0

    non_printable = sum(1 for c in text if ord(c) > 127 and not c.isalpha())
    non_print_ratio = non_printable / total_chars

    score = (raw_ratio * 0.5) + ((1.0 - alpha_ratio) * 0.3) + (non_print_ratio * 0.2)
    return round(min(score, 1.0), 2)


def _is_semantically_complete(text: str) -> bool:
    """Verifica se um texto curto é semanticamente completo.

    Critérios: começa com maiúscula/número, termina com pontuação final,
    tem pelo menos 2 palavras e contém verbo implícito (heurística simplificada).
    """
    stripped = text.strip()
    if len(stripped) < 20:
        return False

    starts_well = bool(re.match(r"^[A-ZÀ-ÚÄ-Ü0-9\"'\(\[]", stripped))
    ends_well = bool(SENTENCE_END_PATTERN.search(stripped))
    word_count = _count_tokens(stripped)

    has_structure = word_count >= 5 and starts_well and ends_well
    return has_structure


def _calc_quality_score(
    text: str,
    ocr_noise: float,
    is_heading_only: bool,
    block_type: str,
) -> float:
    """Score composto de qualidade entre 0.0 (péssimo) e 1.0 (excelente).

    Combina múltiplos sinais para evitar que chunks ruins sejam bem ranqueados
    por embeddings genéricos ou combinação inadequada de score denso/sparse.
    """
    char_count = len(text)
    token_count = _count_tokens(text)

    # --- Penalidades ---
    penalties = 0.0

    # Ruído OCR (peso forte)
    penalties += ocr_noise * 0.35

    # Heading only
    if is_heading_only:
        penalties += 0.30

    # Texto muito curto (menos de 200 chars)
    if char_count < 200:
        shortness = 1.0 - (char_count / 200)
        penalties += shortness * 0.20

    # Baixa densidade de tokens
    if token_count < 15:
        penalties += 0.4

    # URLs dominantes
    urls = URL_PATTERN.findall(text)
    url_char_count = sum(len(u) for u in urls)
    if char_count > 0 and url_char_count / char_count > 0.6:
        penalties += 0.25

    # Tabelas fragmentadas
    if block_type == "table":
        rows = text.count("\n")
        if rows < TABLE_FRAGMENT_MIN_ROWS:
            penalties += 0.20

    score = max(0.0, 1.0 - penalties)
    return round(score, 2)


# ---------------------------------------------------------------------------
# Filtros individuais (composicionais)
# ---------------------------------------------------------------------------

def filter_no_body_text(text: str, block_type: str) -> ChunkQualityVerdict:
    """Rejeita chunks sem corpo textual real.

    Detecta chunks que possuem apenas whitespace, pontuação ou caracteres
    não-alfanuméricos sem conteúdo significativo.
    """
    # Remove espaços, tabs, newlines para avaliar conteúdo real
    stripped = re.sub(r"\s+", "", text)
    alpha_num = re.sub(r"[^a-zA-ZÀ-ÿ0-9]", "", stripped)

    if len(alpha_num) < 10:
        return ChunkQualityVerdict(
            approved=False,
            reason="sem_corpo_textual",
            quality_metrics=QualityMetrics(
                quality_score=0.0,
                token_count=_count_tokens(text),
                char_count=len(text),
                rejection_reason="Chunk sem corpo textual significativo",
            ),
        )
    return ChunkQualityVerdict()


def filter_heading_only(text: str, block_type: str) -> ChunkQualityVerdict:
    """Rejeita chunks que contêm apenas título ou heading isolado.

    Evita chunks compostos só por texto curto em formato de heading,
    sem conteúdo de parágrafo associado.
    """
    stripped = text.strip()
    is_heading = (
        len(stripped) <= HEADING_ONLY_MAX_CHARS
        and "\n" not in stripped
        and not SENTENCE_END_PATTERN.search(stripped)
    )

    if is_heading:
        return ChunkQualityVerdict(
            approved=False,
            reason="apenas_titulo",
            quality_metrics=QualityMetrics(
                quality_score=0.05,
                token_count=_count_tokens(text),
                char_count=len(text),
                is_heading_only=True,
                rejection_reason="Chunk contém apenas título sem corpo",
            ),
        )
    return ChunkQualityVerdict()


def filter_min_tokens(
    text: str,
    block_type: str,
    min_tokens: int = MIN_TOKENS_DEFAULT,
) -> ChunkQualityVerdict:
    """Rejeita chunks com menos de `min_tokens` tokens.
    """
    token_count = _count_tokens(text)

    if token_count < min_tokens:
        return ChunkQualityVerdict(
            approved=False,
            reason="abaixo_do_minimo_de_tokens",
            quality_metrics=QualityMetrics(
                quality_score=0.05,
                token_count=token_count,
                char_count=len(text),
                rejection_reason=f"Chunk com {token_count} tokens (mínimo: {min_tokens})",
            ),
        )
    return ChunkQualityVerdict()


def filter_min_chars(
    text: str,
    block_type: str,
    min_chars: int = MIN_CHARS_DEFAULT,
) -> ChunkQualityVerdict:
    """Rejeita chunks com menos de `min_chars` caracteres.

    Exceção: chunks semanticamente completos (frases bem formadas) são
    aceitos mesmo abaixo do limite.
    """
    char_count = len(text.strip())

    if char_count < min_chars:
        if _is_semantically_complete(text):
            return ChunkQualityVerdict(
                approved=True,
                reason="abaixo_do_limite_mas_semanticamente_completo",
                quality_metrics=QualityMetrics(
                    quality_score=0.55,
                    token_count=_count_tokens(text),
                    char_count=char_count,
                ),
            )
        return ChunkQualityVerdict(
            approved=False,
            reason="abaixo_do_minimo_de_caracteres",
            quality_metrics=QualityMetrics(
                quality_score=0.10,
                token_count=_count_tokens(text),
                char_count=char_count,
                rejection_reason=f"Chunk com {char_count} chars (mínimo: {min_chars})",
            ),
        )
    return ChunkQualityVerdict()


def filter_url_only(text: str, block_type: str) -> ChunkQualityVerdict:
    """Rejeita chunks cujo conteúdo é composto predominantemente por URLs.

    Se > 85% do conteúdo não-whitespace são URLs, o chunk é descartado.
    """
    stripped = text.strip()
    if not stripped:
        return ChunkQualityVerdict(approved=False, reason="vazio")

    urls = URL_PATTERN.findall(stripped)
    url_chars = sum(len(u) for u in urls)
    non_ws_chars = len(re.sub(r"\s+", "", stripped))

    if non_ws_chars > 0 and url_chars / non_ws_chars > 0.85:
        return ChunkQualityVerdict(
            approved=False,
            reason="apenas_urls",
            quality_metrics=QualityMetrics(
                quality_score=0.05,
                token_count=_count_tokens(text),
                char_count=len(text),
                rejection_reason="Chunk composto predominantemente por URLs",
            ),
        )
    return ChunkQualityVerdict()


def filter_table_fragment(text: str, block_type: str) -> ChunkQualityVerdict:
    """Rejeita pedaços isolados de tabela sem contexto suficiente.

    Tabelas com menos de 2 linhas de dados e sem caption/texto
    introdutório são descartadas.
    """
    if block_type != "table":
        return ChunkQualityVerdict()

    lines = [ln.strip() for ln in text.strip().split("\n") if ln.strip()]
    data_lines = [ln for ln in lines if "|" in ln or "\t" in ln]

    # Remove linhas de separador markdown (ex: |---|---|)
    data_lines = [
        ln for ln in data_lines
        if not re.match(r"^\|?[\s\-:]+\|", ln)
    ]

    has_caption = any(
        ln.lower().startswith("tabela") or ln.lower().startswith("table")
        for ln in lines[:2]
    )

    if len(data_lines) < TABLE_FRAGMENT_MIN_ROWS and not has_caption:
        text_outside_table = " ".join(
            ln for ln in lines if "|" not in ln and "\t" not in ln
        )
        if len(text_outside_table.strip()) < 50:
            return ChunkQualityVerdict(
                approved=False,
                reason="fragmento_tabela_isolado",
                quality_metrics=QualityMetrics(
                    quality_score=0.10,
                    token_count=_count_tokens(text),
                    char_count=len(text),
                    rejection_reason="Fragmento de tabela isolado sem contexto",
                ),
            )
    return ChunkQualityVerdict()


def filter_ocr_noise(text: str, block_type: str, threshold: float = 0.65) -> ChunkQualityVerdict:
    """Rejeita chunks com ruído de OCR acima do threshold.

    Textos com muitos caracteres anômalos, letras espaçadas ou artefatos
    de digitalização são filtrados para não poluir o índice.
    """
    noise_score = _calc_ocr_noise_score(text)

    if noise_score >= threshold:
        return ChunkQualityVerdict(
            approved=False,
            reason="ruido_ocr_excessivo",
            quality_metrics=QualityMetrics(
                quality_score=round(max(0.0, 1.0 - noise_score), 2),
                token_count=_count_tokens(text),
                char_count=len(text),
                ocr_noise_score=noise_score,
                rejection_reason=f"Score de ruído OCR={noise_score} (threshold={threshold})",
            ),
        )
    return ChunkQualityVerdict()


# ---------------------------------------------------------------------------
# Orquestrador de filtros
# ---------------------------------------------------------------------------

# Ordem importa: filtros mais baratos e eliminatórios primeiro
DEFAULT_FILTERS = [
    filter_no_body_text,
    filter_url_only,
    filter_heading_only,
    filter_table_fragment,
    filter_ocr_noise,
    filter_min_tokens,
    filter_min_chars,  # último — respeita exceção de completude semântica
]


def apply_all_filters(
    raw_text: str,
    block_type: str = "paragraph",
    min_chars: int = MIN_CHARS_DEFAULT,
    min_tokens: int = MIN_TOKENS_DEFAULT,
    ocr_noise_threshold: float = 0.65,
    filters: list | None = None,
) -> ChunkQualityVerdict:
    """Aplica todos os filtros de qualidade em sequência ao texto bruto.

    O texto avaliado é o conteúdo ANTES da junção do header de seção.
    Retorna o primeiro veredito de rejeição ou um veredito de aprovação
    com métricas de qualidade para enriquecer o payload.

    Args:
        raw_text:  Texto limpo do chunk (sem header de seção).
        block_type: Tipo do bloco ('paragraph', 'table', 'paragraph_part').
        min_chars:  Mínimo de caracteres para aprovação (padrão 300).
        min_tokens: Mínimo de tokens para aprovação (padrão 10).
        ocr_noise_threshold: Threshold de ruído OCR para rejeição (padrão 0.65).
        filters:   Lista custom de filtros (para composição). None = DEFAULT_FILTERS.

    Returns:
        ChunkQualityVerdict com approved=True e métricas, ou approved=False.
    """
    active_filters = filters if filters is not None else DEFAULT_FILTERS

    for f in active_filters:
        if f is filter_min_chars:
            verdict = f(raw_text, block_type, min_chars=min_chars)
        elif f is filter_min_tokens:
            verdict = f(raw_text, block_type, min_tokens=min_tokens)
        elif f is filter_ocr_noise:
            verdict = f(raw_text, block_type, threshold=ocr_noise_threshold)
        else:
            verdict = f(raw_text, block_type)

        if not verdict.approved:
            return verdict

    # Todos passaram — calcula métricas finais
    ocr_noise = _calc_ocr_noise_score(raw_text)
    token_count = _count_tokens(raw_text)
    char_count = len(raw_text)
    is_heading = char_count <= HEADING_ONLY_MAX_CHARS and "\n" not in raw_text.strip()
    quality = _calc_quality_score(raw_text, ocr_noise, is_heading, block_type)

    # Detecção de palavras para correção via LLM
    strange = detectar_palavras_estranhas(raw_text)
    broken = detectar_palavras_quebradas(raw_text)

    return ChunkQualityVerdict(
        approved=True,
        reason="aprovado",
        quality_metrics=QualityMetrics(
            quality_score=quality,
            token_count=token_count,
            ocr_noise_score=ocr_noise,
            is_heading_only=is_heading,
            char_count=char_count,
            strange_words=strange,
            broken_words=broken,
            needs_llm_repair=len(strange) > 0 or len(broken) > 0,
        ),
    )


# ---------------------------------------------------------------------------
# Detecção de palavras estranhas / quebradas (para correção via LLM)
# ---------------------------------------------------------------------------

def _normalizar_texto(texto: str) -> str:
    """Remove acentos e coloca em minúsculas para análise heurística."""
    texto = texto.lower()
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    return texto


def _eh_palavra_estranha(palavra: str) -> bool:
    """Detecta se uma palavra parece estranha por heurísticas de OCR.

    Identifica junções indevidas, ruído de caracteres, sequências
    de consoantes impossíveis e padrões de digitalização defeituosa.
    """
    normalizada = _normalizar_texto(palavra)

    # Remove pontuação das extremidades
    normalizada = re.sub(r"^[^\w]+|[^\w]+$", "", normalizada)

    if not normalizada or len(normalizada) <= 2:
        return False

    # Caracteres não alfanuméricos no meio
    if re.search(r"[^a-z0-9\-]", normalizada):
        return True

    # Palavra muito longa → possível junção de termos (ex: ESTRATÉGIASPEDAGÓGICAS)
    if len(normalizada) > 28:
        return True

    # Sequência longa de consoantes (impossível em português)
    if re.search(r"[bcdfghjklmnpqrstvwxyz]{5,}", normalizada):
        return True

    # Poucas vogais em palavra longa
    vogais = sum(1 for c in normalizada if c in "aeiou")
    if len(normalizada) >= 8 and vogais / len(normalizada) < 0.25:
        return True

    # Muitos caracteres repetidos (ex: "aaaa", "llll")
    if re.search(r"(.)\1{3,}", normalizada):
        return True

    # Mistura estranha letras/números (ex: "abc123def")
    if re.search(r"[a-z]+\d+[a-z]+|\d+[a-z]+\d+", normalizada):
        return True

    return False


def detectar_palavras_estranhas(texto: str) -> list[str]:
    """Retorna palavras suspeitas encontradas no texto.

    Usa heurísticas de proporção de vogais, comprimento, sequências
    de consoantes e padrões de OCR para identificar anomalias.
    """
    palavras = re.findall(r"\b[\wÀ-ÿ\-]+\b", texto)
    return [p for p in palavras if _eh_palavra_estranha(p)]


def detectar_palavras_quebradas(texto: str) -> list[str]:
    """Detecta sequências de letras/pedaços curtos separados por espaço.

    Captura padrões de OCR que fragmentam palavras:
      'Ta be l a'  →  detectado
      'Co m pa rat ivo'  →  detectado

    Exige pelo menos 3 fragmentos de 1-2 chars para reduzir falsos
    positivos com preposições legítimas ('que as despesas' NÃO detecta).
    """
    padrao = r"\b(?:[A-Za-zÀ-ÿ]{1,2}\s+){3,}[A-Za-zÀ-ÿ]{1,8}\b"
    achados = re.findall(padrao, texto)
    return [a.strip() for a in achados]
