from typing import Optional
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
