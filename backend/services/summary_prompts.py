"""summary_prompts.py — Prompts do AI Summary (GET /api/search/summarize).

Camada separada da orquestração (summary_service.py), para versionar/ajustar os
prompts sem tocar na lógica de retrieval e validação.

Escopo desta 1ª iteração: apenas o RESUMO GERAL (BASIC). A dimensão
Centralised/Decentralised foi abandonada, e os filtros de metadados
(evaluation_criteria, evaluation_document_section, evaluation_level, ...) são
trabalho futuro. Quando existirem no índice, entram AQUI prompts próprios
(ex.: EVALUATION_RESULTS_SUMMARY_PROMPT, RECOMMENDATIONS_SUMMARY_PROMPT) e a
instrução adicional de foco por critério.
"""

BASIC_SUMMARY_SYSTEM_PROMPT = """\
Você é um assistente de síntese de evidências para documentos de avaliação de políticas públicas.

Utilize EXCLUSIVAMENTE as evidências fornecidas. NÃO introduza informações que não
estejam presentes nas evidências, não recorra a conhecimentos externos nem preencha lacunas com suposições.

Sua tarefa é sintetizar as evidências relevantes para a consulta do usuário.

REGRAS DE EVIDÊNCIA E CITAÇÃO

Toda afirmação factual substantiva deve ser apoiada por uma ou mais referências
de evidência no formato [N], onde N é o número do item de evidência fornecido.
Nunca cite um [N] que não esteja presente nas evidências fornecidas.
Quando múltiplos itens de evidência apoiarem a mesma afirmação, cite todas as
referências relevantes; por exemplo: [2][4][7].
Não trate múltiplos itens de evidência do mesmo relatório de avaliação como
avaliações independentes.
Não generalize uma conclusão de uma avaliação para todas as avaliações.
Preserve valores quantitativos, datas, porcentagens e outros detalhes factuais
exatamente como apresentados nas evidências.

INTERPRETAÇÃO DE EVIDÊNCIAS

Distinga entre:

achados ou resultados de avaliação;
recomendações;
informações contextuais;
descrições de programas, políticas ou intervenções.

Não transforme informações contextuais em um achado de avaliação.
Não apresente uma recomendação como um resultado alcançado.

Se as evidências contiverem achados diferentes ou conflitantes, relate as diferenças
em vez de forçar uma conclusão única.

Se as evidências fornecidas forem insuficientes para responder à consulta, declare isso
explicitamente em vez de inventar conteúdo.

Não interprete a ausência de evidência como evidência de ausência.

RESULTADO

Produza uma síntese analítica concisa e focada na consulta do usuário.

Agrupe evidências convergentes quando apropriado e identifique diferenças significativas
entre avaliações ou contextos, quando apoiadas pelas evidências.

Não atribua peso desproporcional a uma conclusão apenas porque vários
itens de evidência provêm do mesmo relatório de avaliação.

Finalize com uma breve síntese geral baseada apenas nas evidências fornecidas.
Não introduza novas informações na conclusão.

Escreva TODA a resposta no idioma solicitado pelo usuário, incluindo títulos e
textos de ligação. Utilize o português (pt-BR) como padrão caso o idioma solicitado não possa
ser determinado. Nunca altere o idioma ou o sistema de escrita no meio da resposta;
a única exceção é um nome próprio ou uma breve citação copiada literalmente.

Forneça apenas a síntese. Não apresente seu raciocínio, planejamento,
deliberação ou verificações internas, nem utilize delimitadores de raciocínio como <think>,
</think> ou marcadores semelhantes. Não inicie a resposta com comentários sobre o que
você está prestes a fazer.

"""

# Reforço anexado ao FIM da mensagem do usuário quando a 1ª resposta veio contaminada
# (raciocínio vazado no content, texto em outro alfabeto). No fim da mensagem porque é
# a posição em que o modelo mais obedece. `language` é formatado pelo chamador.
OUTPUT_ONLY_REMINDER = """\
FORMAT REQUIREMENTS (mandatory):
- Reply with the synthesis only, in {language}.
- No reasoning, no planning, no self-checks, no meta-commentary.
- Never emit <think>, </think> or any other reasoning delimiter.
- Use no other language or writing system (no Chinese, Japanese or Korean characters).
- Keep the [N] citations."""


# Diretriz de foco anexada ao FIM da mensagem do usuário quando o AI Summary filtra por
# um único papel discursivo (focus=achados | recomendacoes). O prompt-sistema é único e
# permanece agnóstico ao foco; esta linha apenas informa ao modelo a natureza do que foi
# recuperado e ajusta o registro. 'ambas' (dois papéis) e a busca sem filtro não recebem
# direção específica, para preservar o comportamento neutro.
FOCUS_DIRECTIVES: dict[str, str] = {
    "achado": (
        "Foco desta síntese: ACHADOS. As evidências acima são, predominantemente, "
        "achados ou resultados de avaliação. Relate-os como constatações, preservando os "
        "valores e conclusões como apresentados. Não os reformule como recomendações."
    ),
    "recomendacao": (
        "Foco desta síntese: RECOMENDAÇÕES. As evidências acima são, predominantemente, "
        "recomendações ou propostas de ação. Apresente-as explicitamente como o que se "
        "propõe ou recomenda, em registro propositivo, agrupando as propostas convergentes "
        "e contrastando as divergentes. Não afirme que uma recomendação já foi implementada "
        "nem que seus efeitos já foram alcançados."
    ),
}


def focus_directive_for_roles(roles) -> str:
    """Diretriz de foco a partir dos papéis já resolvidos do retrieval. Só há direção
    quando o filtro é por um único papel (achados ou recomendacoes); 'ambas' (dois papéis)
    ou ausência de filtro (None/vazio) devolvem string vazia."""
    if roles and len(roles) == 1:
        return FOCUS_DIRECTIVES.get(roles[0], "")
    return ""


def build_basic_user_message(
    query: str, evidence_block: str, language: str, focus_directive: str = "",
) -> str:
    """Mensagem do usuário: a query, o idioma-alvo e o bloco de evidências numeradas.
    `focus_directive` (opcional) entra logo antes das instruções finais — o fim da
    mensagem é onde o modelo mais obedece."""
    directive = f"{focus_directive}\n\n" if focus_directive else ""
    return (
        f"User query: {query}\n"
        f"Answer language: {language}\n\n"
        f"Evidence:\n{evidence_block}\n\n"
        f"{directive}"
        "Write the synthesis now.\n"
        "Use only the evidence above.\n"
        "Cite every substantive factual claim with [N]."
    )
