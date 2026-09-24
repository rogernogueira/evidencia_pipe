"""Saneamento da resposta do AI Summary (backend/services/summary_service.py).

Modelo de "thinking" (DeepSeek v4, Qwen, Kimi) devolve o raciocínio embutido no
`content` quando o provedor não separa `reasoning_content`: o que chega ao usuário é um
`</think>` órfão seguido do rascunho, em geral em chinês. Aqui exercitamos as três
formas que aparecem na prática, o detector de contaminação e a segunda tentativa do
`summarize` — sem LLM nem Qdrant reais.
"""

import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from backend.services import summary_service as svc  # noqa: E402

RASCUNHO_CN = "好的，我需要分析这些证据并用葡萄牙语回答用户的问题。"
LIMPO = "O programa foi avaliado e ampliou a cobertura [1][2]."


# --------------------------------------------------------------------------
# _strip_reasoning
# --------------------------------------------------------------------------
@pytest.mark.parametrize("bruto, esperado", [
    # fechamento órfão: o abridor foi consumido pelo template do chat
    (f"{RASCUNHO_CN}</think>\n\n{LIMPO}", LIMPO),
    # bloco fechado
    (f"<think>{RASCUNHO_CN}</think>{LIMPO}", LIMPO),
    # variação de tag e de espaçamento
    (f"< thinking >{RASCUNHO_CN}</ thinking >\n{LIMPO}", LIMPO),
    # delimitador não-XML (Kimi)
    (f"◁think▷{RASCUNHO_CN}◁/think▷{LIMPO}", LIMPO),
    (f"<|begin_of_thought|>{RASCUNHO_CN}<|end_of_thought|>{LIMPO}", LIMPO),
    # dois blocos
    (f"<think>a</think>{LIMPO}<think>b</think>", LIMPO),
])
def test_remove_o_raciocinio_vazado(bruto, esperado):
    texto, removeu = svc._strip_reasoning(bruto)
    assert texto == esperado
    assert removeu is True


def test_abertura_sem_fechamento_descarta_o_resto():
    """Resposta truncada ainda no raciocínio: dali para frente nada é resposta."""
    texto, removeu = svc._strip_reasoning(f"{LIMPO}\n<think>{RASCUNHO_CN}")
    assert texto == LIMPO and removeu is True


def test_resposta_limpa_passa_intacta():
    texto, removeu = svc._strip_reasoning(LIMPO)
    assert texto == LIMPO and removeu is False


def test_nao_confunde_com_texto_que_menciona_a_tag():
    """`<think>` dentro de um bloco de código da evidência não deve zerar a resposta —
    só o padrão de tag isolado é tratado. (Documenta o limite conhecido: a heurística
    é textual.)"""
    texto, _ = svc._strip_reasoning("Use a palavra think normalmente [1].")
    assert texto == "Use a palavra think normalmente [1]."


# --------------------------------------------------------------------------
# detector de contaminação
# --------------------------------------------------------------------------
def test_cjk_em_idioma_nao_cjk_e_contaminacao():
    assert svc._contaminado(RASCUNHO_CN, "pt-BR") is True


def test_cjk_e_legitimo_quando_o_idioma_pedido_e_cjk():
    assert svc._contaminado(RASCUNHO_CN, "zh-CN") is False


def test_nome_proprio_em_ideograma_nao_e_contaminacao():
    """Dois ideogramas num parágrafo inteiro ficam abaixo do teto — não se mutila
    citação legítima da evidência."""
    texto = ("O programa foi avaliado em 北京 e ampliou a cobertura em três regiões, "
             "com melhoria de 12% no indicador central segundo a evidência [1].")
    assert svc._cjk_ratio(texto) < svc._CJK_MAX_RATIO
    assert svc._contaminado(texto, "pt-BR") is False


def test_marcador_remanescente_e_contaminacao():
    assert svc._contaminado(f"{LIMPO} </think>", "pt-BR") is True


def test_sanitize_remove_raciocinio_e_citacao_inexistente():
    texto = svc._sanitize(f"{RASCUNHO_CN}</think> Resultado [1][9].", language="pt-BR", max_index=2)
    assert texto == "Resultado [1]."


# --------------------------------------------------------------------------
# _call_llm: thinking desligado na origem, com fallback
# --------------------------------------------------------------------------
class _ClienteFake:
    """Dublê do cliente OpenAI: registra as chamadas e pode recusar a primeira."""

    def __init__(self, *, recusa_extra_body=False, content="ok"):
        self.recusa_extra_body = recusa_extra_body
        self.content = content
        self.chamadas = []
        self.chat = type("C", (), {"completions": self})()

    def create(self, **kw):
        self.chamadas.append(kw)
        if self.recusa_extra_body and "extra_body" in kw:
            import httpx
            from openai import BadRequestError
            resp = httpx.Response(400, request=httpx.Request("POST", "http://provedor/v1"))
            raise BadRequestError("unknown parameter: thinking", response=resp, body=None)
        msg = type("M", (), {"content": self.content})()
        return type("R", (), {"choices": [type("Ch", (), {"message": msg})()]})()


@pytest.fixture
def cliente(monkeypatch):
    """Instala o cliente fake e zera a memória de suporte ao parâmetro."""
    def instalar(**kw):
        c = _ClienteFake(**kw)
        monkeypatch.setattr(svc, "_get_client", lambda: c)
        monkeypatch.setattr(svc, "_thinking_param_ok", None, raising=False)
        return c
    return instalar


def test_chamada_pede_o_raciocinio_desligado(cliente, monkeypatch):
    monkeypatch.setattr(svc, "LLM_SUMMARY_DISABLE_THINKING", True)
    c = cliente()

    assert svc._call_llm("sys", "user") == "ok"
    assert c.chamadas[0]["extra_body"] == {"thinking": {"type": "disabled"}}


def test_provedor_que_recusa_o_parametro_nao_quebra_o_resumo(cliente, monkeypatch):
    """400 por causa do parâmetro → refaz sem ele e para de enviá-lo nas próximas."""
    monkeypatch.setattr(svc, "LLM_SUMMARY_DISABLE_THINKING", True)
    c = cliente(recusa_extra_body=True)

    assert svc._call_llm("sys", "user") == "ok"
    assert "extra_body" in c.chamadas[0] and "extra_body" not in c.chamadas[1]

    assert svc._call_llm("sys", "user") == "ok"
    assert "extra_body" not in c.chamadas[2]  # não insiste


def test_flag_desligada_nao_manda_o_parametro(cliente, monkeypatch):
    monkeypatch.setattr(svc, "LLM_SUMMARY_DISABLE_THINKING", False)
    c = cliente()

    svc._call_llm("sys", "user")
    assert "extra_body" not in c.chamadas[0]


# --------------------------------------------------------------------------
# summarize: segunda tentativa
# --------------------------------------------------------------------------
class _Ponto:
    def __init__(self, i):
        self.score = 0.9
        self.payload = {"chunk_id": f"c{i}", "document_id": f"doc{i}", "item_uuid": f"u{i}",
                        "item_handle": f"123/{i}", "section": "Resultados", "page": i,
                        "content": f"Evidência número {i}."}


class _SemanticFake:
    async def search_points(self, query, limit=5, type="hybrid", profile="", uuid=None):
        return [_Ponto(1), _Ponto(2)]


@pytest.fixture
def respostas(monkeypatch):
    """Enfileira respostas do LLM e registra as mensagens enviadas."""
    enviadas = []

    def instalar(*saidas):
        fila = list(saidas)

        def fake(system_prompt, user_message):
            enviadas.append(user_message)
            return fila.pop(0)

        monkeypatch.setattr(svc, "_call_llm", fake)
        return enviadas

    return instalar


@pytest.mark.anyio
async def test_summarize_refaz_uma_vez_quando_vem_contaminado(respostas):
    enviadas = respostas(f"{RASCUNHO_CN}\n{RASCUNHO_CN}", LIMPO)

    r = await svc.SummaryService(_SemanticFake()).summarize("avaliação do programa")

    assert r.summary == LIMPO
    assert len(enviadas) == 2
    assert "FORMAT REQUIREMENTS" in enviadas[1] and "pt-BR" in enviadas[1]


@pytest.mark.anyio
async def test_summarize_nao_refaz_quando_a_1a_resposta_esta_boa(respostas):
    enviadas = respostas(f"<think>{RASCUNHO_CN}</think>{LIMPO}")

    r = await svc.SummaryService(_SemanticFake()).summarize("avaliação do programa")

    assert r.summary == LIMPO
    assert len(enviadas) == 1  # o saneamento resolveu; nada de 2ª chamada


@pytest.mark.anyio
async def test_summarize_refaz_quando_o_saneamento_esvazia_a_resposta(respostas):
    enviadas = respostas(f"<think>{RASCUNHO_CN}", LIMPO)

    r = await svc.SummaryService(_SemanticFake()).summarize("avaliação do programa")

    assert r.summary == LIMPO and len(enviadas) == 2


@pytest.fixture
def anyio_backend():
    return "asyncio"
