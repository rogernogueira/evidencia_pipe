# MinerU 4.0.0a5 — imagem de AVALIAÇÃO, paralela à de produção (tmp/Dockerfile).
#
# Produção roda 3.4.4 (última estável) na imagem `evidencia_mineru:local`, portas
# 8011 (VLM) e 8012 (pipeline). Esta imagem NÃO a substitui: sobe na 8013, sob o
# profile "mineru-v4" do compose, para comparar a extração das mesmas páginas antes
# de decidir migrar. O pipeline continua apontando para a 8012 (MINERU_API_URL).
#
# Diferenças em relação ao Dockerfile de produção:
#   - versão FIXADA via ARG (lá é `mineru[core]>=3.4.0`, que não é reprodutível);
#   - extra `standard`, não `core`: a 4.0 reorganizou os extras — `core` não existe
#     mais, e é `standard` que traz o vLLM (`vllm<0.22.0,>=0.10.1.1`, satisfeito
#     pelo vLLM 0.21.0 da imagem base);
#   - `--pre`, porque 4.0.0a5 é pré-lançamento;
#   - checagem de versão no fim do build: falha alto em vez de gerar uma imagem que
#     silenciosamente não é a versão pedida.
#
# ATENÇÃO ao pareamento cliente↔servidor: a CLI do MinerU no host (3.4.4, via
# pyproject) fala com este servidor por HTTP. Versões diferentes podem divergir de
# protocolo. Para exercitar ESTA imagem, use um cliente 4.0.0a5 — não o venv de
# produção. Ver deploy/mineru-v4/README.md.
#
# Build (o download dos modelos leva dezenas de minutos e ~30 GB):
#   docker compose --profile mineru-v4 build mineru-v4
#   docker compose --profile mineru-v4 build --build-arg MINERU_VERSION=4.0.0a6 mineru-v4

FROM vllm/vllm-openai:v0.21.0

# Precisa ser redeclarado após o FROM para ficar visível nos RUN.
ARG MINERU_VERSION=4.0.0a5
ARG MINERU_EXTRA=standard
# Tier de modelos da 4.0 (flash | basic | standard). Acompanha o extra: `standard`
# é o que inclui o vLLM.
ARG MINERU_TIER=standard

# libgl para o opencv; fontes Noto para CJK nos PDFs.
RUN apt-get update && \
    apt-get install -y \
        fonts-noto-core \
        fonts-noto-cjk \
        fontconfig \
        libgl1 && \
    fc-cache -fv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install -U --pre \
        "mineru[${MINERU_EXTRA}]==${MINERU_VERSION}" --break-system-packages && \
    python3 -m pip cache purge

# Falha o build se a versão instalada não for a pedida (extra inválido, resolução
# inesperada do pip, cache velho). Sem isto, a imagem mentiria sobre sua versão.
RUN python3 - <<PY
import importlib.metadata as m, sys
esperada = "${MINERU_VERSION}"
instalada = m.version("mineru")
print(f"mineru instalado: {instalada} (esperado: {esperada})")
if instalada != esperada:
    sys.exit(f"ERRO: imagem construiu {instalada}, não {esperada}")
PY

# A 4.0 REESCREVEU a CLI: `mineru-models-download` e `mineru-api` não existem mais.
# Os entry points agora são só `mineru`, `mineru-kit` e `mineru-gradio`, com
# subcomandos (typer). Equivalências verificadas no wheel 4.0.0a5:
#   mineru-models-download -s huggingface -m all
#     → mineru-kit models download --source huggingface --tier standard
#   mineru-api --host H --port P
#     → mineru-kit api-server --host H --port P
#   mineru-api --enable-vlm-preload false
#     → é o DEFAULT da 4.0 (`--preload-models` existe e vem desligado)
# O tier acompanha o extra instalado (standard = com vLLM).
RUN /bin/bash -c "mineru-kit models download --source huggingface --tier ${MINERU_TIER}"

ENTRYPOINT ["/bin/bash", "-c", "export MINERU_MODEL_SOURCE=local && exec \"$@\"", "--"]
