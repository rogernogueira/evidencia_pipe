# MinerU em CPU — backend `pipeline` apenas (DocAnalysis/OCR clássicos). NÃO carrega
# o VLM (vLLM é CUDA-only), equivalente ao `--enable-vlm-preload false` da versão GPU.
#
# ESBOÇO: a imagem de produção (tmp/Dockerfile) parte de `vllm/vllm-openai` (CUDA);
# esta parte de python-slim + torch CPU para rodar sem placa. É ORDENS DE MAGNITUDE
# mais lenta — serve para validar o fluxo fim-a-fim sem GPU, não para reindexar acervo.
FROM python:3.12-slim

# libgl p/ o opencv; fontes Noto p/ documentos (inclui CJK); curl p/ healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
        fonts-noto-core fonts-noto-cjk fontconfig libgl1 curl \
    && fc-cache -fv && apt-get clean && rm -rf /var/lib/apt/lists/*

# Torch CPU ANTES do mineru: sem o índice CPU o pip puxa a wheel CUDA (~2,5 GB).
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch torchvision

# Mesma pinagem da produção: >=3.4,<4 (não pular para o 4.x alpha).
RUN pip install --no-cache-dir 'mineru[core]>=3.4.0,<4' && pip cache purge

# Só os modelos do pipeline (layout/OCR) — sem os pesos do VLM, que não sobem em CPU.
RUN mineru-models-download -s huggingface -m pipeline

ENV MINERU_MODEL_SOURCE=local \
    MINERU_DEVICE_MODE=cpu

EXPOSE 8010
# `--enable-vlm-preload false`: não tenta carregar o VLM no startup.
CMD ["mineru-api", "--host", "0.0.0.0", "--port", "8010", "--enable-vlm-preload", "false"]
