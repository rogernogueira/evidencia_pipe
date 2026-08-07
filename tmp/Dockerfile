# containers isolam as versoes.
FROM vllm/vllm-openai:v0.21.0

# libgl para o opencv; fontes Noto para documentos com CJK
RUN apt-get update && \
    apt-get install -y \
        fonts-noto-core \
        fonts-noto-cjk \
        fontconfig \
        libgl1 && \
    fc-cache -fv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Pinagem: >=3.4,<4 (linha validada; nao pular para o 4.x alpha)
RUN python3 -m pip install -U 'mineru[core]>=3.4.0,<4' --break-system-packages && \
    python3 -m pip cache purge

# Baixa todos os modelos (layout, OCR, VLM) para dentro da imagem
RUN /bin/bash -c "mineru-models-download -s huggingface -m all"

ENTRYPOINT ["/bin/bash", "-c", "export MINERU_MODEL_SOURCE=local && exec \"$@\"", "--"]
