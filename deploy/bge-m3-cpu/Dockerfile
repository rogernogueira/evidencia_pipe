# bge-m3 em CPU servindo o protocolo do vLLM (ver app.py). Alternativa ao
# evidencia_bge_m3:local para hosts sem GPU nenhuma.
FROM python:3.12-slim

ARG EMBED_MODEL=BAAI/bge-m3

# --index-url do PyTorch CPU: sem isso o pip traz a wheel CUDA (~2,5 GB de
# bibliotecas nvidia que a imagem nunca usa).
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
      --extra-index-url https://pypi.org/simple \
      -r /tmp/requirements.txt

# Pesos assados na imagem, igual ao Dockerfile do lado GPU: o runtime não depende
# do cache do host nem da rede, e a revisão fica congelada junto com a imagem.
# Aqui o `sparse_linear.pt` NÃO é opcional — é o head esparso.
RUN python3 -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('${EMBED_MODEL}', ignore_patterns=['onnx/*', 'imgs/*', '*.jpg', '*.png'])"

ENV HF_HUB_OFFLINE=1
ENV EMBED_MODEL=${EMBED_MODEL}

COPY app.py /app/app.py
WORKDIR /app

# Um worker só: o modelo ocupa ~4,6 GB em float32 e cada worker carregaria a sua
# própria cópia. Paralelismo vem das threads do torch (OMP_NUM_THREADS).
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
