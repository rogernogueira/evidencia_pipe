# containers isolam as versoes.
ARG VLLM_VERSION=v0.26.0
FROM vllm/vllm-openai:${VLLM_VERSION}

# Pinagem: a arquitetura `BgeM3EmbeddingModel` (dense + sparse) so existe a partir
# da 0.26. Em versoes anteriores o vLLM nem reconhece o nome ("has no vLLM
# implementation") ou carrega o bge-m3 como XLM-R puro, sem o head esparso.
ARG EMBED_MODEL=BAAI/bge-m3

# Baixa o modelo para dentro da imagem: pesos + tokenizer + os configs de
# sentence-transformers (de onde vem o pooling CLS+norm) + os heads secundarios
# sparse_linear.pt / colbert_linear.pt, que sao o que o BgeM3EmbeddingModel le.
# Fora: onnx/ (copia redundante dos pesos, ~2 GB) e as imagens do README.
RUN python3 -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('${EMBED_MODEL}', ignore_patterns=['onnx/*', 'imgs/*', '*.jpg', '*.png'])"

# Sem rede em runtime: os pesos ja estao na imagem. Se faltar algo, falha na hora
# de subir em vez de baixar silenciosamente uma revisao diferente da validada.
ENV HF_HUB_OFFLINE=1

# Herda o ENTRYPOINT da base (vllm.entrypoints.openai.api_server); as flags de
# cada instancia (--pooler-config.task etc.) vem do `command` do compose.
