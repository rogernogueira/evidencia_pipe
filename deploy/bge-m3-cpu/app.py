"""Servidor CPU do bge-m3 que fala o MESMO protocolo dos contêineres vLLM.

Existe para o cenário sem GPU alguma: `vllm/vllm-openai` é CUDA-only e o TEI não
serve o esparso do bge-m3 (o `/embed_sparse` dele exige um modelo `ForMaskedLM`,
e aqui o head esparso é o `sparse_linear.pt`, um linear à parte). Este processo
reimplementa as duas pooling tasks em PyTorch CPU, com as rotas que o
`backend/services/embedder.py` já consome — o cliente não muda.

Rotas (as quatro que o cliente usa):
    GET  /health          — healthcheck do compose
    GET  /v1/models       — `health_check()` exige ver EMBED_API_MODEL aqui
    POST /v1/embeddings   — denso  (equivale a --pooler-config '{"task":"embed"}')
    POST /pooling         — esparso (equivale a task token_classify)

UM processo serve as DUAS tasks. Os dois contêineres do lado GPU existem por
limitação do servidor HTTP do vLLM (uma task por instância), não do modelo —
então aqui `EMBED_API_URL` e `EMBED_API_SPARSE_URL` podem apontar para a MESMA
porta, e o modelo é carregado uma vez só.

Paridade com o que já está indexado
-----------------------------------
Os 20.895 pontos da collection vieram do bge-m3 via FlagEmbedding/vLLM. As duas
contas são reproduzidas aqui exatamente:

  denso   = L2_normalize(last_hidden_state[:, 0])        (CLS + norma, como o
            1_Pooling/config.json do repo e como o FlagEmbedding)
  esparso = relu(sparse_linear(last_hidden_state))       (um peso por token)

Não usamos a API do FlagEmbedding porque ela recebe TEXTO e devolve o dicionário
já montado, enquanto o contrato daqui é receber TOKEN IDS e devolver um peso por
token — é isso que garante o alinhamento exato peso↔token no cliente. A conta é a
mesma; o que muda é a fronteira. O `sparse_linear.pt` carregado é o do próprio
repositório do modelo.

Como o vLLM, descartamos BOS/EOS antes de responder: o cliente espera len(pesos)
== len(input_ids) - 2. O descarte de tokens especiais restantes e de pesos <= 0
continua no cliente (`_to_lexical_weights`), que é onde sempre esteve.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Dict, List, Sequence, Union

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer

MODEL_ID = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
# O cliente manda esse nome no campo "model" e o health_check exige vê-lo em
# /v1/models — tem de casar com EMBED_API_MODEL do .env.
SERVED_MODEL_NAME = os.getenv("EMBED_API_MODEL", MODEL_ID)
MAX_LEN = int(os.getenv("EMBED_MAX_LEN", "8192"))
# Teto de tokens por passada no modelo. O cliente pode mandar um lote de 100
# chunks; em CPU, materializar 100 × 8192 × 1024 floats de uma vez estoura a RAM.
# Fatiamos por orçamento de tokens, preservando a ordem de entrada.
MAX_BATCH_TOKENS = int(os.getenv("EMBED_CPU_MAX_BATCH_TOKENS", "16384"))

_MODEL: Any = None
_TOKENIZER: Any = None
_SPARSE_LINEAR: Any = None
_PAD_ID = 1  # XLM-R; sobrescrito pelo tokenizer no load
# Serializa a inferência: o paralelismo em CPU vem das threads intra-op do torch,
# não de requisições concorrentes. Sem o lock, N requisições simultâneas brigam
# pelos mesmos núcleos e multiplicam o pico de RAM.
_INFER_LOCK = threading.Lock()

app = FastAPI(title="bge-m3 CPU (dense + sparse)")


def _load() -> None:
    """Carrega modelo, tokenizer e o head esparso. Chamado no startup."""
    global _MODEL, _TOKENIZER, _SPARSE_LINEAR, _PAD_ID

    threads = int(os.getenv("OMP_NUM_THREADS", "0"))
    if threads > 0:
        torch.set_num_threads(threads)

    started = time.time()
    print(f"[bge-m3-cpu] carregando {MODEL_ID} (float32, {torch.get_num_threads()} threads)…",
          flush=True)

    _TOKENIZER = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if _TOKENIZER.pad_token_id is not None:
        _PAD_ID = _TOKENIZER.pad_token_id

    # float32: a CPU não tem caminho eficiente para fp16, e várias ops nem têm
    # kernel. O custo é ~2,3 GB → ~4,6 GB de RAM.
    _MODEL = AutoModel.from_pretrained(MODEL_ID, dtype=torch.float32).eval()

    # sparse_linear.pt é um Linear(hidden_size, 1) salvo como state_dict solto —
    # não vem no from_pretrained porque não faz parte da arquitetura XLM-R.
    from huggingface_hub import hf_hub_download

    weights_path = hf_hub_download(MODEL_ID, "sparse_linear.pt")
    state = torch.load(weights_path, map_location="cpu")
    linear = torch.nn.Linear(_MODEL.config.hidden_size, 1)
    linear.load_state_dict(state)
    _SPARSE_LINEAR = linear.eval()

    print(f"[bge-m3-cpu] pronto em {time.time() - started:.1f}s", flush=True)


@app.on_event("startup")
def _startup() -> None:
    _load()


# --------------------------------------------------------------------------- #
# Entrada
# --------------------------------------------------------------------------- #
InputT = Union[str, List[str], List[int], List[List[int]]]


def _normalize_input(value: InputT) -> List[List[int]]:
    """Aceita o que o cliente manda (lista de listas de token ids) e, por
    robustez, também texto — tokenizado aqui com o mesmo truncamento."""
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or not value:
        raise HTTPException(status_code=400, detail="`input` vazio ou de tipo inválido.")

    if all(isinstance(item, str) for item in value):
        enc = _TOKENIZER(value, add_special_tokens=True, truncation=True, max_length=MAX_LEN)
        return enc["input_ids"]

    if all(isinstance(item, int) for item in value):  # uma sequência só, sem aninhar
        value = [value]

    batch: List[List[int]] = []
    for item in value:
        if not isinstance(item, list) or not all(isinstance(t, int) for t in item):
            raise HTTPException(
                status_code=400,
                detail="`input` deve ser texto, lista de textos ou lista de listas de token ids.",
            )
        if len(item) < 2:
            raise HTTPException(
                status_code=400,
                detail="cada entrada precisa de ao menos BOS e EOS (2 tokens).",
            )
        batch.append(item[:MAX_LEN])
    return batch


def _slices(batch: Sequence[List[int]]) -> List[range]:
    """Fatia o lote por orçamento de tokens (contando o padding da fatia)."""
    groups: List[range] = []
    start = 0
    longest = 0
    for i, ids in enumerate(batch):
        candidate = max(longest, len(ids))
        if i > start and candidate * (i - start + 1) > MAX_BATCH_TOKENS:
            groups.append(range(start, i))
            start, longest = i, len(ids)
        else:
            longest = candidate
    groups.append(range(start, len(batch)))
    return groups


# --------------------------------------------------------------------------- #
# Inferência
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def _forward(batch: List[List[int]], want_dense: bool) -> tuple[List[List[float]], List[List[float]]]:
    """Uma passada por fatia. Devolve (densos, pesos_por_token) — a lista não
    pedida volta vazia. Os pesos já vêm sem BOS/EOS, como o vLLM entrega."""
    dense_out: List[List[float]] = []
    sparse_out: List[List[float]] = []

    with _INFER_LOCK:
        for group in _slices(batch):
            chunk = [batch[i] for i in group]
            width = max(len(ids) for ids in chunk)
            input_ids = torch.full((len(chunk), width), _PAD_ID, dtype=torch.long)
            attention = torch.zeros((len(chunk), width), dtype=torch.long)
            for row, ids in enumerate(chunk):
                input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
                attention[row, : len(ids)] = 1

            hidden = _MODEL(input_ids=input_ids, attention_mask=attention).last_hidden_state

            if want_dense:
                dense = F.normalize(hidden[:, 0], p=2, dim=-1)
                dense_out.extend(dense.tolist())
            else:
                weights = torch.relu(_SPARSE_LINEAR(hidden)).squeeze(-1)
                for row, ids in enumerate(chunk):
                    # [1:len-1] descarta BOS/EOS; o padding fica fora por construção.
                    sparse_out.append(weights[row, 1 : len(ids) - 1].tolist())

    return dense_out, sparse_out


# --------------------------------------------------------------------------- #
# Rotas
# --------------------------------------------------------------------------- #
class EmbeddingsRequest(BaseModel):
    input: InputT
    model: str | None = None


class PoolingRequest(BaseModel):
    input: InputT
    model: str | None = None
    task: str | None = None


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok" if _MODEL is not None else "loading"}


@app.get("/v1/models")
def models() -> Dict[str, Any]:
    return {
        "object": "list",
        "data": [{"id": SERVED_MODEL_NAME, "object": "model", "owned_by": "evidencia_pipe"}],
    }


@app.post("/v1/embeddings")
def embeddings(request: EmbeddingsRequest) -> Dict[str, Any]:
    batch = _normalize_input(request.input)
    vectors, _ = _forward(batch, want_dense=True)
    return {
        "object": "list",
        "model": SERVED_MODEL_NAME,
        "data": [
            {"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(vectors)
        ],
        "usage": {"prompt_tokens": sum(len(ids) for ids in batch)},
    }


@app.post("/pooling")
def pooling(request: PoolingRequest) -> Dict[str, Any]:
    # O cliente sempre manda token_classify. Recusamos o resto em vez de devolver
    # silenciosamente o vetor errado — o desalinhamento só apareceria lá na frente.
    if request.task not in (None, "token_classify"):
        raise HTTPException(
            status_code=400,
            detail=f"task {request.task!r} não suportada; este servidor só faz 'token_classify'.",
        )
    batch = _normalize_input(request.input)
    _, weights = _forward(batch, want_dense=False)
    return {
        "object": "list",
        "model": SERVED_MODEL_NAME,
        "data": [{"index": i, "data": w} for i, w in enumerate(weights)],
        "usage": {"prompt_tokens": sum(len(ids) for ids in batch)},
    }
