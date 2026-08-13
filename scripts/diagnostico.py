#!/usr/bin/env python3
"""Diagnóstico das dependências externas do evidencia_pipe.

Feito para rodar NO SERVIDOR, inclusive num host onde o `uv sync` ainda não
correu: usa só a biblioteca padrão. Não precisa de venv, de `transformers` nem
de `qdrant-client`.

    python3 scripts/diagnostico.py                  # lê o .env do repositório
    python3 scripts/diagnostico.py --env /etc/evidencia/.env
    python3 scripts/diagnostico.py --api-port 8181  # também checa a API local

Cada linha sai como [OK], [AVISO] ou [FALHA]. Falha = indexação ou busca não
funcionam; aviso = degradação ou coisa que só morde num cenário específico.
Sai com código 1 se houver qualquer falha, para dar para usar em automação.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

# Token ids de "indicadores de avaliacao institucional do ensino superior" pelo
# tokenizer do BAAI/bge-m3. Fixos aqui para o round-trip real de embedding não
# depender do `transformers` estar instalado no servidor.
PROBE_IDS = [0, 202334, 8, 96436, 123142, 67897, 54, 48617, 14597, 2]
PROBE_INNER = len(PROBE_IDS) - 2  # o servidor descarta BOS/EOS
DENSE_DIM = 1024

falhas = 0
avisos = 0


def linha(status: str, titulo: str, detalhe: str = "") -> None:
    global falhas, avisos
    if status == "FALHA":
        falhas += 1
    elif status == "AVISO":
        avisos += 1
    print(f"  [{status:5}] {titulo}" + (f" — {detalhe}" if detalhe else ""))


def secao(titulo: str) -> None:
    print(f"\n{titulo}\n" + "-" * len(titulo))


# --------------------------------------------------------------------------- #
# .env
# --------------------------------------------------------------------------- #
def carrega_env(caminho: str) -> Dict[str, str]:
    """Parser mínimo de .env — não expande variáveis nem executa nada."""
    valores: Dict[str, str] = {}
    if not os.path.isfile(caminho):
        return valores
    with open(caminho, encoding="utf-8") as fh:
        for bruta in fh:
            linha_txt = bruta.strip()
            if not linha_txt or linha_txt.startswith("#") or "=" not in linha_txt:
                continue
            chave, _, valor = linha_txt.partition("=")
            valor = valor.strip()
            if len(valor) >= 2 and valor[0] == valor[-1] and valor[0] in "\"'":
                valor = valor[1:-1]
            valores[chave.strip()] = valor
    return valores


def cfg(env: Dict[str, str], chave: str, padrao: str = "") -> str:
    """Precedência igual à do backend: variável de ambiente vence o .env."""
    return (os.environ.get(chave) or env.get(chave) or padrao).strip()


# --------------------------------------------------------------------------- #
# HTTP / TCP sem dependências
# --------------------------------------------------------------------------- #
def http_get(url: str, timeout: float = 10.0) -> Tuple[int, bytes]:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, resp.read()


def http_json(url: str, timeout: float = 10.0) -> Any:
    return json.loads(http_get(url, timeout)[1])


def http_post_json(url: str, payload: dict, timeout: float = 120.0) -> Any:
    req = urllib.request.Request(
        url, method="POST", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def tcp_aberto(host: str, porta: int, timeout: float = 5.0) -> bool:
    try:
        with socket.create_connection((host, porta), timeout=timeout):
            return True
    except OSError:
        return False


def hostport(url: str, porta_padrao: int) -> Tuple[str, int]:
    p = urlparse(url if "//" in url else f"//{url}", scheme="http")
    return p.hostname or "127.0.0.1", p.port or porta_padrao


def motivo(exc: Exception) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        return f"HTTP {exc.code}"
    if isinstance(exc, urllib.error.URLError):
        return f"inalcançável ({exc.reason})"
    return f"{type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------- #
# Verificações
# --------------------------------------------------------------------------- #
def checa_qdrant(env: Dict[str, str]) -> Optional[int]:
    """Devolve a dimensão do vetor denso da collection, se der para ler."""
    url = cfg(env, "QDRANT_URL", "http://127.0.0.1:6333").rstrip("/")
    colecao = cfg(env, "QDRANT_COLLECTION", "evidencia_chunks")
    secao(f"Qdrant — {url} (collection {colecao})")

    try:
        raiz = http_json(f"{url}/")
        linha("OK", "servidor no ar", f"versão {raiz.get('version', '?')}")
    except Exception as exc:
        linha("FALHA", "servidor inalcançável", motivo(exc))
        print("          → sem Qdrant, busca e indexação não funcionam.")
        return None

    try:
        dados = http_json(f"{url}/collections/{colecao}")["result"]
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            try:
                existentes = [c["name"] for c in http_json(f"{url}/collections")["result"]["collections"]]
            except Exception:
                existentes = []
            linha("FALHA", f"collection '{colecao}' não existe",
                  f"o servidor tem: {', '.join(existentes) or 'nenhuma'}")
            print("          → confira QDRANT_COLLECTION, ou restaure o snapshot.")
        else:
            linha("FALHA", "erro ao ler a collection", motivo(exc))
        return None
    except Exception as exc:
        linha("FALHA", "erro ao ler a collection", motivo(exc))
        return None

    pontos = dados.get("points_count")
    params = dados.get("config", {}).get("params", {})
    densos = params.get("vectors", {}) or {}
    esparsos = params.get("sparse_vectors", {}) or {}

    linha("OK" if pontos else "AVISO", "collection existe", f"{pontos} pontos")
    if not pontos:
        print("          → collection vazia: a busca não vai retornar nada.")

    dim = None
    if "dense" in densos:
        dim = densos["dense"].get("size")
        distancia = densos["dense"].get("distance")
        ok = dim == DENSE_DIM and str(distancia).lower() == "cosine"
        linha("OK" if ok else "AVISO", "vetor 'dense'", f"dim {dim}, distância {distancia}")
    else:
        linha("FALHA", "vetor nomeado 'dense' ausente", f"tem: {list(densos) or 'nenhum'}")

    if "sparse" in esparsos:
        linha("OK", "vetor 'sparse' presente")
    else:
        linha("FALHA", "vetor nomeado 'sparse' ausente", f"tem: {list(esparsos) or 'nenhum'}")
        print("          → busca híbrida (RRF) não funciona sem ele.")

    return dim


def checa_embedding(env: Dict[str, str], dim_colecao: Optional[int]) -> None:
    dense_url = cfg(env, "EMBED_API_URL", "http://127.0.0.1:8000").rstrip("/")
    sparse_url = cfg(env, "EMBED_API_SPARSE_URL", "http://127.0.0.1:8001").rstrip("/")
    modelo = cfg(env, "EMBED_API_MODEL", "BAAI/bge-m3")
    secao(f"Embedding — dense {dense_url} | sparse {sparse_url}")
    print(f"  modelo esperado em /v1/models: {modelo!r}")

    if dense_url == sparse_url:
        print("  (mesma URL para as duas tasks — é o esperado com o serviço bge-m3-cpu)")

    # --- /v1/models nos dois: é o que o health_check() do backend exige ------
    # Divergência de NOME é por endpoint: dense e sparse podem ser servidores
    # diferentes, e um nome errado num não explica nada no outro. Só conta como
    # divergência quando a lista foi lida e o nome não estava nela — servidor
    # inalcançável é outra causa.
    nome_divergente = {"dense": False, "sparse": False}
    for rotulo, base in (("dense", dense_url), ("sparse", sparse_url)):
        try:
            servidos = [m.get("id") for m in http_json(f"{base}/v1/models").get("data", [])]
        except Exception as exc:
            linha("FALHA", f"{rotulo}: /v1/models", motivo(exc))
            print(f"          → o backend loga 'API de embedding indisponível' e falha.")
            continue
        if modelo in servidos:
            linha("OK", f"{rotulo}: /v1/models", f"serve {servidos}")
        else:
            nome_divergente[rotulo] = True
            linha("FALHA", f"{rotulo}: nome do modelo não confere",
                  f"serve {servidos}, esperado {modelo!r}")
            print(f"          → ajuste EMBED_API_MODEL={servidos[0] if servidos else '<nome>'} "
                  "no .env (é só o rótulo servido, não troca o modelo).")
        if dense_url == sparse_url:
            nome_divergente["sparse"] = nome_divergente["dense"]
            break  # mesma instância; não repetir a checagem

    def consequencia(rotulo: str, exc: Exception) -> str:
        """O nome errado faz o POST voltar 404 — isso é consequência, não um
        defeito à parte. Mas só quando o servidor RESPONDEU: connection refused
        tem causa própria e não pode herdar essa nota."""
        if nome_divergente[rotulo] and isinstance(exc, urllib.error.HTTPError):
            return "  (consequência do nome acima)"
        return ""

    # --- round-trip real do denso -------------------------------------------
    try:
        resp = http_post_json(f"{dense_url}/v1/embeddings",
                              {"model": modelo, "input": [PROBE_IDS]})
        vetor = resp["data"][0]["embedding"]
    except Exception as exc:
        linha("FALHA", "dense: /v1/embeddings", motivo(exc) + consequencia("dense", exc))
        vetor = None
    if vetor is not None:
        norma = sum(x * x for x in vetor) ** 0.5
        if len(vetor) != DENSE_DIM:
            linha("FALHA", "dense: dimensão inesperada", f"{len(vetor)}, esperado {DENSE_DIM}")
        elif abs(norma - 1.0) > 1e-3:
            linha("AVISO", "dense: vetor não normalizado", f"norma {norma:.4f}")
            print("          → o índice usa CLS+L2; norma != 1 sugere pooling diferente.")
        else:
            linha("OK", "dense: /v1/embeddings", f"dim {len(vetor)}, norma {norma:.4f}")
        if dim_colecao and len(vetor) != dim_colecao:
            linha("FALHA", "dimensão do embedding × collection",
                  f"API devolve {len(vetor)}, collection espera {dim_colecao}")

    # --- round-trip real do esparso -----------------------------------------
    try:
        resp = http_post_json(f"{sparse_url}/pooling",
                              {"model": modelo, "input": [PROBE_IDS], "task": "token_classify"})
        pesos = resp["data"][0]["data"]
    except Exception as exc:
        linha("FALHA", "sparse: /pooling task=token_classify", motivo(exc) + consequencia("sparse", exc))
        if not nome_divergente["sparse"]:
            print("          → sem esparso não há busca híbrida NEM indexação: o")
            print("            embedder chama os dois endpoints em toda operação.")
            print("            Se o servidor dense só tem a task 'embed', falta a")
            print("            segunda instância (ou use o serviço bge-m3-cpu).")
        pesos = None
    if pesos is not None:
        if len(pesos) == PROBE_INNER:
            linha("OK", "sparse: /pooling", f"{len(pesos)} pesos, alinhados aos tokens internos")
        else:
            linha("FALHA", "sparse: desalinhamento",
                  f"{len(pesos)} pesos para {PROBE_INNER} tokens internos")
            print("          → o servidor precisa rodar com --pooler-config.task")
            print("            token_classify; outra task devolve o vetor errado.")


def checa_redis(env: Dict[str, str]) -> None:
    url = cfg(env, "REDIS_URL", "redis://127.0.0.1:6379")
    host, porta = hostport(url, 6379)
    secao(f"Redis — {host}:{porta}")
    if not tcp_aberto(host, porta):
        linha("FALHA", "porta fechada")
        print("          → broker Celery, job_store e lock da GPU dependem dele.")
        return
    try:
        with socket.create_connection((host, porta), timeout=5) as sock:
            sock.sendall(b"PING\r\n")
            resposta = sock.recv(64)
        if resposta.startswith(b"+PONG"):
            linha("OK", "PING", "+PONG")
        elif b"NOAUTH" in resposta or b"AUTH" in resposta:
            linha("AVISO", "exige autenticação", resposta.decode(errors="replace").strip())
        else:
            linha("AVISO", "resposta inesperada", resposta.decode(errors="replace").strip())
    except OSError as exc:
        linha("FALHA", "PING falhou", str(exc))


def checa_http_simples(titulo: str, url: str, caminho: str, dica: str,
                       critico: bool = True) -> None:
    secao(f"{titulo} — {url}")
    try:
        status, _ = http_get(f"{url.rstrip('/')}{caminho}")
        linha("OK", f"GET {caminho}", f"HTTP {status}")
    except Exception as exc:
        linha("FALHA" if critico else "AVISO", f"GET {caminho}", motivo(exc))
        print(f"          → {dica}")


def checa_minio(env: Dict[str, str]) -> None:
    backend = cfg(env, "ARTIFACT_STORE_BACKEND", "minio").lower()
    endpoint = cfg(env, "MINIO_ENDPOINT", "127.0.0.1:9000")
    seguro = cfg(env, "MINIO_SECURE", "false").lower() in {"1", "true", "yes", "on"}
    if backend != "minio":
        secao("MinIO")
        linha("AVISO", "backend de artefatos não é minio", f"ARTIFACT_STORE_BACKEND={backend}")
        return
    url = f"{'https' if seguro else 'http'}://{endpoint}"
    checa_http_simples("MinIO", url, "/minio/health/live",
                       "sem MinIO o pipeline não grava artefatos (PDF, markdown, chunks).")


def checa_api_local(porta: int) -> None:
    base = f"http://127.0.0.1:{porta}"
    secao(f"API local — {base}")
    try:
        status, _ = http_get(f"{base}/health", timeout=5)
        linha("OK", "GET /health", f"HTTP {status}")
    except Exception as exc:
        linha("AVISO", "GET /health", motivo(exc))
        print("          → a API pode não estar rodando neste host, ou a porta é outra.")
        return
    try:
        dados = http_json(f"{base}/api/search/status", timeout=15)
    except Exception as exc:
        linha("AVISO", "GET /api/search/status", motivo(exc))
        return
    semantico = dados.get("semantic")
    linha("OK" if semantico else "FALHA", "/api/search/status",
          f"semantic={semantico}")
    if not semantico:
        print("          → a API não alcança Qdrant ou a API de embedding;")
        print("            as seções acima dizem qual dos dois.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--env", default=os.path.join(raiz, ".env"),
                        help="caminho do .env (padrão: o do repositório)")
    parser.add_argument("--api-port", type=int, default=None,
                        help="porta da API local, para checar /api/search/status")
    args = parser.parse_args()

    print("=" * 72)
    print("  diagnóstico evidencia_pipe")
    print("=" * 72)

    env = carrega_env(args.env)
    secao("Configuração")
    if env:
        linha("OK", ".env lido", f"{args.env} ({len(env)} variáveis)")
    else:
        # Sem o .env, tudo abaixo mede os DEFAULTS, não o que a aplicação usa —
        # e um default que por acaso responde vira um falso "OK". Vale falha.
        linha("FALHA", ".env não encontrado ou vazio", args.env)
        print("          → o diagnóstico abaixo mede os DEFAULTS do código, não a")
        print("            configuração real da aplicação. Ache o arquivo certo e")
        print("            repita com --env; portas e URLs devem ser outras:")
        # Só sugere o caminho relativo ao repositório se o script estiver mesmo
        # dentro dele — copiado para o home, `raiz` viraria /home e o palpite
        # seria lixo.
        candidato = os.path.join(raiz, ".env")
        if os.path.isdir(os.path.join(raiz, "backend")):
            print(f"              ls -l {candidato}")
        else:
            print("              sudo find / -name '.env' -path '*evidencia*' 2>/dev/null")
        print("              systemctl cat evidencia-api 2>/dev/null | grep -i environment")
    for chave in ("QDRANT_URL", "QDRANT_COLLECTION", "EMBED_API_URL",
                  "EMBED_API_SPARSE_URL", "EMBED_API_MODEL", "MINERU_API_URL",
                  "MINERU_BACKEND", "REDIS_URL", "MINIO_ENDPOINT", "DSPACE_URL"):
        valor = cfg(env, chave)
        print(f"    {chave:22} {valor or '(default do código)'}")

    dim = checa_qdrant(env)
    checa_embedding(env, dim)
    checa_redis(env)
    checa_http_simples("MinerU", cfg(env, "MINERU_API_URL", "http://127.0.0.1:8010"),
                       "/docs", "sem MinerU não há extração; a busca no que já foi "
                                "indexado continua funcionando.")
    checa_minio(env)
    checa_http_simples("DSpace", cfg(env, "DSPACE_URL", "https://rdapp.comais.uft.edu.br"),
                       "/server/api", "sem DSpace não há ingestão de novos itens.",
                       critico=False)
    if args.api_port:
        checa_api_local(args.api_port)

    print("\n" + "=" * 72)
    if falhas:
        print(f"  {falhas} falha(s) e {avisos} aviso(s) — veja as setas acima.")
    elif avisos:
        print(f"  Sem falhas. {avisos} aviso(s).")
    else:
        print("  Tudo OK.")
    print("=" * 72)
    return 1 if falhas else 0


if __name__ == "__main__":
    sys.exit(main())
