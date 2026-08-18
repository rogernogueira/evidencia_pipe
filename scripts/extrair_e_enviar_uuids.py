#!/usr/bin/env python3
"""Colhe os UUIDs dos itens do DSpace e dispara a ingestão na API do RDAPP.

    pip install requests lxml aiohttp

Três etapas:

  1. OAI-PMH `ListIdentifiers` (metadataPrefix=oai_dc) em devrdapp.ibict.br,
     seguindo o `resumptionToken` até o fim → lista de handles.
  2. Para cada handle, a REST do DSpace → uuid do item. Tenta
     `/server/api/core/handles/{handle}` (`_embedded.indexableObject.uuid`) e cai
     para `/server/api/pid/find?id=hdl:{handle}`, que é a rota que o devrdapp
     atende hoje. Sequencial, com pausa, para não afogar o servidor.
  3. `POST /api/files/dspace/item/{uuid}` na API do RDAPP, assíncrono, no máximo
     10 requisições simultâneas.

O CSV (`itens_uuid.csv`, colunas handle,uuid,status_envio,erro) é escrito e
descarregado linha por linha, então sobrevive a Ctrl+C ou queda no meio. Rodar
de novo relê o CSV e pula o que já foi enviado com sucesso (HTTP 2xx); handle
que falhou é tentado outra vez e ganha uma linha nova — vale a última.

    python scripts/extrair_e_enviar_uuids.py
    python scripts/extrair_e_enviar_uuids.py --limite 20 --sem-envio
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import os
import sys
import time
from typing import Dict, Iterator, List, Optional, Set, Tuple
from urllib.parse import quote

import aiohttp
import requests
from lxml import etree

DSPACE_BASE = "https://devrdapp.ibict.br"
OAI_ENDPOINT = f"{DSPACE_BASE}/server/oai/request"
RDAPP_BASE = "https://api.rdapp.comais.uft.edu.br"
CSV_PATH = "itens_uuid.csv"
CSV_COLUNAS = ["handle", "uuid", "status_envio", "erro"]

OAI_NS = {"oai": "http://www.openarchives.org/OAI/2.0/"}
USER_AGENT = "evidencia-pipe/extrair_e_enviar_uuids"

CONCORRENCIA = 10
TENTATIVAS = 3
ESPERA_BASE = 2.0  # segundos; dobra a cada tentativa (2s, 4s)
PAUSA_DSPACE = 0.2  # entre consultas à REST do DSpace
TIMEOUT_HTTP = 60
# O POST de ingestão resolve o bundle ORIGINAL no DSpace antes de responder 202,
# então é bem mais lento que uma leitura: um item inexistente já leva ~17s.
TIMEOUT_ENVIO = 180


# --------------------------------------------------------------------------- #
# CSV                                                                          #
# --------------------------------------------------------------------------- #

class RegistroCsv:
    """Append-only, com flush por linha: o que já foi processado está no disco."""

    def __init__(self, caminho: str):
        self.caminho = caminho
        novo = not os.path.exists(caminho) or os.path.getsize(caminho) == 0
        self._f = open(caminho, "a", newline="", encoding="utf-8")
        self._w = csv.writer(self._f)
        if novo:
            self._w.writerow(CSV_COLUNAS)
            self._f.flush()

    def registrar(self, handle: str, uuid: str, status: str, erro: str = "") -> None:
        self._w.writerow([handle, uuid, status, erro])
        self._f.flush()

    def fechar(self) -> None:
        self._f.close()


def _sucesso(status: str) -> bool:
    try:
        return 200 <= int(status) <= 299
    except (TypeError, ValueError):
        return False


def ler_ja_enviados(caminho: str) -> Tuple[Set[str], Set[str]]:
    """Handles e uuids que já constam no CSV com status de sucesso."""
    handles: Set[str] = set()
    uuids: Set[str] = set()
    if not os.path.exists(caminho):
        return handles, uuids
    try:
        with open(caminho, newline="", encoding="utf-8") as f:
            for linha in csv.DictReader(f):
                if _sucesso((linha.get("status_envio") or "").strip()):
                    if linha.get("handle"):
                        handles.add(linha["handle"].strip())
                    if linha.get("uuid"):
                        uuids.add(linha["uuid"].strip())
    except OSError as e:
        print(f"[aviso] não consegui reler {caminho}: {e}", file=sys.stderr)
    return handles, uuids


# --------------------------------------------------------------------------- #
# Etapa 1 — OAI-PMH                                                            #
# --------------------------------------------------------------------------- #

def extrair_handle(identificador: str) -> str:
    """`oai:devrdapp.ibict.br:123456789/359` → `123456789/359`."""
    if not identificador:
        return ""
    handle = identificador.rsplit(":", 1)[-1].strip()
    return handle if "/" in handle else ""


def _get_oai(sessao: requests.Session, params: Dict[str, str]) -> bytes:
    """GET no endpoint OAI com retentativa em erro de rede, 429 e 5xx."""
    for tentativa in range(1, TENTATIVAS + 1):
        try:
            r = sessao.get(OAI_ENDPOINT, params=params, timeout=(10, TIMEOUT_HTTP))
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(f"HTTP {r.status_code}")
            r.raise_for_status()
            return r.content
        except (requests.RequestException, requests.HTTPError) as e:
            if tentativa == TENTATIVAS:
                raise RuntimeError(f"OAI-PMH falhou após {TENTATIVAS} tentativas: {e}") from e
            espera = ESPERA_BASE * tentativa
            print(f"  [retry {tentativa}/{TENTATIVAS}] OAI: {e} — aguardando {espera:.0f}s")
            time.sleep(espera)
    raise RuntimeError("inalcançável")


def listar_handles(sessao: requests.Session, prefixo: str = "oai_dc") -> Iterator[str]:
    """Percorre o ListIdentifiers inteiro, página por página."""
    params: Dict[str, str] = {"verb": "ListIdentifiers", "metadataPrefix": prefixo}
    pagina = 0
    total = 0
    while True:
        xml = _get_oai(sessao, params)
        pagina += 1
        raiz = etree.fromstring(xml)

        erro = raiz.find("oai:error", OAI_NS)
        if erro is not None:
            codigo = erro.get("code", "?")
            if codigo == "noRecordsMatch":
                return
            raise RuntimeError(f"OAI-PMH devolveu erro {codigo}: {(erro.text or '').strip()}")

        cabecalhos = raiz.findall("oai:ListIdentifiers/oai:header", OAI_NS)
        for cab in cabecalhos:
            if cab.get("status") == "deleted":
                continue
            handle = extrair_handle(cab.findtext("oai:identifier", namespaces=OAI_NS) or "")
            if handle:
                total += 1
                yield handle
        print(f"  página {pagina}: {len(cabecalhos)} identificadores (acumulado: {total})")

        token_el = raiz.find("oai:ListIdentifiers/oai:resumptionToken", OAI_NS)
        token = (token_el.text or "").strip() if token_el is not None else ""
        if not token:
            return
        # Nas páginas seguintes o token substitui o metadataPrefix (spec OAI-PMH).
        params = {"verb": "ListIdentifiers", "resumptionToken": token}
        time.sleep(PAUSA_DSPACE)


# --------------------------------------------------------------------------- #
# Etapa 2 — handle → uuid                                                      #
# --------------------------------------------------------------------------- #

# Rota de resolução que funcionou na última vez (ver resolver_uuid).
_ROTA_PREFERIDA = ["core/handles"]


def _uuid_do_json(dados: dict) -> str:
    """`core/handles` embala o item em _embedded.indexableObject; `pid/find`
    redireciona para o próprio item, que traz uuid na raiz."""
    if not isinstance(dados, dict):
        return ""
    embutido = dados.get("_embedded") or {}
    alvo = embutido.get("indexableObject") if isinstance(embutido, dict) else None
    if isinstance(alvo, dict) and alvo.get("uuid"):
        return str(alvo["uuid"])
    return str(dados.get("uuid") or dados.get("id") or "")


def _consultar_dspace(sessao: requests.Session, url: str) -> Tuple[str, str]:
    """(uuid, erro) para uma URL. Retenta rede, 429 e 5xx; 404 é definitivo."""
    for tentativa in range(1, TENTATIVAS + 1):
        try:
            r = sessao.get(url, timeout=(10, TIMEOUT_HTTP), headers={"Accept": "application/json"})
            if r.status_code == 404:
                return "", "HTTP 404"
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(f"HTTP {r.status_code}")
            r.raise_for_status()
            uuid = _uuid_do_json(r.json())
            return (uuid, "") if uuid else ("", "resposta sem uuid")
        except ValueError as e:  # JSON inválido
            return "", f"resposta não-JSON: {e}"
        except requests.RequestException as e:
            if tentativa == TENTATIVAS:
                return "", f"falha de acesso: {e}"
            time.sleep(ESPERA_BASE * tentativa)
    return "", "inalcançável"


def resolver_uuid(sessao: requests.Session, handle: str) -> Tuple[str, str]:
    """Devolve (uuid, erro). uuid vazio == não resolveu.

    Duas rotas, na ordem: `/server/api/core/handles/{handle}` (a canônica do
    DSpace 7) e, se ela não existir nesta instância, `/server/api/pid/find`, que
    responde 302 para /core/items/{uuid} — é a que o devrdapp atende hoje.
    """
    codificado = quote(handle, safe="")
    rotas = {
        "core/handles": f"{DSPACE_BASE}/server/api/core/handles/{codificado}",
        "pid/find": f"{DSPACE_BASE}/server/api/pid/find?id=hdl:{codificado}",
    }
    # A rota que funcionou antes vai primeiro: numa instância sem `core/handles`,
    # isso poupa um 404 por handle (e o jitter de vários segundos que ele às vezes
    # pega). Se ela falhar, as outras ainda são tentadas.
    ordem = sorted(rotas, key=lambda nome: nome != _ROTA_PREFERIDA[0])
    erros: List[str] = []
    for nome in ordem:
        uuid, erro = _consultar_dspace(sessao, rotas[nome])
        if uuid:
            _ROTA_PREFERIDA[0] = nome
            return uuid, ""
        erros.append(f"{nome}: {erro}")
    return "", "; ".join(erros)


# --------------------------------------------------------------------------- #
# Etapa 3 — envio assíncrono para a API do RDAPP                               #
# --------------------------------------------------------------------------- #

async def enviar_uuid(
    sessao: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    handle: str,
    uuid: str,
    force: bool = False,
) -> Tuple[str, str, str, str]:
    """(handle, uuid, status_envio, erro). Retenta rede, 429 e 5xx.

    `force=True` manda `?force=true`, que reprocessa o item ignorando os artefatos
    já existentes (§32) — usado pelo sincronizar_novos_itens.py em item modificado.
    """
    url = f"{RDAPP_BASE}/api/files/dspace/item/{uuid}"
    if force:
        url += "?force=true"
    async with sem:
        for tentativa in range(1, TENTATIVAS + 1):
            try:
                async with sessao.post(url) as resp:
                    corpo = (await resp.text())[:300].replace("\n", " ").strip()
                    if 200 <= resp.status <= 299:
                        return handle, uuid, str(resp.status), ""
                    if resp.status == 429 or resp.status >= 500:
                        if tentativa == TENTATIVAS:
                            return handle, uuid, str(resp.status), corpo
                        espera = ESPERA_BASE * tentativa
                        cabecalho = resp.headers.get("Retry-After")
                        if resp.status == 429 and cabecalho and cabecalho.isdigit():
                            espera = min(float(cabecalho), 30.0)
                        await asyncio.sleep(espera)
                        continue
                    # 4xx que não é 429: erro definitivo (item sem PDF, uuid inválido).
                    return handle, uuid, str(resp.status), corpo
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if tentativa == TENTATIVAS:
                    return handle, uuid, "ERRO_REDE", f"{type(e).__name__}: {e}"
                await asyncio.sleep(ESPERA_BASE * tentativa)
    return handle, uuid, "ERRO_REDE", "inalcançável"


# --------------------------------------------------------------------------- #
# Orquestração                                                                 #
# --------------------------------------------------------------------------- #

async def processar(args: argparse.Namespace) -> int:
    handles_ok, uuids_ok = ler_ja_enviados(args.csv)
    if handles_ok:
        print(f"CSV existente: {len(handles_ok)} handle(s) já enviado(s) com sucesso — serão pulados.\n")

    registro = RegistroCsv(args.csv)
    http = requests.Session()
    http.headers["User-Agent"] = USER_AGENT

    encontrados = 0
    pulados = 0
    resolvidos = 0
    sem_uuid = 0
    enviados = 0
    falhas = 0
    vistos: Set[str] = set(uuids_ok)  # nunca reenviar o mesmo uuid
    pendentes: Set[asyncio.Task] = set()

    def colher(concluidas) -> None:
        nonlocal enviados, falhas
        for tarefa in concluidas:
            handle, uuid, status, erro = tarefa.result()
            registro.registrar(handle, uuid, status, erro)
            if _sucesso(status):
                enviados += 1
            else:
                falhas += 1
                print(f"  [falha] {handle} ({uuid}): {status} {erro}")

    print(f"Colhendo identificadores em {OAI_ENDPOINT} ...")
    tempo = time.monotonic()

    timeout = aiohttp.ClientTimeout(total=TIMEOUT_ENVIO)
    conector = aiohttp.TCPConnector(limit=CONCORRENCIA)
    async with aiohttp.ClientSession(
        timeout=timeout, connector=conector, headers={"User-Agent": USER_AGENT}
    ) as sessao:
        sem = asyncio.Semaphore(CONCORRENCIA)
        try:
            for handle in listar_handles(http, args.metadata_prefix):
                encontrados += 1
                if handle in handles_ok:
                    pulados += 1
                    continue
                if args.limite and (resolvidos + sem_uuid) >= args.limite:
                    print(f"\n[limite] parando após {args.limite} handle(s) novo(s).")
                    break

                uuid, erro = await asyncio.to_thread(resolver_uuid, http, handle)
                if not uuid:
                    sem_uuid += 1
                    registro.registrar(handle, "", "SEM_UUID", erro)
                    print(f"  [sem uuid] {handle}: {erro}")
                    continue
                resolvidos += 1
                if uuid in vistos:
                    print(f"  [duplicado] {handle} → {uuid} já visto, não reenviado")
                    continue
                vistos.add(uuid)

                if args.sem_envio:
                    registro.registrar(handle, uuid, "NAO_ENVIADO", "--sem-envio")
                else:
                    pendentes.add(asyncio.create_task(enviar_uuid(sessao, sem, handle, uuid)))

                if resolvidos % 25 == 0:
                    print(
                        f"  progresso: {encontrados} handles | {resolvidos} uuids | "
                        f"{enviados} enviados | {falhas} falhas | {len(pendentes)} em voo"
                    )

                # Não deixa a fila de envio crescer sem limite: drena o que já terminou.
                if len(pendentes) >= CONCORRENCIA * 4:
                    prontas, pendentes = await asyncio.wait(
                        pendentes, return_when=asyncio.FIRST_COMPLETED
                    )
                    colher(prontas)

                await asyncio.sleep(PAUSA_DSPACE)

            if pendentes:
                print(f"\nAguardando {len(pendentes)} envio(s) em voo ...")
                prontas, _ = await asyncio.wait(pendentes)
                pendentes = set()
                colher(prontas)
        except KeyboardInterrupt:
            print("\n[interrompido] cancelando envios em voo; o CSV já está no disco.")
            for tarefa in pendentes:
                tarefa.cancel()
        except RuntimeError as e:
            print(f"\n[erro] {e}", file=sys.stderr)
            if pendentes:
                prontas, _ = await asyncio.wait(pendentes)
                colher(prontas)
            registro.fechar()
            return 1

    registro.fechar()
    decorrido = time.monotonic() - tempo

    print("\n" + "=" * 62)
    print(f"handles encontrados no OAI-PMH ... {encontrados}")
    print(f"pulados (sucesso no CSV) ........ {pulados}")
    print(f"uuids resolvidos ................ {resolvidos}")
    print(f"handles sem uuid ................ {sem_uuid}")
    print(f"enviados com sucesso (2xx) ...... {enviados}")
    print(f"falhas de envio ................. {falhas}")
    print(f"tempo ........................... {decorrido:.1f}s")
    print(f"CSV ............................. {os.path.abspath(args.csv)}")
    print("=" * 62)
    return 0 if falhas == 0 and sem_uuid == 0 else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default=CSV_PATH, help=f"arquivo de saída (padrão: {CSV_PATH})")
    p.add_argument("--metadata-prefix", default="oai_dc", help="metadataPrefix do OAI-PMH")
    p.add_argument("--limite", type=int, default=0, help="processa só N handles novos (0 = todos)")
    p.add_argument("--sem-envio", action="store_true", help="só resolve os uuids, não chama a API")
    args = p.parse_args()
    try:
        return asyncio.run(processar(args))
    except KeyboardInterrupt:
        print("\n[interrompido]")
        return 130


if __name__ == "__main__":
    sys.exit(main())
