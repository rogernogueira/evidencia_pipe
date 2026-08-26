#!/usr/bin/env python3
"""Sincroniza os itens NOVOS do DSpace com a API do RDAPP — feito para o cron.

    pip install requests lxml aiohttp     # as mesmas do extrair_e_enviar_uuids.py

É a segunda versão do `extrair_e_enviar_uuids.py`. Aquele varre o acervo inteiro a
cada execução; este roda de meia em meia hora e só olha o que mudou desde a última
vez, usando **colheita seletiva** do OAI-PMH (`&from=`, que o devrdapp atende com
granularidade de segundos — veja `verb=Identify`).

O que muda em relação à v1, e por quê:

  * **Marca d'água.** O estado (`estado_sincronizacao.json`) guarda o maior
    `datestamp` já colhido. A execução seguinte pede só `from=marca - sobreposição`
    (48 h por padrão). A sobreposição existe porque o índice OAI do DSpace é
    alimentado por um `dspace oai import` periódico: um item pode aparecer na
    colheita horas depois do `datestamp` que carrega. Re-listar é barato — quem já
    está no CSV com 2xx é descartado antes de qualquer chamada ao DSpace.
  * **Varredura completa periódica.** A cada 7 dias (`--dias-completo`) ignora a
    marca d'água e relê tudo. É a rede de segurança contra qualquer buraco: item
    que escapou da janela, item que esgotou as retentativas.
  * **Fila de retentativa.** Handle que falhou fica no estado com um contador e é
    retentado em toda execução, independentemente da janela; some da fila depois de
    5 tentativas (e aí a varredura completa o pega). Por isso a marca d'água pode
    avançar sem risco de perder item.
  * **Trava.** `flock` num arquivo: se a execução anterior ainda está rodando, esta
    sai na hora, sem erro. Cron a cada 30 min com uma carga que leva 40 min não
    empilha processos.
  * **Checagem antes de colher.** Consulta `GET /api/status`; se a capacidade
    `ingestao` estiver indisponível (MinIO fora, Redis fora, worker parado), pula a
    rodada **sem** avançar a marca d'água — nada se perde, e não se queima uma
    janela mandando POST para um pipeline que não consegue processar. Essa rota
    exige Bearer de admin do DSpace: com `RDAPP_ADMIN_TOKEN` (ou `--token`) a
    checagem acontece — o que serve para execução manual, já que o token do DSpace
    é de vida curta e no cron o normal é ele já ter expirado. Sem token válido a
    rodada SEGUE e o log registra que não deu para checar: 401 é "não enxerguei",
    não "o pipeline está fora", e pular por isso pararia a sincronização para sempre.
  * **Log em arquivo, silêncio no stdout.** Tudo vai para `logs/sincronizacao.log`
    (rotativo). No stdout/stderr só sai WARNING+, então o cron **só manda e-mail
    quando há problema**. Com `--verboso` o log também aparece no terminal.
  * **Prazo.** `--tempo-maximo` (1 h por padrão) encerra a rodada com elegância:
    drena os envios em voo, salva o estado e não avança a marca d'água, então o que
    faltou entra na próxima.

O CSV é o **mesmo** da v1 (`itens_uuid.csv`) — os dois scripts compartilham o
histórico e nenhum reenvia o que o outro já enviou com sucesso.

Códigos de saída (o cron avisa pelo e-mail quando != 0):

    0  tudo certo — inclusive "nada novo" e "outra instância já está rodando"
    1  rodou, mas algum item falhou (está na fila de retentativa)
    2  erro fatal (OAI-PMH inacessível, estado ilegível)
    3  rodada pulada: a API não consegue ingerir agora

Uso:

    python scripts/sincronizar_novos_itens.py                 # o que o cron chama
    python scripts/sincronizar_novos_itens.py --verboso       # à mão, vendo tudo
    python scripts/sincronizar_novos_itens.py --mostrar-estado
    python scripts/sincronizar_novos_itens.py --completo      # varredura completa já
    python scripts/sincronizar_novos_itens.py --crontab       # a linha do crontab
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import fcntl
import io
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from logging.handlers import RotatingFileHandler
from typing import Dict, Iterator, List, Optional, Set, Tuple

import aiohttp
import requests
from lxml import etree

# A v1 é a biblioteca desta v2: resolução de handle→uuid, POST com retentativa,
# CSV append-only e leitura do que já foi enviado vêm de lá, sem cópia.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extrair_e_enviar_uuids as base  # noqa: E402

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ESTADO_PADRAO = os.path.join(RAIZ, "estado_sincronizacao.json")
TRAVA_PADRAO = os.path.join(RAIZ, ".sincronizacao.lock")
LOG_PADRAO = os.path.join(RAIZ, "logs", "sincronizacao.log")

SOBREPOSICAO_HORAS = 48   # o índice OAI do DSpace é atualizado em lote, com atraso
DIAS_COMPLETO = 7         # varredura completa de segurança
TEMPO_MAXIMO = 3600       # segundos; 0 desliga
MAX_TENTATIVAS_PENDENTE = 5
LOG_BYTES = 5_000_000
LOG_BACKUPS = 5

log = logging.getLogger("sincronizacao")


# --------------------------------------------------------------------------- #
# Tempo                                                                        #
# --------------------------------------------------------------------------- #

def agora() -> datetime:
    return datetime.now(timezone.utc)


def iso(momento: datetime) -> str:
    """Formato exigido pela granularidade do repositório: YYYY-MM-DDThh:mm:ssZ."""
    return momento.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def para_dt(texto: Optional[str]) -> Optional[datetime]:
    """Lê um datestamp do OAI (ou do estado). Tolera `YYYY-MM-DD` sem hora."""
    if not texto:
        return None
    bruto = texto.strip()
    try:
        momento = datetime.fromisoformat(bruto.replace("Z", "+00:00"))
    except ValueError:
        try:
            momento = datetime.strptime(bruto, "%Y-%m-%d")
        except ValueError:
            return None
    return momento if momento.tzinfo else momento.replace(tzinfo=timezone.utc)


# --------------------------------------------------------------------------- #
# Estado                                                                       #
# --------------------------------------------------------------------------- #

@dataclass
class Estado:
    """O que precisa sobreviver entre execuções do cron.

    `pendentes` é a fila de retentativa: handle → {uuid, tentativas, erro, desde}.
    """

    caminho: str
    marca_dagua: Optional[str] = None
    ultima_execucao: Optional[str] = None
    ultima_execucao_ok: Optional[str] = None
    ultimo_completo: Optional[str] = None
    pendentes: Dict[str, dict] = field(default_factory=dict)
    ultimo_resumo: dict = field(default_factory=dict)

    @classmethod
    def carregar(cls, caminho: str) -> "Estado":
        if not os.path.exists(caminho):
            return cls(caminho=caminho)
        with open(caminho, encoding="utf-8") as f:
            dados = json.load(f)          # JSON quebrado é erro fatal, não silêncio:
        if not isinstance(dados, dict):   # rodar sem estado reenviaria o acervo todo.
            raise ValueError(f"{caminho}: esperava um objeto JSON")
        return cls(
            caminho=caminho,
            marca_dagua=dados.get("marca_dagua"),
            ultima_execucao=dados.get("ultima_execucao"),
            ultima_execucao_ok=dados.get("ultima_execucao_ok"),
            ultimo_completo=dados.get("ultimo_completo"),
            pendentes=dados.get("pendentes") or {},
            ultimo_resumo=dados.get("ultimo_resumo") or {},
        )

    def salvar(self) -> None:
        """Grava por tmp+rename: um kill no meio nunca deixa estado pela metade."""
        dados = {
            "marca_dagua": self.marca_dagua,
            "ultima_execucao": self.ultima_execucao,
            "ultima_execucao_ok": self.ultima_execucao_ok,
            "ultimo_completo": self.ultimo_completo,
            "pendentes": self.pendentes,
            "ultimo_resumo": self.ultimo_resumo,
        }
        pasta = os.path.dirname(os.path.abspath(self.caminho))
        os.makedirs(pasta, exist_ok=True)
        temporario = f"{self.caminho}.tmp"
        with open(temporario, "w", encoding="utf-8") as f:
            json.dump(dados, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporario, self.caminho)

    def marcar_falha(self, handle: str, uuid: str, erro: str) -> bool:
        """Enfileira/incrementa a retentativa. Devolve True se desistiu do handle."""
        registro = self.pendentes.get(handle) or {"tentativas": 0, "desde": iso(agora())}
        registro["tentativas"] = int(registro.get("tentativas", 0)) + 1
        registro["uuid"] = uuid or registro.get("uuid", "")
        registro["erro"] = erro[:300]
        registro["ultima_tentativa"] = iso(agora())
        if registro["tentativas"] >= MAX_TENTATIVAS_PENDENTE:
            self.pendentes.pop(handle, None)
            return True
        self.pendentes[handle] = registro
        return False


# --------------------------------------------------------------------------- #
# Trava, log e configuração                                                    #
# --------------------------------------------------------------------------- #

@contextlib.contextmanager
def travar(caminho: str) -> Iterator[bool]:
    """flock não-bloqueante: `False` quando outra execução já está em curso."""
    arquivo = open(caminho, "a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(arquivo.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            arquivo.seek(0)
            yield False
            return
        arquivo.seek(0)
        arquivo.truncate()
        arquivo.write(f"{os.getpid()} {iso(agora())}\n")
        arquivo.flush()
        try:
            yield True
        finally:
            fcntl.flock(arquivo.fileno(), fcntl.LOCK_UN)
    finally:
        arquivo.close()


class SaidaParaLog(io.TextIOBase):
    """Adaptador de `print` → logger: os prints da v1 caem no arquivo de log."""

    def __init__(self, destino: logging.Logger, nivel: int):
        self._log = destino
        self._nivel = nivel

    def write(self, texto: str) -> int:
        for linha in texto.splitlines():
            if linha.strip():
                self._log.log(self._nivel, "%s", linha.rstrip())
        return len(texto)

    def flush(self) -> None:  # pragma: no cover - exigido pela interface
        pass


def configurar_log(caminho: str, verboso: bool) -> None:
    log.setLevel(logging.DEBUG)
    log.handlers.clear()
    formato = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%Y-%m-%d %H:%M:%S")

    # Console em stderr e só WARNING+: é isso que faz o cron mandar e-mail apenas
    # quando algo deu errado.
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.INFO if verboso else logging.WARNING)
    console.setFormatter(formato)
    log.addHandler(console)

    if caminho and caminho != "-":
        try:
            pasta = os.path.dirname(os.path.abspath(caminho))
            os.makedirs(pasta, exist_ok=True)
            arquivo = RotatingFileHandler(
                caminho, maxBytes=LOG_BYTES, backupCount=LOG_BACKUPS, encoding="utf-8"
            )
            arquivo.setLevel(logging.INFO)
            arquivo.setFormatter(formato)
            log.addHandler(arquivo)
        except OSError as e:
            print(f"[aviso] sem log em arquivo ({caminho}): {e}", file=sys.stderr)


def configurar_bases(dspace: str, rdapp: str) -> None:
    """Aponta a v1 para outras instâncias (`--dspace-base`/`--rdapp-base`).

    As funções da v1 leem esses nomes no módulo a cada chamada, então reatribuí-los
    aqui basta; `OAI_ENDPOINT` é derivado e precisa acompanhar.
    """
    base.DSPACE_BASE = dspace.rstrip("/")
    base.OAI_ENDPOINT = f"{base.DSPACE_BASE}/server/oai/request"
    base.RDAPP_BASE = rdapp.rstrip("/")


def checar_ingestao(rdapp_base: str, token: str = "") -> Tuple[bool, str]:
    """A API consegue ingerir agora? (`capabilities.ingestao` de GET /api/status).

    Retorna (pode_seguir, motivo). Duas situações diferentes cabem em `pode_seguir=True`:
    a capacidade está disponível (motivo vazio) ou **não foi possível checar** (motivo
    preenchido) — quem chama registra a segunda e segue.

    O `/api/status` exige Bearer de administrador do DSpace (ver DEPLOY.md §8.2) e
    este script roda sem sessão. Com RDAPP_ADMIN_TOKEN a checagem funciona como
    sempre; sem ele, um 401/403 significa "não enxerguei", não "o pipeline está
    fora" — pular a rodada por isso pararia a sincronização para sempre.
    """
    url = f"{rdapp_base}/api/status?artifacts=false"
    headers = {"User-Agent": base.USER_AGENT}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        r = requests.get(url, timeout=(10, 30), headers=headers)
    except requests.RequestException as e:
        return False, f"{url} inacessível: {type(e).__name__}: {e}"
    if r.status_code in (401, 403):
        return True, (f"{url} respondeu HTTP {r.status_code} — a rota exige admin do "
                      "DSpace e não há token (defina RDAPP_ADMIN_TOKEN ou use --token)")
    try:
        dados = r.json()
    except ValueError:
        return (200 <= r.status_code <= 299), f"{url} respondeu HTTP {r.status_code} sem JSON"
    capacidade = (dados.get("capabilities") or {}).get("ingestao") or {}
    if capacidade.get("available") is True:
        return True, ""
    if not capacidade:  # API antiga, sem `capabilities`: vale o código HTTP
        return (200 <= r.status_code <= 299), f"HTTP {r.status_code}"
    bloqueios = ", ".join(capacidade.get("blocked_by") or []) or dados.get("status", "?")
    return False, f"capacidade `ingestao` indisponível — bloqueada por: {bloqueios}"


# --------------------------------------------------------------------------- #
# Colheita seletiva                                                            #
# --------------------------------------------------------------------------- #

def listar_cabecalhos(
    sessao: requests.Session, prefixo: str, desde: Optional[str]
) -> Iterator[Tuple[str, str]]:
    """(handle, datestamp) do ListIdentifiers, opcionalmente a partir de `desde`.

    Igual ao `listar_handles` da v1, com duas diferenças: aceita `from=` e devolve
    também o datestamp — é dele que sai a marca d'água.
    """
    params: Dict[str, str] = {"verb": "ListIdentifiers", "metadataPrefix": prefixo}
    if desde:
        params["from"] = desde
    pagina = 0
    total = 0
    while True:
        xml = base._get_oai(sessao, params)
        pagina += 1
        raiz = etree.fromstring(xml)

        erro = raiz.find("oai:error", base.OAI_NS)
        if erro is not None:
            codigo = erro.get("code", "?")
            if codigo == "noRecordsMatch":
                return
            raise RuntimeError(f"OAI-PMH devolveu erro {codigo}: {(erro.text or '').strip()}")

        cabecalhos = raiz.findall("oai:ListIdentifiers/oai:header", base.OAI_NS)
        for cab in cabecalhos:
            if cab.get("status") == "deleted":
                continue
            handle = base.extrair_handle(cab.findtext("oai:identifier", namespaces=base.OAI_NS) or "")
            datestamp = (cab.findtext("oai:datestamp", namespaces=base.OAI_NS) or "").strip()
            if handle:
                total += 1
                yield handle, datestamp
        log.info("OAI página %d: %d identificadores (acumulado: %d)", pagina, len(cabecalhos), total)

        token_el = raiz.find("oai:ListIdentifiers/oai:resumptionToken", base.OAI_NS)
        token = (token_el.text or "").strip() if token_el is not None else ""
        if not token:
            return
        params = {"verb": "ListIdentifiers", "resumptionToken": token}
        time.sleep(base.PAUSA_DSPACE)


def _definitivo(status: str) -> bool:
    """Erro que não adianta retentar: 4xx que não seja 429."""
    try:
        codigo = int(status)
    except (TypeError, ValueError):
        return False
    return 400 <= codigo <= 499 and codigo != 429


@dataclass
class Candidato:
    handle: str
    datestamp: str = ""
    uuid: str = ""            # já conhecido, quando vem da fila de retentativa
    force: bool = False       # reprocessar item modificado (POST ...?force=true)
    origem: str = "novo"      # novo | modificado | retentativa


def montar_janela(estado: Estado, args: argparse.Namespace) -> Tuple[bool, Optional[str]]:
    """(varredura_completa, valor do `from`). Completa na 1ª vez e a cada N dias."""
    if args.completo or not estado.marca_dagua:
        return True, None
    if args.dias_completo:
        ultimo = para_dt(estado.ultimo_completo)
        if ultimo is None or agora() - ultimo >= timedelta(days=args.dias_completo):
            log.info("varredura completa periódica (última: %s)", estado.ultimo_completo or "nunca")
            return True, None
    marca = para_dt(estado.marca_dagua)
    if marca is None:
        log.warning("marca d'água ilegível (%r) — caindo para varredura completa", estado.marca_dagua)
        return True, None
    return False, iso(marca - timedelta(hours=args.sobreposicao))


# --------------------------------------------------------------------------- #
# Rodada                                                                       #
# --------------------------------------------------------------------------- #

async def rodar(args: argparse.Namespace, estado: Estado) -> int:
    inicio = time.monotonic()
    prazo = inicio + args.tempo_maximo if args.tempo_maximo else None

    if not args.sem_envio and not args.sem_checagem:
        pode, motivo = checar_ingestao(args.rdapp_base, args.token)
        if not pode:
            log.warning("rodada pulada — %s", motivo)
            estado.ultima_execucao = iso(agora())
            estado.salvar()
            return 3
        if motivo:  # deu para seguir, mas a checagem não foi feita (ver checar_ingestao)
            log.warning("seguindo sem a checagem prévia — %s", motivo)

    handles_ok, uuids_ok = base.ler_ja_enviados(args.csv)
    completo, desde = montar_janela(estado, args)
    log.info(
        "janela: %s | CSV com %d handle(s) já enviado(s) | fila de retentativa: %d",
        "varredura completa" if completo else f"from={desde}", len(handles_ok), len(estado.pendentes),
    )

    http = requests.Session()
    http.headers["User-Agent"] = base.USER_AGENT

    # A fila de retentativa entra primeiro e não depende da janela.
    candidatos: List[Candidato] = [
        Candidato(handle, uuid=(dados or {}).get("uuid", ""), origem="retentativa")
        for handle, dados in sorted(estado.pendentes.items())
        if handle not in handles_ok
    ]
    ja_na_fila = {c.handle for c in candidatos}
    for handle in list(estado.pendentes):
        if handle in handles_ok:      # a v1 (ou uma rodada anterior) já resolveu
            estado.pendentes.pop(handle, None)

    marca_anterior = para_dt(estado.marca_dagua)
    maior_datestamp: Optional[datetime] = None
    vistos = 0
    pulados = 0

    try:
        for handle, datestamp in listar_cabecalhos(http, args.metadata_prefix, desde):
            vistos += 1
            momento = para_dt(datestamp)
            if momento and (maior_datestamp is None or momento > maior_datestamp):
                maior_datestamp = momento
            if handle in ja_na_fila:
                continue
            if handle in handles_ok:
                # Item já enviado que voltou a aparecer: só reprocessa se mudou DEPOIS
                # da última marca d'água — a sobreposição de 48 h reapresenta itens
                # que já estavam em dia, e eles não devem ser reenviados.
                modificado = bool(momento and marca_anterior and momento > marca_anterior)
                if args.reprocessar_modificados and modificado:
                    candidatos.append(Candidato(handle, datestamp, force=True, origem="modificado"))
                else:
                    pulados += 1
                continue
            candidatos.append(Candidato(handle, datestamp, origem="novo"))
    except RuntimeError as e:
        log.error("colheita OAI-PMH falhou: %s", e)
        estado.ultima_execucao = iso(agora())
        estado.salvar()
        return 2

    truncado = False
    if args.max_itens and len(candidatos) > args.max_itens:
        log.info("%d candidatos — processando %d nesta rodada", len(candidatos), args.max_itens)
        candidatos = candidatos[: args.max_itens]
        truncado = True

    novos = sum(1 for c in candidatos if c.origem == "novo")
    modificados = sum(1 for c in candidatos if c.origem == "modificado")
    retentativas = sum(1 for c in candidatos if c.origem == "retentativa")
    log.info(
        "%d identificador(es) na janela | %d novo(s), %d modificado(s), %d retentativa(s), %d sem mudança",
        vistos, novos, modificados, retentativas, pulados,
    )

    if not candidatos:
        estado.ultima_execucao = estado.ultima_execucao_ok = iso(agora())
        if maior_datestamp:
            estado.marca_dagua = iso(maior_datestamp)
        if completo:
            estado.ultimo_completo = iso(agora())
        estado.ultimo_resumo = {"quando": iso(agora()), "nada_novo": True, "vistos": vistos}
        estado.salvar()
        log.info("nada a fazer (%d identificador(es) sem novidade)", vistos)
        return 0

    registro = base.RegistroCsv(args.csv)
    enviados = 0
    falhas = 0
    sem_uuid = 0
    desistidos: List[str] = []
    por_handle = {c.handle: c for c in candidatos}
    ja_enviados_uuid: Set[str] = set(uuids_ok)
    pendentes_tarefas: Set[asyncio.Task] = set()
    prazo_estourado = False

    def colher(concluidas) -> None:
        nonlocal enviados, falhas
        for tarefa in concluidas:
            handle, uuid, status, erro = tarefa.result()
            registro.registrar(handle, uuid, status, erro)
            if base._sucesso(status):
                enviados += 1
                estado.pendentes.pop(handle, None)
                log.info("enviado %s → %s (HTTP %s)", handle, uuid, status)
            else:
                falhas += 1
                origem = por_handle[handle].origem if handle in por_handle else "?"
                if _definitivo(status):
                    # 4xx que não é 429 (item sem PDF no bundle ORIGINAL, uuid inválido):
                    # retentar de meia em meia hora não muda nada. Fica só no CSV, e a
                    # varredura completa semanal dá a nova chance — se um PDF aparecer.
                    estado.pendentes.pop(handle, None)
                elif estado.marcar_falha(handle, uuid, f"HTTP {status}: {erro}"):
                    desistidos.append(handle)
                log.warning("falha no envio de %s (%s, %s): %s %s", handle, uuid, origem, status, erro)

    timeout = aiohttp.ClientTimeout(total=base.TIMEOUT_ENVIO)
    conector = aiohttp.TCPConnector(limit=base.CONCORRENCIA)
    async with aiohttp.ClientSession(
        timeout=timeout, connector=conector, headers={"User-Agent": base.USER_AGENT}
    ) as sessao:
        sem = asyncio.Semaphore(base.CONCORRENCIA)
        try:
            for indice, candidato in enumerate(candidatos, 1):
                if prazo and time.monotonic() > prazo:
                    prazo_estourado = True
                    log.warning(
                        "tempo máximo (%ds) atingido em %d/%d — o resto fica para a próxima rodada",
                        args.tempo_maximo, indice - 1, len(candidatos),
                    )
                    break

                uuid = candidato.uuid
                if not uuid:
                    uuid, erro = await asyncio.to_thread(base.resolver_uuid, http, candidato.handle)
                    if not uuid:
                        sem_uuid += 1
                        registro.registrar(candidato.handle, "", "SEM_UUID", erro)
                        if estado.marcar_falha(candidato.handle, "", erro):
                            desistidos.append(candidato.handle)
                        log.warning("sem uuid para %s: %s", candidato.handle, erro)
                        continue
                    candidato.uuid = uuid

                if uuid in ja_enviados_uuid and not candidato.force:
                    log.info("%s → %s já enviado (outro handle do mesmo item)", candidato.handle, uuid)
                    estado.pendentes.pop(candidato.handle, None)
                    continue
                ja_enviados_uuid.add(uuid)

                if args.sem_envio:
                    registro.registrar(candidato.handle, uuid, "NAO_ENVIADO", "--sem-envio")
                    continue

                pendentes_tarefas.add(asyncio.create_task(
                    base.enviar_uuid(sessao, sem, candidato.handle, uuid, force=candidato.force)
                ))
                if len(pendentes_tarefas) >= base.CONCORRENCIA * 4:
                    prontas, pendentes_tarefas = await asyncio.wait(
                        pendentes_tarefas, return_when=asyncio.FIRST_COMPLETED
                    )
                    colher(prontas)
                await asyncio.sleep(base.PAUSA_DSPACE)

            if pendentes_tarefas:
                log.info("aguardando %d envio(s) em voo", len(pendentes_tarefas))
                prontas, _ = await asyncio.wait(pendentes_tarefas)
                pendentes_tarefas = set()
                colher(prontas)
        finally:
            if pendentes_tarefas:      # KeyboardInterrupt/SIGTERM no meio da rodada
                for tarefa in pendentes_tarefas:
                    tarefa.cancel()
            registro.fechar()

    # A marca d'água só avança quando a rodada viu a janela INTEIRA e processou tudo
    # o que ela trouxe. Truncar por `--max-itens` ou por prazo mantém a marca parada,
    # e o resto reaparece na próxima rodada. Falha de item não trava a marca: ela vai
    # para a fila de retentativa.
    avancou = not truncado and not prazo_estourado
    if avancou and maior_datestamp:
        anterior = estado.marca_dagua
        estado.marca_dagua = iso(maior_datestamp)
        if anterior != estado.marca_dagua:
            log.info("marca d'água: %s → %s", anterior or "(nenhuma)", estado.marca_dagua)
    if avancou and completo:
        estado.ultimo_completo = iso(agora())

    for handle in desistidos:
        log.warning(
            "desisti de %s após %d tentativas — a varredura completa (a cada %d dias) tenta de novo",
            handle, MAX_TENTATIVAS_PENDENTE, args.dias_completo or DIAS_COMPLETO,
        )

    decorrido = time.monotonic() - inicio
    resumo = {
        "quando": iso(agora()),
        "janela": "completa" if completo else f"from={desde}",
        "vistos": vistos,
        "novos": novos,
        "modificados": modificados,
        "retentativas": retentativas,
        "enviados": enviados,
        "falhas": falhas,
        "sem_uuid": sem_uuid,
        "pendentes": len(estado.pendentes),
        "segundos": round(decorrido, 1),
        "marca_dagua_avancou": avancou,
    }
    estado.ultima_execucao = iso(agora())
    if falhas == 0 and sem_uuid == 0:
        estado.ultima_execucao_ok = estado.ultima_execucao
    estado.ultimo_resumo = resumo
    estado.salvar()

    log.info(
        "fim: %d enviado(s), %d falha(s), %d sem uuid, %d na fila, %.1fs",
        enviados, falhas, sem_uuid, len(estado.pendentes), decorrido,
    )
    if falhas or sem_uuid:
        log.warning(
            "%d item(ns) não entraram (%d falha(s) de envio, %d sem uuid) — %d na fila de retentativa",
            falhas + sem_uuid, falhas, sem_uuid, len(estado.pendentes),
        )
        return 1
    return 0


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

def imprimir_crontab(args: argparse.Namespace) -> None:
    venv = os.path.join(RAIZ, ".venv", "bin", "python")
    python = venv if os.path.exists(venv) else sys.executable
    script = os.path.join(RAIZ, "scripts", os.path.basename(__file__))
    print("# evidencia_pipe — itens novos do DSpace, de 30 em 30 minutos.")
    print("# `crontab -e` (do usuário que roda o pipeline) e cole as duas linhas:")
    print("MAILTO=root   # e-mail só quando o script sai != 0; o log fica no arquivo")
    print(f"*/30 * * * * cd {RAIZ} && {python} {script}")
    print()
    print(f"# log:    {args.log}")
    print(f"# estado: {args.estado}   (--mostrar-estado para ler)")


def mostrar_estado(estado: Estado) -> None:
    print(f"arquivo ................. {estado.caminho}")
    print(f"marca d'água ............ {estado.marca_dagua or '(nenhuma — a próxima é completa)'}")
    print(f"última execução ......... {estado.ultima_execucao or '(nunca)'}")
    print(f"última sem falhas ....... {estado.ultima_execucao_ok or '(nunca)'}")
    print(f"última varredura completa {estado.ultimo_completo or '(nunca)'}")
    print(f"fila de retentativa ..... {len(estado.pendentes)}")
    for handle, dados in sorted(estado.pendentes.items()):
        print(f"  {handle}  tentativas={dados.get('tentativas')}  {str(dados.get('erro'))[:80]}")
    if estado.ultimo_resumo:
        print("último resumo:")
        for chave, valor in estado.ultimo_resumo.items():
            print(f"  {chave} = {valor}")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--csv", default=os.path.join(RAIZ, base.CSV_PATH), help="ledger compartilhado com a v1")
    p.add_argument("--estado", default=ESTADO_PADRAO, help="JSON com marca d'água e fila")
    p.add_argument("--trava", default=TRAVA_PADRAO, help="arquivo de flock (uma execução por vez)")
    p.add_argument("--log", default=LOG_PADRAO, help="arquivo de log rotativo ('-' desliga)")
    p.add_argument("--verboso", action="store_true", help="também joga o log no console")
    p.add_argument("--completo", action="store_true", help="ignora a marca d'água e relê tudo")
    p.add_argument("--dias-completo", type=int, default=DIAS_COMPLETO,
                   help=f"varredura completa a cada N dias (0 desliga; padrão {DIAS_COMPLETO})")
    p.add_argument("--sobreposicao", type=int, default=SOBREPOSICAO_HORAS,
                   help=f"horas relidas antes da marca d'água (padrão {SOBREPOSICAO_HORAS})")
    p.add_argument("--max-itens", type=int, default=0, help="teto de itens por rodada (0 = sem teto)")
    p.add_argument("--tempo-maximo", type=int, default=TEMPO_MAXIMO,
                   help=f"segundos até encerrar com elegância (0 desliga; padrão {TEMPO_MAXIMO})")
    p.add_argument("--reprocessar-modificados", action="store_true",
                   help="reenvia com force=true o item já enviado cujo datestamp mudou")
    p.add_argument("--sem-envio", action="store_true", help="só resolve os uuids, não chama a API")
    p.add_argument("--sem-checagem", action="store_true", help="não consulta GET /api/status antes")
    p.add_argument("--metadata-prefix", default="oai_dc", help="metadataPrefix do OAI-PMH")
    p.add_argument("--dspace-base", default=os.environ.get("DSPACE_URL", base.DSPACE_BASE))
    p.add_argument("--rdapp-base", default=os.environ.get("RDAPP_API_URL", base.RDAPP_BASE))
    p.add_argument("--token", default=os.environ.get("RDAPP_ADMIN_TOKEN", ""),
                   help="Bearer de admin do DSpace para a checagem prévia (padrão: "
                        "$RDAPP_ADMIN_TOKEN). Token do DSpace expira: sem um válido, a "
                        "checagem é pulada e a rodada segue")
    p.add_argument("--crontab", action="store_true", help="imprime a linha do crontab e sai")
    p.add_argument("--mostrar-estado", action="store_true", help="imprime o estado atual e sai")
    args = p.parse_args()

    if args.crontab:
        imprimir_crontab(args)
        return 0

    configurar_log(args.log, args.verboso)
    configurar_bases(args.dspace_base, args.rdapp_base)

    try:
        estado = Estado.carregar(args.estado)
    except (OSError, ValueError, json.JSONDecodeError) as e:
        log.error("estado ilegível em %s: %s (mova o arquivo para forçar uma varredura completa)",
                  args.estado, e)
        return 2

    if args.mostrar_estado:
        mostrar_estado(estado)
        return 0

    with travar(args.trava) as livre:
        if not livre:
            log.info("outra execução ainda está rodando (%s) — saindo", args.trava)
            return 0
        # Os `print` da v1 (retentativas do OAI, progresso) viram linhas de log.
        with contextlib.redirect_stdout(SaidaParaLog(log, logging.INFO)):
            try:
                return asyncio.run(rodar(args, estado))
            except KeyboardInterrupt:
                log.warning("interrompido — o CSV e o estado já estão no disco")
                return 130
            except Exception as e:  # nada de traceback silencioso num job de cron
                log.exception("erro fatal: %s", e)
                return 2


if __name__ == "__main__":
    sys.exit(main())
