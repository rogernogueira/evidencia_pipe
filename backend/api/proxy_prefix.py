"""Acomoda um proxy reverso que REMOVE o prefixo do caminho.

Contexto (ver DEPLOY.md §8.1): as rotas públicas deste backend já nascem com o
prefixo `/api` (`/api/files/...`, `/api/search/...`), então o proxy da borda
deveria repassar `/api` sem reescrever. A borda do IBICT foi configurada com
`ProxyPass /api/ → http://<host>:8020/`, que TIRA o prefixo. Consequência: um
`GET https://devrdapp.ibict.br/api/files/active` chega aqui como `/files/active`
e devolve 404 — as rotas reais só respondem no caminho duplicado `/api/api/...`.

Enquanto a borda não for corrigida (a máquina é de outro time), este middleware
recoloca o prefixo: `/files/active` → `/api/files/active`. Só é ativado com
`PROXY_STRIPPED_PREFIX=/api` no .env; sem a variável, nada muda.

O segundo efeito do strip é pior que o 404: com `/api/` mapeado para a raiz do
backend, tudo que estava fora do proxy de propósito passou a responder na
internet — `/docs`, `/redoc`, `/openapi.json`, as rotas administrativas
`/internal/*` e o mount `/output`. Recolocar o prefixo NÃO fecha esse buraco,
então o middleware também recusa esse conjunto quando o request veio pela borda.
"""

from __future__ import annotations

from typing import Iterable, Iterator, Optional, Set

from starlette.types import ASGIApp, Receive, Scope, Send

from backend.core import config as settings
from backend.core.logger import log

# O que nunca deveria ser alcançável pela internet. `/health` fica de fora: não
# revela nada e serve de health check para o próprio proxy.
ADMIN_PATHS = ("/docs", "/redoc", "/openapi.json", "/internal", "/output")

# Corpo idêntico ao 404 do FastAPI: para quem sonda de fora, a rota bloqueada
# é indistinguível de uma que não existe.
_NOT_FOUND = b'{"detail":"Not Found"}'


def _e_admin(path: str) -> bool:
    return any(path == p or path.startswith(p + "/") for p in ADMIN_PATHS)


class StrippedPrefixMiddleware:
    """Recoloca `prefix` nos caminhos que o proxy da borda removeu.

    Reescreve apenas o que é reconhecidamente do router com prefixo: o primeiro
    segmento do caminho tem de estar em `segmentos` (calculado das rotas
    registradas, ver `segmentos_do_router`). Assim `/health`, `/internal/*` e
    `/output/*`, que existem na raiz de verdade, nunca são mexidos.
    """

    def __init__(self, app: ASGIApp, *, prefix: str, segmentos: Iterable[str],
                 guarda_admin: bool = True) -> None:
        self.app = app
        self.prefix = prefix
        self.segmentos: Set[str] = set(segmentos)
        self.guarda_admin = guarda_admin

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path: str = scope.get("path", "")

        # O X-Forwarded-For é adicionado pelo mod_proxy em todo request que ele
        # repassa, e o cliente não tem como impedir isso — ele é o sinal de
        # "veio de fora". Quem fala direto com a porta 8020 (rede interna,
        # scripts, systemd) não manda o header e continua com acesso total.
        if self.guarda_admin and _e_admin(path) and self._veio_do_proxy(scope):
            await self._nao_encontrado(send)
            return

        primeiro = path.split("/", 2)[1] if path.startswith("/") else ""
        if primeiro in self.segmentos and not path.startswith(self.prefix + "/"):
            scope = dict(scope)
            scope["path"] = self.prefix + path
            # raw_path é o caminho percent-encoded como veio na linha de request;
            # deixá-lo dessincronizado do path faz o Starlette montar redirect e
            # url_for com o caminho antigo.
            bruto = scope.get("raw_path")
            if bruto:
                scope["raw_path"] = self.prefix.encode() + bruto

        await self.app(scope, receive, send)

    @staticmethod
    def _veio_do_proxy(scope: Scope) -> bool:
        return any(nome == b"x-forwarded-for" for nome, _ in scope.get("headers", []))

    @staticmethod
    async def _nao_encontrado(send: Send) -> None:
        await send({
            "type": "http.response.start",
            "status": 404,
            "headers": [(b"content-type", b"application/json"),
                        (b"content-length", str(len(_NOT_FOUND)).encode())],
        })
        await send({"type": "http.response.body", "body": _NOT_FOUND})


def caminhos_registrados(no, vistos: Optional[set] = None) -> Iterator[str]:
    """Todos os caminhos de rota alcançáveis a partir de um app ou router.

    A varredura é recursiva porque o FastAPI (≥0.139) não materializa mais o
    `include_router()` na hora: `app.routes` guarda um marcador e as rotas de
    verdade seguem dentro do router original — ler só `app.routes` devolveria
    `/health`, `/docs` e nada de `/api/...`. Também é deliberadamente tolerante:
    segue quem expõe `.routes` ou `.original_router` e ignora o resto, para não
    quebrar quando esses detalhes internos mudarem de nome.
    """
    if vistos is None:
        vistos = set()
    if id(no) in vistos:
        return
    vistos.add(id(no))

    caminho = getattr(no, "path", None)
    if isinstance(caminho, str) and caminho.startswith("/"):
        yield caminho

    incluido = getattr(no, "original_router", None)
    if incluido is not None:
        yield from caminhos_registrados(incluido, vistos)

    try:
        filhas = list(getattr(no, "routes", None) or [])
    except Exception:  # pragma: no cover - .routes pode ser property de terceiros
        filhas = []
    for filha in filhas:
        yield from caminhos_registrados(filha, vistos)


def segmentos_do_router(app, prefix: str) -> Set[str]:
    """Primeiros segmentos sob `prefix` que podem ser recolocados com segurança.

    Ex.: com as rotas `/api/files/...` e `/api/search/...`, devolve
    {'files', 'search'} — e o middleware passa a reescrever só `/files/...` e
    `/search/...`. Segmentos que JÁ existem na raiz são descartados: reescrever
    esconderia a rota real, e um 404 novo é pior que o 404 que se quer curar.
    """
    sob_prefixo: Set[str] = set()
    na_raiz: Set[str] = set()
    for caminho in caminhos_registrados(app):
        if caminho.startswith(prefix + "/"):
            sob_prefixo.add(caminho[len(prefix) + 1:].split("/", 1)[0])
        elif caminho != "/":
            na_raiz.add(caminho.split("/", 2)[1])

    conflitos = sob_prefixo & na_raiz
    if conflitos:
        log.warning("[proxy] segmentos ignorados por já existirem na raiz: %s",
                    ", ".join(sorted(conflitos)))
    return sob_prefixo - na_raiz


def instala_acomodacao_de_proxy(app) -> None:
    """Liga o middleware se PROXY_STRIPPED_PREFIX estiver configurado."""
    prefix = settings.PROXY_STRIPPED_PREFIX
    if not prefix:
        return

    segmentos = segmentos_do_router(app, prefix)
    if not segmentos:
        log.warning("[proxy] PROXY_STRIPPED_PREFIX=%s, mas nenhuma rota usa esse "
                    "prefixo — middleware não instalado.", prefix)
        return

    app.add_middleware(StrippedPrefixMiddleware, prefix=prefix, segmentos=segmentos,
                       guarda_admin=settings.PROXY_GUARD_ADMIN)
    log.info("[proxy] acomodando proxy que remove '%s': %s → %s/... (guarda das "
             "rotas administrativas: %s)", prefix, sorted(segmentos), prefix,
             "on" if settings.PROXY_GUARD_ADMIN else "OFF")
