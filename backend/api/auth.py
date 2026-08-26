"""Autorização administrativa das rotas de ingestão — quem autoriza é o DSpace.

As rotas de `/api/files/*` mostram e mexem na fila de ingestão do repositório:
nomes de arquivo, UUIDs, mensagens de erro e o reprocessamento (escrita). Elas
passaram a exigir o MESMO Bearer que o dspace-angular já usa, e a API pergunta ao
DSpace se aquele token é de um administrador. Não há chave própria nem segredo
compartilhado: manter um segundo universo de permissões, desalinhado do DSpace,
é pior que não ter nenhum. O mesmo vale para o `/api/status`, que descreve a
infraestrutura inteira. Abertos ficam a busca (`/api/search/*`) e o `/health`.

São DUAS chamadas ao DSpace, e a segunda não é redundante:

  1. GET /api/authn/status
     Diz QUEM é o dono do token (e se ele ainda vale) — não diz o que ele pode.

  2. GET /api/authz/authorizations/search/object?uri={site}&feature=administratorOf
     É o mesmo endpoint que a UI do DSpace usa para decidir o que mostrar ao
     administrador (FeatureID.AdministratorOf). Sem o parâmetro `eperson`, o
     backend responde sobre o DONO DO TOKEN. Resultado não vazio ⇒ é admin.

O `{site}` é o self href do Site, resolvido uma vez por processo em
GET /api/core/sites (`_embedded.sites[0]._links.self.href`) — ver `site_self_href`.

Os códigos de resposta são contrato com o front (ele decide entre "sua sessão
expirou", "sua conta não pode ver isso" e "tente de novo"):

  401  sem token, token inválido ou expirado        → relogar, sem retry
  403  token válido, mas a conta não é admin        → sem retry
  503  DSpace fora do ar / timeout na validação     → retry com backoff

O ponto crítico é o 503. Falha ao VALIDAR não é sessão inválida: devolver 401
quando o DSpace está reiniciando manda o administrador relogar à toa, e o login
vai falhar pelo mesmo motivo. Só o próprio DSpace produz 401 aqui.

O token nunca é logado, guardado em claro nem devolvido em mensagem de erro — o
cache é indexado pelo SHA-256 dele, e os logs mostram no máximo 8 caracteres
desse hash, que serve para correlacionar requisições da mesma sessão.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Optional

import httpx
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from backend.core import config as settings
from backend.core.logger import log_api

# auto_error=False: com o padrão (True) o FastAPI responde 403 quando falta o
# header — e 403 significa "não é admin" para o front, que não manda relogar.
# Sem token quem responde somos nós, com 401.
_bearer = HTTPBearer(auto_error=False, description="Bearer do DSpace (o mesmo do dspace-angular)")

# Cabeçalho do 401: RFC 6750. O front não depende dele, mas quem chama a API por
# curl/cliente HTTP genérico descobre por aí o esquema esperado.
_WWW_AUTH = {"WWW-Authenticate": "Bearer"}

# token_sha256 → (expira_em_monotonic, identidade). Só validações BEM SUCEDIDAS
# entram: token inválido não é cacheado (não vale a pena) e 403 tampouco (o front
# não repete). Processo único e sem thread: dict simples basta — o job_store é que
# precisa ser compartilhado entre API e workers, isto aqui não.
_cache: dict[str, tuple[float, dict]] = {}

# Self href do Site, resolvido sob demanda e mantido pelo processo todo (é fixo
# no DSpace). Falha na resolução NÃO é memorizada: vira 503 e tenta de novo.
_site_self_href: Optional[str] = None

# Só para não repetir o aviso de "auth desligada" a cada requisição.
_avisou_desligado = False


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _apelido(chave: str) -> str:
    """Prefixo do hash, para o log poder correlacionar requisições sem expor nada."""
    return chave[:8]


def _cache_get(chave: str) -> Optional[dict]:
    entrada = _cache.get(chave)
    if entrada is None:
        return None
    expira_em, identidade = entrada
    if expira_em <= time.monotonic():
        _cache.pop(chave, None)
        return None
    return identidade


def _cache_set(chave: str, identidade: dict) -> None:
    if len(_cache) >= settings.ADMIN_AUTH_CACHE_MAX_ENTRIES:
        agora = time.monotonic()
        for k in [k for k, (exp, _) in _cache.items() if exp <= agora]:
            _cache.pop(k, None)
        # Ainda cheio (todas válidas): descarta a mais próxima de expirar.
        if len(_cache) >= settings.ADMIN_AUTH_CACHE_MAX_ENTRIES:
            _cache.pop(min(_cache, key=lambda k: _cache[k][0]), None)
    _cache[chave] = (time.monotonic() + settings.ADMIN_AUTH_CACHE_TTL_SECONDS, identidade)


def limpar_cache() -> None:
    """Esvazia o cache de sessões e o self href memorizado (usado nos testes)."""
    global _site_self_href, _avisou_desligado
    _cache.clear()
    _site_self_href = None
    _avisou_desligado = False


def _client() -> httpx.AsyncClient:
    """Ponto único de criação do cliente HTTP (os testes trocam por um MockTransport)."""
    return httpx.AsyncClient(timeout=settings.ADMIN_AUTH_TIMEOUT_SECONDS)


def _sessao_invalida() -> HTTPException:
    return HTTPException(status_code=401, detail="Sessão DSpace inválida ou expirada.",
                         headers=_WWW_AUTH)


def _indisponivel() -> HTTPException:
    return HTTPException(status_code=503,
                         detail="Serviço de autenticação (DSpace) indisponível — tente de novo.")


async def site_self_href(client: httpx.AsyncClient) -> str:
    """Self href do Site — o `uri` que a checagem de `administratorOf` recebe.

    É construído pelo DSpace (inclui o host que ELE conhece), então não dá para
    montar na mão a partir de DSPACE_SERVER_URL: um host diferente do que o
    backend usa faz a busca de autorização devolver vazio, e todo admin viraria
    403. Resolve uma vez por processo; erro aqui não é memorizado.
    """
    global _site_self_href
    if _site_self_href:
        return _site_self_href

    resp = await client.get(f"{settings.DSPACE_SERVER_URL}/api/core/sites",
                            headers={"Accept": "application/json"})
    resp.raise_for_status()
    sites = (resp.json().get("_embedded") or {}).get("sites") or []
    href = ((sites[0].get("_links") or {}).get("self") or {}).get("href") if sites else None
    if not href:
        log_api.error("[auth] GET /api/core/sites não trouxe o self href do Site — "
                      "sem ele não há como perguntar quem é administrador.")
        raise _indisponivel()

    _site_self_href = href
    log_api.info("[auth] Site do DSpace resolvido: %s", href)
    return href


async def preload_site_self_href() -> None:
    """Resolve o self href do Site no startup (best-effort — ver `site_self_href`).

    Serve só para a primeira requisição administrativa não pagar a resolução (e
    para o problema aparecer no log da subida, não no primeiro 503). Se o DSpace
    estiver fora quando a API sobe, o servidor sobe do mesmo jeito e a resolução
    acontece na primeira requisição.
    """
    if not settings.ADMIN_AUTH_ENABLED:
        log_api.warning("[auth] ADMIN_AUTH_ENABLED=false — /api/files/* SEM autorização "
                        "(só use assim em desenvolvimento).")
        return
    try:
        async with _client() as c:
            await site_self_href(c)
    except Exception as exc:  # pragma: no cover - a sonda é best-effort
        log_api.warning("[auth] não resolvi o Site do DSpace na subida (%s: %s) — "
                        "será resolvido na primeira requisição administrativa.",
                        type(exc).__name__, exc)


async def _validar_no_dspace(token: str, chave: str) -> dict:
    """Pergunta ao DSpace se o token vale e se o dono é administrador do Site."""
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    try:
        async with _client() as c:
            base = settings.DSPACE_SERVER_URL
            status = await c.get(f"{base}/api/authn/status", headers=headers)
            # O DSpace responde 200 + authenticated=false para token inválido, mas
            # 401 quando o token está malformado/expirado — os dois são o mesmo caso.
            if status.status_code in (401, 403):
                raise _sessao_invalida()
            status.raise_for_status()
            corpo: dict[str, Any] = status.json()
            if not corpo.get("authenticated"):
                raise _sessao_invalida()

            autorizacoes = await c.get(
                f"{base}/api/authz/authorizations/search/object",
                params={"uri": await site_self_href(c), "feature": "administratorOf"},
                headers=headers,
            )
            autorizacoes.raise_for_status()
            total = ((autorizacoes.json().get("page") or {}).get("totalElements")) or 0
    except HTTPException:
        raise
    except (httpx.HTTPError, ValueError, TypeError, KeyError, IndexError) as exc:
        # httpx.HTTPError cobre timeout, DNS/conexão e raise_for_status; ValueError
        # cobre o corpo que não é JSON (uma página de erro do proxy, por exemplo).
        log_api.warning("[auth] falha ao validar a sessão %s no DSpace (%s: %s) → 503",
                        _apelido(chave), type(exc).__name__, exc)
        raise _indisponivel() from exc

    if total < 1:
        log_api.info("[auth] sessão %s é válida, mas não é administradora → 403", _apelido(chave))
        raise HTTPException(status_code=403,
                            detail="Requer administrador do repositório DSpace.")

    return _identidade(corpo, chave)


def _identidade(corpo: dict, chave: str) -> dict:
    """O que se sabe de quem chamou — para trilha de auditoria, não para decisão.

    Cuidado com `_links.eperson`: no /authn/status ele é sempre o MESMO
    `.../api/authn/status/eperson` (um atalho que só resolve com o token de quem
    perguntou), então não identifica ninguém — por isso não é usado aqui. O que
    identifica de fato é o eperson embutido, quando o DSpace o manda; fora isso
    sobra `sessao`, o prefixo do hash do token, que serve para correlacionar as
    requisições de uma mesma sessão no log sem expor o token.
    """
    eperson = ((corpo.get("_embedded") or {}).get("eperson") or {})
    return {
        "eperson_uuid": eperson.get("uuid") or corpo.get("id"),
        "eperson_email": eperson.get("email"),
        "sessao": _apelido(chave),
    }


async def dspace_admin(
    cred: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> dict:
    """Dependência das rotas administrativas: exige um Bearer de admin do DSpace.

    Uso: `@router.get("/api/files/failures", dependencies=[Depends(dspace_admin)])`
    (ou receba o retorno com `admin: dict = Depends(dspace_admin)` — ver `_identidade`).
    """
    global _avisou_desligado
    if not settings.ADMIN_AUTH_ENABLED:
        if not _avisou_desligado:
            log_api.warning("[auth] ADMIN_AUTH_ENABLED=false — rota administrativa "
                            "servida sem autorização (só use assim em desenvolvimento).")
            _avisou_desligado = True
        return {"eperson_uuid": None, "eperson_email": None, "sessao": "auth-desligada"}

    if cred is None or cred.scheme.lower() != "bearer" or not cred.credentials.strip():
        raise HTTPException(status_code=401, detail="Sessão DSpace ausente.", headers=_WWW_AUTH)

    chave = _hash(cred.credentials)
    identidade = _cache_get(chave)
    if identidade is not None:
        return identidade

    try:
        identidade = await _validar_no_dspace(cred.credentials, chave)
    except HTTPException as exc:
        if exc.status_code == 401:
            # Em geral já não há entrada (só se revalida depois de ela expirar), mas
            # se houver — sessão encerrada no DSpace — ela morre aqui, sem esperar o TTL.
            _cache.pop(chave, None)
        raise

    _cache_set(chave, identidade)
    log_api.info("[auth] sessão %s validada como administradora (cache por %.0fs)",
                 _apelido(chave), settings.ADMIN_AUTH_CACHE_TTL_SECONDS)
    return identidade
