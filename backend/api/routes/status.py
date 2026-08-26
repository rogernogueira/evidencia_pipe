"""Estado da infraestrutura numa consulta.

    GET /api/status              → todos os componentes + capacidades derivadas
    GET /api/status/{component}  → um componente só (polling barato e dirigido)

A sondagem vive em backend/services/infra_status.py; aqui ficam a rota, a
autorização, o nível de detalhe e o código HTTP.

Autorização
-----------
As duas rotas exigem Bearer de administrador do DSpace (backend/api/auth.py), em
QUALQUER caminho — pela borda ou direto na porta do backend. O status descreve a
infraestrutura inteira (o que está fora, quantos chunks existem, quais capacidades
estão bloqueadas): é informação de operação, não de consulta pública.

Consequência para quem opera: os `curl` de acompanhamento precisam do token (ver
DEPLOY.md §9), e a sincronização periódica, que consulta `capabilities.ingestao`
antes de colher, degrada para "não consegui checar" quando não tem token — ela
segue a rodada em vez de pulá-la (scripts/sincronizar_novos_itens.py). Liveness sem
token continua existindo: `/health`.

Nível de detalhe
----------------
A resposta separa métricas (contagens, flags, nome de modelo — sempre presentes) de
topologia (`internal`: URLs, versões, hostnames, PIDs, caminhos). `internal` só é
serializado para quem tem por que vê-lo:

  - requests que NÃO vieram da borda (sem `X-Forwarded-For`, ou seja, rede interna,
    script, systemd) — a mesma convenção que barra a raiz administrativa em
    backend/api/proxy_prefix.py; ou
  - requests com `X-Internal-Token` igual a INTERNAL_API_TOKEN.

Por que não simplesmente pôr a rota inteira sob `/internal`: um painel de operação
consome a API pela borda (é assim que o front alcança o backend), e um status que só
responde na rede interna não serve para ele. O Bearer resolve QUEM pode consultar; o
nível de detalhe é outra pergunta e continua valendo por cima dele — nem todo admin
autenticado precisa receber URLs, hostnames e PIDs dentro do navegador. `detail_level`
na resposta diz qual dos dois níveis veio, para o consumidor não confundir campo
ausente com problema.

As opções caras (`fresh`, `probe`) também são restritas ao nível completo: sem isso um
painel aberto na tela forçaria broadcast de Celery, varredura do bucket e round-trip
de embedding a cada recarga — o Bearer diz que a pessoa pode consultar, não que a
consulta possa custar o que quiser. Quem chega pela borda não força remedição: a
resposta vem do cache enquanto ele vale, então uma consulta frequente custa, no
máximo, uma medição por STATUS_CACHE_TTL_SECONDS.

Código HTTP: 200 quando o pipeline está de pé (mesmo degradado) e 503 quando um
componente CRÍTICO está fora, para um monitor externo poder olhar só o status da
resposta. O corpo é o mesmo JSON nos dois casos.
"""

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from backend.api.auth import dspace_admin
from backend.api.proxy_prefix import veio_da_borda
from backend.core import config as settings
from backend.core.logger import log_api
from backend.services import infra_status

router = APIRouter(tags=["status"])


def _detalhe_completo(request: Request, token: str | None) -> bool:
    """Quem enxerga a topologia (`internal`) e pode usar `fresh`/`probe`."""
    if settings.INTERNAL_API_TOKEN and token == settings.INTERNAL_API_TOKEN:
        return True
    return not veio_da_borda(request.scope)


@router.get("/api/status", dependencies=[Depends(dspace_admin)])
def infra_status_geral(
    request: Request,
    fresh: bool = Query(default=False, description="Ignora o cache e remede tudo (só no detalhe completo)"),
    probe: bool = Query(default=False, description="Sondagens ativas: round-trip real de embedding (só no detalhe completo)"),
    artifacts: bool = Query(default=True, description="Contar os artefatos no MinIO (listagem que cresce com o acervo)"),
    x_internal_token: str | None = Header(default=None),
) -> JSONResponse:
    """Status de toda a infraestrutura do pipeline: Qdrant (e quantos chunks), MinIO (e
    quantos artefatos), MinerU, a API de embedding do bge-m3, o LLM de enriquecimento
    (ligado ou não, com qual modelo), Redis, os workers Celery e suas filas, a GPU, o
    DSpace, o Flower, os índices de job e o disco.

    Além do estado por componente, devolve o que o sistema CONSEGUE fazer agora
    (`capabilities`: busca, ingestão, extração, indexação, enriquecimento) com os
    componentes que bloqueiam cada uma, e a configuração que define a semântica do
    índice (`config`: chunking, embedding, filtros de busca).
    """
    completo = _detalhe_completo(request, x_internal_token)
    opcoes = infra_status.Options(probe=probe and completo, artifacts=artifacts)
    snapshot, idade = infra_status.collect(opcoes, fresh=fresh and completo)
    corpo = snapshot.to_dict(include_internal=completo, age_seconds=idade)

    log_api.info("GET /api/status → %s (%d bloqueando, cache %.1fs, detalhe %s)",
                 corpo["status"], len(corpo["blocking"]), idade, corpo["detail_level"])
    return JSONResponse(corpo, status_code=503 if corpo["status"] == infra_status.DOWN else 200)


@router.get("/api/status/{component}", dependencies=[Depends(dspace_admin)])
def infra_status_componente(
    request: Request,
    component: str,
    fresh: bool = Query(default=False, description="Ignora o cache e remede (só no detalhe completo)"),
    probe: bool = Query(default=False, description="Sondagens ativas (só no detalhe completo)"),
    artifacts: bool = Query(default=True, description="Contar os artefatos no MinIO"),
    x_internal_token: str | None = Header(default=None),
) -> JSONResponse:
    """Status de UM componente, sem pagar as outras doze sondagens.

    Serve para acompanhar de perto o que se está mexendo — `/api/status/celery` depois
    de subir um worker, com `fresh=true` a cada tentativa. O cache é o mesmo do
    snapshot (por componente), então repetir a consulta não custa uma medição por
    requisição; `age_seconds` diz a idade da resposta.

    503 quando o componente é crítico e está fora; 404 quando o nome não existe.
    """
    completo = _detalhe_completo(request, x_internal_token)
    if component not in infra_status.COMPONENT_ORDER:
        raise HTTPException(
            status_code=404,
            detail=f"componente desconhecido: {component!r} "
                   f"(existem: {', '.join(infra_status.COMPONENT_ORDER)})",
        )
    opcoes = infra_status.Options(probe=probe and completo, artifacts=artifacts)
    comp, idade = infra_status.check_cached(component, opcoes, fresh=fresh and completo)
    fora = comp.status == infra_status.DOWN and comp.critical
    corpo = {**comp.to_dict(include_internal=completo),
             "age_seconds": round(idade, 3),
             "detail_level": "full" if completo else "public"}
    return JSONResponse(corpo, status_code=503 if fora else 200)
