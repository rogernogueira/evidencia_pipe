"""Rotas de ingestão do evidencia_pipe (pipeline v2 — artefatos no MinIO).

Expõe a cadeia OBRIGATÓRIA (3 estágios) a partir de um UUID do DSpace:
  1. baixar_dspace   → baixa o PDF e o grava no MinIO (source/original.pdf).
  2. extrair_mineru  → markdown + content_list_v2.json + imagens → MinIO.
  3. indexar_qdrant  → chunks.jsonl + embeddings bge-m3 → Qdrant.

O enrich por LLM é DESACOPLADO: NÃO faz parte da chain obrigatória. Quando há
provedor configurado e LLM_ENRICH_AUTO está ligado, um follow-up opcional
(enrich_after_index) é anexado APÓS a indexação — ele gera metadata_candidates.json
e propaga os metadados ao Qdrant (set_payload), sem que o índice dependa do LLM.
Também pode ser disparado sob demanda em POST /api/files/enrich/{job_id}.

A API só ENFILEIRA (responde 202); os workers executam. A chain transporta apenas
um PipelineContext leve; o conteúdo vive no MinIO e é descoberto pelo manifesto.

Item ainda indisponível no DSpace também é caso de FILA, não de erro: em vez do 502
que perdia o pedido, a ingestão de item enfileira resolver_item_dspace, que reconsulta
o DSpace com backoff (DSPACE_ITEM_RETRY_*) até os PDFs aparecerem. A espera é
acompanhada pelo registro `item:{uuid}` no job_store (ver backend/services/ingest_service.py).

Os endpoints de status/resultado NÃO retornam artefatos completos — só um resumo.
Para baixar um artefato use o endpoint interno de URL pré-assinada
(backend/api/routes/artifacts.py).

AUTORIZAÇÃO: as rotas que MOSTRAM a fila (active/succeeded/failures/status/result)
e a que a reprocessa exigem um Bearer de administrador do DSpace — ver
backend/api/auth.py, que também protege o `/api/status`. A busca (`/api/search/*`) e
o `/health` continuam abertos.
As rotas de ENFILEIRAMENTO (`POST /api/files/dspace/...`) e o `POST /api/files/enrich`
seguem abertas: quem as chama é a sincronização automática
(scripts/sincronizar_novos_itens.py), que roda no próprio host e não tem sessão DSpace.
O destino delas NÃO é o Bearer — é sair do acesso externo e ficar alcançáveis só pela
rede local. Ver o comentário em cada uma e DEPLOY.md §8.2.
"""

import urllib.error

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse

from backend.api.auth import dspace_admin
from backend.core.logger import log_api
from backend.core.schemas import (
    ART_METADATA_CANDIDATES,
    CTX_STAGE_EXTRACTED,
    PipelineContext,
)
from backend.services import ingest_service as ingest
from backend.services import llm_enrich_service as llm_enrich
from backend.services import pipeline_stages as stages
from backend.services.dspace_service import item_ainda_indisponivel, resolve_item_pdfs
from backend.services.job_store import (
    get_job,
    list_active,
    list_failed,
    list_succeeded,
    set_status,
)

router = APIRouter()


# ABERTA POR ORA, MAS NÃO PARA SEMPRE: o plano é tirá-la do acesso externo e deixá-la
# alcançável só pela rede local (borda/firewall), não com Bearer. Quem a chama é a
# sincronização periódica (scripts/sincronizar_novos_itens.py), que roda no próprio
# host e não tem sessão DSpace para apresentar — exigir admin aqui derrubaria a
# ingestão automática. Ver a nota de AUTORIZAÇÃO no topo do módulo e DEPLOY.md §8.2.
@router.post("/api/files/dspace/item/{uuid}")
def ingest_dspace_item(uuid: str, force: bool = Query(default=False)) -> JSONResponse:
    """Resolve os PDFs do bundle ORIGINAL de um item DSpace e enfileira uma chain
    Celery por PDF. Cada PDF vira um documento com manifesto e prefixo próprios no
    MinIO (§31). `force=true` reprocessa ignorando artefatos existentes (§32).

    Item ainda NÃO disponível para download (em submissão/workflow, embargo, DSpace
    fora do ar, PDF ainda não anexado): a resposta NÃO é mais 502/422. O pedido é
    ENFILEIRADO — a task resolver_item_dspace reconsulta o DSpace com backoff
    (DSPACE_ITEM_RETRY_*) e dispara as chains assim que os PDFs aparecerem. A resposta
    é 202 com `status="aguardando_dspace"`, e a espera é acompanhada em
    `GET /api/files/status/item:{uuid}` (ou em `GET /api/files/active`). Esgotadas as
    tentativas, o item vai para `GET /api/files/failures`. Erro definitivo do DSpace
    (ex.: 400 de UUID malformado) continua virando 502 na hora."""
    log_api.info("Ingestão de item DSpace solicitada: uuid=%s force=%s", uuid, force)
    try:
        pdfs = resolve_item_pdfs(uuid)
    except Exception as e:
        if not item_ainda_indisponivel(e):
            raise _erro_dspace_definitivo(uuid, e)
        return _aguardar_item(uuid, force, e)

    jobs = ingest.enqueue_item_pdfs(uuid, pdfs, force)
    return JSONResponse(
        status_code=202,
        content={
            "message": f"{len(jobs)} PDF(s) do item enfileirado(s) para processamento",
            "item_uuid": uuid,
            "status": "na_fila",
            "jobs": jobs,
        },
    )


def _erro_dspace_definitivo(uuid: str, exc: Exception) -> HTTPException:
    """Traduz um erro NÃO transitório do DSpace na resposta de erro da API (mesma
    semântica de antes: 502 para falha de acesso, 422 para item sem PDF)."""
    if isinstance(exc, urllib.error.HTTPError):
        return HTTPException(status_code=502, detail=f"DSpace retornou HTTP {exc.code} para o item {uuid}.")
    if isinstance(exc, urllib.error.URLError):
        return HTTPException(status_code=502, detail=f"Falha ao acessar o DSpace: {exc.reason}")
    if isinstance(exc, ValueError):
        return HTTPException(status_code=422, detail=str(exc))
    log_api.error("Falha inesperada ao resolver o item %s: %s", uuid, exc)
    return HTTPException(status_code=502, detail=f"Falha ao resolver o item {uuid}: {exc}")


def _aguardar_item(uuid: str, force: bool, exc: Exception) -> JSONResponse:
    """Item ainda indisponível: coloca a ingestão na fila (ou reaproveita a espera já
    em andamento) e responde 202 — nunca 502."""
    key = ingest.item_job_key(uuid)
    motivo = f"{type(exc).__name__}: {exc}"
    em_espera = get_job(key)

    if not force and ingest.item_wait_is_alive(em_espera):
        log_api.info("Item %s já está aguardando o DSpace — não reenfileirado.", uuid)
        espera = ingest.item_wait_summary(em_espera)
        mensagem = "Item ainda não disponível no DSpace — já havia uma ingestão aguardando na fila."
    else:
        espera = ingest.schedule_item_resolution(uuid, force=force, attempt=1, error=motivo)
        mensagem = ("Item ainda não disponível no DSpace — ingestão enfileirada; "
                    f"nova tentativa em {espera['next_retry_in_seconds']}s.")

    return JSONResponse(
        status_code=202,
        content={
            "message": mensagem,
            "item_uuid": uuid,
            "status": ingest.PENDING_STAGE,
            "job_id": key,
            "status_url": f"/api/files/status/{key}",
            "reason": motivo,
            "retry": espera,
            "jobs": [],
        },
    )


# ABERTA POR ORA, MAS NÃO PARA SEMPRE: o plano é tirá-la do acesso externo e deixá-la
# alcançável só pela rede local (borda/firewall), não com Bearer. Quem a chama é a
# sincronização periódica (scripts/sincronizar_novos_itens.py), que roda no próprio
# host e não tem sessão DSpace para apresentar — exigir admin aqui derrubaria a
# ingestão automática. Ver a nota de AUTORIZAÇÃO no topo do módulo e DEPLOY.md §8.2.
@router.post("/api/files/dspace/{uuid}")
def ingest_dspace_bitstream(uuid: str, force: bool = Query(default=False)) -> JSONResponse:
    """Enfileira a chain completa para um bitstream avulso. Diferente da v1, o
    download NÃO é mais síncrono: ele roda no worker e grava direto no MinIO — o
    status (inclusive erro de download) é acompanhado em /status ou no Flower."""
    log_api.info("Ingestão de bitstream DSpace solicitada: uuid=%s force=%s", uuid, force)
    job_id = uuid
    filename = f"{uuid}.pdf"
    set_status(job_id, "na_fila", filename=filename, source="dspace", bitstream_uuid=uuid)
    ingest.enqueue_chain(uuid, filename, job_id, "", "", force)

    return JSONResponse(
        status_code=202,
        content={
            "message": "Processamento enfileirado",
            "job_id": job_id,
            "filename": filename,
            "bitstream_uuid": uuid,
            "status": "na_fila",
            "status_url": f"/api/files/status/{job_id}",
            "result_url": f"/api/files/result/{job_id}",
        },
    )


@router.get("/api/files/status/{job_id}", dependencies=[Depends(dspace_admin)])
def job_status(job_id: str) -> JSONResponse:
    """Status resumido do job (do job_store). Não retorna artefatos."""
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' desconhecido.")
    return JSONResponse(job)


@router.get("/api/files/result/{job_id}", dependencies=[Depends(dspace_admin)])
def job_result(job_id: str) -> JSONResponse:
    """Resultado resumido do job (§30) — NÃO devolve o conteúdo dos artefatos.

    Para baixar o markdown/JSON/etc., use o endpoint interno de URL pré-assinada.
    """
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' desconhecido.")

    status = job.get("status", "desconhecido")
    if status == "processando":
        raise HTTPException(status_code=409, detail=f"Job '{job_id}' ainda em processamento.")
    if status == "erro":
        raise HTTPException(status_code=422, detail=f"Job '{job_id}' falhou: {job.get('error')}")

    pipeline_id = job.get("pipeline_id")
    document_id = job.get("document_id", job_id)
    return JSONResponse({
        "job_id": job_id,
        "status": status,
        "documents": [{
            "document_id": document_id,
            "status": status,
            "chunk_count": job.get("n_chunks"),
            "indexed_count": job.get("indexed_count"),
            "artifact_id": f"{pipeline_id}/{document_id}" if pipeline_id else None,
        }],
    })


# ABERTA POR ORA, MAS NÃO PARA SEMPRE: mesmo destino das rotas de enfileiramento —
# sai do acesso externo e fica restrita à rede local. Esta é a primeira candidata:
# cada chamada gasta LLM, e nenhum consumidor legítimo dela vem da internet.
@router.post("/api/files/enrich/{job_id}")
def enrich_job_metadata(
    job_id: str,
    uuid: str = Query(default="", description="UUID do item no DSpace (opcional)"),
) -> JSONResponse:
    """Aciona a LLM (provedor configurável via LLM_ENRICH_*) sobre o markdown do job
    (lido do MinIO) e devolve os metadados candidatos, persistindo-os em
    enrichment/metadata_candidates.json. Se o doc já estiver indexado, os metadados
    também são propagados ao Qdrant (set_payload) — enrich desacoplado da indexação."""
    log_api.info("POST /api/files/enrich/%s uuid=%r", job_id, uuid)
    if not llm_enrich.is_available():
        raise HTTPException(
            status_code=503,
            detail="Step de LLM indisponível: configure LLM_ENRICH_API_KEY (ou o legado DEEPSEEK_API_KEY).",
        )

    job = get_job(job_id)
    manifest_uri = (job or {}).get("artifact_manifest_uri")
    pipeline_id = (job or {}).get("pipeline_id")
    document_id = (job or {}).get("document_id", job_id)

    # Fluxo legado (sem manifesto): enriquecimento local a partir de output/.
    if not manifest_uri or not pipeline_id:
        try:
            meta = llm_enrich.enrich_job(job_id, uuid=uuid)
            return JSONResponse(meta.model_dump())
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            log_api.error("Enriquecimento LLM (legado) falhou para %s: %s", job_id, e)
            raise HTTPException(status_code=502, detail=f"Falha no step de LLM: {e}")

    # Fluxo v2 (MinIO): reusa stage_enrich (force) e devolve os metadados persistidos.
    from backend.services.artifact_store import get_artifact_store
    from backend.services.manifest_repository import get_manifest_repository

    ctx = PipelineContext.model_validate({
        "pipeline_id": pipeline_id,
        "job_id": job_id,
        "item_uuid": uuid,
        "document_id": document_id,
        "artifact_manifest_uri": manifest_uri,
        "current_stage": CTX_STAGE_EXTRACTED,
        "force": True,
    })
    stages.stage_enrich(ctx)

    repo = get_manifest_repository()
    store = get_artifact_store()
    manifest = repo.load(pipeline_id, document_id)
    ref = manifest.artifacts.get(ART_METADATA_CANDIDATES)
    if ref is None:
        raise HTTPException(status_code=502, detail="Falha no step de LLM: metadados não gerados.")
    return JSONResponse(store.read_json(ref.object_key))


@router.get("/api/files/active", dependencies=[Depends(dspace_admin)])
def list_active_jobs(limit: int = Query(default=100, ge=1, le=1000)) -> JSONResponse:
    """Lista os **jobs em execução** (mais recentes primeiro): os que estão `na_fila`
    ou `processando`. Devolve os IDs em `job_ids` e um resumo por job (status, estágio
    atual, arquivo, último update) — sem artefatos. Jobs concluídos ou com erro saem
    do índice automaticamente (erros ficam em `GET /api/files/failures`)."""
    jobs = list_active(limit)
    return JSONResponse({
        "count": len(jobs),
        "job_ids": [j["job_id"] for j in jobs],
        "jobs": [
            {
                "job_id": j["job_id"],
                "status": j.get("status"),
                "stage": j.get("stage"),
                "filename": j.get("filename"),
                "item_uuid": j.get("item_uuid"),
                "updated_at": j.get("updated_at"),
            }
            for j in jobs
        ],
    })


@router.get("/api/files/succeeded", dependencies=[Depends(dspace_admin)])
def list_succeeded_jobs(limit: int = Query(default=100, ge=1, le=1000)) -> JSONResponse:
    """Lista os **últimos jobs bem sucedidos** (mais recentes primeiro): concluídos e
    indexados sem erro. Devolve os IDs em `job_ids` e um resumo por job (contagens de
    chunks/pontos indexados, `artifact_id`, `updated_at`) — sem artefatos. Um job que
    extraiu mas falhou ao indexar NÃO entra aqui (fica em `GET /api/files/failures`).

    A janela é limitada pelo TTL do job_store (`JOBSTORE_TTL`): jobs cujo registro
    expirou saem da lista."""
    jobs = list_succeeded(limit)
    return JSONResponse({
        "count": len(jobs),
        "job_ids": [j["job_id"] for j in jobs],
        "jobs": [
            {
                "job_id": j["job_id"],
                "status": j.get("status"),
                "filename": j.get("filename"),
                "item_uuid": j.get("item_uuid"),
                "chunk_count": j.get("n_chunks"),
                "indexed_count": j.get("indexed_count"),
                "artifact_id": (
                    f"{j['pipeline_id']}/{j.get('document_id', j['job_id'])}"
                    if j.get("pipeline_id") else None
                ),
                "updated_at": j.get("updated_at"),
            }
            for j in jobs
        ],
    })


@router.get("/api/files/failures", dependencies=[Depends(dspace_admin)])
def list_failures(limit: int = Query(default=100, ge=1, le=1000)) -> JSONResponse:
    """Lista os jobs na **fila de falhas** (mais recentes primeiro) — jobs que
    falharam num estágio ou concluíram com erro de índice. Cada item é o registro
    do job (status/stage/error). Reprocessar via `POST /api/files/reprocess/{job_id}`."""
    jobs = list_failed(limit)
    return JSONResponse({"count": len(jobs), "jobs": jobs})


@router.post("/api/files/reprocess/{job_id}")
def reprocess_job(
    job_id: str,
    force: bool = Query(default=True),
    admin: dict = Depends(dspace_admin),
) -> JSONResponse:
    """Re-enfileira a chain de ingestão de um job que falhou, reusando a origem
    (bitstream/item) registrada no job_store. `force=true` (padrão) ignora artefatos
    existentes e reprocessa do zero; `force=false` reaproveita etapas já concluídas
    (idempotência por manifesto/SHA — útil p.ex. para reindexar sem re-extrair).

    Para o registro de um ITEM que ficou indisponível no DSpace (`item:{uuid}`, ver
    POST /api/files/dspace/item/{uuid}) o que se reenfileira é a RESOLUÇÃO do item —
    a contagem de tentativas recomeça e a primeira é imediata."""
    log_api.info("POST /api/files/reprocess/%s force=%s [admin=%s sessao=%s]", job_id, force,
                 admin.get("eperson_email") or admin.get("eperson_uuid") or "?",
                 admin.get("sessao"))
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' desconhecido.")

    # Registro de ITEM que esgotou as tentativas de espera (`item:{uuid}`): não há
    # bitstream para reprocessar — o que se reenfileira é a resolução no DSpace.
    if ingest.is_pending_item(job):
        item_uuid = job.get("item_uuid") or job_id.removeprefix(ingest.ITEM_KEY_PREFIX)
        espera = ingest.schedule_item_resolution(item_uuid, force=force, attempt=1,
                                                 error=job.get("error") or "", delay=0)
        return JSONResponse(
            status_code=202,
            content={
                "message": "Resolução do item reenfileirada — o DSpace será consultado de novo.",
                "item_uuid": item_uuid,
                "job_id": job_id,
                "status": ingest.PENDING_STAGE,
                "retry": espera,
                "status_url": f"/api/files/status/{job_id}",
            },
        )

    bs_uuid = job.get("bitstream_uuid")
    if not bs_uuid:
        raise HTTPException(
            status_code=422,
            detail=(f"Job '{job_id}' não tem bitstream de origem registrado "
                    "(ex.: ingestão local legada) — reprocesse re-enviando a ingestão do item."),
        )

    filename = job.get("filename") or f"{job_id}.pdf"
    item_uuid = job.get("item_uuid") or ""
    item_handle = job.get("item_handle") or ""

    set_status(
        job_id, "na_fila", filename=filename, source=job.get("source"),
        item_uuid=item_uuid, item_handle=item_handle, bitstream_uuid=bs_uuid,
    )
    ingest.enqueue_chain(bs_uuid, filename, job_id, item_uuid, item_handle, force)

    return JSONResponse(
        status_code=202,
        content={
            "message": "Reprocessamento enfileirado",
            "job_id": job_id,
            "filename": filename,
            "bitstream_uuid": bs_uuid,
            "force": force,
            "status": "na_fila",
            "status_url": f"/api/files/status/{job_id}",
        },
    )
