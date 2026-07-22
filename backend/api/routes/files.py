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

Os endpoints de status/resultado NÃO retornam artefatos completos — só um resumo.
Para baixar um artefato use o endpoint interno de URL pré-assinada
(backend/api/routes/artifacts.py).
"""

import urllib.error
from pathlib import Path

from celery import chain
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from backend.core import config as settings
from backend.core.logger import log_api
from backend.core.schemas import (
    ART_METADATA_CANDIDATES,
    CTX_STAGE_EXTRACTED,
    PipelineContext,
)
from backend.services import llm_enrich_service as llm_enrich
from backend.services import pipeline_stages as stages
from backend.services.dspace_service import resolve_item_pdfs
from backend.services.job_store import clear_failed, get_job, list_failed, set_status
from backend.tasks import baixar_dspace, enrich_after_index, extrair_mineru, indexar_qdrant

router = APIRouter()


def _build_chain(bs_uuid, filename, job_id, item_uuid, item_handle, force):
    """Cadeia OBRIGATÓRIA (3 estágios) — o download roda no worker e grava no MinIO.
    O enrich NÃO entra aqui: é anexado como follow-up opcional em _enqueue_chain."""
    return chain(
        baixar_dspace.s(bs_uuid, filename, job_id=job_id, item_uuid=item_uuid,
                        item_handle=item_handle, force=force),
        extrair_mineru.s(),
        indexar_qdrant.s(),
    )


def _enqueue_chain(bs_uuid, filename, job_id, item_uuid, item_handle, force):
    """Enfileira a chain obrigatória e, quando o enrich está habilitado e há provedor
    LLM configurado, anexa enrich_after_index como follow-up DESACOPLADO (link) que
    roda APÓS a indexação — o índice nunca espera nem depende do LLM."""
    clear_failed(job_id)  # (re)enfileirar supera uma falha anterior
    sig = _build_chain(bs_uuid, filename, job_id, item_uuid, item_handle, force)
    link = None
    if settings.LLM_ENRICH_AUTO and llm_enrich.is_available():
        link = enrich_after_index.s()
    sig.apply_async(link=link)


@router.post("/api/files/dspace/item/{uuid}")
def ingest_dspace_item(uuid: str, force: bool = Query(default=False)) -> JSONResponse:
    """Resolve os PDFs do bundle ORIGINAL de um item DSpace e enfileira uma chain
    Celery por PDF. Cada PDF vira um documento com manifesto e prefixo próprios no
    MinIO (§31). `force=true` reprocessa ignorando artefatos existentes (§32)."""
    log_api.info("Ingestão de item DSpace solicitada: uuid=%s force=%s", uuid, force)
    try:
        pdfs = resolve_item_pdfs(uuid)
    except urllib.error.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"DSpace retornou HTTP {e.code} para o item {uuid}.")
    except urllib.error.URLError as e:
        raise HTTPException(status_code=502, detail=f"Falha ao acessar o DSpace: {e.reason}")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    jobs = []
    for pdf in pdfs:
        bs_uuid = pdf["bitstream_uuid"]
        filename = pdf["filename"]
        item_handle = pdf.get("item_handle") or ""
        job_id = Path(filename).stem

        set_status(
            job_id, "na_fila", filename=filename,
            source="dspace-item", item_uuid=uuid, item_handle=item_handle, bitstream_uuid=bs_uuid,
        )
        _enqueue_chain(bs_uuid, filename, job_id, uuid, item_handle, force)

        jobs.append({
            "job_id": job_id,
            "filename": filename,
            "bitstream_uuid": bs_uuid,
            "status_url": f"/api/files/status/{job_id}",
            "result_url": f"/api/files/result/{job_id}",
        })

    return JSONResponse(
        status_code=202,
        content={
            "message": f"{len(jobs)} PDF(s) do item enfileirado(s) para processamento",
            "item_uuid": uuid,
            "status": "na_fila",
            "jobs": jobs,
        },
    )


@router.post("/api/files/dspace/{uuid}")
def ingest_dspace_bitstream(uuid: str, force: bool = Query(default=False)) -> JSONResponse:
    """Enfileira a chain completa para um bitstream avulso. Diferente da v1, o
    download NÃO é mais síncrono: ele roda no worker e grava direto no MinIO — o
    status (inclusive erro de download) é acompanhado em /status ou no Flower."""
    log_api.info("Ingestão de bitstream DSpace solicitada: uuid=%s force=%s", uuid, force)
    job_id = uuid
    filename = f"{uuid}.pdf"
    set_status(job_id, "na_fila", filename=filename, source="dspace", bitstream_uuid=uuid)
    _enqueue_chain(uuid, filename, job_id, "", "", force)

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


@router.get("/api/files/status/{job_id}")
def job_status(job_id: str) -> JSONResponse:
    """Status resumido do job (do job_store). Não retorna artefatos."""
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' desconhecido.")
    return JSONResponse(job)


@router.get("/api/files/result/{job_id}")
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


@router.get("/api/files/failures")
def list_failures(limit: int = Query(default=100, ge=1, le=1000)) -> JSONResponse:
    """Lista os jobs na **fila de falhas** (mais recentes primeiro) — jobs que
    falharam num estágio ou concluíram com erro de índice. Cada item é o registro
    do job (status/stage/error). Reprocessar via `POST /api/files/reprocess/{job_id}`."""
    jobs = list_failed(limit)
    return JSONResponse({"count": len(jobs), "jobs": jobs})


@router.post("/api/files/reprocess/{job_id}")
def reprocess_job(job_id: str, force: bool = Query(default=True)) -> JSONResponse:
    """Re-enfileira a chain de ingestão de um job que falhou, reusando a origem
    (bitstream/item) registrada no job_store. `force=true` (padrão) ignora artefatos
    existentes e reprocessa do zero; `force=false` reaproveita etapas já concluídas
    (idempotência por manifesto/SHA — útil p.ex. para reindexar sem re-extrair)."""
    log_api.info("POST /api/files/reprocess/%s force=%s", job_id, force)
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' desconhecido.")

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
    _enqueue_chain(bs_uuid, filename, job_id, item_uuid, item_handle, force)

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
