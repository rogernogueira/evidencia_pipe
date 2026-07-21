"""Rotas de ingestão do evidencia_pipe.

Reproduz o endpoint do minerU e expõe a cadeia completa (4 estágios) a partir de
um UUID do DSpace:
  1. Resolve e baixa os PDFs do bundle ORIGINAL do item (DSpace REST API).
  2. Extração via MinerU (markdown + _content_list_v2.json).
  3. Chunking + embeddings bge-m3 (dense+sparse) → upsert no Qdrant.
  4. Metadados por LLM (DeepSeek) → _metadata_llm.json.

O trabalho pesado roda em BackgroundTasks; os POST de ingestão respondem 202.

Endpoint principal: POST /api/files/dspace/item/{uuid}
"""

import urllib.error
from pathlib import Path

from celery import chain
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse, PlainTextResponse

from backend.core.logger import log_api
from backend.services.dspace_service import download_bitstream, resolve_item_pdfs
from backend.services import llm_enrich_service as llm_enrich
from backend.services.job_store import set_status, get_job, find_markdown, markdown_url
from backend.tasks import baixar_dspace, extrair_mineru, enrich_llm, indexar_qdrant

router = APIRouter()

# PDFs baixados do DSpace ficam aqui antes da extração.
DATA_DIR = Path("data")


@router.post("/api/files/dspace/item/{uuid}")
def ingest_dspace_item(uuid: str) -> JSONResponse:
    """Resolve os PDFs do bundle ORIGINAL de um item DSpace e enfileira uma chain
    Celery por PDF (baixar → MinerU → LLM → Qdrant).

    O `resolve` roda síncrono (rápido, só metadados REST) — por isso a resposta já
    lista os job_id reais. O download e o resto rodam nos workers; o status é
    acompanhado em /api/files/status/{job_id}. Erros de download deixam de ser
    síncronos: o job vai para "erro" (visível no status e no Flower).
    """
    log_api.info("Ingestão de item DSpace solicitada: uuid=%s", uuid)
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
        chain(
            baixar_dspace.s(bs_uuid, filename, job_id=job_id, item_uuid=uuid, item_handle=item_handle),
            extrair_mineru.s(),
            enrich_llm.s(),
            indexar_qdrant.s(),
        ).apply_async()

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
def ingest_dspace_bitstream(uuid: str) -> JSONResponse:
    """Baixa o PDF de um bitstream avulso e enfileira a chain de processamento.

    Sem `resolve` para descobrir o nome, o download é feito síncrono aqui (é um
    bitstream único e explícito), preservando job_id = stem do arquivo e os erros
    de download síncronos (502). Os estágios MinerU→LLM→Qdrant rodam nos workers.
    """
    log_api.info("Ingestão de bitstream DSpace solicitada: uuid=%s", uuid)
    try:
        pdf_path = download_bitstream(uuid, DATA_DIR)
    except urllib.error.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"DSpace retornou HTTP {e.code} para o bitstream {uuid}.")
    except urllib.error.URLError as e:
        raise HTTPException(status_code=502, detail=f"Falha ao acessar o DSpace: {e.reason}")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    job_id = pdf_path.stem
    set_status(job_id, "na_fila", filename=pdf_path.name, source="dspace", bitstream_uuid=uuid)
    ctx = {"job_id": job_id, "pdf_path": str(pdf_path), "item_uuid": "", "item_handle": ""}
    chain(
        extrair_mineru.s(ctx),
        enrich_llm.s(),
        indexar_qdrant.s(),
    ).apply_async()

    return JSONResponse(
        status_code=202,
        content={
            "message": "Bitstream baixado; processamento enfileirado",
            "job_id": job_id,
            "filename": pdf_path.name,
            "bitstream_uuid": uuid,
            "status": "na_fila",
            "status_url": f"/api/files/status/{job_id}",
            "result_url": f"/api/files/result/{job_id}",
        },
    )


@router.get("/api/files/status/{job_id}")
def job_status(job_id: str) -> JSONResponse:
    """Status do job de processamento. Faz fallback pro filesystem se o registro foi perdido (restart)."""
    job = get_job(job_id)
    if job is None:
        md = find_markdown(job_id)
        if md is not None:
            return JSONResponse({"job_id": job_id, "status": "concluido", "md_url": markdown_url(md)})
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' desconhecido.")
    return JSONResponse(job)


@router.get("/api/files/result/{job_id}")
def job_result(job_id: str) -> PlainTextResponse:
    """Retorna o conteúdo do markdown extraído (text/markdown)."""
    md = find_markdown(job_id)
    if md is None:
        job = get_job(job_id)
        if job and job.get("status") == "processando":
            raise HTTPException(status_code=409, detail=f"Job '{job_id}' ainda em processamento.")
        if job and job.get("status") == "erro":
            raise HTTPException(status_code=422, detail=f"Job '{job_id}' falhou: {job.get('error')}")
        raise HTTPException(status_code=404, detail=f"Markdown para '{job_id}' não encontrado.")
    return PlainTextResponse(md.read_text(encoding="utf-8"), media_type="text/markdown; charset=utf-8")


@router.post("/api/files/enrich/{job_id}")
def enrich_job_metadata(
    job_id: str,
    uuid: str = Query(default="", description="UUID do item no DSpace (opcional)"),
) -> JSONResponse:
    """Aciona a LLM (DeepSeek) sobre o markdown do job e devolve os metadados candidatos.

    Salva o resultado em output/<doc>/<doc>_metadata_llm.json e retorna o objeto.
    """
    log_api.info("POST /api/files/enrich/%s uuid=%r", job_id, uuid)
    if not llm_enrich.is_available():
        raise HTTPException(status_code=503, detail="Step de LLM indisponível: configure DEEPSEEK_API_KEY.")
    try:
        meta = llm_enrich.enrich_job(job_id, uuid=uuid)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        log_api.error("Enriquecimento LLM falhou para %s: %s", job_id, e)
        raise HTTPException(status_code=502, detail=f"Falha no step de LLM: {e}")
    return JSONResponse(meta.model_dump())
