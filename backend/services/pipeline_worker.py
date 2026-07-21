import csv
from pathlib import Path
import traceback

from backend.core.config import OUTPUT_DIR
from backend.core.logger import log
from backend.services.mineru_service import process_pdf
from backend.services.job_store import set_status, find_markdown, markdown_url
from backend.services import llm_enrich_service as llm_enrich
from backend.indexing.index_chunks import index_single_document

LOG_PROCESS_FILE = Path("relatorio_processamento.csv")
LOG_EMBED_FILE = Path("relatorio_embeddings.csv")

def write_process_log(result: dict):
    file_exists = LOG_PROCESS_FILE.exists()
    with open(LOG_PROCESS_FILE, 'a', newline='', encoding='utf-8') as f:
        fieldnames = [
            "arquivo", "tempo_processamento_s", "quantidade_paginas", 
            "quantidade_imagens_extraidas", "quantidade_tabelas_extraidas",
            "ram_max_mb", "vram_max_mb", "status"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)

def write_embed_log(result: dict):
    file_exists = LOG_EMBED_FILE.exists()
    with open(LOG_EMBED_FILE, 'a', newline='', encoding='utf-8') as f:
        fieldnames = [
            "doc_id", "timestamp", "n_chunks", "total_chars", "chunk_time_s",
            "embed_time_s", "upsert_time_s", "total_time_s", "chars_per_s",
            "ram_delta_mb", "ram_peak_mb", "vram_peak_mb", "avg_sparse_tokens",
            "status"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)

def run_pdf_pipeline(pdf_path: Path, item_uuid: str = "", item_handle: str = ""):
    """
    Worker job executado em background para processar o PDF inteiro
    e indexá-lo no Qdrant sem travar o endpoint.

    item_uuid/item_handle (ingestão a partir de um item DSpace) são propagados
    aos chunks do Qdrant e ao metadata da LLM.
    """
    job_id = pdf_path.stem
    log.info("Pipeline Background Iniciado: %s", pdf_path.name)
    set_status(job_id, "processando", filename=pdf_path.name)
    try:
        # 1. Extração via MinerU
        proc_result = process_pdf(pdf_path, OUTPUT_DIR)
        write_process_log(proc_result)

        if proc_result["status"] != "Sucesso":
            log.error("Pipeline abortado para %s: Falha no MinerU.", pdf_path.name)
            set_status(job_id, "erro", error="Falha na extração do MinerU")
            return

        # 2. Busca pelo JSON recém-criado (fica em <basename>/<método>/, ex: hybrid_auto/)
        json_matches = list((OUTPUT_DIR / job_id).rglob(f"{job_id}_content_list_v2.json"))

        if not json_matches:
            log.warning("JSON de conteúdo não encontrado em: %s", OUTPUT_DIR / job_id)
            set_status(job_id, "erro", error="JSON de conteúdo não encontrado após extração")
            return
        json_path = json_matches[0]

        # MD pronto a partir daqui — a indexação é best-effort (não invalida o MD)
        md_path = find_markdown(job_id)
        md_url = markdown_url(md_path) if md_path else None

        # 3. Enriquecimento de metadados por LLM (DeepSeek) — best-effort.
        # Quando disponível, roda antes da indexação para que os chunks recebam
        # no payload os metadados derivados da análise do documento.
        if llm_enrich.is_available():
            try:
                log.info("Iniciando enriquecimento por LLM para: %s", job_id)
                meta = llm_enrich.enrich_job(job_id, uuid=item_uuid)
                set_status(
                    job_id, "processando", md_url=md_url,
                    metadata_json=meta.arquivo_json, metadata_revisar=meta.revisar,
                )
            except Exception as llm_exc:
                log.error("Enriquecimento LLM falhou para %s: %s", job_id, llm_exc)
                log.error(traceback.format_exc())
                set_status(job_id, "processando", md_url=md_url, llm_error=str(llm_exc))

        # 4. Indexação e Embedding (Qdrant)
        try:
            log.info("Iniciando indexação (chunking/embedding) para: %s", json_path.name)
            embed_result = index_single_document(json_path, item_uuid=item_uuid, item_handle=item_handle)
            write_embed_log(embed_result)
            set_status(
                job_id,
                "concluido",
                md_url=md_url,
                n_chunks=embed_result.get("n_chunks") if isinstance(embed_result, dict) else None,
            )
            log.info("Pipeline Background Concluído para: %s", pdf_path.name)
        except Exception as idx_exc:
            log.error("Indexação falhou para %s: %s", pdf_path.name, idx_exc)
            log.error(traceback.format_exc())
            set_status(job_id, "concluido", md_url=md_url, index_error=str(idx_exc))

    except Exception as e:
        log.error("Pipeline Background falhou criticamente: %s", e)
        log.error(traceback.format_exc())
        set_status(job_id, "erro", error=str(e))
