"""Relatórios CSV do pipeline (métricas por-documento).

Escreve os logs de processamento (MinerU) e de embeddings (índice), consumidos pelas
etapas em backend/services/pipeline_stages.py. Não contém mais o fluxo legado
`run_pdf_pipeline` (removido — o único caminho de ingestão é a chain Celery).
"""

import csv
from pathlib import Path

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
        # extrasaction="ignore": tolera chaves extras no dict (ex.: métricas novas)
        # sem quebrar o relatório CSV legado.
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
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
        # extrasaction="ignore": o result do chunker estrutural traz chaves extras
        # (chunking_strategy/version/config_hash, tokenizer_name, structure_source,
        # chunking_report) que não entram neste CSV flat — ignora em vez de quebrar.
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)
