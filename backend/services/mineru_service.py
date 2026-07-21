import time
import subprocess
import threading
from pathlib import Path
import psutil
from pypdf import PdfReader

from backend.core.config import MINERU_API_URL
from backend.core.logger import log

def get_vram_usage():
    """Retorna o uso atual de VRAM da GPU (em MB) usando nvidia-smi"""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True
        )
        lines = [int(x.strip()) for x in out.strip().split('\n') if x.strip()]
        return max(lines) if lines else 0
    except Exception:
        return 0

def monitor_resources(pid, stats, stop_event):
    """Monitora o pico de uso de RAM e VRAM em uma thread separada"""
    max_ram_mb = 0
    max_vram_mb = 0

    try:
        proc = psutil.Process(pid)
    except psutil.NoSuchProcess:
        proc = None

    while not stop_event.is_set():
        ram_mb = 0
        if proc and proc.is_running():
            try:
                ram_mb = proc.memory_info().rss / (1024 * 1024)
                for child in proc.children(recursive=True):
                    ram_mb += child.memory_info().rss / (1024 * 1024)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        if ram_mb > max_ram_mb:
            max_ram_mb = ram_mb
        
        vram_mb = get_vram_usage()
        if vram_mb > max_vram_mb:
            max_vram_mb = vram_mb
            
        time.sleep(1)
        
    stats['max_ram_mb'] = max_ram_mb
    stats['max_vram_mb'] = max_vram_mb

def count_pages(pdf_path):
    """Conta as páginas do PDF usando pypdf"""
    try:
        reader = PdfReader(str(pdf_path))
        return len(reader.pages)
    except Exception:
        return 0

def process_pdf(pdf_path: Path, output_dir: Path) -> dict:
    """Processa o PDF usando o mineru e coleta métricas"""
    pages = count_pages(pdf_path)
    basename = pdf_path.stem
    
    file_output_dir = output_dir / basename
    file_output_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "/root/.local/bin/uv", "run", "mineru",
        "-p", str(pdf_path),
        "-o", str(output_dir),
        "--api-url", MINERU_API_URL,
        "-m", "auto",
        "-l", "latin",
        "-b", "hybrid-auto-engine"
    ]
    
    stats = {'max_ram_mb': 0, 'max_vram_mb': 0}
    stop_event = threading.Event()
    
    log.info("Iniciando MinerU: %s (%d páginas)", pdf_path.name, pages)
    start_time = time.time()
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    monitor_thread = threading.Thread(target=monitor_resources, args=(process.pid, stats, stop_event))
    monitor_thread.start()
    
    stdout, stderr = process.communicate()
    processing_time = time.time() - start_time
    
    stop_event.set()
    monitor_thread.join()

    # Contagens de Extrações (a saída do MinerU fica em <basename>/<método>/, ex: hybrid_auto/)
    images_count = 0
    tables_count = 0

    md_files = list(file_output_dir.rglob("*.md"))
    if md_files:
        try:
            content = md_files[0].read_text(encoding="utf-8")
            tables_count = content.count("|---|") + content.count("<table>")
        except Exception:
            pass

    images_dirs = list(file_output_dir.rglob("images"))
    if images_dirs:
        images_count = sum(1 for item in images_dirs[0].iterdir() if item.is_file())
    elif md_files:
        try:
            images_count = md_files[0].read_text(encoding="utf-8").count("![")
        except Exception:
            pass

    status = "Sucesso" if process.returncode == 0 else "Erro"
    if status == "Erro":
        log.error("Erro ao processar %s. Logs: %s...", pdf_path.name, stderr[:200])

    metrics = {
        "arquivo": pdf_path.name,
        "tempo_processamento_s": round(processing_time, 2),
        "quantidade_paginas": pages,
        "quantidade_imagens_extraidas": images_count,
        "quantidade_tabelas_extraidas": tables_count,
        "ram_max_mb": round(stats['max_ram_mb'], 2),
        "vram_max_mb": round(stats['max_vram_mb'], 2),
        "status": status
    }
    
    return metrics
