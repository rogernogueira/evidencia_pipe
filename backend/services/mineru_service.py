import os
import signal
import time
import subprocess
import threading
from contextlib import contextmanager, nullcontext
from pathlib import Path
import psutil
from pypdf import PdfReader

from backend.core import config as settings
from backend.core.config import MINERU_API_URL
from backend.core.logger import log


@contextmanager
def _mineru_gpu_lease(pdf_path: Path, task_id, document_id):
    """Adquire o recurso de GPU imediatamente antes do subprocesso do MinerU usar
    CUDA e o mantém enquanto o subprocesso (e seus filhos) roda. Fica FORA deste
    lock: download, resolução DSpace, criação de diretórios, contagem de páginas,
    validações posteriores e escrita de CSV. Import tardio para não acoplar o
    módulo à lib quando GPU_MANAGER_ENABLED=false."""
    from backend.services.gpu_manager import get_gpu_manager

    manager = get_gpu_manager()
    with manager.acquire(
        resource=settings.GPU_RESOURCE_NAME,
        service="mineru",
        priority=settings.MINERU_GPU_PRIORITY,
        task_id=task_id,
        document_id=document_id or pdf_path.stem,
        metadata={"pdf_path": str(pdf_path), "operation": "pdf-extraction"},
    ) as lease:
        yield lease


def _terminate_process_group(process: subprocess.Popen, grace_seconds: float = 10.0) -> None:
    """Encerra o subprocesso do MinerU E seus filhos (que seguram a VRAM). Usa o
    grupo de processos criado por start_new_session=True: SIGTERM, aguarda, e por
    fim SIGKILL. Só então o lock deve ser liberado (senão a VRAM continuaria presa)."""
    if process.poll() is not None:
        return
    try:
        pgid = os.getpgid(process.pid)
    except ProcessLookupError:  # pragma: no cover
        return
    log.warning("MinerU: encerrando grupo de processos pgid=%s (SIGTERM)...", pgid)
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:  # pragma: no cover
        return
    try:
        process.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        log.warning("MinerU: grupo pgid=%s não terminou; enviando SIGKILL.", pgid)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:  # pragma: no cover
        pass
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:  # pragma: no cover
        log.error("MinerU: grupo pgid=%s ainda ativo após SIGKILL.", pgid)

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

def process_pdf(pdf_path: Path, output_dir: Path, *, task_id=None, document_id=None) -> dict:
    """Processa o PDF usando o mineru e coleta métricas.

    O subprocesso do MinerU roda sob o lock da GPU (gpu_resource_manager). O lock é
    adquirido imediatamente antes do subprocesso e mantido enquanto ele (e seus
    filhos) usam CUDA. Em falha/timeout, o grupo de processos é encerrado ANTES de
    liberar o lock (a VRAM não é liberada só por soltar o lock). Contagem de páginas,
    criação de diretório e leitura de métricas de saída ficam fora do lock."""
    pages = count_pages(pdf_path)          # fora do lock (só lê o PDF)
    basename = pdf_path.stem

    file_output_dir = output_dir / basename
    file_output_dir.mkdir(parents=True, exist_ok=True)  # fora do lock

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

    lease_cm = (
        _mineru_gpu_lease(pdf_path, task_id, document_id)
        if settings.GPU_MANAGER_ENABLED else nullcontext()
    )
    with lease_cm:
        # start_new_session=True → novo grupo de processos, para poder matar o
        # MinerU e TODOS os seus filhos (que seguram a VRAM) de uma vez.
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            start_new_session=True,
        )
        monitor_thread = threading.Thread(target=monitor_resources, args=(process.pid, stats, stop_event))
        monitor_thread.start()
        mineru_timeout = settings.MINERU_TIMEOUT_SECONDS or None
        try:
            stdout, stderr = process.communicate(timeout=mineru_timeout)
        except subprocess.TimeoutExpired:
            # Wall-clock estourado: encerra o grupo (libera a VRAM) e falha limpo.
            log.error("MinerU: timeout de %ss excedido para %s — encerrando subprocesso.",
                      mineru_timeout, pdf_path.name)
            _terminate_process_group(process)
            stop_event.set()
            monitor_thread.join()
            raise RuntimeError(f"MinerU excedeu o timeout de {mineru_timeout}s para {pdf_path.name}")
        except BaseException:
            # Erro/cancelamento (inclui SoftTimeLimitExceeded do Celery): encerra o
            # grupo e AGUARDA antes de liberar o lock (preserva o lease do `with`).
            log.error("MinerU: exceção durante a execução de %s — encerrando subprocesso.", pdf_path.name)
            _terminate_process_group(process)
            stop_event.set()
            monitor_thread.join()
            raise
        processing_time = time.time() - start_time
        stop_event.set()
        monitor_thread.join()
    # A partir daqui o lock já foi liberado; o resto (leitura de arquivos de saída,
    # contagem de imagens/tabelas, montagem das métricas) não usa a GPU.

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
