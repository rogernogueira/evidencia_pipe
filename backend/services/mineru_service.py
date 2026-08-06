"""Estágio 2 — extração via MinerU, falando HTTP direto com o contêiner.

O MinerU roda EXCLUSIVAMENTE no serviço Docker (`mineru-api`, ver docker-compose).
Este módulo é cliente da API dele: submete o PDF, aguarda a tarefa e extrai o ZIP de
resultado no `output_dir`.

Antes este módulo chamava `uv run mineru --api-url …` num subprocesso. A CLI era só
um cliente HTTP do MESMO contêiner — mas obrigava o host a instalar `mineru[all]`
(~3 GB de torch/vllm para, na prática, fazer requisições) e criava um acoplamento de
versão cliente↔servidor que já quebrou em produção (contêiner legado em 3.2.0 contra
cliente 3.4.4). Falar HTTP direto elimina os dois problemas.

Equivalência verificada (3.4.4, backend pipeline, mesmo servidor): o
`content_list_v2.json` que vem no ZIP do servidor é IDÊNTICO ao que a CLI gerava
localmente, inclusive no layout de diretórios `<stem>/auto/`. O contrato com
`stage_mineru` (que faz rglob por `*_content_list_v2.json`, `*.md` e `images/`)
permanece intacto.

Atenção ao formato: `return_content_list=true` sozinho devolve o content_list **v1**
(plano). O v2 — o que `document_blocks.py` parseia — só vem com
`response_format_zip=true`, dentro do ZIP.
"""

import io
import subprocess
import threading
import time
import zipfile
from contextlib import contextmanager, nullcontext
from pathlib import Path

import httpx
from pypdf import PdfReader

from backend.core import config as settings
from backend.core.config import MINERU_API_URL
from backend.core.logger import log

# Intervalo de polling da tarefa no servidor.
_POLL_INTERVAL_SECONDS = 3.0
# Estados terminais da API de tarefas do MinerU 3.4.x.
_DONE_STATES = frozenset({"completed", "failed", "error"})
_FAILED_STATES = frozenset({"failed", "error"})


@contextmanager
def _mineru_gpu_lease(pdf_path: Path, task_id, document_id):
    """Adquire o recurso de GPU enquanto o contêiner processa o PDF. Quem usa CUDA é
    o servidor, mas a GPU é a MESMA — o lock continua serializando o acesso contra o
    bge-m3. Ficam FORA do lock: download, criação de diretórios, contagem de páginas
    e a extração do ZIP. Import tardio para não acoplar o módulo à lib quando
    GPU_MANAGER_ENABLED=false."""
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


def get_vram_usage():
    """Uso atual de VRAM da GPU (MB) via nvidia-smi. Continua medindo a GPU INTEIRA
    (como antes) — o processo que consome agora é o do contêiner, não um filho nosso."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True
        )
        lines = [int(x.strip()) for x in out.strip().split('\n') if x.strip()]
        return max(lines) if lines else 0
    except Exception:
        return 0


def monitor_vram(stats, stop_event):
    """Amostra o pico de VRAM enquanto o contêiner processa."""
    max_vram_mb = 0
    while not stop_event.is_set():
        vram_mb = get_vram_usage()
        if vram_mb > max_vram_mb:
            max_vram_mb = vram_mb
        time.sleep(1)
    stats['max_vram_mb'] = max_vram_mb


def count_pages(pdf_path):
    """Conta as páginas do PDF usando pypdf"""
    try:
        reader = PdfReader(str(pdf_path))
        return len(reader.pages)
    except Exception:
        return 0


def _client() -> httpx.Client:
    """Cliente HTTP para o mineru-api. O timeout de leitura é generoso porque o
    servidor responde só ao fim do parse quando o resultado é grande."""
    return httpx.Client(
        base_url=MINERU_API_URL.rstrip("/"),
        timeout=httpx.Timeout(300.0, connect=10.0),
    )


def _submit(client: httpx.Client, pdf_path: Path) -> str:
    """Submete o PDF e devolve o task_id.

    `response_format_zip=true` é OBRIGATÓRIO: é a única forma de o servidor devolver
    o `content_list_v2.json` (sem ele vem o content_list v1, que o parser descarta)."""
    # dict (não lista de tuplas): o httpx só monta o multipart corretamente assim.
    # `lang_list` é array na API — vai como lista, que o httpx repete no corpo.
    data = {
        "backend": settings.MINERU_BACKEND,
        "parse_method": settings.MINERU_METHOD,
        "lang_list": [settings.MINERU_LANG],
        "return_md": "true",
        "return_content_list": "true",
        "return_images": "true",
        "response_format_zip": "true",
    }
    with pdf_path.open("rb") as fh:
        resp = client.post(
            "/tasks", data=data,
            files=[("files", (pdf_path.name, fh, "application/pdf"))],
        )
    resp.raise_for_status()
    task_id = resp.json().get("task_id")
    if not task_id:
        raise RuntimeError(f"mineru-api não devolveu task_id: {resp.text[:200]}")
    return task_id


def _wait(client: httpx.Client, task_id: str, deadline: float | None, pdf_name: str) -> str:
    """Aguarda a tarefa e devolve o estado final.

    Mantém a semântica antiga de `process_pdf`: falha do parse NÃO levanta (vira
    status "Erro", e quem decide é o `stage_mineru`); timeout levanta RuntimeError,
    como o subprocesso fazia.

    A API 3.4.x NÃO expõe cancelamento — em timeout apenas paramos de aguardar; o
    servidor segue processando até concluir sozinho. Difere do comportamento antigo,
    em que matávamos o grupo de processos local e liberávamos a VRAM na hora."""
    while True:
        resp = client.get(f"/tasks/{task_id}")
        resp.raise_for_status()
        body = resp.json()
        state = str(body.get("status", "")).lower()
        if state in _FAILED_STATES:
            log.error("MinerU: tarefa %s falhou para %s: %s",
                      task_id, pdf_name, body.get("error") or body)
            return state
        if state in _DONE_STATES:
            return state
        if deadline is not None and time.time() > deadline:
            log.error(
                "MinerU: timeout aguardando a tarefa %s de %s. O servidor CONTINUA "
                "processando (a API não permite cancelar).", task_id, pdf_name,
            )
            raise RuntimeError(
                f"MinerU excedeu o timeout de {settings.MINERU_TIMEOUT_SECONDS}s para {pdf_name}"
            )
        time.sleep(_POLL_INTERVAL_SECONDS)


def _extract_zip(raw: bytes, output_dir: Path) -> None:
    """Extrai o ZIP no output_dir, reproduzindo o layout `<stem>/auto/…` que a CLI
    criava. Rejeita caminhos que escapem do destino (zip slip)."""
    if raw[:2] != b"PK":
        raise RuntimeError(
            f"mineru-api devolveu {len(raw)} byte(s) que não são um ZIP "
            f"(início: {raw[:60]!r}). Confira response_format_zip."
        )
    output_dir = output_dir.resolve()
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            dest = (output_dir / member.filename).resolve()
            if not dest.is_relative_to(output_dir):
                raise RuntimeError(f"entrada suspeita no ZIP do MinerU: {member.filename!r}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, dest.open("wb") as out:
                out.write(src.read())


def process_pdf(pdf_path: Path, output_dir: Path, *, task_id=None, document_id=None) -> dict:
    """Extrai o PDF no contêiner MinerU e materializa a saída em `output_dir`.

    O lock de GPU (gpu_resource_manager) cobre submissão + espera, porque é nesse
    intervalo que o contêiner usa CUDA. Ficam fora do lock: contagem de páginas,
    extração do ZIP e a contagem de imagens/tabelas."""
    pages = count_pages(pdf_path)          # fora do lock (só lê o PDF)
    basename = pdf_path.stem

    file_output_dir = output_dir / basename
    file_output_dir.mkdir(parents=True, exist_ok=True)  # fora do lock

    log.info("MinerU: -m %s -l %s -b %s (api=%s)",
             settings.MINERU_METHOD, settings.MINERU_LANG, settings.MINERU_BACKEND, MINERU_API_URL)
    log.info("Iniciando MinerU: %s (%d páginas)", pdf_path.name, pages)

    stats = {'max_vram_mb': 0}
    stop_event = threading.Event()
    start_time = time.time()
    timeout_s = settings.MINERU_TIMEOUT_SECONDS or None
    deadline = (start_time + timeout_s) if timeout_s else None

    lease_cm = (
        _mineru_gpu_lease(pdf_path, task_id, document_id)
        if settings.GPU_MANAGER_ENABLED else nullcontext()
    )
    status = "Sucesso"
    raw = b""
    with lease_cm:
        monitor_thread = threading.Thread(target=monitor_vram, args=(stats, stop_event))
        monitor_thread.start()
        # Timeout e erro de transporte PROPAGAM (o subprocesso também propagava);
        # o `with lease_cm` garante a liberação do lock. Só a falha do parte-servidor
        # vira status "Erro", como o returncode != 0 fazia antes.
        try:
            with _client() as client:
                server_task = _submit(client, pdf_path)
                log.info("MinerU: tarefa %s submetida para %s", server_task, pdf_path.name)
                if _wait(client, server_task, deadline, pdf_path.name) in _FAILED_STATES:
                    status = "Erro"
                else:
                    result = client.get(f"/tasks/{server_task}/result")
                    result.raise_for_status()
                    raw = result.content
        finally:
            stop_event.set()
            monitor_thread.join()
        processing_time = time.time() - start_time
    # A partir daqui o lock já foi liberado; nada abaixo usa a GPU.

    if status == "Sucesso":
        _extract_zip(raw, output_dir)

    # Contagens de extrações (a saída fica em <basename>/<método>/, ex: auto/)
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

    return {
        "arquivo": pdf_path.name,
        "tempo_processamento_s": round(processing_time, 2),
        "quantidade_paginas": pages,
        "quantidade_imagens_extraidas": images_count,
        "quantidade_tabelas_extraidas": tables_count,
        # A extração roda no contêiner: o host não tem mais um subprocesso cuja RSS
        # medir. A chave é mantida para não quebrar o CSV/relatório e o schema do
        # processing_metrics.json. Para RAM do extrator, olhe `docker stats`.
        "ram_max_mb": 0.0,
        "vram_max_mb": round(stats['max_vram_mb'], 2),
        "status": status,
    }
