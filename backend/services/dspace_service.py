import json
import re
import urllib.request
import urllib.error
import urllib.parse
from pathlib import Path

from backend.core.config import DSPACE_URL
from backend.core.logger import log


def _bitstream_url(uuid: str) -> str:
    return f"{DSPACE_URL}/server/api/core/bitstreams/{uuid}/content"


def _get_json(url: str) -> dict:
    """GET JSON da API REST do DSpace (anônimo, sem auth)."""
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _is_pdf_bitstream(bs: dict) -> bool:
    """True se o bitstream é um PDF (por mimetype ou extensão do nome)."""
    mime = ((bs.get("metadata") or {}).get("dc.format.mimetype") or [{}])
    mime_value = (mime[0].get("value") if mime else "") or ""
    if "pdf" in mime_value.lower():
        return True
    name = (bs.get("name") or "").lower()
    return name.endswith(".pdf")


def resolve_item_pdfs(item_uuid: str) -> list[dict]:
    """Resolve os bitstreams PDF do bundle ORIGINAL de um item DSpace.

    Retorna [{bitstream_uuid, filename, item_handle}] (um por PDF). Levanta
    ValueError se o item não tiver nenhum PDF no ORIGINAL, e propaga
    urllib.error.HTTPError/URLError em falhas de rede/HTTP.
    """
    item_url = f"{DSPACE_URL}/server/api/core/items/{item_uuid}"
    log.info("Resolvendo PDFs do item DSpace: %s", item_url)
    item = _get_json(item_url)
    item_handle = item.get("handle")

    bundles_href = (
        item.get("_links", {}).get("bundles", {}).get("href")
        or f"{DSPACE_URL}/server/api/core/items/{item_uuid}/bundles"
    )
    bundles = _get_json(bundles_href).get("_embedded", {}).get("bundles", [])

    pdfs: list[dict] = []
    for b in bundles:
        if b.get("name") != "ORIGINAL":
            continue
        bs_href = b.get("_links", {}).get("bitstreams", {}).get("href")
        if not bs_href:
            continue
        bss = _get_json(bs_href).get("_embedded", {}).get("bitstreams", [])
        for bs in bss:
            if not _is_pdf_bitstream(bs):
                continue
            bs_uuid = bs.get("uuid")
            if not bs_uuid:
                continue
            pdfs.append({
                "bitstream_uuid": bs_uuid,
                "filename": bs.get("name") or f"{bs_uuid}.pdf",
                "item_handle": item_handle,
            })

    if not pdfs:
        raise ValueError(f"Item {item_uuid} não tem nenhum PDF no bundle ORIGINAL.")
    log.info("Item %s: %d PDF(s) encontrado(s) no ORIGINAL.", item_uuid, len(pdfs))
    return pdfs


def _filename_from_headers(headers, uuid: str) -> str:
    """Extrai o nome original do Content-Disposition; fallback para <uuid>.pdf."""
    disposition = headers.get("Content-Disposition", "") or ""
    match = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^"\r\n;]+)"?', disposition)
    name = match.group(1).strip() if match else ""
    name = urllib.parse.unquote(name) if name else ""
    # Sanitiza: só o basename, sem separadores de caminho.
    name = Path(name).name
    if not name:
        name = f"{uuid}.pdf"
    if not name.lower().endswith(".pdf"):
        name = f"{name}.pdf"
    return name


def download_bitstream(uuid: str, dest_dir: Path, filename: str | None = None) -> Path:
    """
    Baixa o PDF de um bitstream do DSpace para dest_dir e retorna o caminho.

    Se `filename` for informado, força o nome de destino (usado pela ingestão de
    item, onde o job_id é derivado do nome resolvido ANTES do download — garante
    que job_id === stem do arquivo === doc_id em todo o pipeline). Caso contrário,
    deriva do Content-Disposition (comportamento do endpoint de bitstream avulso).

    Levanta ValueError se o conteúdo não for um PDF e urllib.error.HTTPError/
    URLError em falhas de rede.
    """
    url = _bitstream_url(uuid)
    log.info("Baixando bitstream do DSpace: %s", url)

    dest_dir.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"Accept": "application/pdf,*/*"})

    with urllib.request.urlopen(req, timeout=120) as resp:
        content_type = (resp.headers.get("Content-Type") or "").lower()
        if filename:
            filename = Path(filename).name  # sanitiza: só o basename
            if not filename.lower().endswith(".pdf"):
                filename = f"{filename}.pdf"
        else:
            filename = _filename_from_headers(resp.headers, uuid)
        # DSpace retorna text/html quando o bitstream não existe ou exige auth.
        if "pdf" not in content_type and not filename.lower().endswith(".pdf"):
            raise ValueError(
                f"Bitstream {uuid} não é um PDF (Content-Type={content_type!r})."
            )
        dest_path = dest_dir / filename
        with dest_path.open("wb") as f:
            while chunk := resp.read(1 << 16):
                f.write(chunk)

    log.info("Bitstream %s salvo em %s (%d bytes)", uuid, dest_path, dest_path.stat().st_size)
    return dest_path
