#!/usr/bin/env python3
"""Teste de carga do sistema de filas do evidencia_pipe.

Enfileira uma lista de UUIDs de ITEM DSpace via POST /api/files/dspace/item/{uuid}
(cada item resolve os PDFs do bundle ORIGINAL → 1 job por PDF), acompanha cada job
em GET /api/files/status/{job_id} até chegar a um estado terminal (concluido | erro)
e, no fim, gera um relatório (CSV + JSON) por job + um resumo agregado. Também cruza
com GET /api/files/failures.

Só usa a stdlib (urllib) — nenhuma dependência externa.

Uso:
    python scripts/test_queue_ingest.py                      # usa a lista embutida
    python scripts/test_queue_ingest.py --base-url http://127.0.0.1:8020
    python scripts/test_queue_ingest.py --force              # reprocessa do zero
    python scripts/test_queue_ingest.py --uuids-file ids.txt # 1 uuid por linha
    python scripts/test_queue_ingest.py --timeout 0          # sem teto de espera

Sinal de saída: 0 se todos os jobs concluíram sem erro; 1 se houve erro/timeout.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

# Estados terminais do job_store.
TERMINAL = {"concluido", "erro"}

# Lista de UUIDs de ITEM a testar. Sobrescreva com --uuids-file.
UUIDS = [
    "16e38276-b275-452b-991f-fe4de674f0bf",
    "204cf0e7-c5bc-4aaa-b92a-ba968c9275a4",
    "5e3ac040-3817-4f33-8137-f70860822bbb",
    "9d7ffedf-8269-432f-81e2-591dc5382c18",
    "b43b8c3a-52aa-4349-b98e-753a5343c14f",
    "a2bc03b6-977a-47d0-824d-c147358bc59c",
    "b6f55d46-d3d0-4cf7-af02-56538ab42b2c",
    "c2a58b05-72e6-46e1-b75f-a0edfdeec5a7",
    "ec9f155c-dfd0-468c-a353-11416a74b407",
    "0b0f615e-67df-41e5-9242-21344551018d",
    "4092e2c6-e015-4249-9374-b7b45a1ffa13",
    "4ac1452f-ff27-4763-a4ed-81b1614a13f9",
    "7b438e6a-0fb4-4c22-b090-27b6848120dc",
    "ba9d6eaa-acb0-4e81-83e4-44ba8d261cb9",
    "e2e3afa8-6981-489d-ae4b-1da6a2302825",
    "18afcaea-816b-4672-a13c-c4eae5aa898f",
    "0b6ab157-640d-4b83-a838-1bfce607325e",
    "8411105e-3c0a-4fe8-9488-51f46d146317",
    "b0eafefa-39a0-4a96-b66c-27325dff7cf3",
    "e99e53ba-eb61-48e9-b443-d436b226df08",
    "411c8b11-e641-4e66-be41-436dbbaf2f56",
    "3f932daa-c2a3-433f-8f26-11dc06450ae3",
    "937018ac-d05d-4499-81f4-75fbf50c7061",
    "b1bf4600-ff97-42bd-8479-d3545430cb92",
    "b8ba4956-3ace-450b-85fa-e4bf568ed139",
    "e668f011-13c6-4315-83fa-f10db8644445",
    "0f0d2a5f-5de0-4f08-bed8-08041c905a68",
    "a829603b-5de1-428b-88e3-a4d444e777f7",
    "4643b6fe-0b8a-4040-bf2c-1ecec5b91369",
    "c028f5aa-fa6b-453a-8768-ca4dc72bed38",
    "a8e91624-bcf4-4d27-b84e-be57d1f18a9e",
    "872cdfc2-9181-4272-a51d-51bf1fac8add",
    "cbf8e1cf-70c7-454c-a3c8-08b870665666",
    "e64c637a-4293-41e4-acaf-195165f72638",
    "3423e166-354a-4fe4-9fbb-bacc6fd3813d",
    "bfc42420-f408-440c-93da-865c8f7652a6",
    "a390f24b-e663-4616-9b81-eb8dd4e59353",
    "378d35a5-742e-4b35-8681-71939e2b82f6",
    "2e9cda6a-1781-4a1d-a3e0-4b589abc3590",
    "d83701f0-9236-4056-b596-351df98bc438",
    "b9f562a5-fd3a-42a9-aa10-62e1715203e0",
    "756ba7f3-0611-49dd-ae39-e8d2aada5584",
    "aeb4cd25-bf52-42fd-8089-fa9bb58e1221",
    "dcb45c24-c15f-470f-9d42-7c31ad5d6cca",
    "18e64d15-06a8-4cda-b566-21b560fa9d4e",
    "8d5df489-336a-43cc-b706-8aefe9fdd4ee",
    "60a9e4c0-8252-48e5-bc29-e0f0adae0a19",
    "f206e6dd-5fbc-4676-81a7-13785c03437e",
    "94b772fa-8829-41ec-9014-336eddd39497",
    "33a542fc-5402-40f8-9f31-c7fbc41d6aeb",
    "3acbc177-6a6c-4251-9023-c6fcccd72431",
    "74e4ec13-fbd4-4768-bb4d-f000d58c6c31",
    "e757173f-75fb-4b1c-97db-47b13505a3dc",
    "29ce3360-840f-4a64-badf-efd99f91adcd",
    "e1f6e79c-bcbf-4e0e-8ebd-ca8bfd7c030e",
    "31d3608e-6681-4259-9688-cdd87637dd42",
    "544e3ca7-97d0-484d-b24c-25182b5aebd8",
    "1f661403-2fa8-4709-bfb8-560520acfb52",
    "df4a5702-c8b1-4e3a-9d2c-1f2841b6fdda",
    "b7253992-d846-47b0-aef4-49266360c335",
    "d59f0049-631d-447a-b786-667914fc1379",
    "59cfbd5a-1116-4187-8ee4-374f782cbe0f",
    "3bceebfe-99ac-437c-a40e-e2051ae24b27",
    "66d2d10f-7f1b-43a0-8770-8ac24006711d",
]


def _request(method: str, url: str, timeout: float = 30.0) -> tuple[int, object]:
    """HTTP simples. Retorna (status_code, corpo_json_ou_texto). status_code=0 em
    falha de conexão."""
    req = urllib.request.Request(url, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", "replace")
            code = resp.getcode()
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        code = e.code
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return 0, {"_error": str(e)}
    try:
        return code, json.loads(body)
    except json.JSONDecodeError:
        return code, body


def enqueue_item(base_url: str, item_uuid: str, force: bool) -> dict:
    """POST /api/files/dspace/item/{uuid}. Cada item resolve N PDFs → N jobs.

    Retorna: {item_uuid, http, ok, detail, jobs:[{job_id, filename, bitstream_uuid}]}.
    """
    q = "?force=true" if force else ""
    code, body = _request("POST", f"{base_url}/api/files/dspace/item/{item_uuid}{q}")
    jobs = []
    detail = None
    if code == 202 and isinstance(body, dict):
        for j in body.get("jobs", []):
            jobs.append({
                "job_id": j.get("job_id"),
                "filename": j.get("filename"),
                "bitstream_uuid": j.get("bitstream_uuid"),
            })
    else:
        detail = body.get("detail") if isinstance(body, dict) else str(body)
    ok = code == 202 and len(jobs) > 0
    return {"item_uuid": item_uuid, "http": code, "ok": ok, "detail": detail, "jobs": jobs}


def get_status(base_url: str, job_id: str) -> dict:
    code, body = _request("GET", f"{base_url}/api/files/status/{job_id}")
    if code == 200 and isinstance(body, dict):
        return body
    return {"status": "desconhecido", "_http": code, "_body": body}


def get_failures(base_url: str) -> list:
    code, body = _request("GET", f"{base_url}/api/files/failures?limit=2000")
    if code == 200 and isinstance(body, dict):
        return body.get("jobs", [])
    return []


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    ap = argparse.ArgumentParser(description="Teste de carga da fila do evidencia_pipe (endpoint de ITEM).")
    ap.add_argument("--base-url", default="http://127.0.0.1:8020")
    ap.add_argument("--force", action="store_true", help="reprocessa do zero (?force=true)")
    ap.add_argument("--poll-interval", type=float, default=10.0, help="segundos entre rondas de polling")
    ap.add_argument("--timeout", type=float, default=14400.0, help="teto total de espera em s (0 = sem teto)")
    ap.add_argument("--enqueue-workers", type=int, default=8, help="concorrência do enfileiramento")
    ap.add_argument("--uuids-file", help="arquivo com 1 uuid de item por linha (sobrescreve a lista embutida)")
    ap.add_argument("--out-prefix", default=None, help="prefixo dos arquivos de relatório")
    args = ap.parse_args()

    base_url = args.base_url.rstrip("/")
    uuids = UUIDS
    if args.uuids_file:
        with open(args.uuids_file, encoding="utf-8") as f:
            uuids = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = args.out_prefix or f"relatorio_fila_{stamp}"

    print(f"[{_now()}] Alvo: {base_url} | {len(uuids)} item(ns) | force={args.force}")

    # Sanidade: a API está de pé?
    code, _ = _request("GET", f"{base_url}/openapi.json", timeout=10)
    if code == 0:
        print(f"ERRO: não consegui falar com a API em {base_url}. A app está no ar (porta 8020)?")
        return 1

    # ---------------- Fase 1: enfileirar (por item) ----------------
    t_enq = time.time()
    print(f"[{_now()}] Enfileirando itens…")
    with ThreadPoolExecutor(max_workers=max(1, args.enqueue_workers)) as ex:
        enq = list(ex.map(lambda u: enqueue_item(base_url, u, args.force), uuids))

    # Achata os jobs e mapeia job_id -> contexto do item.
    job_ctx: dict[str, dict] = {}
    for e in enq:
        for j in e["jobs"]:
            jid = j["job_id"]
            if jid:
                job_ctx[jid] = {"item_uuid": e["item_uuid"], "filename": j["filename"],
                                "bitstream_uuid": j["bitstream_uuid"]}
    n_items_ok = sum(1 for e in enq if e["ok"])
    n_jobs = len(job_ctx)
    print(f"[{_now()}] Itens aceitos: {n_items_ok}/{len(enq)} | jobs (PDFs) gerados: {n_jobs} "
          f"(em {time.time()-t_enq:.1f}s)")
    for e in enq:
        if not e["ok"]:
            print(f"  ! item {e['item_uuid']}: HTTP {e['http']} — {e['detail']}")

    # ---------------- Fase 2: acompanhar (por job) ----------------
    t_started = {jid: time.time() for jid in job_ctx}
    results: dict[str, dict] = {}
    pending = set(t_started)
    t_poll0 = time.time()
    print(f"[{_now()}] Acompanhando {len(pending)} job(s)…")
    while pending:
        if args.timeout and (time.time() - t_poll0) > args.timeout:
            print(f"[{_now()}] Timeout global ({args.timeout}s) — {len(pending)} job(s) ainda pendente(s).")
            break
        for job_id in list(pending):
            st = get_status(base_url, job_id)
            results[job_id] = st
            if st.get("status") in TERMINAL:
                pending.discard(job_id)
        done = len(t_started) - len(pending)
        counts = _tally(results)
        print(f"[{_now()}] {done}/{len(t_started)} terminais | "
              f"concluido={counts['concluido']} erro={counts['erro']} "
              f"processando={counts['processando']} na_fila={counts['na_fila']} | "
              f"restam {len(pending)}")
        if pending:
            time.sleep(max(1.0, args.poll_interval))

    now = time.time()
    for job_id, st in results.items():
        st["_seconds"] = round(now - t_started.get(job_id, now), 1)

    failures_idx = {j.get("job_id") for j in get_failures(base_url)}

    # ---------------- Fase 3: relatório ----------------
    rows = []
    # Itens que nem geraram job (falha de resolução / sem PDF).
    for e in enq:
        if not e["ok"]:
            rows.append(_row(item_uuid=e["item_uuid"], job_id=None, filename=None,
                             bitstream_uuid=None, enqueue_http=e["http"], enqueue_ok=False,
                             st={"status": "nao_enfileirado"}, detail=e["detail"],
                             in_failure_queue=False))
    # Jobs (PDFs) rastreados.
    for job_id, ctx in job_ctx.items():
        st = results.get(job_id, {"status": "sem_status"})
        rows.append(_row(item_uuid=ctx["item_uuid"], job_id=job_id, filename=ctx["filename"],
                         bitstream_uuid=ctx["bitstream_uuid"], enqueue_http=202, enqueue_ok=True,
                         st=st, detail=None, in_failure_queue=job_id in failures_idx))

    csv_path = f"{prefix}.csv"
    json_path = f"{prefix}.json"
    fields = list(rows[0].keys()) if rows else []
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    summary = _summary(rows, n_items=len(enq), n_items_ok=n_items_ok,
                       elapsed_total=round(time.time() - t_enq, 1))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "jobs": rows}, f, ensure_ascii=False, indent=2)

    _print_summary(summary, rows, csv_path, json_path)

    bad = [r for r in rows if r["status"] != "concluido" or r["index_error"] or r["in_failure_queue"]]
    return 0 if not bad else 1


def _row(*, item_uuid, job_id, filename, bitstream_uuid, enqueue_http, enqueue_ok, st, detail, in_failure_queue):
    return {
        "item_uuid": item_uuid,
        "job_id": job_id,
        "filename": filename,
        "bitstream_uuid": bitstream_uuid,
        "enqueue_http": enqueue_http,
        "enqueue_ok": enqueue_ok,
        "status": st.get("status"),
        "stage": st.get("stage"),
        "n_chunks": st.get("n_chunks"),
        "indexed_count": st.get("indexed_count"),
        "index_error": st.get("index_error"),
        "error": detail or st.get("error"),
        "warnings_count": st.get("warnings_count"),
        "in_failure_queue": in_failure_queue,
        "seconds_to_terminal": st.get("_seconds"),
        "updated_at": st.get("updated_at"),
    }


def _tally(results: dict) -> dict:
    c = {"concluido": 0, "erro": 0, "processando": 0, "na_fila": 0, "outros": 0}
    for st in results.values():
        s = st.get("status")
        c[s if s in c else "outros"] += 1
    return c


def _summary(rows: list, n_items: int, n_items_ok: int, elapsed_total: float) -> dict:
    def cnt(pred):
        return sum(1 for r in rows if pred(r))
    jobs = [r for r in rows if r["job_id"]]
    tempos = [r["seconds_to_terminal"] for r in jobs
              if isinstance(r["seconds_to_terminal"], (int, float)) and r["status"] in TERMINAL]
    total_chunks = sum(r["n_chunks"] for r in jobs if isinstance(r["n_chunks"], int))
    return {
        "itens_total": n_items,
        "itens_resolvidos_ok": n_items_ok,
        "itens_sem_pdf_ou_erro": n_items - n_items_ok,
        "jobs_total": len(jobs),
        "concluido_ok": cnt(lambda r: r["job_id"] and r["status"] == "concluido" and not r["index_error"]),
        "concluido_com_index_error": cnt(lambda r: r["status"] == "concluido" and r["index_error"]),
        "erro": cnt(lambda r: r["status"] == "erro"),
        "pendente_ou_desconhecido": cnt(lambda r: r["job_id"] and r["status"] not in TERMINAL),
        "na_fila_de_falhas": cnt(lambda r: r["in_failure_queue"]),
        "total_chunks_indexados": total_chunks,
        "tempo_medio_terminal_s": round(sum(tempos) / len(tempos), 1) if tempos else None,
        "tempo_max_terminal_s": round(max(tempos), 1) if tempos else None,
        "tempo_total_s": elapsed_total,
    }


def _print_summary(summary: dict, rows: list, csv_path: str, json_path: str) -> None:
    print("\n" + "=" * 64)
    print("RELATÓRIO — teste de fila (endpoint de item)")
    print("=" * 64)
    for k, v in summary.items():
        print(f"  {k:<28} {v}")
    problematicos = [r for r in rows if r["status"] != "concluido" or r["index_error"]]
    if problematicos:
        print("\n  Com problema:")
        for r in problematicos:
            motivo = r["error"] or r["index_error"] or f"status={r['status']}"
            ref = r["job_id"] or f"item:{r['item_uuid']}"
            print(f"    - {ref}  [{r['status']}]  {str(motivo)[:120]}")
    print(f"\n  CSV : {csv_path}")
    print(f"  JSON: {json_path}")
    print("=" * 64)


if __name__ == "__main__":
    sys.exit(main())
