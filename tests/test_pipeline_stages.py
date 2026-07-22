"""Testes das etapas do pipeline com MinIO fake + externos mockados (§46/§47).

Exercita o fluxo real download→mineru→index (chain obrigatória) + enrich DESACOPLADO
em memória, sem DSpace/MinerU/LLM/BGE-M3/Qdrant reais, e verifica que:
  - cada etapa grava os artefatos no MinIO e atualiza o manifesto;
  - a indexação NÃO depende do enrich (roda sem LLM);
  - o PipelineContext trafegado é pequeno e sem chaves proibidas (§47);
  - retries reutilizam artefatos válidos (idempotência).
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.core import config as settings  # noqa: E402
from backend.core.schemas import (  # noqa: E402
    ART_CHUNKS,
    ART_MINERU_CONTENT_LIST,
    ART_MINERU_MARKDOWN,
    validate_chain_payload_size,
)
from tests.fakes import make_minio_store  # noqa: E402


@pytest.fixture
def wired(monkeypatch, tmp_path):
    """Injeta o store fake e reseta o singleton do ManifestRepository."""
    import backend.services.artifact_store as store_mod
    import backend.services.manifest_repository as repo_mod

    store = make_minio_store()
    monkeypatch.setattr(store_mod, "_store", store)
    monkeypatch.setattr(repo_mod, "_repo", None)
    monkeypatch.setattr(settings, "ARTIFACT_TEMP_DIR", str(tmp_path / "work"))

    # DSpace: escreve um "PDF" local.
    import backend.services.dspace_service as dspace

    def fake_download(bs_uuid, dest_dir, filename=None):
        dest_dir.mkdir(parents=True, exist_ok=True)
        p = dest_dir / (filename or f"{bs_uuid}.pdf")
        p.write_bytes(b"%PDF-1.4 conteudo de teste")
        return p

    monkeypatch.setattr(dspace, "download_bitstream", fake_download)

    # MinerU: cria a estrutura de saída esperada.
    import backend.services.mineru_service as mineru

    def fake_process(pdf_local, out_dir, *, task_id=None, document_id=None):
        doc_out = out_dir / document_id / "hybrid_auto"
        doc_out.mkdir(parents=True)
        (doc_out / f"{document_id}_content_list_v2.json").write_text(
            json.dumps([{"type": "text", "text": "olá", "page_idx": 0}]), encoding="utf-8")
        (doc_out / f"{document_id}.md").write_text("# Título\n\nCorpo do documento.", encoding="utf-8")
        imgs = doc_out / "images"
        imgs.mkdir()
        (imgs / "a.png").write_bytes(b"PNG1")
        (imgs / "b.png").write_bytes(b"PNG2")
        return {"arquivo": f"{document_id}.pdf", "status": "Sucesso",
                "quantidade_paginas": 5, "quantidade_imagens_extraidas": 2}

    monkeypatch.setattr(mineru, "process_pdf", fake_process)

    # CSVs: no-op.
    import backend.services.report_logs as pw

    monkeypatch.setattr(pw, "write_process_log", lambda r: None)
    monkeypatch.setattr(pw, "write_embed_log", lambda r: None)

    # Indexação: grava o chunks.jsonl e devolve métricas pequenas.
    import backend.indexing.index_chunks as idx

    def fake_index(json_local, *, doc_id, chunks_jsonl_path=None,
                   chunking_report_path=None, **kwargs):
        if chunks_jsonl_path is not None:
            chunks_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            with open(chunks_jsonl_path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"chunk_id": "x", "text": "olá", "chunk_index": 0}) + "\n")
        report = {"chunk_count": 1, "structure_source": "mineru_json"}
        if chunking_report_path is not None:
            chunking_report_path.parent.mkdir(parents=True, exist_ok=True)
            chunking_report_path.write_text(json.dumps(report), encoding="utf-8")
        return {"doc_id": doc_id, "n_chunks": 1, "status": "ok",
                "chunking_strategy": "structural_tokens", "chunking_config_hash": "abc",
                "tokenizer_name": "BAAI/bge-m3", "chunking_report": report}

    monkeypatch.setattr(idx, "index_materialized_document", fake_index)
    return store


def _assert_light(ctx):
    msg = ctx.to_message()
    size = validate_chain_payload_size(msg, settings.CELERY_CHAIN_MAX_PAYLOAD_BYTES)
    assert size <= settings.CELERY_CHAIN_MAX_PAYLOAD_BYTES


def test_full_flow_no_llm(wired, monkeypatch):
    import backend.services.llm_enrich_service as llm

    monkeypatch.setattr(llm, "is_available", lambda: False)
    from backend.services import pipeline_stages as stages

    ctx = stages.stage_download(bs_uuid="bs-1", filename="documento-1.pdf", job_id="documento-1",
                                item_uuid="item-1", item_handle="h/1")
    _assert_light(ctx)
    assert wired.exists("artifacts/" + str(ctx.pipeline_id) + "/documento-1/source/original.pdf")

    ctx = stages.stage_mineru(ctx)
    _assert_light(ctx)
    from backend.services.manifest_repository import get_manifest_repository

    m = get_manifest_repository().load_from_uri(ctx.artifact_manifest_uri)
    assert ART_MINERU_CONTENT_LIST in m.artifacts
    assert ART_MINERU_MARKDOWN in m.artifacts
    assert m.metrics["images_count"] == 2
    # imagens sob prefixo (não uma ref por imagem no manifesto principal)
    assert m.artifacts["mineru_images"].object_count == 2

    # Indexação DESACOPLADA: roda direto após o mineru, SEM enrich/LLM.
    summary = stages.stage_index(ctx)
    validate_chain_payload_size(summary, settings.CELERY_CHAIN_MAX_PAYLOAD_BYTES)
    assert summary["status"] == "completed"
    assert summary["chunk_count"] == 1
    m = get_manifest_repository().load_from_uri(ctx.artifact_manifest_uri)
    assert m.status == "COMPLETED"
    assert ART_CHUNKS in m.artifacts
    assert m.is_stage_completed("indexing")

    # Enrich (fora da chain) sem LLM → no-op, não altera o índice já concluído.
    ctx = stages.stage_enrich(ctx)
    _assert_light(ctx)


def test_enrich_writes_metadata(wired, monkeypatch):
    import backend.services.llm_enrich_service as llm
    from backend.core.schemas import DocumentMetadata

    monkeypatch.setattr(llm, "is_available", lambda: True)
    monkeypatch.setattr(llm, "enrich_markdown",
                        lambda md, doc_id, uuid="", raw_sink=None: DocumentMetadata(doc_id=doc_id, titulo_candidato="T"))
    from backend.services import pipeline_stages as stages

    ctx = stages.stage_download(bs_uuid="bs-2", filename="doc-2.pdf", job_id="doc-2")
    ctx = stages.stage_mineru(ctx)
    ctx = stages.stage_enrich(ctx)
    from backend.services.manifest_repository import get_manifest_repository

    m = get_manifest_repository().load_from_uri(ctx.artifact_manifest_uri)
    assert "metadata_candidates" in m.artifacts
    assert m.is_stage_completed("enrichment")


def test_enrich_after_index_propagates_to_qdrant(wired, monkeypatch):
    """Enrich DESACOPLADO rodando APÓS a indexação propaga os metadados ao Qdrant
    (push_llm_metadata_to_qdrant), sem re-embedar."""
    import backend.services.llm_enrich_service as llm
    import backend.indexing.index_chunks as idx
    from backend.core.schemas import DocumentMetadata

    monkeypatch.setattr(llm, "is_available", lambda: True)
    monkeypatch.setattr(llm, "enrich_markdown",
                        lambda md, doc_id, uuid="", raw_sink=None: DocumentMetadata(doc_id=doc_id, titulo_candidato="T"))
    pushed = {}
    monkeypatch.setattr(idx, "push_llm_metadata_to_qdrant",
                        lambda doc_id, payload: pushed.update(doc_id=doc_id, payload=payload) or 3)

    from backend.services import pipeline_stages as stages

    ctx = stages.stage_download(bs_uuid="bs-4", filename="doc-4.pdf", job_id="doc-4")
    ctx = stages.stage_mineru(ctx)
    stages.stage_index(ctx)              # índice primeiro (desacoplado)
    stages.stage_enrich(ctx)             # enrich depois → propaga ao Qdrant

    assert pushed.get("doc_id") == "doc-4"
    assert pushed["payload"]["metadata_source"] == "llm"
    assert pushed["payload"]["titulo_candidato"] == "T"


def test_download_retry_reuses_source(wired):
    from backend.services import pipeline_stages as stages
    from backend.services.manifest_repository import get_manifest_repository

    ctx = stages.stage_download(bs_uuid="bs-3", filename="doc-3.pdf", job_id="doc-3")
    rev1 = get_manifest_repository().load_from_uri(ctx.artifact_manifest_uri).revision
    # segunda chamada (retry) NÃO deve re-baixar nem re-gravar (reuso idempotente)
    ctx2 = stages.stage_download(bs_uuid="bs-3", filename="doc-3.pdf", job_id="doc-3")
    rev2 = get_manifest_repository().load_from_uri(ctx2.artifact_manifest_uri).revision
    assert rev2 == rev1  # sem novo update de manifesto
    assert ctx2.pipeline_id == ctx.pipeline_id  # pipeline_id determinístico
