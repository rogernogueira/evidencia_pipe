"""Testes do PipelineContext e da barreira de payload da chain (§44.26-28)."""

import os
import sys
import uuid

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.core.schemas import (  # noqa: E402
    ForbiddenChainPayloadError,
    PipelineContext,
    PipelinePayloadTooLargeError,
    validate_chain_payload_size,
)


def _ctx():
    return PipelineContext(
        pipeline_id=uuid.uuid4(),
        job_id="doc-1",
        item_uuid="item-uuid",
        bitstream_uuid="bs-uuid",
        document_id="doc-1",
        artifact_manifest_uri="minio://evidencia-pipe/artifacts/p/doc-1/manifest.json",
        current_stage="downloaded",
    )


def test_context_json_serializable_and_small():
    msg = _ctx().to_message()
    assert isinstance(msg["pipeline_id"], str)  # UUID → str
    size = validate_chain_payload_size(msg, 16384)
    assert size < 1024


def test_context_roundtrip():
    ctx = _ctx()
    assert PipelineContext.model_validate(ctx.to_message()) == ctx


def test_payload_too_large():
    with pytest.raises(PipelinePayloadTooLargeError):
        validate_chain_payload_size({"blob": "x" * 20000}, 16384)


@pytest.mark.parametrize("key", ["markdown", "chunks", "embeddings", "mineru_json", "pdf_bytes", "raw_response"])
def test_forbidden_keys_rejected(key):
    with pytest.raises(ForbiddenChainPayloadError):
        validate_chain_payload_size({key: "small"}, 16384)


def test_binary_value_rejected():
    with pytest.raises(ForbiddenChainPayloadError):
        validate_chain_payload_size({"foo": b"\x00\x01"}, 16384)


def test_non_serializable_rejected():
    with pytest.raises(ForbiddenChainPayloadError):
        validate_chain_payload_size({"foo": {1, 2, 3}}, 16384)


def test_context_forbids_extra_fields():
    with pytest.raises(Exception):
        PipelineContext.model_validate({
            "pipeline_id": str(uuid.uuid4()), "job_id": "d", "document_id": "d",
            "artifact_manifest_uri": "minio://b/k", "current_stage": "x",
            "markdown": "conteúdo gigante proibido",  # extra field
        })
