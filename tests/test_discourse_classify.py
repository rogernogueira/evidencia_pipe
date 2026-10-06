"""Testes do classificador discursivo (discourse_classify_service) e da propagação
por ponto (push_discourse_roles_to_qdrant), com LLM e Qdrant mockados."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import backend.services.discourse_classify_service as svc  # noqa: E402


# --------------------------------------------------------------------------
# ChunkLabels — validação/normalização da saída do LLM
# --------------------------------------------------------------------------

def test_descarta_classe_fora_da_taxonomia():
    m = svc.ChunkLabels.model_validate({"labels": ["achado", "inexistente"], "confidence": {}})
    assert m.labels == ["achado"]


def test_outro_nunca_combina():
    m = svc.ChunkLabels.model_validate({"labels": ["achado", "outro"], "confidence": {}})
    assert m.labels == ["achado"]


def test_vazio_vira_outro():
    assert svc.ChunkLabels.model_validate({"labels": [], "confidence": {}}).labels == ["outro"]
    # Só classe inválida também colapsa para outro.
    assert svc.ChunkLabels.model_validate({"labels": ["xpto"], "confidence": {}}).labels == ["outro"]


def test_confianca_clampa_e_filtra():
    m = svc.ChunkLabels.model_validate(
        {"labels": ["achado"], "confidence": {"achado": 1.4, "xpto": 0.5, "recomendacao": -0.2}}
    )
    assert m.confidence == {"achado": 1.0, "recomendacao": 0.0}  # xpto descartado; clamp [0,1]


def test_duplicatas_removidas_preservando_ordem():
    m = svc.ChunkLabels.model_validate({"labels": ["achado", "achado", "metodologia"], "confidence": {}})
    assert m.labels == ["achado", "metodologia"]


# --------------------------------------------------------------------------
# classify_chunks — concorrência + best-effort por chunk
# --------------------------------------------------------------------------

@pytest.fixture
def habilitado(monkeypatch):
    monkeypatch.setattr(svc, "is_available", lambda: True)


def test_classify_chunks_omite_falhas(habilitado, monkeypatch):
    def fake_one(rec):
        if rec["point_id"] == "p2":
            raise RuntimeError("boom")
        return {"labels": ["achado"], "confidence": {}}

    monkeypatch.setattr(svc, "classify_one", fake_one)
    out = svc.classify_chunks([
        {"point_id": "p1", "text": "a"},
        {"point_id": "p2", "text": "b"},
        {"point_id": "p3", "text": "c"},
    ])
    assert set(out) == {"p1", "p3"}  # p2 falhou → omitido
    assert out["p1"]["labels"] == ["achado"]


def test_classify_chunks_ignora_sem_texto_ou_id(habilitado, monkeypatch):
    monkeypatch.setattr(svc, "classify_one", lambda rec: {"labels": ["outro"], "confidence": {}})
    out = svc.classify_chunks([
        {"point_id": "p1", "text": "   "},   # texto vazio → ignorado
        {"point_id": "", "text": "x"},        # sem id → ignorado
        {"point_id": "p4", "text": "ok"},
    ])
    assert set(out) == {"p4"}


def test_classify_chunks_vazio_nao_exige_disponibilidade(monkeypatch):
    # Sem registros válidos, nem toca no is_available (retorna {} direto).
    monkeypatch.setattr(svc, "is_available", lambda: False)
    assert svc.classify_chunks([]) == {}


def test_classify_chunks_indisponivel_levanta(monkeypatch):
    monkeypatch.setattr(svc, "is_available", lambda: False)
    with pytest.raises(RuntimeError):
        svc.classify_chunks([{"point_id": "p1", "text": "a"}])


# --------------------------------------------------------------------------
# push_discourse_roles_to_qdrant — set_payload por ponto
# --------------------------------------------------------------------------

class _FakeQdrant:
    def __init__(self):
        self.ops = None

    def batch_update_points(self, collection_name, update_operations, wait=True):
        self.ops = update_operations


def test_push_grava_payload_por_ponto(monkeypatch):
    import backend.indexing.index_chunks as idx

    fake = _FakeQdrant()
    monkeypatch.setattr(idx, "_get_indexer", lambda: (fake, None, None))

    n = idx.push_discourse_roles_to_qdrant("doc-x", {
        "p1": {"labels": ["achado"], "confidence": {"achado": 0.9}},
        "p2": {"labels": [], "confidence": {}},  # sem rótulo → pulado
        "p3": {"labels": ["recomendacao"], "confidence": {}, "model": "m", "source": "s"},
    })

    assert n == 2  # p2 pulado
    por_ponto = {op.set_payload.points[0]: op.set_payload.payload for op in fake.ops}
    assert set(por_ponto) == {"p1", "p3"}
    assert por_ponto["p1"]["discourse_role"] == ["achado"]
    assert por_ponto["p1"]["discourse_role_conf"] == {"achado": 0.9}
    assert por_ponto["p1"]["discourse_role_source"] == svc.SOURCE  # default
    assert por_ponto["p3"]["discourse_role_model"] == "m"
    assert por_ponto["p3"]["discourse_role_source"] == "s"  # override respeitado


def test_push_vazio_nao_chama_qdrant(monkeypatch):
    import backend.indexing.index_chunks as idx

    fake = _FakeQdrant()
    monkeypatch.setattr(idx, "_get_indexer", lambda: (fake, None, None))
    assert idx.push_discourse_roles_to_qdrant("doc-x", {}) == 0
    assert fake.ops is None
