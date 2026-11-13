"""Predictor loading and contract."""

import shutil

import pytest

from reclamoruta.contracts import ContractError, Prediction, artifact_digest
from reclamoruta.predictor import Predictor

TEXTS = ["card arrive please", "refund missing now", "rate exchange today"]


@pytest.fixture(params=["baseline", "distilbert"])
def artifact(request, baseline_artifact, tiny_artifact):
    return baseline_artifact if request.param == "baseline" else tiny_artifact


@pytest.fixture
def copy(artifact, tmp_path):
    return shutil.copytree(artifact, tmp_path / "copy")


def test_loads_version_and_catalog(artifact):
    predictor = Predictor.load(artifact, device="cpu")
    assert predictor.model_version in ("baseline-tfidf", "tiny-v1")
    assert predictor.categories == ["Refund_not_showing_up", "card_arrival", "exchange_rate"]
    assert len(artifact_digest(artifact)) == 64


def test_empty_batch(artifact):
    assert Predictor.load(artifact).predict([]) == []


def test_predictions_follow_the_contract(artifact):
    predictor = Predictor.load(artifact)
    predictions = predictor.predict(TEXTS)
    assert len(predictions) == 3
    for prediction in predictions:
        assert isinstance(prediction, Prediction)
        assert prediction.model_version == predictor.model_version
        assert prediction.top3[0] == (prediction.label, prediction.score)
        assert len(prediction.top3) == 3
        assert sum(score for _, score in prediction.top3) == pytest.approx(1.0)


def test_baseline_predicts_the_fixture(baseline_artifact):
    labels = [p.label for p in Predictor.load(baseline_artifact).predict(TEXTS)]
    assert labels == ["card_arrival", "Refund_not_showing_up", "exchange_rate"]


def test_order_is_kept_across_batches(artifact):
    predictor = Predictor.load(artifact)
    texts = [TEXTS[i % 3] + " hello" * (i % 4) for i in range(70)]
    together = predictor.predict(texts)
    for text, prediction in zip(texts, together):
        alone = predictor.predict([text])[0]
        assert prediction.label == alone.label
        assert prediction.score == pytest.approx(alone.score, abs=1e-5)


def test_rejects_modified_file(copy):
    with (copy / "label_to_id.json").open("a", encoding="utf-8") as stream:
        stream.write(" ")
    with pytest.raises(ContractError):
        Predictor.load(copy)


def test_rejects_missing_file(copy):
    (copy / "metrics.json").unlink()
    with pytest.raises(ContractError):
        Predictor.load(copy)


def test_rejects_extra_file(copy):
    (copy / "extra.bin").write_bytes(b"x")
    with pytest.raises(ContractError):
        Predictor.load(copy)


def test_rejects_missing_or_broken_manifest(copy, tmp_path):
    (copy / "manifest.json").write_text("{", encoding="utf-8")
    with pytest.raises(ContractError):
        Predictor.load(copy)
    with pytest.raises(ContractError):
        Predictor.load(tmp_path / "nothing")
