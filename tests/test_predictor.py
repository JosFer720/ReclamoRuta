"""Predictor loading and contract."""

import json
import math
import shutil

import pytest

from reclamoruta.contracts import (ContractError, Prediction, artifact_digest, bind_decision,
                                   route_message, validate_routes)
from reclamoruta.predictor import Predictor, write_manifest

TEXTS = ["card arrive please", "refund missing now", "rate exchange today"]


@pytest.fixture(params=["baseline", "distilbert"])
def artifact(request, baseline_artifact, tiny_artifact):
    return baseline_artifact if request.param == "baseline" else tiny_artifact


@pytest.fixture
def copy(artifact, tmp_path):
    return shutil.copytree(artifact, tmp_path / "copy")


@pytest.fixture
def bert_tokenizer_artifact(tiny_artifact, tmp_path):
    """DistilBERT artifact with tokenizer metadata produced by Transformers 5."""
    path = shutil.copytree(tiny_artifact, tmp_path / "bert-tokenizer")
    config_path = path / "tokenizer_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["tokenizer_class"] = "BertTokenizer"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    write_manifest(path, manifest["model_version"], manifest["kind"],
                   manifest["data_manifest_sha256"])
    return path


def test_predicts_distilbert_with_bert_tokenizer_metadata(bert_tokenizer_artifact):
    predictions = Predictor.load(bert_tokenizer_artifact).predict(TEXTS)
    assert len(predictions) == 3
    for prediction in predictions:
        assert prediction.model_version == "tiny-v1"
        assert len(prediction.top3) == 3
        assert sum(score for _, score in prediction.top3) == pytest.approx(1.0)


def test_training_encode_accepts_distilbert_with_bert_tokenizer_metadata(bert_tokenizer_artifact):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from reclamoruta.train import _encode

    tokenizer = AutoTokenizer.from_pretrained(bert_tokenizer_artifact, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        bert_tokenizer_artifact, local_files_only=True)
    batch = _encode(tokenizer, TEXTS, 16, "cpu")
    output = model(**batch, labels=torch.tensor([1, 0, 2]))
    assert output.logits.shape == (3, 3)
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert model.classifier.weight.grad is not None


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


def test_write_manifest_requires_data_hash(tmp_path):
    (tmp_path / "model.pkl").write_bytes(b"x")
    for digest in (None, "", "x" * 64):
        with pytest.raises(ValueError):
            write_manifest(tmp_path, "v", "baseline", digest)
    assert not (tmp_path / "manifest.json").exists()


# --- Calibration ----------------------------------------------------------

def test_uncalibrated_by_default(artifact):
    assert Predictor.load(artifact).calibration_sha256 is None


def test_temperature_rescales_scores(artifact):
    raw = Predictor.load(artifact)
    calibrated = raw.with_calibration(3.0, "c" * 64)
    assert calibrated.calibration_sha256 == "c" * 64
    assert calibrated.model_version == raw.model_version
    for before, after in zip(raw.predict(TEXTS), calibrated.predict(TEXTS)):
        # fixture has 3 classes
        logits = {label: math.log(score) / 3.0 for label, score in before.top3}
        total = sum(math.exp(value) for value in logits.values())
        expected = {label: math.exp(value) / total for label, value in logits.items()}
        assert after.label == before.label
        assert dict(after.top3) == pytest.approx(expected)
        assert after.score < before.score


def test_temperature_one_keeps_scores(artifact):
    raw = Predictor.load(artifact)
    same = raw.with_calibration(1.0, "c" * 64)
    for before, after in zip(raw.predict(TEXTS), same.predict(TEXTS)):
        assert dict(after.top3) == pytest.approx(dict(before.top3))


@pytest.mark.parametrize("temperature", [0, -1.0, math.nan, math.inf, True, "2"])
def test_invalid_temperature_is_rejected(baseline_artifact, temperature):
    with pytest.raises(ContractError):
        Predictor.load(baseline_artifact).with_calibration(temperature, "c" * 64)


def test_calibration_changes_the_review_decision(baseline_artifact):
    raw = Predictor.load(baseline_artifact)
    calibrated = raw.with_calibration(3.0, "c" * 64)
    raw_score = raw.predict(TEXTS[:1])[0].score
    calibrated_score = calibrated.predict(TEXTS[:1])[0].score
    threshold = round((raw_score + calibrated_score) / 2, 6)
    assert calibrated_score < threshold <= raw_score
    routes = validate_routes({"version": "t", "status": "t",
                              "routes": {"r": {"name": "R", "description": "d"}},
                              "intents": {label: "r" for label in raw.categories}}, raw.categories)
    digest = artifact_digest(baseline_artifact)

    def decision(calibration):
        return bind_decision({
            "status": "validated", "split_role": "validation_threshold",
            "model_version": raw.model_version, "artifact_sha256": digest,
            "threshold": threshold, "calibration": calibration, "assumptions_version": "s-v1",
        }, model_version=raw.model_version, artifact_sha256=digest)

    plain = route_message(TEXTS[0], raw, decision({"method": "none", "fitted": False}), routes)
    fitted = decision({"method": "temperature", "fitted": True,
                       "split_role": "validation_calibration", "artifact_sha256": "c" * 64})
    scaled = route_message(TEXTS[0], calibrated, fitted, routes)
    assert (plain.needs_review, plain.calibrated) == (False, False)
    assert (scaled.needs_review, scaled.calibrated) == (True, True)
    assert scaled.score == pytest.approx(calibrated_score)
