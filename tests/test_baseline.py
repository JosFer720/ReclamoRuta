"""Baseline pipeline, metrics and artifact."""

import hashlib
import json
import pickle
import shutil
from pathlib import Path

import pytest

from reclamoruta import baseline

FILES = ["train.csv", "validation_model.csv", "label_to_id.json"]


def test_pipeline_matches_specification(baseline_artifact):
    with (baseline_artifact / "model.pkl").open("rb") as stream:
        tfidf, classifier = pickle.load(stream).steps
    assert tfidf[1].ngram_range == (1, 2)
    assert (tfidf[1].min_df, tfidf[1].max_features) == (2, 30000)
    logistic = classifier[1].estimator
    assert (logistic.C, logistic.solver, logistic.max_iter) == (4, "liblinear", 300)


def test_artifact_files_and_manifest(baseline_artifact):
    names = {path.name for path in baseline_artifact.iterdir()}
    assert names == {"model.pkl", "label_to_id.json", "metrics.json", "manifest.json"}
    manifest = json.loads((baseline_artifact / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["kind"] == "baseline"
    assert manifest["model_version"] == baseline.MODEL_VERSION
    assert set(manifest["files"]) == names - {"manifest.json"}


def test_metrics_on_separable_fixture(baseline_artifact):
    metrics = json.loads((baseline_artifact / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["n_train"] == 24 and metrics["n_validation_model"] == 9
    assert metrics["validation_model"] == {"accuracy": 1.0, "macro_f1": 1.0}


def test_reads_only_training_files(data_dir, tmp_path, monkeypatch):
    seen = []
    original = Path.read_bytes

    def spy(path):
        if path.parent == data_dir:
            seen.append(path.name)
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", spy)
    baseline.train_baseline(data_dir, tmp_path / "out")
    assert set(seen) == {*FILES, "manifest.json"}


def test_artifact_records_verified_data_manifest(data_dir, baseline_artifact):
    manifest = json.loads((baseline_artifact / "manifest.json").read_text(encoding="utf-8"))
    expected = hashlib.sha256((data_dir / "manifest.json").read_bytes()).hexdigest()
    assert manifest["data_manifest_sha256"] == expected


@pytest.mark.parametrize("name", FILES)
def test_modified_data_fails_before_training(tampered, tmp_path, monkeypatch, name):
    calls = []
    monkeypatch.setattr(baseline, "make_pipeline", lambda *steps: calls.append(steps))
    with pytest.raises(baseline.DataIntegrityError, match=name):
        baseline.train_baseline(tampered(name), tmp_path / "out")
    assert calls == []
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("manifest", [None, "{", "[]", '{"output_sha256": null}',
                                      '{"output_sha256": {}}'])
def test_missing_or_invalid_data_manifest_is_rejected(data_dir, tmp_path, manifest):
    data = shutil.copytree(data_dir, tmp_path / "data")
    if manifest is None:
        (data / "manifest.json").unlink()
    else:
        (data / "manifest.json").write_text(manifest, encoding="utf-8")
    with pytest.raises(baseline.DataIntegrityError):
        baseline.train_baseline(data, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_never_overwrites_artifact(data_dir, baseline_artifact):
    with pytest.raises(FileExistsError):
        baseline.train_baseline(data_dir, baseline_artifact)


def test_cli(data_dir, tmp_path, capsys):
    assert baseline.main(["--data", str(data_dir), "--out", str(tmp_path / "out")]) == 0
    assert json.loads(capsys.readouterr().out)["model_version"] == baseline.MODEL_VERSION
    assert baseline.main(["--data", str(tmp_path / "missing"), "--out", str(tmp_path / "x")]) == 2
    assert not (tmp_path / "x").exists()
