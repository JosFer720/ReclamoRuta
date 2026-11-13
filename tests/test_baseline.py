"""Baseline pipeline, metrics and artifact."""

import json
import pickle

import pytest

from reclamoruta import baseline


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


def test_reads_only_train_and_validation_model(data_dir, tmp_path, monkeypatch):
    seen = []
    original = baseline.read_split
    monkeypatch.setattr(baseline, "read_split",
                        lambda path, name, catalog: seen.append(name) or original(path, name, catalog))
    baseline.train_baseline(data_dir, tmp_path / "out")
    assert seen == ["train", "validation_model"]


def test_never_overwrites_artifact(data_dir, baseline_artifact):
    with pytest.raises(FileExistsError):
        baseline.train_baseline(data_dir, baseline_artifact)


def test_cli(data_dir, tmp_path, capsys):
    assert baseline.main(["--data", str(data_dir), "--out", str(tmp_path / "out")]) == 0
    assert json.loads(capsys.readouterr().out)["model_version"] == baseline.MODEL_VERSION
    assert baseline.main(["--data", str(tmp_path / "missing"), "--out", str(tmp_path / "x")]) == 2
