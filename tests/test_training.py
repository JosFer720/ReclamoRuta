"""Tiny-model training checks."""

import json
import math

import pytest
import torch
from transformers import AutoModel, AutoModelForSequenceClassification

from reclamoruta import train


def load_config(path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def encoder_changed(artifact, tiny_base) -> bool:
    before = AutoModel.from_pretrained(tiny_base).state_dict()
    after = AutoModelForSequenceClassification.from_pretrained(artifact).base_model.state_dict()
    return any(not torch.equal(before[name], after[name]) for name in before)


def test_head_has_77_outputs(make_config):
    model = train.build_model(load_config(make_config()), 77)
    assert model.classifier.out_features == 77


def test_v1_optimizer_gets_only_the_head(make_config):
    model = train.build_model(load_config(make_config(freeze_encoder=True)), 3)
    assert not any(p.requires_grad for p in model.base_model.parameters())
    optimized = {id(p) for group in train.build_optimizer(model, 0.01).param_groups
                 for p in group["params"]}
    head = [*model.pre_classifier.parameters(), *model.classifier.parameters()]
    assert optimized == {id(p) for p in head}


def test_v2_optimizer_gets_every_parameter(make_config):
    model = train.build_model(load_config(make_config(freeze_encoder=False)), 3)
    optimized = sum(len(group["params"]) for group in train.build_optimizer(model, 0.01).param_groups)
    assert optimized == len(list(model.parameters()))


def test_v1_updates_head_and_keeps_encoder(data_dir, make_config, tiny_base, tmp_path,
                                          monkeypatch):
    initial = {}
    original = train.build_model

    def spy(config, num_labels):
        model = original(config, num_labels)
        initial.update({name: value.clone() for name, value in model.state_dict().items()
                        if not name.startswith("distilbert.")})
        return model

    monkeypatch.setattr(train, "build_model", spy)
    summary = train.train_model(make_config(), data_dir, tmp_path / "out")
    trained = AutoModelForSequenceClassification.from_pretrained(tmp_path / "out").state_dict()
    assert initial and any(not torch.equal(initial[name], trained[name]) for name in initial)
    assert not encoder_changed(tmp_path / "out", tiny_base)
    assert all(math.isfinite(row["train_loss"]) and math.isfinite(row["val_loss"])
               for row in summary["curves"])


def test_v2_updates_encoder(data_dir, make_config, tiny_base, tmp_path):
    train.train_model(make_config(freeze_encoder=False), data_dir, tmp_path / "out")
    assert encoder_changed(tmp_path / "out", tiny_base)


def test_early_stopping_keeps_best_epoch(data_dir, make_config, tmp_path):
    # zero lr never improves
    config = make_config(lr=0.0, max_epochs=5, patience=2)
    summary = train.train_model(config, data_dir, tmp_path / "out")
    assert summary["best"]["epoch"] == 1
    assert len(summary["curves"]) == 3


def test_same_seed_same_curves(data_dir, make_config, tmp_path):
    runs = [train.train_model(make_config(), data_dir, tmp_path / name) for name in ("a", "b")]
    losses = [[row["train_loss"] for row in run["curves"]] for run in runs]
    assert losses[0] == losses[1]


def test_reads_only_train_and_validation_model(data_dir, make_config, tmp_path, monkeypatch):
    seen = []
    original = train.read_split
    monkeypatch.setattr(train, "read_split",
                        lambda path, name, catalog: seen.append(name) or original(path, name, catalog))
    train.train_model(make_config(), data_dir, tmp_path / "out")
    assert seen == ["train", "validation_model"]


def test_artifact_files_and_summary(tiny_artifact):
    names = {path.name for path in tiny_artifact.iterdir()}
    assert {"model.safetensors", "config.json", "tokenizer.json", "label_to_id.json",
            "train_config.json", "metrics.json", "manifest.json"} <= names
    manifest = json.loads((tiny_artifact / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["kind"] == "distilbert" and manifest["model_version"] == "tiny-v1"
    assert set(manifest["files"]) == names - {"manifest.json"}
    metrics = json.loads((tiny_artifact / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["truncated_texts"] == 0 and metrics["max_tokens"] == 5
    assert metrics["precision"] == "fp32"


def test_never_overwrites_artifact(data_dir, make_config, tiny_artifact):
    with pytest.raises(FileExistsError):
        train.train_model(make_config(), data_dir, tiny_artifact)


def test_cli(data_dir, make_config, tmp_path, capsys):
    args = ["--config", str(make_config()), "--data", str(data_dir)]
    assert train.main([*args, "--out", str(tmp_path / "out")]) == 0
    assert json.loads(capsys.readouterr().out)["model_version"] == "tiny-v1"
    assert train.main([*args, "--out", str(tmp_path / "out")]) == 2
