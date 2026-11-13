"""Tiny offline fixtures."""

import csv
import json

import pytest

WORDS = {"card_arrival": ["card", "arrive", "delivery"],
         "Refund_not_showing_up": ["refund", "missing", "money"],
         "exchange_rate": ["rate", "exchange", "currency"]}
FILLER = ["please", "help", "now", "today", "thanks", "hello"]


def _rows(count: int, offset: int = 0) -> list[tuple[str, str]]:
    return [(f"{words[i % 3]} {words[(i + 1) % 3]} {FILLER[(i + offset) % 6]}", label)
            for label, words in WORDS.items() for i in range(count)]


def _write_split(path, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("id", "text", "category", "group_id", "split"))
        writer.writerows((f"{path.stem}:{i}", text, label, str(i), path.stem)
                         for i, (text, label) in enumerate(rows))


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):
    """Prepared splits without test files."""
    path = tmp_path_factory.mktemp("data")
    _write_split(path / "train.csv", _rows(8))
    _write_split(path / "validation_model.csv", _rows(3, offset=3))
    labels = {label: index for index, label in enumerate(sorted(WORDS))}
    (path / "label_to_id.json").write_text(json.dumps(labels), encoding="utf-8")
    (path / "manifest.json").write_text(json.dumps({"fixture": True}), encoding="utf-8")
    return path


@pytest.fixture(scope="session")
def tiny_base(tmp_path_factory):
    """Random one-layer DistilBERT, no download."""
    import torch
    from transformers import DistilBertConfig, DistilBertModel, DistilBertTokenizer

    path = tmp_path_factory.mktemp("base")
    tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *FILLER,
              *(word for words in WORDS.values() for word in words)]
    # vocab file works 4.x–5.x
    (path / "vocab.txt").write_text("\n".join(tokens) + "\n", encoding="utf-8")
    DistilBertTokenizer.from_pretrained(path).save_pretrained(path)
    torch.manual_seed(0)
    DistilBertModel(DistilBertConfig(
        vocab_size=len(tokens), dim=16, n_layers=1, n_heads=2, hidden_dim=32,
        max_position_embeddings=32)).save_pretrained(path)
    return path


@pytest.fixture(scope="session")
def make_config(tiny_base, tmp_path_factory):
    def make(**changes) -> "Path":
        config = {"model_version": "tiny-v1", "base_model": str(tiny_base), "revision": None,
                  "freeze_encoder": True, "lr": 0.01, "max_epochs": 2, "patience": 2,
                  "warmup_ratio": 0.1, "max_length": 16, "batch_size": 4, "accumulation": 2, "seed": 42, **changes}
        path = tmp_path_factory.mktemp("config") / "train.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        return path
    return make


@pytest.fixture(scope="session")
def baseline_artifact(data_dir, tmp_path_factory):
    from reclamoruta.baseline import train_baseline

    path = tmp_path_factory.mktemp("artifacts") / "baseline"
    train_baseline(data_dir, path)
    return path


@pytest.fixture(scope="session")
def tiny_artifact(data_dir, make_config, tmp_path_factory):
    from reclamoruta.train import train_model

    path = tmp_path_factory.mktemp("artifacts") / "tiny-v1"
    train_model(make_config(), data_dir, path)
    return path
