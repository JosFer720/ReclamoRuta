"""Train the TF-IDF baseline."""

import argparse
import csv
import hashlib
import io
import json
import pickle
import sys
import time
from pathlib import Path

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline

from reclamoruta.predictor import write_manifest

MODEL_VERSION = "baseline-tfidf"
TRAINING_FILES = ("train.csv", "validation_model.csv", "label_to_id.json")


class DataIntegrityError(ValueError):
    """Data does not match manifest."""


def load_training_data(data_dir: Path) -> dict:
    """Verify hashes, then parse."""
    data_dir = Path(data_dir)
    try:
        manifest = (data_dir / "manifest.json").read_bytes()
        expected = json.loads(manifest)["output_sha256"]
        files = {name: (data_dir / name).read_bytes() for name in TRAINING_FILES}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise DataIntegrityError("Manifiesto o archivos de datos ausentes o inválidos") from exc
    for name, content in files.items():
        if not isinstance(expected, dict) or hashlib.sha256(content).hexdigest() != expected.get(name):
            raise DataIntegrityError(f"{name} no coincide con el manifiesto de datos")
    catalog = json.loads(files["label_to_id.json"])
    return {
        "catalog": catalog,
        "train": _split(files["train.csv"], catalog),
        "validation_model": _split(files["validation_model.csv"], catalog),
        "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
    }


def _split(content: bytes, catalog: dict[str, int]) -> tuple[list[str], list[int]]:
    rows = list(csv.DictReader(io.StringIO(content.decode("utf-8"), newline="")))
    return [row["text"] for row in rows], [catalog[row["category"]] for row in rows]


def scores(labels: list[int], predictions: list[int]) -> dict:
    return {"accuracy": float(accuracy_score(labels, predictions)),
            "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0))}


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def train_baseline(data_dir: Path, output_dir: Path) -> dict:
    output_dir = Path(output_dir)
    data = load_training_data(data_dir)
    catalog = data["catalog"]
    train_texts, train_labels = data["train"]
    val_texts, val_labels = data["validation_model"]
    pipeline = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30000),
        OneVsRestClassifier(LogisticRegression(C=4, solver="liblinear", max_iter=300)),
    )
    started = time.perf_counter()
    pipeline.fit(train_texts, train_labels)
    summary = {
        "model_version": MODEL_VERSION,
        "fit_seconds": time.perf_counter() - started,
        "train": scores(train_labels, pipeline.predict(train_texts).tolist()),
        "validation_model": scores(val_labels, pipeline.predict(val_texts).tolist()),
        "n_train": len(train_texts), "n_validation_model": len(val_texts),
        "versions": {"scikit-learn": sklearn.__version__},
    }
    # never overwrite an artifact
    output_dir.mkdir(parents=True)
    with (output_dir / "model.pkl").open("wb") as stream:
        pickle.dump(pipeline, stream)
    write_json(output_dir / "label_to_id.json", catalog)
    write_json(output_dir / "metrics.json", summary)
    write_manifest(output_dir, MODEL_VERSION, "baseline", data["manifest_sha256"])
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        summary = train_baseline(args.data, args.out)
    except (OSError, KeyError, ValueError) as error:
        print(json.dumps({"error": type(error).__name__}), file=sys.stderr)
        return 2
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
