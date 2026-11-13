"""Train the TF-IDF baseline."""

import argparse
import csv
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


def load_catalog(data_dir: Path) -> dict[str, int]:
    return json.loads((Path(data_dir) / "label_to_id.json").read_text(encoding="utf-8"))


def read_split(data_dir: Path, name: str, catalog: dict[str, int]) -> tuple[list[str], list[int]]:
    """Texts and label ids."""
    with (Path(data_dir) / f"{name}.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return [row["text"] for row in rows], [catalog[row["category"]] for row in rows]


def scores(labels: list[int], predictions: list[int]) -> dict:
    return {"accuracy": float(accuracy_score(labels, predictions)),
            "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0))}


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def train_baseline(data_dir: Path, output_dir: Path) -> dict:
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    catalog = load_catalog(data_dir)
    train_texts, train_labels = read_split(data_dir, "train", catalog)
    val_texts, val_labels = read_split(data_dir, "validation_model", catalog)
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
    write_manifest(output_dir, MODEL_VERSION, "baseline", data_dir)
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
