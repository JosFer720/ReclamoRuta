"""Verified artifact loading and prediction."""

import hashlib
import json
import pickle
import re
import time
from pathlib import Path

from reclamoruta.contracts import MANIFEST_NAME, ContractError, Prediction

_BATCH = 32


def _sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _hashes(artifact_dir: Path) -> dict:
    return {path.name: _sha256(path) for path in sorted(artifact_dir.iterdir())
            if path.name != MANIFEST_NAME}


def write_manifest(artifact_dir: Path, model_version: str, kind: str,
                   data_manifest_sha256: str) -> dict:
    """Call after writing other files."""
    if not isinstance(data_manifest_sha256, str) \
            or re.fullmatch(r"[0-9a-f]{64}", data_manifest_sha256) is None:
        raise ValueError("Falta el hash verificado del manifiesto de datos")
    manifest = {
        "schema_version": 1, "model_version": model_version, "kind": kind,
        "data_manifest_sha256": data_manifest_sha256,
        "files": _hashes(Path(artifact_dir)),
    }
    (Path(artifact_dir) / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def read_manifest(artifact_dir: Path) -> dict:
    """Reject missing, extra, modified files."""
    try:
        manifest = json.loads((artifact_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
        found = _hashes(artifact_dir)
    except (OSError, ValueError) as exc:
        raise ContractError(f"Artefacto incompleto: {MANIFEST_NAME} ausente o ilegible") from exc
    if not isinstance(manifest, dict) or not found or manifest.get("files") != found:
        raise ContractError("Artefacto corrupto: los archivos no coinciden con el manifiesto")
    return manifest


def _torch_scorer(artifact_dir: Path, device: str):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(artifact_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(artifact_dir, local_files_only=True)
    model.to(device).eval()
    max_length = json.loads((artifact_dir / "train_config.json").read_text(encoding="utf-8"))["max_length"]

    def probabilities(texts: list[str]) -> list[list[float]]:
        batch = tokenizer(texts, padding=True, truncation=True, max_length=max_length,
                          return_tensors="pt").to(device)
        with torch.no_grad():
            return torch.softmax(model(**batch).logits, dim=1).tolist()

    return probabilities


class Predictor:
    """Baseline and DistilBERT, same interface."""

    def __init__(self, model_version: str, categories: list[str], probabilities):
        self.model_version = model_version
        self.categories = categories
        self._probabilities = probabilities

    @classmethod
    def load(cls, artifact_dir: Path, device: str = "cpu") -> "Predictor":
        artifact_dir = Path(artifact_dir)
        manifest = read_manifest(artifact_dir)
        catalog = json.loads((artifact_dir / "label_to_id.json").read_text(encoding="utf-8"))
        categories = sorted(catalog, key=catalog.get)
        if manifest["kind"] == "baseline":
            # hash already verified
            with (artifact_dir / "model.pkl").open("rb") as stream:
                probabilities = pickle.load(stream).predict_proba
        else:
            probabilities = _torch_scorer(artifact_dir, device)
        return cls(manifest["model_version"], categories, probabilities)

    def predict(self, texts: list[str]) -> list[Prediction]:
        """Keeps input order."""
        if not texts:
            return []
        started = time.perf_counter()
        rows = [row for start in range(0, len(texts), _BATCH)
                for row in self._probabilities(texts[start:start + _BATCH])]
        elapsed_ms = (time.perf_counter() - started) * 1000 / len(texts)
        return [self._prediction(row, elapsed_ms) for row in rows]

    def _prediction(self, row, elapsed_ms: float) -> Prediction:
        best = sorted(range(len(row)), key=lambda index: -row[index])[:3]
        top3 = [(self.categories[index], min(1.0, float(row[index]))) for index in best]
        return Prediction(top3[0][0], top3[0][1], top3, self.model_version, elapsed_ms)
