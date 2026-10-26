"""Pure classification metrics; no prediction text or artifacts are persisted."""

from collections.abc import Mapping, Sequence
import math
from numbers import Real
from typing import Protocol


class PredictionLike(Protocol):
    """Structural, read-only subset of the classifier prediction contract."""

    @property
    def label(self) -> str: ...

    @property
    def score(self) -> float: ...

    @property
    def top3(self) -> list[tuple[str, float]]: ...

    @property
    def model_version(self) -> str: ...

    @property
    def elapsed_ms(self) -> float: ...


def _nonempty_string(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _label_universe(labels: Sequence[str], categories: Sequence[str] | None) -> list[str]:
    if not labels:
        raise ValueError("labels must be nonempty")
    for label in labels:
        _nonempty_string(label, "label")
    if isinstance(categories, (str, bytes)):
        raise ValueError("categories must be a sequence of category names")
    universe = sorted(set(labels)) if categories is None else list(categories)
    if not universe:
        raise ValueError("categories must be nonempty")
    for category in universe:
        _nonempty_string(category, "category")
    if len(set(universe)) != len(universe):
        raise ValueError("categories must be unique")
    if any(label not in universe for label in labels):
        raise ValueError("gold label is outside categories")
    return universe


def _metrics_from_matrix(matrix: list[list[int]], categories: Sequence[str]) -> dict:
    """Rows are gold labels, columns predictions; zero denominators yield zero."""
    total = sum(sum(row) for row in matrix)
    per_class = {}
    for index, category in enumerate(categories):
        tp = matrix[index][index]
        support = sum(matrix[index])
        fp = sum(row[index] for row in matrix) - tp
        fn = support - tp
        per_class[category] = {
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / support if support else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "support": support, "tp": tp, "fp": fp, "fn": fn,
        }
    return {
        "accuracy": sum(matrix[i][i] for i in range(len(categories))) / total,
        "macro_f1": math.fsum(row["f1"] for row in per_class.values()) / len(categories),
        "per_class": per_class,
        "confusion_matrix": {"labels": list(categories), "matrix": matrix},
    }


def _percentile(ordered_values: Sequence[float], fraction: float) -> float:
    """Linear interpolation at (n - 1) * fraction (Hyndman-Fan type 7)."""
    position = (len(ordered_values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    weight = position - lower
    return ordered_values[lower] * (1 - weight) + ordered_values[upper] * weight


def evaluate_predictions(
    labels: list[str],
    predictions: list[PredictionLike],
    *,
    categories: Sequence[str] | None = None,
    routes: Mapping[str, str] | None = None,
    n_bins: int = 10,
) -> dict:
    """Return JSON-safe metrics for aligned, nonempty gold and prediction rows.

    Supply the full category catalog for a fixed evaluation universe. Otherwise
    the sorted observed gold labels define the universe; unseen predicted labels
    are rejected. Explicit category order is preserved, including absent classes
    whose F1 is zero in the macro average. Latency quantiles use linear type-7
    interpolation. Top-label ECE uses [low, high) bins, with 1 in the last bin.
    An optional routing map must cover every category, with nonempty route names.
    Empty top3 is accepted; present rankings must agree with label and score.
    """
    if len(labels) != len(predictions):
        raise ValueError("labels and predictions must have equal lengths")
    universe = _label_universe(labels, categories)
    if isinstance(n_bins, bool) or not isinstance(n_bins, int) or n_bins <= 0:
        raise ValueError("n_bins must be a positive integer")
    indices = {label: index for index, label in enumerate(universe)}
    if routes is not None:
        if set(routes) != set(universe):
            raise ValueError("routes must map exactly the evaluation categories")
        for route in routes.values():
            _nonempty_string(route, "route")

    matrix = [[0] * len(universe) for _ in universe]
    bin_counts = [0] * n_bins
    bin_correct = [0] * n_bins
    bin_confidence = [0.0] * n_bins
    latencies = []
    versions = set()
    routing_correct = 0
    for gold, prediction in zip(labels, predictions):
        _nonempty_string(prediction.label, "predicted label")
        if prediction.label not in indices:
            raise ValueError("predicted label is outside categories; supply a full catalog")
        score = _finite_number(prediction.score, "score")
        if not 0 <= score <= 1:
            raise ValueError("score must be between 0 and 1")
        elapsed = _finite_number(prediction.elapsed_ms, "elapsed_ms")
        if elapsed < 0:
            raise ValueError("elapsed_ms must be nonnegative")
        _nonempty_string(prediction.model_version, "model_version")
        versions.add(prediction.model_version)
        ranking = prediction.top3
        if not isinstance(ranking, (list, tuple)) or len(ranking) > 3:
            raise ValueError("top3 must contain at most three ranked pairs")
        seen = set()
        previous_score = 1.0
        for rank, pair in enumerate(ranking):
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError("top3 entries must be label-score pairs")
            category, confidence = pair
            _nonempty_string(category, "top3 label")
            confidence = _finite_number(confidence, "top3 score")
            if category not in indices or category in seen:
                raise ValueError("top3 contains unknown or duplicate labels")
            if not 0 <= confidence <= previous_score:
                raise ValueError("top3 scores must be probabilities in descending order")
            if rank == 0 and (category != prediction.label or confidence != score):
                raise ValueError("top3 first entry must equal the prediction label and score")
            seen.add(category)
            previous_score = confidence
        correct = gold == prediction.label
        matrix[indices[gold]][indices[prediction.label]] += 1
        bin_index = min(int(score * n_bins), n_bins - 1)
        bin_counts[bin_index] += 1
        bin_correct[bin_index] += correct
        bin_confidence[bin_index] += score
        latencies.append(elapsed)
        if routes is not None:
            routing_correct += routes[gold] == routes[prediction.label]
    if len(versions) != 1:
        raise ValueError("predictions must use a single model_version")

    bins = []
    for index, count in enumerate(bin_counts):
        bins.append({
            "lower": index / n_bins, "upper": (index + 1) / n_bins,
            "count": count,
            "accuracy": bin_correct[index] / count if count else 0.0,
            "confidence": bin_confidence[index] / count if count else 0.0,
        })
    latencies.sort()
    result = _metrics_from_matrix(matrix, universe)
    result.update({
        "n_examples": len(labels), "model_version": next(iter(versions)),
        "ece": math.fsum(
            row["count"] / len(labels) * abs(row["accuracy"] - row["confidence"])
            for row in bins
        ),
        "calibration_bins": bins,
        "latency_ms": {"p50": _percentile(latencies, 0.5), "p95": _percentile(latencies, 0.95)},
    })
    if routes is not None:
        result["routing_accuracy"] = routing_correct / len(labels)
    return result
