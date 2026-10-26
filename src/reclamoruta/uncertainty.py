"""Paired cluster bootstrap with deterministic local random state."""

from collections.abc import Iterator, Sequence
import math
import random

from reclamoruta.evaluate import (
    _finite_number,
    _label_universe,
    _metrics_from_matrix,
    _nonempty_string,
    _percentile,
)


def _group_rows(group_ids: list[str], n_rows: int, n_boot: int, seed: int) -> list[list[int]]:
    if not group_ids or len(group_ids) != n_rows:
        raise ValueError("group_ids and values must be nonempty and aligned")
    if isinstance(n_boot, bool) or not isinstance(n_boot, int) or n_boot <= 0:
        raise ValueError("n_boot must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    groups: dict[str, list[int]] = {}
    for index, group in enumerate(group_ids):
        _nonempty_string(group, "group id")
        groups.setdefault(group, []).append(index)
    return list(groups.values())


def _draw_rows(groups: list[list[int]], rng: random.Random) -> Iterator[int]:
    # Yield rows instead of allocating expanded samples (large duplicate groups).
    for _ in groups:
        yield from groups[rng.randrange(len(groups))]


def _summary(delta: float, replicates: list[float]) -> dict:
    replicates.sort()
    interval = [_percentile(replicates, 0.025), _percentile(replicates, 0.975)]
    if not math.isfinite(delta) or not all(math.isfinite(value) for value in interval):
        raise ValueError("metric difference exceeds finite numeric range")
    return {"delta": delta, "ci95": interval}


def _metadata(n_rows: int, n_groups: int, n_boot: int, seed: int) -> dict:
    return {
        "difference": "v2 - v1", "method": "paired_cluster_percentile",
        "confidence_level": 0.95, "percentile_method": "linear_type_7",
        "n_examples": n_rows, "n_groups": n_groups, "n_boot": n_boot, "seed": seed,
    }


def _mean_difference(first: Sequence[float], second: Sequence[float], rows: Iterator[int]) -> float:
    # Pair before summing to preserve small changes beside large common values.
    # Scale differences to avoid overflow while streaming whole clusters.
    scale = max(max(map(abs, first)), max(map(abs, second)), 1.0)
    count = 0

    def scaled_differences() -> Iterator[float]:
        nonlocal count
        for index in rows:
            count += 1
            difference = second[index] - first[index]
            if math.isfinite(difference):
                yield difference / scale
            else:
                yield second[index] / scale - first[index] / scale

    total = math.fsum(scaled_differences())
    delta = (total / count) * scale
    if not math.isfinite(delta):
        raise ValueError("metric difference exceeds finite numeric range")
    return delta


def paired_bootstrap(
    group_ids: list[str],
    values_v1: list[float],
    values_v2: list[float],
    n_boot: int = 1000,
    seed: int = 42,
) -> dict:
    """Bootstrap a pooled row-mean difference, always oriented v2 minus v1.

    Group IDs are sampled in first-occurrence order: draw the original number
    of distinct groups with replacement, including every row in each draw.
    Unequal group sizes therefore retain their pooled row weights. The same rows
    are used for both versions. CI endpoints are the 2.5 and 97.5 percentiles,
    linearly interpolated at (n_boot - 1) * quantile. Only one expanded sample is
    streamed at a time; O(n_boot) scalar differences are stored for percentiles.
    """
    if len(values_v1) != len(values_v2):
        raise ValueError("values_v1 and values_v2 must have equal lengths")
    groups = _group_rows(group_ids, len(values_v1), n_boot, seed)
    first = [_finite_number(value, "values_v1") for value in values_v1]
    second = [_finite_number(value, "values_v2") for value in values_v2]
    delta = _mean_difference(first, second, iter(range(len(first))))
    rng = random.Random(seed)
    replicates = [
        _mean_difference(first, second, _draw_rows(groups, rng)) for _ in range(n_boot)
    ]
    return {
        **_metadata(len(first), len(groups), n_boot, seed),
        **_summary(delta, replicates),
    }


def paired_classification_bootstrap(
    group_ids: list[str],
    labels: list[str],
    predicted_v1: list[str],
    predicted_v2: list[str],
    *,
    categories: Sequence[str] | None = None,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict:
    """Recompute accuracy and macro-F1 differences on paired cluster samples.

    The category universe is fixed before resampling (explicit catalog or sorted
    observed gold labels), and absent classes contribute zero F1. Macro-F1 is
    recomputed from confusion counts, never averaged from per-example F1 values.
    Each group must have one gold label, as duplicate normalized messages within
    a class represent a single cluster. Inputs contain labels only, never text.
    """
    if len(labels) != len(predicted_v1) or len(labels) != len(predicted_v2):
        raise ValueError("gold and predicted labels must have equal lengths")
    groups = _group_rows(group_ids, len(labels), n_boot, seed)
    universe = _label_universe(labels, categories)
    index = {label: position for position, label in enumerate(universe)}
    for prediction in [*predicted_v1, *predicted_v2]:
        _nonempty_string(prediction, "predicted label")
        if prediction not in index:
            raise ValueError("predicted label is outside categories")
    if any(len({labels[row] for row in group}) != 1 for group in groups):
        raise ValueError("each group must have a consistent gold label")

    def differences(rows: Iterator[int]) -> tuple[float, float]:
        matrix_first = [[0] * len(universe) for _ in universe]
        matrix_second = [[0] * len(universe) for _ in universe]
        for row in rows:
            gold = index[labels[row]]
            matrix_first[gold][index[predicted_v1[row]]] += 1
            matrix_second[gold][index[predicted_v2[row]]] += 1
        first = _metrics_from_matrix(matrix_first, universe)
        second = _metrics_from_matrix(matrix_second, universe)
        return second["accuracy"] - first["accuracy"], second["macro_f1"] - first["macro_f1"]

    delta_accuracy, delta_f1 = differences(iter(range(len(labels))))
    accuracy_replicates = []
    f1_replicates = []
    rng = random.Random(seed)
    for _ in range(n_boot):
        accuracy, f1 = differences(_draw_rows(groups, rng))
        accuracy_replicates.append(accuracy)
        f1_replicates.append(f1)
    return {
        **_metadata(len(labels), len(groups), n_boot, seed),
        "categories": universe,
        "accuracy": _summary(delta_accuracy, accuracy_replicates),
        "macro_f1": _summary(delta_f1, f1_replicates),
    }
