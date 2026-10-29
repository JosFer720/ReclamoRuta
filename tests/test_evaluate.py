"""Behavioral checks for held-out classification reporting."""

from dataclasses import dataclass, replace
import json

import pytest


@dataclass
class Prediction:
    label: str
    score: float
    top3: list[tuple[str, float]]
    model_version: str
    elapsed_ms: float


def prediction(label="a", score=1.0, elapsed_ms=10.0):
    return Prediction(label, score, [(label, score)], "v1", elapsed_ms)


def evaluate(*args, **kwargs):
    from reclamoruta.evaluate import evaluate_predictions

    return evaluate_predictions(*args, **kwargs)


def test_perfect_predictions_and_json_serialization():
    result = evaluate(["b", "a"], [prediction("b"), prediction("a")])
    assert result["accuracy"] == result["macro_f1"] == 1.0
    assert result["ece"] == 0.0
    assert result["model_version"] == "v1"
    assert result["n_examples"] == 2
    json.dumps(result, allow_nan=False)


def test_confusion_orientation_per_class_and_fixed_absent_classes():
    result = evaluate(
        ["a", "a", "b", "b"],
        [prediction("a"), prediction("b"), prediction("b"), prediction("b")],
        categories=["b", "a", "c"],
    )
    assert result["accuracy"] == 0.75
    assert result["macro_f1"] == pytest.approx((0.8 + 2 / 3) / 3)
    assert result["confusion_matrix"] == {
        "labels": ["b", "a", "c"],
        "matrix": [[2, 0, 0], [1, 1, 0], [0, 0, 0]],
    }
    assert result["per_class"]["a"] == {
        "precision": 1.0, "recall": 0.5, "f1": 2 / 3,
        "support": 2, "tp": 1, "fp": 0, "fn": 1,
    }
    assert result["per_class"]["c"]["f1"] == 0.0


def test_calibration_bins_include_zero_one_and_bin_edges():
    result = evaluate(
        ["a", "b", "a", "b"],
        [prediction("a", 0), prediction("a", 0.5),
         prediction("a", 1), prediction("b", 1)],
        n_bins=2,
    )
    assert result["ece"] == pytest.approx(0.375)
    assert [entry["count"] for entry in result["calibration_bins"]] == [1, 3]
    assert result["calibration_bins"][0]["accuracy"] == 1
    assert result["calibration_bins"][1]["confidence"] == pytest.approx(5 / 6)


def test_latency_uses_linear_interpolation_and_routes_can_merge_classes():
    result = evaluate(
        ["a", "b"], [prediction("b", elapsed_ms=10), prediction("a", elapsed_ms=30)],
        routes={"a": "same", "b": "same"},
    )
    assert result["latency_ms"] == {"p50": 20.0, "p95": 29.0}
    assert result["accuracy"] == 0.0
    assert result["routing_accuracy"] == 1.0


def test_explicit_categories_allow_predictions_absent_from_gold():
    result = evaluate(["a"], [prediction("b")], categories=["a", "b"])
    assert result["accuracy"] == result["macro_f1"] == 0.0


def test_route_metrics_merge_intents_and_measure_wrong_route_with_stable_order():
    result = evaluate(
        ["a", "b", "c"],
        [prediction("b"), prediction("c"), prediction("c")],
        categories=["c", "b", "a", "d"],
        routes={"c": "z", "b": "m", "a": "m", "d": "a"},
    )
    # a->b is an intent error but correctly routes to m; b->c misroutes.
    assert result["accuracy"] == pytest.approx(1 / 3)
    assert result["routing_accuracy"] == pytest.approx(2 / 3)
    routing = result["routing_metrics"]
    assert routing["accuracy"] == result["routing_accuracy"]
    assert routing["confusion_matrix"] == {
        "labels": ["a", "m", "z"],
        "matrix": [[0, 0, 0], [0, 1, 1], [0, 0, 1]],
    }
    assert routing["per_class"]["m"] == {
        "precision": 1.0, "recall": 0.5, "f1": 2 / 3,
        "support": 2, "tp": 1, "fp": 0, "fn": 1,
    }
    assert routing["per_class"]["z"] == {
        "precision": 0.5, "recall": 1.0, "f1": 2 / 3,
        "support": 1, "tp": 1, "fp": 1, "fn": 0,
    }
    assert routing["per_class"]["a"]["support"] == 0
    assert routing["macro_f1"] == pytest.approx(4 / 9)
    json.dumps(result, allow_nan=False)


def test_route_metrics_are_only_reported_when_routes_are_provided():
    assert "routing_metrics" not in evaluate(["a"], [prediction()])


@pytest.mark.parametrize("labels,predictions,kwargs", [
    ([], [], {}),
    (["a"], [], {}),
    (["a"], [prediction("b")], {}),
    (["a"], [prediction()], {"categories": ["b"]}),
    (["a"], [prediction()], {"categories": []}),
    (["a"], [prediction()], {"categories": ["a", "a"]}),
    ([""], [prediction()], {}),
    (["a"], [prediction()], {"n_bins": 0}),
    (["a"], [prediction()], {"n_bins": True}),
    (["a"], [prediction()], {"n_bins": 2.5}),
    (["a"], [prediction()], {"routes": {}}),
    (["a"], [prediction()], {"routes": {"a": ""}}),
    (["a"], [prediction()], {"categories": ["a", "b"], "routes": {"a": "x"}}),
])
def test_invalid_evaluation_inputs_are_rejected(labels, predictions, kwargs):
    with pytest.raises(ValueError):
        evaluate(labels, predictions, **kwargs)


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -0.1, 1.1, True])
def test_invalid_confidences_are_rejected(score):
    with pytest.raises(ValueError):
        evaluate(["a"], [prediction(score=score)])


@pytest.mark.parametrize("elapsed", [float("nan"), float("inf"), -1, True])
def test_invalid_latency_is_rejected(elapsed):
    with pytest.raises(ValueError):
        evaluate(["a"], [prediction(elapsed_ms=elapsed)])


@pytest.mark.parametrize("top3", [
    [("b", 0.9)], [("a", 0.8)], [("a", 0.9), ("a", 0.1)],
    [("a", 0.9), ("b", float("nan"))],
    [("a", 0.9), ("outside", 0.1)],
    [("a", 0.9), ("b", 1.0)],
])
def test_inconsistent_rankings_are_rejected(top3):
    with pytest.raises(ValueError):
        evaluate(["a"], [replace(prediction(score=0.9), top3=top3)], categories=["a", "b"])


def test_empty_optional_ranking_and_homogeneous_model_version():
    assert evaluate(["a"], [replace(prediction(), top3=[])])["accuracy"] == 1
    with pytest.raises(ValueError):
        evaluate(["a", "a"], [prediction(), replace(prediction(), model_version="v2")])
    with pytest.raises(ValueError):
        evaluate(["a"], [replace(prediction(), model_version="")])
