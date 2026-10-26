"""Cluster-aware paired uncertainty must preserve duplicate groups."""

import json
import random

import pytest


def bootstrap(*args, **kwargs):
    from reclamoruta.uncertainty import paired_bootstrap

    return paired_bootstrap(*args, **kwargs)


def classification(*args, **kwargs):
    from reclamoruta.uncertainty import paired_classification_bootstrap

    return paired_classification_bootstrap(*args, **kwargs)


def test_identical_values_have_exact_zero_interval_and_json_output():
    result = bootstrap(["x", "x", "y"], [1, 0, 1], [1, 0, 1])
    assert result["delta"] == 0
    assert result["ci95"] == [0.0, 0.0]
    assert result["difference"] == "v2 - v1"
    assert result["n_groups"] == 2
    assert result["n_boot"] == 1000
    json.dumps(result, allow_nan=False)


def test_local_seed_is_reproducible_and_does_not_change_global_rng():
    random.seed(31)
    before = random.getstate()
    result = bootstrap(["x", "y", "z"], [0, 1, 0], [1, 0, 1], n_boot=101, seed=7)
    assert random.getstate() == before
    assert result == bootstrap(["x", "y", "z"], [0, 1, 0], [1, 0, 1], n_boot=101, seed=7)
    assert result["delta"] == pytest.approx(1 / 3)


def test_resampling_complete_unequal_clusters_uses_row_weighted_mean():
    # Only two possible constant-group samples give endpoints 0 and 1.
    # With seed=4 and three replicates the draws are (x,y), (x,y), (y,x),
    # all yielding pooled mean 1/3; independent rows or group means differ.
    result = bootstrap(["x", "x", "y"], [0, 0, 0], [0, 0, 1], n_boot=3, seed=4)
    assert result["delta"] == pytest.approx(1 / 3)
    assert result["ci95"] == pytest.approx([1 / 3, 1 / 3])


def test_positive_improvement_has_positive_interval():
    result = bootstrap(["x", "y"], [0, 0], [1, 1], n_boot=10)
    assert result["delta"] == 1
    assert result["ci95"] == [1, 1]


def test_paired_difference_preserves_small_changes_beside_large_common_values():
    result = bootstrap(["x", "x"], [1e16, 1], [1e16, 2], n_boot=3)
    assert result["delta"] == 0.5
    assert result["ci95"] == [0.5, 0.5]


def test_large_finite_values_produce_finite_output_or_clear_validation_error():
    result = bootstrap(["x", "x"], [1e308, -1e308], [1e308, -1e308], n_boot=3)
    assert result["delta"] == 0
    json.dumps(result, allow_nan=False)
    with pytest.raises(ValueError, match="finite numeric range"):
        bootstrap(["x"], [-1e308], [1e308], n_boot=3)


@pytest.mark.parametrize("groups,first,second,kwargs", [
    ([], [], [], {}),
    (["x"], [0], [], {}),
    ([""], [0], [1], {}),
    ([None], [0], [1], {}),
    (["x"], [float("nan")], [1], {}),
    (["x"], [0], [float("inf")], {}),
    (["x"], [True], [1], {}),
    (["x"], [0], [1], {"n_boot": 0}),
    (["x"], [0], [1], {"n_boot": True}),
    (["x"], [0], [1], {"n_boot": 1.5}),
    (["x"], [0], [1], {"seed": True}),
    (["x"], [0], [1], {"seed": 1.5}),
])
def test_invalid_paired_inputs_are_rejected(groups, first, second, kwargs):
    with pytest.raises(ValueError):
        bootstrap(groups, first, second, **kwargs)


def test_classification_recomputes_macro_f1_over_fixed_categories():
    result = classification(
        ["x", "x", "y"], ["a", "a", "b"],
        ["a", "a", "a"], ["a", "a", "b"], n_boot=3, seed=4,
    )
    assert result["accuracy"]["delta"] == pytest.approx(1 / 3)
    assert result["macro_f1"]["delta"] == pytest.approx(0.6)
    assert result["macro_f1"]["ci95"] == pytest.approx([0.6, 0.6])


def test_classification_identical_predictions_have_zero_intervals():
    result = classification(["x", "y"], ["a", "b"], ["a", "b"], ["a", "b"])
    assert result["accuracy"]["ci95"] == [0, 0]
    assert result["macro_f1"]["ci95"] == [0, 0]
    assert result["categories"] == ["a", "b"]
    json.dumps(result, allow_nan=False)


def test_classification_absent_categories_remain_in_resample_macro():
    result = classification(
        ["x", "y"], ["a", "a"], ["b", "b"], ["a", "a"],
        categories=["a", "b", "c"], n_boot=4,
    )
    assert result["macro_f1"]["delta"] == pytest.approx(1 / 3)
    assert result["macro_f1"]["ci95"] == pytest.approx([1 / 3, 1 / 3])


@pytest.mark.parametrize("groups,labels,first,second,kwargs", [
    ([], [], [], [], {}),
    (["x"], ["a"], [], ["a"], {}),
    (["x"], ["a"], ["z"], ["a"], {}),
    (["x"], ["a"], ["a"], ["a"], {"categories": ["b"]}),
    (["x", "x"], ["a", "b"], ["a", "b"], ["a", "b"], {}),
])
def test_invalid_classification_inputs_are_rejected(groups, labels, first, second, kwargs):
    with pytest.raises(ValueError):
        classification(groups, labels, first, second, **kwargs)
