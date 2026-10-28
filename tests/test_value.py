"""Hand-calculated economic examples; no fitted models or test predictions."""

import importlib
import json
from pathlib import Path

import pytest


def value_module():
    # This assertion gives the initial RED an explicit missing-feature failure.
    assert importlib.util.find_spec("reclamoruta.value") is not None, "economic module missing"
    return importlib.import_module("reclamoruta.value")


@pytest.fixture
def assumptions():
    return {
        "t_manual": 30, "t_aceptar": 5, "t_corregir": 60, "t_revision": 30,
        "costo_hora": 36, "q_detectado": 1,
        "costo_error_ruta_no_detectado": 100, "costo_inferencia": 2,
        "costo_operacion": 3,
    }


def test_exact_time_fixture_and_net_value(assumptions):
    result = value_module().compute_value(80, 5, 5, 15, assumptions)
    assert result["N"] == 100
    assert result["T_base"] == 3000
    assert result["T_assist"] == 1175
    assert result["ahorro_horas"] == pytest.approx(1825 / 3600)
    assert result["valor_neto"] == pytest.approx(13.25)
    assert result["coverage"] == 0.85
    assert result["selective_error"] == pytest.approx(5 / 85)


def test_no_detection_retains_residual_route_risk(assumptions):
    assumptions["q_detectado"] = 0
    result = value_module().compute_value(80, 5, 3, 15, assumptions)
    assert result["T_assist"] == 875
    assert result["riesgo_residual"] == 300
    assert result["valor_neto"] == pytest.approx(-283.75)


def test_empty_and_all_review_are_defined_without_fake_zero_error(assumptions):
    empty = value_module().compute_value(0, 0, 0, 0, assumptions)
    assert empty["coverage"] == 0
    assert empty["selective_error"] is None
    assert empty["valor_neto"] == -5
    review = value_module().compute_value(0, 0, 0, 2, assumptions)
    assert review["T_base"] == review["T_assist"] == 60
    assert review["selective_error"] is None


def test_negative_time_savings_are_not_clipped(assumptions):
    assumptions["t_aceptar"] = 100
    result = value_module().compute_value(1, 0, 0, 0, assumptions)
    assert result["ahorro_horas"] == pytest.approx(-70 / 3600)
    assert result["valor_neto"] == pytest.approx(-5.7)


@pytest.mark.parametrize("bad", [True, False, -1, 1.2, "1", float("nan"), float("inf")])
@pytest.mark.parametrize("index", range(4))
def test_counts_are_nonnegative_integers_not_bools(assumptions, bad, index):
    counts = [2, 1, 1, 1]
    counts[index] = bad
    with pytest.raises(ValueError):
        value_module().compute_value(*counts, assumptions)


def test_wrong_route_count_cannot_exceed_wrong_intentions(assumptions):
    with pytest.raises(ValueError):
        value_module().compute_value(2, 1, 2, 0, assumptions)


@pytest.mark.parametrize("field", [
    "t_manual", "t_aceptar", "t_corregir", "t_revision", "costo_hora",
    "costo_error_ruta_no_detectado", "costo_inferencia", "costo_operacion", "q_detectado",
])
@pytest.mark.parametrize("bad", [True, "0", None, -1, float("nan"), float("inf")])
def test_assumptions_fail_closed_on_invalid_numbers(assumptions, field, bad):
    assumptions[field] = bad
    with pytest.raises(ValueError):
        value_module().compute_value(2, 1, 1, 0, assumptions)


@pytest.mark.parametrize("field", [
    "t_manual", "t_aceptar", "t_corregir", "t_revision", "costo_hora", "q_detectado",
    "costo_error_ruta_no_detectado", "costo_inferencia", "costo_operacion",
])
def test_missing_assumptions_are_not_silent_zero(assumptions, field):
    del assumptions[field]
    with pytest.raises(ValueError):
        value_module().compute_value(1, 0, 0, 0, assumptions)


def test_detection_probability_cannot_exceed_one(assumptions):
    assumptions["q_detectado"] = 1.01
    with pytest.raises(ValueError):
        value_module().compute_value(1, 0, 0, 0, assumptions)


@pytest.fixture
def value_config(assumptions):
    return {
        "status": "validated", "assumptions_version": "fixture-v1",
        "units": {"time": "seconds", "currency": "GTQ", "cost_scope": "batch"},
        "assumptions": assumptions,
        "evidence": {name: {"source": "synthetic test fixture", "estimated": True}
                     for name in assumptions},
    }


def test_curve_distinguishes_intention_errors_from_route_errors(assumptions):
    assumptions["q_detectado"] = 0
    curve = value_module().threshold_curve(
        ["a", "a", "b"], [0.8, 0.8, 0.2], ["b", "c", "b"],
        {"a": "team1", "b": "team1", "c": "team2"}, assumptions,
        thresholds=[0, 0.8, 0.9],
    )
    assert [row["threshold"] for row in curve] == [0, 0.8, 0.9]
    assert curve[0]["accepted_correct"] == 1
    assert curve[1]["accepted_wrong"] == 2
    assert curve[1]["accepted_wrong_route"] == 1
    assert curve[1]["reviewed"] == 1
    assert curve[1]["riesgo_residual"] == 100
    assert curve[2]["coverage"] == 0
    assert curve[2]["selective_error"] is None


def test_curve_default_grid_includes_no_abstention(assumptions):
    curve = value_module().threshold_curve([], [], [], {"a": "team"}, assumptions)
    assert len(curve) == 21
    assert curve[0]["threshold"] == 0
    assert curve[-1]["threshold"] == 1
    assert all(row["N"] == 0 for row in curve)


@pytest.mark.parametrize("role", ["train", "test", "validation_model", "calibration", None])
def test_threshold_search_refuses_other_split_roles(assumptions, role):
    with pytest.raises(ValueError):
        value_module().threshold_curve([], [], [], {"a": "team"}, assumptions, split_role=role)


@pytest.mark.parametrize("labels,scores,predictions,routes,thresholds", [
    (["a"], [], ["a"], {"a": "team"}, [0.5]),
    (["a"], [True], ["a"], {"a": "team"}, [0.5]),
    (["a"], [float("nan")], ["a"], {"a": "team"}, [0.5]),
    (["a"], [float("inf")], ["a"], {"a": "team"}, [0.5]),
    (["a"], [1.1], ["a"], {"a": "team"}, [0.5]),
    (["a"], [-0.1], ["a"], {"a": "team"}, [0.5]),
    (["a"], [0.8], ["unknown"], {"a": "team"}, [0.5]),
    (["unknown"], [0.8], ["a"], {"a": "team"}, [0.5]),
    (["a"], [0.8], ["a"], {"a": ""}, [0.5]),
    (["a"], [0.8], ["a"], {"a": "team", "unused": None}, [0.5]),
    ("a", [0.8], ["a"], {"a": "team"}, [0.5]),
    (["a"], [0.8], ["a"], {"a": "team"}, []),
    (["a"], [0.8], ["a"], {"a": "team"}, [True]),
    (["a"], [0.8], ["a"], {"a": "team"}, [float("nan")]),
])
def test_curve_validates_all_inputs(assumptions, labels, scores, predictions, routes, thresholds):
    with pytest.raises(ValueError):
        value_module().threshold_curve(labels, scores, predictions, routes, assumptions,
                                       thresholds=thresholds)


def test_threshold_selection_binds_artifact_and_compares_no_abstention(value_config):
    result = value_module().select_threshold(
        ["a", "a"], [0.9, 0.2], ["a", "b"], {"a": "x", "b": "y"}, value_config,
        model_version="fixture-model", artifact_sha256="a" * 64,
        calibration={"method": "none", "fitted": False}, thresholds=[0.5, 1.0],
    )
    assert result["status"] == "validated"
    assert result["threshold"] == 0.5
    assert result["model_version"] == "fixture-model"
    assert result["artifact_sha256"] == "a" * 64
    assert result["assumptions_version"] == "fixture-v1"
    assert result["no_abstention"]["accepted"] == 2
    assert result["no_abstention"]["threshold"] == 0
    assert result["selected"]["accepted"] == 1


def test_threshold_ties_choose_highest_threshold_conservatively(value_config):
    result = value_module().select_threshold(
        [], [], [], {"a": "x"}, value_config, model_version="fixture-model",
        artifact_sha256="a" * 64, calibration={"method": "none", "fitted": False},
    )
    assert result["threshold"] == 1
    assert result["tie_break"] == "highest_threshold"


def test_pending_templates_cannot_execute():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/value.json").read_text())
    decision = json.loads((root / "configs/decision.json").read_text())
    with pytest.raises(ValueError):
        value_module().validate_value_config(config)
    with pytest.raises(ValueError):
        value_module().validate_decision(decision)
    with pytest.raises(ValueError):
        value_module().select_threshold([], [], [], {"a": "x"}, config,
            model_version="fixture", artifact_sha256="a" * 64,
            calibration={"method": "none", "fitted": False})


@pytest.mark.parametrize("field,bad", [
    ("status", "pending"), ("assumptions_version", None),
    ("units", {"time": "minutes", "currency": "GTQ", "cost_scope": "batch"}),
    ("units", {"time": "seconds", "currency": "", "cost_scope": "batch"}),
    ("evidence", {}),
])
def test_value_config_requires_validation_units_and_provenance(value_config, field, bad):
    value_config[field] = bad
    with pytest.raises(ValueError):
        value_module().validate_value_config(value_config)


@pytest.mark.parametrize("model,sha,calibration", [
    (None, "a" * 64, {"method": "none", "fitted": False}),
    ("model", None, {"method": "none", "fitted": False}),
    ("model", "not-a-sha", {"method": "none", "fitted": False}),
    ("model", "a" * 64, None),
    ("model", "a" * 64, {"method": "none", "fitted": True}),
    ("model", "a" * 64, {"method": "temperature", "fitted": False}),
])
def test_selection_requires_real_binding_not_placeholder(value_config, model, sha, calibration):
    with pytest.raises(ValueError):
        value_module().select_threshold([], [], [], {"a": "x"}, value_config,
            model_version=model, artifact_sha256=sha, calibration=calibration)


def test_sensitivity_one_at_a_time_and_fractional_volume(assumptions):
    counts = {"accepted_correct": 80, "accepted_wrong": 5,
              "accepted_wrong_route": 5, "reviewed": 15}
    scenarios = value_module().sensitivity_analysis(counts, assumptions)
    assert len(scenarios) == 24
    volume = {row["relative_change"]: row for row in scenarios if row["parameter"] == "volume"}
    assert volume[-0.5]["projected_N"] == 50.0
    assert volume[-0.5]["accepted_wrong"] == 2.5
    assert volume[-0.5]["T_base"] == 1500
    assert volume[-0.5]["costo_inferencia"] == 1
    assert volume[-0.5]["costo_operacion"] == 3
    assert volume[-0.5]["valor_neto"] == pytest.approx(5.125)
    assert volume[0.5]["valor_neto"] == pytest.approx(21.375)
    manual = next(row for row in scenarios
                  if row["parameter"] == "t_manual" and row["relative_change"] == 0.2)
    assert manual["T_base"] == 3600
    assert manual["T_assist"] == 1175


def test_volume_projection_scales_residual_route_risk(assumptions):
    assumptions["q_detectado"] = 0
    counts = {"accepted_correct": 1, "accepted_wrong": 1,
              "accepted_wrong_route": 1, "reviewed": 0}
    scenarios = value_module().sensitivity_analysis(counts, assumptions)
    half = next(row for row in scenarios
                if row["parameter"] == "volume" and row["relative_change"] == -0.5)
    assert half["riesgo_residual"] == 50
    assert half["projected_N"] == 1


@pytest.mark.parametrize("parameter", ["t_manual", "t_aceptar", "t_corregir", "t_revision", "costo_hora"])
def test_relative_sensitivity_requires_positive_baseline(assumptions, parameter):
    assumptions[parameter] = 0
    with pytest.raises(ValueError):
        value_module().sensitivity_analysis(dict(zip(
            ("accepted_correct", "accepted_wrong", "accepted_wrong_route", "reviewed"),
            (1, 0, 0, 0), strict=True)), assumptions)


def test_volume_sensitivity_refuses_empty_batch(assumptions):
    with pytest.raises(ValueError):
        value_module().sensitivity_analysis(
            {"accepted_correct": 0, "accepted_wrong": 0, "accepted_wrong_route": 0, "reviewed": 0},
            assumptions)


def test_curve_accepts_structural_indexed_sequences(assumptions):
    class IndexedSequence:
        def __init__(self, values):
            self.values = values

        def __len__(self):
            return len(self.values)

        def __getitem__(self, index):
            return self.values[index]

    result = value_module().threshold_curve(
        IndexedSequence(["a"]), IndexedSequence([0.5]), IndexedSequence(["a"]),
        {"a": "team"}, assumptions, thresholds=IndexedSequence([0.5]),
    )
    assert result[0]["accepted_correct"] == 1


@pytest.mark.parametrize("role", [None, "test", "train", "validation_model"])
def test_persisted_decision_refuses_missing_or_wrong_selection_role(role):
    decision = {
        "status": "validated", "model_version": "fixture", "artifact_sha256": "b" * 64,
        "threshold": 0.5, "calibration": {"method": "none", "fitted": False},
        "assumptions_version": "fixture-v1", "split_role": role,
    }
    with pytest.raises(ValueError):
        value_module().validate_decision(decision)


def test_calibration_binding_requires_separate_validation_role(value_config):
    fitted = {"method": "temperature", "fitted": True,
              "split_role": "validation_calibration", "artifact_sha256": "b" * 64}
    result = value_module().select_threshold(
        ["a"], [0.5], ["a"], {"a": "team"}, value_config,
        model_version="fixture", artifact_sha256="a" * 64, calibration=fitted,
    )
    assert result["calibration"] == fitted
    fitted["split_role"] = "test"
    with pytest.raises(ValueError):
        value_module().select_threshold(
            ["a"], [0.5], ["a"], {"a": "team"}, value_config,
            model_version="fixture", artifact_sha256="a" * 64, calibration=fitted,
        )


@pytest.mark.parametrize("count", [10**1000, 10**308])
def test_overflow_is_rejected_instead_of_infinite_economic_report(assumptions, count):
    with pytest.raises(ValueError):
        value_module().compute_value(count, 0, 0, 0, assumptions)


def test_invalid_metadata_cannot_be_hidden_by_valid_status(value_config):
    value_config["evidence"]["t_manual"]["source"] = ""
    with pytest.raises(ValueError):
        value_module().validate_value_config(value_config)
    value_config["evidence"]["t_manual"] = {"source": "synthetic fixture", "estimated": 1}
    with pytest.raises(ValueError):
        value_module().validate_value_config(value_config)
