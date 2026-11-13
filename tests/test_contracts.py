"""Shared contracts: prediction shape, message limits, routes and decision binding."""

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CATALOG = json.loads((ROOT / "reports" / "BANKING77_LABEL_TO_ID.json").read_text(encoding="utf-8"))
SHA_A = "a" * 64
SHA_B = "b" * 64


def contracts():
    from reclamoruta import contracts as module

    return module


def make_prediction(label="card_arrival", score=0.8, version="distilbert-v1"):
    c = contracts()
    return c.Prediction(label, score, [(label, score), ("activate_my_card", 0.1)], version, 12.5)


def decision(**changes):
    value = {
        "status": "validated",
        "split_role": "validation_threshold",
        "model_version": "distilbert-v1",
        "artifact_sha256": SHA_A,
        "threshold": 0.6,
        "calibration": {"method": "none", "fitted": False},
        "assumptions_version": "supuestos-v1",
    }
    value.update(changes)
    return value


class CountingPredictor:
    model_version = "distilbert-v1"

    def __init__(self, label="card_arrival", score=0.8):
        self.calls = []
        self.label = label
        self.score = score

    def predict(self, texts):
        self.calls.append(list(texts))
        return [make_prediction(self.label, self.score) for _ in texts]


# --- Prediction -----------------------------------------------------------

def test_prediction_has_agreed_fields_and_is_immutable():
    p = make_prediction()
    assert (p.label, p.score, p.model_version, p.elapsed_ms) == ("card_arrival", 0.8, "distilbert-v1", 12.5)
    assert p.top3[0] == ("card_arrival", 0.8)
    with pytest.raises(Exception):
        p.label = "other"


def test_prediction_satisfies_evaluation_protocol():
    from reclamoruta.evaluate import evaluate_predictions

    result = evaluate_predictions(["card_arrival"], [make_prediction()], categories=list(CATALOG))
    assert result["accuracy"] == 1.0


@pytest.mark.parametrize("score", [-0.01, 1.01, float("nan"), float("inf"), True, "0.5"])
def test_prediction_rejects_scores_outside_unit_interval(score):
    c = contracts()
    with pytest.raises(c.ContractError):
        c.Prediction("card_arrival", score, [], "v1", 1.0)


@pytest.mark.parametrize("kwargs", [
    {"label": ""}, {"model_version": " "}, {"elapsed_ms": -1.0}, {"elapsed_ms": float("nan")},
    {"top3": [("x", 0.5)] * 4}, {"top3": [("card_arrival", 0.3), ("b", 0.5)]},
    {"top3": [("other", 0.8)]},
    {"top3": [("card_arrival", 0.8), ("card_arrival", 0.1)]},
])
def test_prediction_rejects_inconsistent_fields(kwargs):
    c = contracts()
    base = {"label": "card_arrival", "score": 0.8, "top3": [("card_arrival", 0.8)],
            "model_version": "v1", "elapsed_ms": 1.0}
    base.update(kwargs)
    with pytest.raises(c.ContractError):
        c.Prediction(**base)


# --- Message validation ---------------------------------------------------

@pytest.mark.parametrize("text", ["", "   ", "\n\t ", "x" * 2001, "  " + "x" * 2001 + "  ", None, 5])
def test_invalid_messages_are_rejected(text):
    c = contracts()
    with pytest.raises(c.ContractError):
        c.validate_message(text)


def test_message_limits_apply_after_stripping_outer_spaces():
    c = contracts()
    assert c.validate_message("  hi  ") == "hi"
    assert c.validate_message("  " + "x" * 2000 + "  ") == "x" * 2000
    assert c.validate_message("a") == "a"


# --- Routes ---------------------------------------------------------------

def test_repository_route_map_covers_exactly_the_77_intentions():
    c = contracts()
    routes = c.load_routes(ROOT / "configs" / "routes.json", categories=list(CATALOG))
    assert set(routes.intent_to_route) == set(CATALOG)
    assert len(routes.intent_to_route) == 77
    assert 2 <= len(routes.route_names) < 77  # 77 intenciones, no 77 departamentos
    assert set(routes.intent_to_route.values()) == set(routes.route_names)


def routes_config(intents, routes=None):
    return {
        "version": "rutas-test",
        "status": "proposal",
        "routes": routes or {"r1": {"name": "Ruta 1", "description": "d"}},
        "intents": intents,
    }


@pytest.mark.parametrize("config", [
    routes_config({"a": "r1"}),                                     # falta b
    routes_config({"a": "r1", "b": "r1", "c": "r1"}),               # sobra c
    routes_config({"a": "r1", "b": "nope"}),                        # ruta inexistente
    routes_config({"a": "r1", "b": "r1"}, {"r1": {"name": "R", "description": "d"},
                                           "r2": {"name": "S", "description": "d"}}),  # ruta sin uso
    routes_config({"a": "r1", "b": "r1"}, {"r1": {"name": "", "description": "d"}}),
    {"intents": {"a": "r1", "b": "r1"}},
])
def test_route_maps_must_be_complete_and_consistent(config):
    c = contracts()
    with pytest.raises(c.ContractError):
        c.validate_routes(config, categories=["a", "b"])


# --- Decision binding -----------------------------------------------------

def test_decision_binds_to_matching_model_and_artifact():
    c = contracts()
    bound = c.bind_decision(decision(), model_version="distilbert-v1", artifact_sha256=SHA_A)
    assert bound.threshold == 0.6
    assert bound.calibrated is False


@pytest.mark.parametrize("kwargs", [
    {"model_version": "distilbert-v2", "artifact_sha256": SHA_A},
    {"model_version": "distilbert-v1", "artifact_sha256": SHA_B},
])
def test_configuration_of_another_model_is_rejected(kwargs):
    c = contracts()
    with pytest.raises(c.ContractError, match="otro modelo|artefacto"):
        c.bind_decision(decision(), **kwargs)


def test_repository_pending_decision_is_rejected():
    c = contracts()
    pending = json.loads((ROOT / "configs" / "decision.json").read_text(encoding="utf-8"))
    with pytest.raises(c.ContractError):
        c.bind_decision(pending, model_version="distilbert-v1", artifact_sha256=SHA_A)


def test_artifact_digest_is_sha256_of_manifest(tmp_path):
    c = contracts()
    (tmp_path / "manifest.json").write_bytes(b'{"model_version": "v1"}')
    assert c.artifact_digest(tmp_path) == hashlib.sha256(b'{"model_version": "v1"}').hexdigest()
    with pytest.raises(c.ContractError):
        c.artifact_digest(tmp_path / "missing")


# --- Routing one message --------------------------------------------------

def bound(threshold=0.6):
    c = contracts()
    return c.bind_decision(decision(threshold=threshold), model_version="distilbert-v1", artifact_sha256=SHA_A)


def repo_routes():
    return contracts().load_routes(ROOT / "configs" / "routes.json", categories=list(CATALOG))


@pytest.mark.parametrize("text", ["", "   ", "x" * 2001])
def test_invalid_text_never_invokes_predictor(text):
    c = contracts()
    predictor = CountingPredictor()
    with pytest.raises(c.ContractError):
        c.route_message(text, predictor, bound(), repo_routes())
    assert predictor.calls == []


def test_score_equal_to_threshold_allows_suggestion():
    c = contracts()
    result = c.route_message(" where is my card? ", CountingPredictor(score=0.6), bound(0.6), repo_routes())
    assert result.needs_review is False
    assert result.intent == "card_arrival"
    assert result.route == repo_routes().intent_to_route["card_arrival"]


def test_score_below_threshold_requires_review():
    c = contracts()
    predictor = CountingPredictor(score=0.5999)
    result = c.route_message("where is my card?", predictor, bound(0.6), repo_routes())
    assert result.needs_review is True
    assert predictor.calls == [["where is my card?"]]


def test_route_comes_from_route_map_not_from_classifier():
    c = contracts()
    routes = repo_routes()
    for label in ("compromised_card", "top_up_failed", "card_arrival"):
        result = c.route_message("msg", CountingPredictor(label=label), bound(), routes)
        assert result.route == routes.intent_to_route[label]


def test_predictor_of_another_version_or_unknown_label_is_rejected():
    c = contracts()

    class Other(CountingPredictor):
        def predict(self, texts):
            return [make_prediction(version="distilbert-v2")]

    with pytest.raises(c.ContractError, match="otro modelo"):
        c.route_message("msg", Other(), bound(), repo_routes())

    class Unknown(CountingPredictor):
        def predict(self, texts):
            return [c.Prediction("not_an_intent", 0.9, [], "distilbert-v1", 1.0)]

    with pytest.raises(c.ContractError):
        c.route_message("msg", Unknown(), bound(), repo_routes())


def test_predictor_must_return_exactly_one_prediction_per_text():
    c = contracts()

    class Empty(CountingPredictor):
        def predict(self, texts):
            return []

    with pytest.raises(c.ContractError):
        c.route_message("msg", Empty(), bound(), repo_routes())


def test_correction_derives_route_from_map_and_flags_change():
    c = contracts()
    routes = repo_routes()
    result = c.route_message("msg", CountingPredictor(label="card_arrival"), bound(), routes)
    same = c.confirm(result, "card_arrival", routes)
    changed = c.confirm(result, "compromised_card", routes)
    assert same.corrected is False and same.route == result.route
    assert changed.corrected is True and changed.route == routes.intent_to_route["compromised_card"]
    with pytest.raises(c.ContractError):
        c.confirm(result, "not_an_intent", routes)


# --- Calibration ----------------------------------------------------------

def write_calibration(tmp_path, **changes):
    value = {"method": "temperature", "temperature": 2.0, "split_role": "validation_calibration",
             "model_version": "distilbert-v1", "artifact_sha256": SHA_A, **changes}
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def fitted(digest, method="temperature"):
    return {"method": method, "fitted": True, "split_role": "validation_calibration",
            "artifact_sha256": digest}


def load(path, calibration):
    return contracts().load_calibration(
        path, calibration=calibration, model_version="distilbert-v1", artifact_sha256=SHA_A)


def calibrated_bound(digest):
    return contracts().bind_decision(
        decision(calibration=fitted(digest)), model_version="distilbert-v1", artifact_sha256=SHA_A)


def test_calibration_file_returns_verified_temperature(tmp_path):
    path, digest = write_calibration(tmp_path)
    assert load(path, fitted(digest)) == 2.0


def test_missing_calibration_file_is_rejected(tmp_path):
    with pytest.raises(contracts().ContractError, match="calibración"):
        load(tmp_path / "missing.json", fitted("c" * 64))


def test_modified_calibration_file_is_rejected(tmp_path):
    path, digest = write_calibration(tmp_path)
    write_calibration(tmp_path, temperature=0.5)
    with pytest.raises(contracts().ContractError, match="hash"):
        load(path, fitted(digest))


@pytest.mark.parametrize("changes", [
    {"model_version": "distilbert-v2"}, {"artifact_sha256": SHA_B},
    {"split_role": "validation_threshold"}, {"method": "isotonic"},
    {"temperature": 0}, {"temperature": -1.0}, {"temperature": "2"}, {"temperature": True},
    {"temperature": float("nan")}, {"temperature": float("inf")},
])
def test_calibration_for_another_model_or_invalid_temperature_is_rejected(tmp_path, changes):
    path, digest = write_calibration(tmp_path, **changes)
    with pytest.raises(contracts().ContractError):
        load(path, fitted(digest))


def test_unsupported_calibration_method_is_rejected(tmp_path):
    path, digest = write_calibration(tmp_path, method="isotonic")
    with pytest.raises(contracts().ContractError, match="no soportado"):
        load(path, fitted(digest, method="isotonic"))


def test_declared_calibration_not_applied_is_rejected_before_predicting():
    c = contracts()
    predictor = CountingPredictor()
    with pytest.raises(c.ContractError, match="calibración"):
        c.route_message("msg", predictor, calibrated_bound("c" * 64), repo_routes())
    assert predictor.calls == []


@pytest.mark.parametrize("applied,declared", [("d" * 64, "c" * 64), ("c" * 64, None)])
def test_applied_calibration_must_match_decision(applied, declared):
    c = contracts()
    predictor = CountingPredictor()
    predictor.calibration_sha256 = applied
    decision_ = calibrated_bound(declared) if declared else bound()
    with pytest.raises(c.ContractError, match="calibración"):
        c.route_message("msg", predictor, decision_, repo_routes())
    assert predictor.calls == []


def test_matching_applied_calibration_is_marked_calibrated():
    c = contracts()
    predictor = CountingPredictor()
    predictor.calibration_sha256 = "c" * 64
    result = c.route_message("msg", predictor, calibrated_bound("c" * 64), repo_routes())
    assert result.calibrated is True
