"""Streamlit app: runtime loading guards and UI flow with a TEST-ONLY fake predictor."""

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "reports" / "BANKING77_LABEL_TO_ID.json"
ROUTES = ROOT / "configs" / "routes.json"


def app():
    from reclamoruta import app as module

    return module


def contracts():
    from reclamoruta import contracts as module

    return module


# --- Runtime loading --------------------------------------------------------

def make_artifact(tmp_path, version="distilbert-v1"):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "manifest.json").write_text(json.dumps({"model_version": version}), encoding="utf-8")
    digest = hashlib.sha256((artifact / "manifest.json").read_bytes()).hexdigest()
    return artifact, digest


def write_decision(tmp_path, digest, version="distilbert-v1", **changes):
    value = {
        "status": "validated", "split_role": "validation_threshold", "model_version": version,
        "artifact_sha256": digest, "threshold": 0.6,
        "calibration": {"method": "none", "fitted": False}, "assumptions_version": "s-v1",
    }
    value.update(changes)
    path = tmp_path / "decision.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


class LoadedPredictor:
    def __init__(self, version="distilbert-v1"):
        self.model_version = version

    def predict(self, texts):
        raise AssertionError("loading must not classify")


def load(tmp_path, artifact, decision, loader, calibration=None):
    return app().load_runtime(
        artifact_dir=artifact, decision_path=decision, routes_path=ROUTES,
        catalog_path=CATALOG, loader=loader, calibration_path=calibration,
    )


def test_runtime_loads_with_matching_artifact_and_decision(tmp_path):
    artifact, digest = make_artifact(tmp_path)
    runtime = load(tmp_path, artifact, write_decision(tmp_path, digest), lambda d, device: LoadedPredictor())
    assert runtime.decision.threshold == 0.6
    assert runtime.test_double is False
    assert len(runtime.routes.intent_to_route) == 77


def test_missing_artifact_is_reported_without_loading(tmp_path):
    calls = []
    with pytest.raises(app().AppConfigError, match="manifest|artefacto|Artefacto"):
        load(tmp_path, tmp_path / "nope", write_decision(tmp_path, "a" * 64),
             lambda d, device: calls.append(d))
    assert calls == []


def test_pending_repository_decision_is_rejected(tmp_path):
    artifact, _ = make_artifact(tmp_path)
    with pytest.raises(app().AppConfigError, match="[Dd]ecisi"):
        load(tmp_path, artifact, ROOT / "configs" / "decision.json", lambda d, device: LoadedPredictor())


def test_decision_for_another_artifact_is_rejected_before_loading_weights(tmp_path):
    artifact, _ = make_artifact(tmp_path)
    calls = []
    with pytest.raises(app().AppConfigError, match="artefacto"):
        load(tmp_path, artifact, write_decision(tmp_path, "b" * 64),
             lambda d, device: calls.append(d) or LoadedPredictor())
    assert calls == []


def test_decision_for_another_model_version_is_rejected(tmp_path):
    artifact, digest = make_artifact(tmp_path)
    with pytest.raises(app().AppConfigError, match="otro modelo"):
        load(tmp_path, artifact, write_decision(tmp_path, digest, version="distilbert-v2"),
             lambda d, device: LoadedPredictor("distilbert-v1"))


def test_corrupt_weights_become_clear_error_without_fallback(tmp_path):
    artifact, digest = make_artifact(tmp_path)

    def broken(directory, device):
        raise OSError("safetensors header corrupt")

    with pytest.raises(app().AppConfigError, match="No se pudo cargar el modelo"):
        load(tmp_path, artifact, write_decision(tmp_path, digest), broken)


def test_predictor_with_different_catalog_is_rejected(tmp_path):
    artifact, digest = make_artifact(tmp_path)
    predictor = LoadedPredictor()
    predictor.categories = ["a", "b"]
    with pytest.raises(app().AppConfigError, match="catálogo"):
        load(tmp_path, artifact, write_decision(tmp_path, digest), lambda d, device: predictor)


# --- Calibration -----------------------------------------------------------

class CalibratablePredictor(LoadedPredictor):
    calibration_sha256 = None

    def with_calibration(self, temperature, calibration_sha256):
        calibrated = CalibratablePredictor(self.model_version)
        calibrated.temperature = temperature
        calibrated.calibration_sha256 = calibration_sha256
        return calibrated


def write_calibration(tmp_path, digest, **changes):
    value = {"method": "temperature", "temperature": 1.5, "split_role": "validation_calibration",
             "model_version": "distilbert-v1", "artifact_sha256": digest, **changes}
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def calibrated_decision(tmp_path, digest, calibration_sha, method="temperature"):
    return write_decision(tmp_path, digest, calibration={
        "method": method, "fitted": True, "split_role": "validation_calibration",
        "artifact_sha256": calibration_sha,
    })


def test_fitted_calibration_is_applied_to_the_predictor(tmp_path):
    artifact, digest = make_artifact(tmp_path)
    path, sha = write_calibration(tmp_path, digest)
    runtime = load(tmp_path, artifact, calibrated_decision(tmp_path, digest, sha),
                   lambda d, device: CalibratablePredictor(), calibration=path)
    assert runtime.decision.calibrated is True
    assert runtime.predictor.temperature == 1.5
    assert runtime.predictor.calibration_sha256 == sha


def test_fitted_calibration_without_file_is_rejected_before_loading(tmp_path):
    artifact, digest = make_artifact(tmp_path)
    calls = []
    for calibration in (None, tmp_path / "missing.json"):
        with pytest.raises(app().AppConfigError, match="calibración"):
            load(tmp_path, artifact, calibrated_decision(tmp_path, digest, "c" * 64),
                 lambda d, device: calls.append(d) or CalibratablePredictor(), calibration)
    assert calls == []


def test_modified_calibration_file_is_rejected_before_loading(tmp_path):
    artifact, digest = make_artifact(tmp_path)
    path, sha = write_calibration(tmp_path, digest)
    write_calibration(tmp_path, digest, temperature=0.2)
    calls = []
    with pytest.raises(app().AppConfigError, match="hash"):
        load(tmp_path, artifact, calibrated_decision(tmp_path, digest, sha),
             lambda d, device: calls.append(d) or CalibratablePredictor(), path)
    assert calls == []


def test_unsupported_calibration_method_is_rejected(tmp_path):
    artifact, digest = make_artifact(tmp_path)
    path, sha = write_calibration(tmp_path, digest, method="isotonic")
    with pytest.raises(app().AppConfigError, match="no soportado"):
        load(tmp_path, artifact, calibrated_decision(tmp_path, digest, sha, method="isotonic"),
             lambda d, device: CalibratablePredictor(), path)


def test_predictor_that_cannot_calibrate_is_rejected(tmp_path):
    artifact, digest = make_artifact(tmp_path)
    path, sha = write_calibration(tmp_path, digest)
    with pytest.raises(app().AppConfigError, match="calibración"):
        load(tmp_path, artifact, calibrated_decision(tmp_path, digest, sha),
             lambda d, device: LoadedPredictor(), path)


# --- UI with AppTest -----------------------------------------------------------

def fake_app_script():
    """DOBLE DE PRUEBA: predictor falso, nunca usar como demo."""
    import json as _json
    from pathlib import Path as _Path

    import streamlit as st

    from reclamoruta import app as _app
    from reclamoruta import contracts as _c

    root = _Path(_app.__file__).resolve().parents[2]
    catalog = list(_json.loads((root / "reports" / "BANKING77_LABEL_TO_ID.json").read_text(encoding="utf-8")))
    routes = _c.load_routes(root / "configs" / "routes.json", catalog)

    calibration = st.session_state.get("fake_calibration", {"method": "none", "fitted": False})

    class FakePredictor:
        model_version = "fake-v1"
        calibration_sha256 = None if st.session_state.get("fake_skip_calibration") \
            else calibration.get("artifact_sha256")

        def predict(self, texts):
            st.session_state["fake_calls"] = st.session_state.get("fake_calls", 0) + 1
            if st.session_state.get("fake_fail"):
                raise RuntimeError("fallo simulado")
            label = st.session_state.get("fake_label", "card_arrival")
            score = st.session_state.get("fake_score", 0.9)
            return [_c.Prediction(label, score, [(label, score)], "fake-v1", 3.0)]

    decision = _c.BoundDecision("fake-v1", "a" * 64, 0.6, calibration, "s-v1")
    _app.render(_app.Runtime(FakePredictor(), decision, routes, test_double=True))


@pytest.fixture
def at():
    from streamlit.testing.v1 import AppTest

    test = AppTest.from_function(fake_app_script, default_timeout=10)
    test.run()
    assert not test.exception
    return test


def all_text(at):
    parts = []
    for kind in ("markdown", "caption", "info", "warning", "error", "success", "title", "subheader", "metric"):
        for element in getattr(at, kind):
            parts.append(str(getattr(element, "value", "")) + " " + str(getattr(element, "label", "")))
    return "\n".join(parts)


def submit(at, text):
    at.text_area(key="message").input(text)
    at.button(key="classify").click()
    at.run()
    assert not at.exception
    return at


def test_scope_warning_and_test_double_banner_are_visible(at):
    text = all_text(at)
    assert "inglés" in text
    assert "operaciones bancarias" in text
    assert "DOBLE DE PRUEBA" in text


@pytest.mark.parametrize("message", ["", "    "])
def test_empty_input_is_rejected_without_calling_predictor(at, message):
    submit(at, message)
    assert "fake_calls" not in at.session_state
    assert any("vacío" in w.value for w in at.warning)


def test_overlong_input_is_rejected_without_calling_predictor(at):
    # The widget enforces max_chars=2000 itself; contracts.validate_message is the
    # server-side guard (tested in test_contracts.py). Either way: rejected, no call.
    submit(at, "x" * 2001)
    assert "fake_calls" not in at.session_state
    assert at.warning
    assert "result" not in at.session_state


def test_confident_suggestion_shows_intent_route_and_uncalibrated_score(at):
    submit(at, "Where is my new card?")
    text = all_text(at)
    assert at.session_state["fake_calls"] == 1
    assert "card_arrival" in text
    assert "Tarjetas" in text
    assert "sin calibrar" in text
    assert any("Sugerencia" in s.value for s in at.success)


def test_calibrated_score_is_labeled_as_calibrated(at):
    at.session_state["fake_calibration"] = {
        "method": "temperature", "fitted": True, "split_role": "validation_calibration",
        "artifact_sha256": "c" * 64,
    }
    at.run()
    submit(at, "Where is my new card?")
    assert "calibrad" in all_text(at)
    assert "sin calibrar" not in all_text(at)


def test_declared_but_unapplied_calibration_shows_no_score(at):
    at.session_state["fake_calibration"] = {
        "method": "temperature", "fitted": True, "split_role": "validation_calibration",
        "artifact_sha256": "c" * 64,
    }
    at.session_state["fake_skip_calibration"] = True
    at.run()
    submit(at, "Where is my new card?")
    assert "fake_calls" not in at.session_state
    assert "result" not in at.session_state
    assert any("calibración" in w.value for w in at.warning)


def test_low_score_requires_review(at):
    at.session_state["fake_score"] = 0.3
    submit(at, "hmm")
    assert any("Revisión requerida" in w.value for w in at.warning)
    assert not any("Sugerencia aceptable" in s.value for s in at.success)


def test_correction_changes_route_without_new_prediction(at):
    submit(at, "Where is my new card?")
    at.selectbox(key="final_intent").select("compromised_card")
    at.button(key="confirm").click()
    at.run()
    assert not at.exception
    assert at.session_state["fake_calls"] == 1
    text = all_text(at)
    assert "corregida" in text.lower()
    assert "Seguridad" in text


def test_predictor_failure_shows_error_and_no_invented_answer(at):
    at.session_state["fake_fail"] = True
    submit(at, "Where is my new card?")
    assert any("No se pudo clasificar" in e.value for e in at.error)
    assert "card_arrival" not in all_text(at)


def test_main_without_artifact_shows_configuration_error(monkeypatch, tmp_path):
    from streamlit.testing.v1 import AppTest

    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("RECLAMORUTA_ARTIFACT_DIR", str(tmp_path / "missing"))
    test = AppTest.from_file(str(ROOT / "src" / "reclamoruta" / "app.py"), default_timeout=10)
    test.run()
    assert not test.exception
    assert test.error and "artefacto" in test.error[0].value.lower()
    assert not test.text_area
