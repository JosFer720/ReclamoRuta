"""Human timing study: anonymous IDs, no text by default, safe explicit CSV export."""

import csv
import hashlib
import re

import pytest

INTENTS = ["card_arrival", "compromised_card", "top_up_failed"]
ROUTES = {"card_arrival": "tarjetas", "compromised_card": "seguridad", "top_up_failed": "recargas"}


def study():
    from reclamoruta import study as module

    return module


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def session(clock=None, **kwargs):
    s = study()
    return s.StudySession(
        participant_id=s.new_participant_id(), intent_to_route=ROUTES,
        clock=clock or Clock(), **kwargs,
    )


def test_participant_ids_are_anonymous_and_names_rejected():
    s = study()
    pid = s.new_participant_id()
    assert re.fullmatch(r"P-[0-9a-f]{8}", pid)
    assert s.new_participant_id() != pid
    for bad in ["Example User", "example@example.invalid", "P-1234", "", "P-GGGGGGGG"]:
        with pytest.raises(s.StudyError):
            s.StudySession(participant_id=bad, intent_to_route=ROUTES)


def test_assisted_trial_records_time_correction_and_routes_without_text():
    clock = Clock()
    sess = session(clock)
    sess.start("train.csv:12", "asistida", text="my card was stolen", suggested_intent="card_arrival",
                needs_review=True, model_version="v1")
    clock.now = 7.5
    trial = sess.finish("compromised_card")
    assert trial.elapsed_s == pytest.approx(7.5)
    assert trial.corrected is True
    assert trial.final_route == "seguridad"
    assert trial.suggested_route == "tarjetas"
    assert trial.text is None
    assert "stolen" not in repr(sess.trials)


def test_manual_trial_has_no_suggestion():
    clock = Clock()
    sess = session(clock)
    sess.start("m1", "manual")
    clock.now = 3.0
    trial = sess.finish("top_up_failed")
    assert trial.suggested_intent is None and trial.corrected is False
    with pytest.raises(study().StudyError):
        sess.start("m2", "manual", suggested_intent="card_arrival")


def test_text_is_kept_only_when_explicitly_enabled():
    sess = session(store_text=True)
    sess.start("m1", "manual", text="where is my card")
    assert sess.finish("card_arrival").text == "where is my card"


@pytest.mark.parametrize("call", [
    lambda s: s.start("m1", "otra"),
    lambda s: s.start("bad id with spaces", "manual"),
    lambda s: s.start("m1", "asistida"),  # assisted without suggestion
    lambda s: s.finish("card_arrival"),   # not started
])
def test_invalid_trials_rejected(call):
    with pytest.raises(study().StudyError):
        call(session())


def test_unknown_final_intent_and_duplicate_message_rejected():
    sess = session()
    sess.start("m1", "manual")
    with pytest.raises(study().StudyError):
        sess.finish("not_an_intent")
    sess.finish("card_arrival")
    with pytest.raises(study().StudyError):
        sess.start("m1", "manual")


def test_condition_assignment_is_balanced_and_deterministic():
    s = study()
    ids = [f"m{i}" for i in range(10)]
    first = s.assign_conditions(ids, "P-0000abcd", seed=7)
    assert first == s.assign_conditions(ids, "P-0000abcd", seed=7)
    assert sorted(first) == ids
    assert list(first.values()).count("manual") == 5
    assert list(first.values()).count("asistida") == 5


def make_trials():
    clock = Clock()
    sess = session(clock)
    configuration = [("m1", "manual", None, "card_arrival", 10.0), ("m2", "manual", None, "top_up_failed", 12.0),
            ("m3", "asistida", "card_arrival", "card_arrival", 4.0),
            ("m4", "asistida", "card_arrival", "compromised_card", 9.0)]
    for message, condition, suggested, final, seconds in configuration:
        sess.start(message, condition, suggested_intent=suggested, model_version="v1" if suggested else None,
                   needs_review=False if suggested else None)
        clock.now += seconds
        sess.finish(final)
    return sess.trials


def test_summary_maps_times_to_value_parameters():
    summary = study().summarize(make_trials())
    assert summary["manual"]["n"] == 2
    assert summary["manual"]["median_s"] == pytest.approx(11.0)
    assert summary["asistida"]["correction_rate"] == pytest.approx(0.5)
    assert summary["value_inputs"]["t_manual"] == pytest.approx(11.0)
    assert summary["value_inputs"]["t_aceptar"] == pytest.approx(4.0)
    assert summary["value_inputs"]["t_corregir"] == pytest.approx(9.0)
    assert summary["value_inputs"]["t_revision"] is None  # no reviewed trials: never zero
    assert summary["estimated"] is True


def test_export_is_explicit_new_file_and_neutralises_formulas(tmp_path):
    s = study()
    sess = session(store_text=True)
    sess.start("m1", "manual", text="=HYPERLINK(\"http://x\")")
    sess.finish("card_arrival")
    out = tmp_path / "estudio.csv"
    s.export_trials(sess.trials, out)
    rows = list(csv.DictReader(out.open(encoding="utf-8", newline="")))
    assert rows[0]["text"].startswith("'=")
    assert rows[0]["participant_id"].startswith("P-")
    with pytest.raises(s.StudyError):
        s.export_trials(sess.trials, out)  # never overwrite


def test_export_omits_text_column_when_text_not_stored(tmp_path):
    out = tmp_path / "estudio.csv"
    study().export_trials(make_trials(), out)
    header = out.read_text(encoding="utf-8").splitlines()[0]
    assert "text" not in header.split(",")


def test_recording_and_export_do_not_touch_weights_or_test_data(tmp_path):
    weights = tmp_path / "artifact" / "model.safetensors"
    test_csv = tmp_path / "data" / "test_official.csv"
    weights.parent.mkdir()
    test_csv.parent.mkdir()
    weights.write_bytes(b"weights")
    test_csv.write_text("id,text\n", encoding="utf-8")
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (weights, test_csv)}
    study().export_trials(make_trials(), tmp_path / "out" / "estudio.csv")
    after = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (weights, test_csv)}
    assert before == after
    assert sorted(x.name for x in tmp_path.rglob("*") if x.is_file()) == [
        "estudio.csv", "model.safetensors", "test_official.csv"]


def test_study_module_does_not_import_training_code():
    # Fresh interpreter: other test files may legitimately import torch.
    import subprocess
    import sys

    code = ("import sys, reclamoruta.study; "
            "print('reclamoruta.train' in sys.modules or 'torch' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                         env={**__import__("os").environ, "PYTHONPATH": "src"})
    assert out.stdout.strip() == "False"
