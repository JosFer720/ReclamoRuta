"""Behavioral checks for pinned, leakage-free BANKING77 preparation."""

import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from reclamoruta import data


def write_csv(path, rows, header=("text", "category")):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def repin(raw, monkeypatch):
    monkeypatch.setattr(data, "_SOURCE_HASHES", {
        name: hashlib.sha256((raw / name).read_bytes()).hexdigest()
        for name in ("train.csv", "test.csv", "categories.json", "LICENSE")
    })


@pytest.fixture
def raw(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    rows = [(f"message {label} {index}", label)
            for label in ("A!", "b") for index in range(20)]
    rows.extend([("shared", "A!"), (" MESSAGE A! 0 ", "A!")])
    write_csv(raw / "train.csv", rows)
    write_csv(raw / "test.csv", [("shared", "A!"), ("test b", "b"), ("TEST B", "b")])
    (raw / "categories.json").write_text(json.dumps(["b", "A!"]), encoding="utf-8")
    (raw / "LICENSE").write_text("fixture license", encoding="utf-8")
    repin(raw, monkeypatch)
    monkeypatch.setattr(data, "_SOURCE_COUNTS", {"train.csv": 42, "test.csv": 3, "categories.json": 2})
    return raw


def read_rows(out, name):
    with (out / f"{name}.csv").open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_normalization_unifies_compatibility_case_and_whitespace():
    assert data.normalize("  ＢＡＮＫ\tStraße\n\u00a0Café  ") == "bank strasse café"


def test_preparation_removes_leakage_retains_original_ids_and_labels(raw, tmp_path):
    out = tmp_path / "processed"
    manifest = data.prepare_data(raw, out)
    assert manifest["counts"] == {
        "train": 32, "validation": 8, "validation_model": 4,
        "validation_calibration": 2, "validation_threshold": 2,
        "test_official": 3, "test_unique": 2,
    }
    assert manifest["removed"]["train_test_overlap_rows"] == 1
    assert manifest["removed"]["train_duplicate_rows"] == 1
    assert json.loads((out / "label_to_id.json").read_text()) == {"A!": 0, "b": 1}
    train = read_rows(out, "train")
    val = read_rows(out, "validation")
    test = read_rows(out, "test_official")
    assert [row["id"] for row in test] == ["test.csv:0", "test.csv:1", "test.csv:2"]
    assert {row["category"] for row in train} == {"A!", "b"}
    assert {row["category"] for row in val} == {"A!", "b"}
    assert not {row["group_id"] for row in train + val} & {row["group_id"] for row in test}
    assert len({row["group_id"] for row in train + val}) == 40
    assert "train.csv:40" not in {row["id"] for row in train + val}
    assert "train.csv:41" not in {row["id"] for row in train + val}
    assert train[0]["id"] == "train.csv:9"
    assert val[0]["id"] == "train.csv:19"
    assert set(train[0]) == {"id", "text", "category", "group_id", "split"}
    subdivisions = [read_rows(out, name) for name in (
        "validation_model", "validation_calibration", "validation_threshold")]
    sub_ids = [{row["id"] for row in rows} for rows in subdivisions]
    assert set.union(*sub_ids) == {row["id"] for row in val}
    assert sum(map(len, sub_ids)) == len(set.union(*sub_ids))
    eda = json.loads((out / "eda.json").read_text())
    assert eda["raw"]["train"]["normalized_duplicate_rows"] == 1
    assert eda["raw"]["test"]["normalized_duplicate_rows"] == 1
    assert "message A! 0" not in json.dumps(eda)
    for name, digest in manifest["output_sha256"].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest


def test_exact_rerun_is_read_only_and_seed_change_is_collision(raw, tmp_path):
    out = tmp_path / "processed"
    first = data.prepare_data(raw, out)
    timestamps = {path.name: path.stat().st_mtime_ns for path in out.iterdir()}
    assert data.prepare_data(raw, out) == first
    assert timestamps == {path.name: path.stat().st_mtime_ns for path in out.iterdir()}
    with pytest.raises(data.DataError, match="output_collision"):
        data.prepare_data(raw, out, seed=7)


def test_tampered_source_is_rejected_without_writing(raw, tmp_path):
    with (raw / "train.csv").open("a", encoding="utf-8") as stream:
        stream.write("private banking message,A!\n")
    out = tmp_path / "processed"
    with pytest.raises(data.DataError, match="source_hash_mismatch") as caught:
        data.prepare_data(raw, out)
    assert "private" not in str(caught.value)
    assert not out.exists()


@pytest.mark.parametrize("problem", ["empty", "unknown", "schema", "malformed", "categories", "contradictory"])
def test_rejects_invalid_schema_and_semantics_even_if_hash_is_pinned(raw, tmp_path, monkeypatch, problem):
    if problem == "empty":
        write_csv(raw / "test.csv", [("  ", "A!")] * 3)
    elif problem == "unknown":
        write_csv(raw / "test.csv", [("private", "wrong")] * 3)
    elif problem == "schema":
        write_csv(raw / "test.csv", [("private", "A!")] * 3, ("message", "category"))
    elif problem == "malformed":
        (raw / "test.csv").write_text('text,category\n"unterminated,A!\n', encoding="utf-8")
    elif problem == "categories":
        (raw / "categories.json").write_text('["A!", "A!"]', encoding="utf-8")
    else:
        write_csv(raw / "test.csv", [("MESSAGE A! 0", "b"), ("one", "A!"), ("two", "b")])
    repin(raw, monkeypatch)
    with pytest.raises(data.DataError) as caught:
        data.prepare_data(raw, tmp_path / "processed")
    assert "private" not in str(caught.value)


@pytest.mark.parametrize("target", ["same", "child", "parent", "arbitrary", "tampered", "extra"])
def test_never_overwrites_raw_or_unowned_or_modified_outputs(raw, tmp_path, target):
    out = tmp_path / "processed"
    if target == "same":
        out = raw
    elif target == "child":
        out = raw / "processed"
    elif target == "parent":
        out = raw.parent
    elif target == "arbitrary":
        out.mkdir()
        (out / "train.csv").write_text("keep me", encoding="utf-8")
    else:
        data.prepare_data(raw, out)
        (out / ("train.csv" if target == "tampered" else "unowned.txt")).write_text("keep me", encoding="utf-8")
    before = {str(path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    with pytest.raises(data.DataError):
        data.prepare_data(raw, out)
    assert before == {str(path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}


def test_rejects_symbolic_link_outputs(raw, tmp_path):
    target = tmp_path / "outside"
    target.mkdir()
    link = tmp_path / "processed"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("symlink privilege unavailable")
    with pytest.raises(data.DataError, match="unsafe_path"):
        data.prepare_data(raw, link)
    assert not list(target.iterdir())


def test_cli_returns_structured_non_pii_errors(tmp_path):
    environment = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    result = subprocess.run([
        sys.executable, "-m", "reclamoruta.data", "--raw", str(tmp_path / "missing"),
        "--out", str(tmp_path / "processed")], capture_output=True, text=True, env=environment)
    assert result.returncode == 2
    assert json.loads(result.stderr)["error"]["code"] == "source_missing"
    assert not result.stdout


@pytest.mark.skipif(not os.getenv("BANKING77_RAW_DIR"), reason="set BANKING77_RAW_DIR for official pinned data")
def test_official_source_counts_and_no_group_leakage(tmp_path):
    manifest = data.prepare_data(Path(os.environ["BANKING77_RAW_DIR"]), tmp_path / "processed")
    assert manifest["counts"]["train"] == 7959
    assert manifest["counts"]["validation"] == 2033
    assert manifest["counts"]["test_official"] == 3080
    assert manifest["counts"]["test_unique"] == 3079
    assert manifest["removed"]["train_test_overlap_rows"] == 7
    assert manifest["removed"]["train_duplicate_rows"] == 4
