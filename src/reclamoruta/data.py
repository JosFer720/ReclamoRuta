"""Prepare pinned local BANKING77 sources without training or network access.

CSV text and labels retain their original spelling. Normalization is used only
for duplicate grouping and leakage checks. Generated message files belong in an
ignored local directory, never in source control.
"""

import argparse
import csv
import hashlib
import io
import json
import math
import random
import statistics
import sys
import unicodedata
from collections import Counter
from pathlib import Path

_REVISION = "57ec275d8078af65b7731c2a98be812d844a6d6b"
_SOURCE_HASHES = {
    "train.csv": "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b",
    "test.csv": "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d",
    "categories.json": "53261da888122daf2d120d925458631d9619e15d82e56052e7a42e535ce32b63",
    "LICENSE": "7e7170e3cebf88a9f60c7b8421418323c09304da1af4d5e90f4da1dc1c8a2661",
}
_SOURCE_COUNTS = {"train.csv": 10003, "test.csv": 3080, "categories.json": 77}
_FIELDS = ("id", "text", "category", "group_id", "split")
_POLICY = {
    "version": 1,
    "normalization": "Unicode NFKC, casefold, Unicode whitespace collapse",
    "train_test_overlap": "remove every train row whose normalized key occurs in test",
    "train_duplicates": "retain first remaining normalized key in original source order",
    "validation": "sorted categories; original row order; fresh Random(seed) per class; first ceil(n/5)",
    "validation_subsets": "fresh Random(43) per class in validation order; floor(n/2) model, floor(remainder/2) calibration, rest threshold",
    "test": "official untouched; unique view keeps first normalized key",
    "id": "source filename plus colon plus zero-based data row",
    "group_id": "SHA256(normalized text encoded as UTF-8)",
}


class DataError(ValueError):
    """A safe structured failure containing no raw message or exception text."""

    def __init__(self, code: str, *, file: str | None = None, row: int | None = None):
        self.details = {"code": code}
        if file is not None:
            self.details["file"] = file
        if row is not None:
            self.details["row"] = row
        super().__init__(json.dumps(self.details, sort_keys=True))


def normalize(text: str) -> str:
    """Return a comparison key while leaving source text unchanged."""
    if not isinstance(text, str):
        raise DataError("invalid_text_type")
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _reject_links(path: Path) -> None:
    # Windows junctions/reparse points can redirect writes just like symlinks.
    for part in (path, *path.parents):
        if part.is_symlink():
            raise DataError("unsafe_path")
        if part.exists():
            stat = part.lstat()
            if getattr(stat, "st_file_attributes", 0) & 0x400:
                raise DataError("unsafe_path")


def _paths(raw_dir: Path, out_dir: Path) -> tuple[Path, Path]:
    raw, out = Path(raw_dir).absolute(), Path(out_dir).absolute()
    if ".." in raw.parts or ".." in out.parts:
        raise DataError("unsafe_path")
    _reject_links(raw)
    _reject_links(out)
    raw, out = raw.resolve(), out.resolve()
    if raw == out or raw in out.parents or out in raw.parents:
        raise DataError("raw_output_overlap")
    if not raw.is_dir():
        raise DataError("source_missing")
    return raw, out


def _load_sources(raw: Path) -> dict[str, str]:
    result = {}
    for name, expected in _SOURCE_HASHES.items():
        path = raw / name
        _reject_links(path)
        if not path.is_file():
            raise DataError("source_missing", file=name)
        content = path.read_bytes()
        if _sha256(content) != expected:
            raise DataError("source_hash_mismatch", file=name)
        try:
            result[name] = content.decode("utf-8")
        except UnicodeDecodeError:
            raise DataError("source_encoding", file=name) from None
    return result


def _categories(content: str) -> list[str]:
    try:
        labels = json.loads(content)
    except json.JSONDecodeError:
        raise DataError("categories_schema", file="categories.json") from None
    if not isinstance(labels, list) or any(
        not isinstance(label, str) or not label.strip() for label in labels
    ):
        raise DataError("categories_schema", file="categories.json")
    if len(set(labels)) != len(labels):
        raise DataError("duplicate_categories", file="categories.json")
    if len(labels) != _SOURCE_COUNTS["categories.json"]:
        raise DataError("source_count_mismatch", file="categories.json")
    return sorted(labels)


def _rows(content: str, filename: str, categories: list[str]) -> list[dict[str, str]]:
    rows = []
    try:
        reader = csv.reader(io.StringIO(content, newline=""), strict=True)
        if next(reader, None) != ["text", "category"]:
            raise DataError("csv_schema", file=filename)
        for index, fields in enumerate(reader):
            if len(fields) != 2:
                raise DataError("csv_schema", file=filename, row=index)
            text, category = fields
            key = normalize(text)
            if not key or "\x00" in text:
                raise DataError("empty_or_invalid_text", file=filename, row=index)
            if category not in categories:
                raise DataError("unknown_category", file=filename, row=index)
            rows.append({"id": f"{filename}:{index}", "text": text,
                         "category": category, "group_id": _sha256(key.encode("utf-8"))})
    except csv.Error:
        raise DataError("malformed_csv", file=filename) from None
    if len(rows) != _SOURCE_COUNTS[filename]:
        raise DataError("source_count_mismatch", file=filename)
    if {row["category"] for row in rows} != set(categories):
        raise DataError("missing_category", file=filename)
    return rows


def _unique(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen = set()
    result = []
    for row in rows:
        if row["group_id"] not in seen:
            seen.add(row["group_id"])
            result.append(row)
    return result


def _partition(rows: list[dict[str, str]], categories: list[str], seed: int,
               validation: bool = False) -> tuple[list, ...]:
    parts: tuple[list, ...] = ([], [], []) if validation else ([], [])
    for category in categories:
        group = [row for row in rows if row["category"] == category]
        random.Random(seed).shuffle(group)
        if validation:
            model_end = len(group) // 2
            calibration_end = model_end + (len(group) - model_end) // 2
            parts[0].extend(group[:model_end])
            parts[1].extend(group[model_end:calibration_end])
            parts[2].extend(group[calibration_end:])
        else:
            end = math.ceil(len(group) / 5)
            if end == 0 or end >= len(group):
                raise DataError("insufficient_class_rows")
            parts[0].extend(group[end:])
            parts[1].extend(group[:end])
    return parts


def _summary(rows: list[dict[str, str]], categories: list[str]) -> dict:
    counts = {category: 0 for category in categories}
    counts.update(Counter(row["category"] for row in rows))
    lengths = [len(row["text"]) for row in rows]
    return {
        "rows": len(rows), "class_counts": counts,
        "class_count_min": min(counts.values()), "class_count_max": max(counts.values()),
        "empty_text_rows": sum(not normalize(row["text"]) for row in rows),
        "exact_duplicate_rows": len(rows) - len({row["text"] for row in rows}),
        "normalized_duplicate_rows": len(rows) - len({row["group_id"] for row in rows}),
        "character_length": {"min": min(lengths, default=0), "max": max(lengths, default=0),
                             "mean": statistics.mean(lengths) if lengths else 0,
                             "median": statistics.median(lengths) if lengths else 0},
    }


def _json_bytes(value: dict) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _csv_bytes(rows: list[dict[str, str]], split: str) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(dict(row, split=split) for row in rows)
    return stream.getvalue().encode("utf-8")


def _write_outputs(out: Path, outputs: dict[str, bytes]) -> None:
    _reject_links(out)
    if out.exists():
        if not out.is_dir() or {path.name for path in out.iterdir()} != set(outputs):
            raise DataError("output_collision")
        # Exact bytes include our manifest, policy, pins, seed and every output hash.
        # No existing file is rewritten, including on a successful rerun.
        for name, content in outputs.items():
            path = out / name
            _reject_links(path)
            if not path.is_file() or path.stat().st_nlink != 1 or path.read_bytes() != content:
                raise DataError("output_collision")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    _reject_links(out.parent)
    out.mkdir()  # Exclusive directory ownership; a race becomes an error.
    for name, content in outputs.items():
        _reject_links(out)
        with (out / name).open("xb") as stream:
            stream.write(content)


def prepare_data(raw_dir: Path, out_dir: Path, seed: int = 42) -> dict:
    """Validate local source pins, prepare deterministic splits and return manifest.

    Failure raises DataError with safe structured metadata. Validate all source
    rows before creating any output. Existing outputs require exact ownership
    and content agreement; arbitrary files and changed policies are rejected.
    """
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise DataError("invalid_seed")
    try:
        raw, out = _paths(raw_dir, out_dir)
        sources = _load_sources(raw)
        categories = _categories(sources["categories.json"])
        train_raw = _rows(sources["train.csv"], "train.csv", categories)
        test = _rows(sources["test.csv"], "test.csv", categories)
        group_labels: dict[str, str] = {}
        for row in train_raw + test:
            previous = group_labels.setdefault(row["group_id"], row["category"])
            if previous != row["category"]:
                raise DataError("contradictory_normalized_labels")
        test_groups = {row["group_id"] for row in test}
        without_overlap = [row for row in train_raw if row["group_id"] not in test_groups]
        clean_train = _unique(without_overlap)
        train, validation = _partition(clean_train, categories, seed)
        model, calibration, threshold = _partition(validation, categories, 43, validation=True)
        splits = {"train": train, "validation": validation, "validation_model": model,
                  "validation_calibration": calibration, "validation_threshold": threshold,
                  "test_official": test, "test_unique": _unique(test)}
        removed = {
            "train_test_overlap_rows": len(train_raw) - len(without_overlap),
            "train_duplicate_rows": len(without_overlap) - len(clean_train),
            "test_duplicate_rows_in_unique_view": len(test) - len(splits["test_unique"]),
        }
        overlap = {"raw_train_test_groups": len({row["group_id"] for row in train_raw} & test_groups),
                   "train_validation_groups": len({row["group_id"] for row in train} & {row["group_id"] for row in validation}),
                   "prepared_train_test_groups": len({row["group_id"] for row in clean_train} & test_groups)}
        eda = {"raw": {"train": _summary(train_raw, categories), "test": _summary(test, categories)},
               "splits": {name: _summary(rows, categories) for name, rows in splits.items()},
               "removed": removed, "overlap": overlap, "errors": [],
               "manual_semantic_review": "pending", "manual_privacy_review": "pending"}
        outputs = {f"{name}.csv": _csv_bytes(rows, name) for name, rows in splits.items()}
        outputs["label_to_id.json"] = _json_bytes({category: index for index, category in enumerate(categories)})
        outputs["eda.json"] = _json_bytes(eda)
        manifest = {
            "schema_version": 1, "source": {"revision": _REVISION, "sha256": dict(_SOURCE_HASHES)},
            "policy": dict(_POLICY), "seed": seed, "validation_subdivision_seed": 43,
            "counts": {name: len(rows) for name, rows in splits.items()},
            "source_counts": dict(_SOURCE_COUNTS), "removed": removed, "overlap": overlap,
            "output_sha256": {name: _sha256(content) for name, content in outputs.items()},
            "provenance": {"dataset": "PolyAI/banking77", "repository": "https://github.com/PolyAI-LDN/task-specific-datasets",
                           "source_directory": "banking_data", "license_file_sha256": _SOURCE_HASHES["LICENSE"],
                           "manual_semantic_review": "pending", "manual_privacy_review": "pending"},
        }
        outputs["manifest.json"] = _json_bytes(manifest)
        _write_outputs(out, outputs)
        return manifest
    except OSError:
        # Never expose OS exception text, which can include user-controlled paths.
        raise DataError("filesystem_error") from None


def main(argv: list[str] | None = None) -> int:
    """CLI emits machine-readable success or safe error metadata."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    try:
        manifest = prepare_data(args.raw, args.out, args.seed)
    except DataError as error:
        print(json.dumps({"error": error.details}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps({"counts": manifest["counts"], "seed": manifest["seed"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
