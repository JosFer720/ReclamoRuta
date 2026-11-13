"""Fit temperature on the verified calibration split without training a model.

NLL uses the same 1e-12 probability floor as Predictor temperature scaling.
The private package scorer is used here because predict exposes only top three
classes, while calibration requires the complete distribution.
"""

import argparse
import csv
import hashlib
import io
import json
import math
from numbers import Integral, Real
from pathlib import Path

from reclamoruta.contracts import ContractError
from reclamoruta.predictor import Predictor, read_manifest

_ROLE = 'validation_calibration'


def fit_temperature(probabilities, gold_indices) -> dict:
    """Minimize mean NLL deterministically over temperatures in [0.05, 20]."""
    try:
        rows = [list(row) for row in probabilities]
        gold = list(gold_indices)
    except TypeError as exc:
        raise ValueError('Expected a probability matrix and gold indices') from exc
    if not rows or not rows[0] or len(gold) != len(rows):
        raise ValueError('Expected nonempty aligned probabilities and gold indices')
    width = len(rows[0])
    for row, target in zip(rows, gold):
        if len(row) != width or any(isinstance(p, bool) or not isinstance(p, Real)
                or not math.isfinite(p) or not 0 <= p <= 1 for p in row):
            raise ValueError('Probabilities must be finite, rectangular and nonnegative')
        if not math.isclose(sum(row), 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError('Probabilities must sum to one')
        if isinstance(target, bool) or not isinstance(target, Integral) or not 0 <= target < width:
            raise ValueError('Gold indices must be integer class IDs')
    logs = [[math.log(max(float(p), 1e-12)) for p in row] for row in rows]

    def nll(temperature):
        losses = []
        for row, target in zip(logs, gold):
            scaled = [value / temperature for value in row]
            top = max(scaled)
            losses.append(top + math.log(sum(math.exp(value - top) for value in scaled))
                          - scaled[target])
        return sum(losses) / len(losses)

    # T=1 in Predictor preserves probabilities exactly (including zero values).
    before = -sum(row[target] for row, target in zip(logs, gold)) / len(rows)
    left, right = math.log(.05), math.log(20.)
    ratio = (math.sqrt(5) - 1) / 2
    x1, x2 = right - ratio * (right - left), left + ratio * (right - left)
    f1, f2 = nll(math.exp(x1)), nll(math.exp(x2))
    for _ in range(100):
        if f1 <= f2:
            right, x2, f2 = x2, x1, f1
            x1 = right - ratio * (right - left)
            f1 = nll(math.exp(x1))
        else:
            left, x1, f1 = x1, x2, f2
            x2 = left + ratio * (right - left)
            f2 = nll(math.exp(x2))
    candidates = [(before, 1.), (nll(.05), .05), (nll(20.), 20.),
                  (f1, math.exp(x1)), (f2, math.exp(x2))]
    after, temperature = min(candidates, key=lambda item: item[0])
    # Prefer identity for numerically indistinguishable objectives.
    if after >= before - 1e-12:
        after, temperature = before, 1.
    return {'temperature': temperature, 'nll_before': before,
            'nll_after': after, 'n_examples': len(rows)}


def _digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _inputs(artifact: Path, data: Path):
    """Read only calibration rows and catalog; bind their exact bytes."""
    model_manifest = read_manifest(artifact)
    try:
        manifest_bytes = (data / 'manifest.json').read_bytes()
        manifest = json.loads(manifest_bytes)
        digest = _digest(manifest_bytes)
        if model_manifest.get('data_manifest_sha256') != digest:
            raise ContractError('Data manifest does not match model provenance')
        payloads = {}
        for name in (f'{_ROLE}.csv', 'label_to_id.json'):
            content = (data / name).read_bytes()
            if _digest(content) != manifest['output_sha256'][name]:
                raise ContractError(f'Dataset hash mismatch: {name}')
            payloads[name] = content
        catalog = json.loads(payloads['label_to_id.json'])
        if not isinstance(catalog, dict) or not catalog or any(
                not isinstance(label, str) or not label.strip() or isinstance(index, bool)
                or not isinstance(index, int) for label, index in catalog.items()) \
                or sorted(catalog.values()) != list(range(len(catalog))):
            raise ContractError('Invalid label catalog')
        if json.loads((artifact / 'label_to_id.json').read_bytes()) != catalog:
            raise ContractError('Dataset catalog does not match model catalog')
        reader = csv.DictReader(io.StringIO(payloads[f'{_ROLE}.csv'].decode('utf-8')))
        if not {'text', 'category', 'split'} <= set(reader.fieldnames or []):
            raise ContractError('Calibration CSV is missing required columns')
        texts, targets = [], []
        for row in reader:
            if row.get('split') != _ROLE:
                raise ContractError('Only validation_calibration rows are allowed')
            if row.get('category') not in catalog:
                raise ContractError('Calibration CSV contains unknown classes')
            if not isinstance(row.get('text'), str) or not row['text'].strip():
                raise ContractError('Calibration CSV contains empty texts')
            texts.append(row['text'])
            targets.append(catalog[row['category']])
        if not texts:
            raise ContractError('Calibration split is empty')
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        if isinstance(exc, ContractError):
            raise
        raise ContractError('Invalid calibration data or provenance') from exc
    provenance = {'data_manifest_sha256': digest,
                  'split_sha256': _digest(payloads[f'{_ROLE}.csv']),
                  'label_to_id_sha256': _digest(payloads['label_to_id.json'])}
    return model_manifest, catalog, texts, targets, provenance


def calibrate(artifact: Path, data: Path, out: Path, device: str = 'cpu') -> dict:
    """Fit a verified model's calibration temperature and create a new JSON file."""
    artifact, data, out = Path(artifact), Path(data), Path(out)
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    # Keep output separate from immutable model/data inputs, including symlink aliases.
    resolved = out.resolve()
    if any(resolved.is_relative_to(root.resolve()) for root in (artifact, data)):
        raise ContractError('Calibration output must be outside model and data directories')
    model_manifest, catalog, texts, targets, provenance = _inputs(artifact, data)
    predictor = Predictor.load(artifact, device=device)
    if predictor.categories != sorted(catalog, key=catalog.get):
        raise ContractError('Predictor class order does not match dataset catalog')
    probabilities = []
    for start in range(0, len(texts), 32):
        batch = list(predictor._probabilities(texts[start:start + 32]))
        if len(batch) != len(texts[start:start + 32]) or any(len(row) != len(catalog) for row in batch):
            raise ContractError('Predictor returned an invalid probability matrix')
        probabilities.extend(batch)
    result = {**fit_temperature(probabilities, targets), 'schema_version': 1,
              'method': 'temperature', 'split_role': _ROLE,
              'model_version': model_manifest['model_version'],
              'artifact_sha256': _digest((artifact / 'manifest.json').read_bytes()),
              **provenance}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args(argv)
    result = calibrate(args.artifact, args.data, args.out, args.device)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return result


if __name__ == '__main__':
    main()
