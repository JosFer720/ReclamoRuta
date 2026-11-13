"""Freeze calibration/economic decisions on validation, then evaluate held-out test.

Only aggregate reports are persisted. Monetary costs apply to each evaluated
batch as supplied; all economic results are explicitly estimated scenarios.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys

from reclamoruta.calibration import fit_temperature
from reclamoruta.contracts import bind_decision, load_calibration, load_routes
from reclamoruta.data import _reject_links
from reclamoruta.evaluate import evaluate_predictions
from reclamoruta.predictor import Predictor
from reclamoruta.uncertainty import paired_bootstrap, paired_classification_bootstrap
from reclamoruta.value import (
    COUNT_PARAMETERS, compute_value, select_threshold, sensitivity_analysis,
    validate_value_config,
)


def _hash(content):
    return hashlib.sha256(content).hexdigest()


def _read(path):
    path = Path(path)
    _reject_links(path)
    return path.read_bytes()


def _json(path):
    return json.loads(_read(path))


def _write(path, value):
    _reject_links(path)
    with path.open('xb') as stream:
        stream.write((json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode())


def _data(data):
    content = _read(data / 'manifest.json')
    manifest = json.loads(content)
    hashes = manifest.get('output_sha256')
    if not isinstance(hashes, dict):
        raise ValueError('Invalid data manifest')
    catalog_bytes = _verified(data, 'label_to_id.json', hashes)
    catalog = json.loads(catalog_bytes)
    if not isinstance(catalog, dict) or not catalog or any(
        not isinstance(k, str) or not k.strip() or type(v) is not int for k, v in catalog.items()
    ) or sorted(catalog.values()) != list(range(len(catalog))):
        raise ValueError('Invalid category catalog')
    return _hash(content), hashes, catalog


def _verified(data, name, hashes):
    content = _read(data / name)
    if _hash(content) != hashes.get(name):
        raise ValueError(f'Data hash mismatch: {name}')
    return content


def _rows(data, role, hashes, catalog):
    content = _verified(data, f'{role}.csv', hashes)
    reader = csv.DictReader(io.StringIO(content.decode('utf-8'), newline=''), strict=True)
    if reader.fieldnames != ['id', 'text', 'category', 'group_id', 'split']:
        raise ValueError('Invalid split schema')
    rows = list(reader)
    if not rows or any(set(r) != set(reader.fieldnames) or r['split'] != role
                       or r['category'] not in catalog or not r['text'].strip()
                       or not r['group_id'] or not r['id'] for r in rows):
        raise ValueError('Invalid split rows or split role')
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate split row IDs')
    return rows


def _disjoint(first, second):
    if {r['group_id'] for r in first} & {r['group_id'] for r in second}:
        raise ValueError('Leakage between split roles')


def _inputs(data, artifacts, value_config, routes, device):
    data_hash, hashes, catalog = _data(data)
    config = _json(value_config)
    assumptions = validate_value_config(config)
    categories = sorted(catalog, key=catalog.get)
    route_map = load_routes(routes, categories).intent_to_route
    models = {}
    for artifact in artifacts:
        content = _read(artifact / 'manifest.json')
        manifest = json.loads(content)
        if manifest.get('data_manifest_sha256') != data_hash:
            raise ValueError('Model is bound to another data manifest')
        predictor = Predictor.load(artifact, device=device)
        version = predictor.model_version
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', version) or version in models:
            raise ValueError('Invalid or duplicate model version')
        if predictor.categories != categories:
            raise ValueError('Model category catalog differs from data')
        if json.loads(_read(artifact / 'label_to_id.json')) != catalog:
            raise ValueError('Model label IDs differ from data')
        models[version] = (predictor, _hash(content), artifact)
    if not models:
        raise ValueError('At least one model is required')
    return data_hash, hashes, catalog, config, assumptions, route_map, models


def _source():
    files = {p.name: _hash(_read(p)) for p in sorted(Path(__file__).parent.glob('*.py'))}
    try:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).parent,
                                check=True, capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = None
    return {'commit': commit, 'python_sha256': files}


def _output_location(out, data, artifacts, value_config, routes):
    _reject_links(out)
    resolved = out.resolve()
    for input_path in [data, *artifacts, value_config, routes]:
        _reject_links(input_path)
        protected = input_path.resolve()
        if resolved == protected or protected in resolved.parents or resolved in protected.parents:
            raise ValueError('Output/input overlap is forbidden')


def prepare(data, artifacts, value_config, routes, out, *, device='cpu'):
    """Create a new, exclusive freeze using validation subsets only."""
    data, value_config, routes, out = map(Path, (data, value_config, routes, out))
    artifacts = list(map(Path, artifacts))
    _output_location(out, data, artifacts, value_config, routes)
    if out.exists():
        raise FileExistsError('Output already exists')
    data_hash, hashes, catalog = _data(data)
    # Validate split roles before loading any model or creating output.
    calibration_rows = _rows(data, 'validation_calibration', hashes, catalog)
    threshold_rows = _rows(data, 'validation_threshold', hashes, catalog)
    _disjoint(calibration_rows, threshold_rows)
    _, _, _, config, _, route_map, models = _inputs(data, artifacts, value_config, routes, device)
    outputs = {}
    for version, (predictor, model_hash, _) in models.items():
        texts = [r['text'] for r in calibration_rows]
        probabilities = [row for start in range(0, len(texts), 32)
                         for row in predictor._probabilities(texts[start:start + 32])]
        fitted = fit_temperature(probabilities, [catalog[r['category']] for r in calibration_rows])
        calibration = {**fitted, 'method': 'temperature', 'fitted': True,
            'split_role': 'validation_calibration', 'model_version': version,
            'artifact_sha256': model_hash, 'data_manifest_sha256': data_hash}
        # Hash exact serialized bytes later written by _write.
        calibration_hash = _hash((json.dumps(calibration, indent=2, sort_keys=True,
                                             allow_nan=False) + '\n').encode())
        metadata = {'method': 'temperature', 'fitted': True,
                    'split_role': 'validation_calibration', 'artifact_sha256': calibration_hash}
        calibrated = predictor.with_calibration(fitted['temperature'], calibration_hash)
        predictions = calibrated.predict([r['text'] for r in threshold_rows])
        evaluate_predictions([r['category'] for r in threshold_rows], predictions,
                             categories=list(predictor.categories), routes=route_map)
        decision = select_threshold([r['category'] for r in threshold_rows],
            [p.score for p in predictions], [p.label for p in predictions], route_map, config,
            model_version=version, artifact_sha256=model_hash, calibration=metadata,
            split_role='validation_threshold')
        decision.update(economic_config_sha256=_hash(_read(value_config)),
                        data_manifest_sha256=data_hash, routes_sha256=_hash(_read(routes)))
        outputs[version] = (calibration, decision)
    freeze = {'schema_version': 1, 'data_manifest_sha256': data_hash,
              'value_config_sha256': _hash(_read(value_config)),
              'routes_sha256': _hash(_read(routes)), 'source': _source(), 'models': {}}
    out.mkdir(parents=True)
    for version, (calibration, decision) in outputs.items():
        folder = out / version
        folder.mkdir()
        _write(folder / 'calibration.json', calibration)
        _write(folder / 'decision.json', decision)
        freeze['models'][version] = {'manifest_sha256': models[version][1],
            'calibration_sha256': _hash(_read(folder / 'calibration.json')),
            'decision_sha256': _hash(_read(folder / 'decision.json'))}
    _write(out / 'freeze.json', freeze)
    return freeze


def _counts(labels, predictions, threshold, routes):
    counts = dict.fromkeys(COUNT_PARAMETERS, 0)
    for gold, prediction in zip(labels, predictions, strict=True):
        if prediction.score < threshold:
            counts['reviewed'] += 1
        elif gold == prediction.label:
            counts['accepted_correct'] += 1
        else:
            counts['accepted_wrong'] += 1
            counts['accepted_wrong_route'] += routes[gold] != routes[prediction.label]
    return counts


def evaluate(data, artifacts, value_config, routes, out, *, device='cpu', bootstrap=1000):
    """Verify the freeze, then read tests and report fixed-decision outcomes."""
    data, value_config, routes, out = map(Path, (data, value_config, routes, out))
    artifacts = list(map(Path, artifacts))
    _output_location(out, data, artifacts, value_config, routes)
    if isinstance(bootstrap, bool) or not isinstance(bootstrap, int) or bootstrap < 1:
        raise ValueError('bootstrap must be positive')
    if (out / 'evaluation.json').exists():
        raise FileExistsError('Evaluation already exists')
    freeze_content = _read(out / 'freeze.json')
    freeze = json.loads(freeze_content)
    for path, key in [(data / 'manifest.json', 'data_manifest_sha256'),
                      (value_config, 'value_config_sha256'), (routes, 'routes_sha256')]:
        if _hash(_read(path)) != freeze.get(key):
            raise ValueError('Frozen input hash mismatch')
    if _source()['python_sha256'] != freeze['source']['python_sha256']:
        raise ValueError('Source code changed since prepare')
    data_hash, hashes, catalog, config, assumptions, route_map, models = _inputs(
        data, artifacts, value_config, routes, device)
    if set(models) != set(freeze['models']):
        raise ValueError('Model set changed since prepare')
    decisions, calibrated = {}, {}
    for version, (predictor, model_hash, _) in models.items():
        entry = freeze['models'][version]
        if model_hash != entry['manifest_sha256']:
            raise ValueError('Frozen model hash mismatch')
        folder = out / version
        for name in ['calibration', 'decision']:
            if _hash(_read(folder / f'{name}.json')) != entry[f'{name}_sha256']:
                raise ValueError('Frozen decision/calibration hash mismatch')
        decision = _json(folder / 'decision.json')
        bound = bind_decision(decision, model_version=version, artifact_sha256=model_hash)
        if decision.get('economic_config_sha256') != freeze['value_config_sha256']:
            raise ValueError('Decision economics mismatch')
        temperature = load_calibration(folder / 'calibration.json', calibration=bound.calibration,
                                      model_version=version, artifact_sha256=model_hash)
        calibrated[version] = predictor.with_calibration(temperature, bound.calibration_sha256)
        decisions[version] = bound
    # No test bytes have been read above this barrier.
    validation_rows = [*_rows(data, 'validation_calibration', hashes, catalog),
                       *_rows(data, 'validation_threshold', hashes, catalog)]
    test_rows = {role: _rows(data, role, hashes, catalog)
                 for role in ['test_official', 'test_unique']}
    for rows in test_rows.values():
        _disjoint(validation_rows, rows)
    unique = {r['group_id']: r for r in test_rows['test_unique']}
    expected = {}
    for row in test_rows['test_official']:
        expected.setdefault(row['group_id'], row)
    if len(unique) != len(test_rows['test_unique']) or set(unique) != set(expected) or any(
        any(unique[g][k] != expected[g][k] for k in ['id', 'text', 'category']) for g in expected
    ):
        raise ValueError('test_unique must preserve first official row per group')
    report = {'schema_version': 1, 'estimated': True, 'currency': config['units']['currency'],
              'cost_scope': 'batch', 'freeze_sha256': _hash(freeze_content),
              'provenance': freeze, 'splits': {}}
    categories = sorted(catalog, key=catalog.get)
    for role, rows in test_rows.items():
        labels = [r['category'] for r in rows]
        predictions = {v: p.predict([r['text'] for r in rows]) for v, p in calibrated.items()}
        result = {'n_examples': len(rows), 'models': {}}
        for version, outputs in predictions.items():
            metrics = evaluate_predictions(labels, outputs, categories=categories, routes=route_map)
            counts = _counts(labels, outputs, decisions[version].threshold, route_map)
            value = compute_value(**counts, assumptions=assumptions)
            value['selective_route_error'] = (counts['accepted_wrong_route'] / value['accepted']
                                               if value['accepted'] else None)
            result['models'][version] = {'metrics': metrics, 'threshold': decisions[version].threshold,
                'value': value,
                'sensitivity': sensitivity_analysis(counts, assumptions)}
        v1 = [v for v in predictions if re.search(r'(^|[-_.])v1($|[-_.])', v, re.I)]
        v2 = [v for v in predictions if re.search(r'(^|[-_.])v2($|[-_.])', v, re.I)]
        if len(v1) == len(v2) == 1:
            result['paired_v2_minus_v1'] = paired_classification_bootstrap(
                [r['group_id'] for r in rows], labels,
                [p.label for p in predictions[v1[0]]], [p.label for p in predictions[v2[0]]],
                categories=categories, n_boot=bootstrap)
            # Equal batch costs cancel in the paired difference. Bootstrap case
            # contributions conditional on these fixed assumed economics only.
            per_case_assumptions = {**assumptions, 'costo_inferencia': 0., 'costo_operacion': 0.}
            contributions = {}
            for version in [v1[0], v2[0]]:
                contributions[version] = [compute_value(
                    **_counts([gold], [prediction], decisions[version].threshold, route_map),
                    assumptions=per_case_assumptions)['valor_neto']
                    for gold, prediction in zip(labels, predictions[version], strict=True)]
            paired = paired_bootstrap([r['group_id'] for r in rows],
                contributions[v1[0]], contributions[v2[0]], n_boot=bootstrap)
            result['paired_estimated_value_v2_minus_v1'] = {
                'estimated': True, 'currency': config['units']['currency'],
                'interpretation': 'Sampling uncertainty conditional on fixed estimated costs; not measured commercial savings',
                'mean_per_message': paired,
                'batch': {'N': len(rows), 'delta': paired['delta'] * len(rows),
                          'ci95': [x * len(rows) for x in paired['ci95']]}}
        report['splits'][role] = result
    _write(out / 'evaluation.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['prepare', 'evaluate'], required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--artifacts', type=Path, nargs='+', required=True)
    parser.add_argument('--value-config', type=Path, required=True)
    parser.add_argument('--routes', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--bootstrap', type=int, default=1000)
    args = parser.parse_args(argv)
    try:
        operation = prepare if args.stage == 'prepare' else evaluate
        kwargs = {'device': args.device}
        if args.stage == 'evaluate':
            kwargs['bootstrap'] = args.bootstrap
        operation(args.data, args.artifacts, args.value_config, args.routes, args.out, **kwargs)
    except (OSError, ValueError, KeyError, TypeError, csv.Error):
        print(json.dumps({'error': 'finalization_failed', 'stage': args.stage}), file=sys.stderr)
        return 2
    print(json.dumps({'stage': args.stage, 'status': 'complete'}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
