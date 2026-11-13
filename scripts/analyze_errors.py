"""Describe frozen test_unique errors without retaining text or row predictions."""
import argparse
import json
import math
from pathlib import Path
import re
import sys

from reclamoruta.contracts import bind_decision, load_calibration, load_routes
from reclamoruta.evaluate import evaluate_predictions
from reclamoruta.finalize import (_data, _hash, _json, _output_location, _read,
                                  _rows, _source, _write)
from reclamoruta.predictor import Predictor, read_manifest

WORD_PATTERN = r"[A-Za-z0-9]+(?:['\u2019][A-Za-z0-9]+)*"
NEGATIONS = ('no', 'not', 'never', 'cannot', "can't", "don't", "doesn't", "didn't",
             "won't", "isn't", "aren't", "wasn't", "weren't", "haven't", "hasn't",
             "hadn't", "couldn't", "wouldn't", "shouldn't", "mustn't", "needn't")
LENGTH_GROUPS = ['words_0_5', 'words_6_10', 'words_11_20', 'words_21_plus']
STYLES = ['question', 'digit', 'negation', 'first_person']
GROUPS = LENGTH_GROUPS + [f'{name}_{value}' for name in STYLES for value in ['yes', 'no']]
DEFINITIONS = {
    'word_pattern': WORD_PATTERN, 'length_inclusive_bounds': [[0, 5], [6, 10], [11, 20], [21, None]],
    'question': 'ASCII ? present', 'digit': 'ASCII 0-9 present',
    'negation': list(NEGATIONS), 'first_person': ['i', 'my', 'me'],
    'matching': 'Whole regex tokens, case-insensitive; curly apostrophe normalized to ASCII',
    'groups': GROUPS, 'ece': '10 equal-width bins of calibrated top-label confidence',
    'acceptance': 'calibrated score >= frozen validation threshold',
    'interpretation': 'Descriptive overlapping style groups; class mix confounds comparisons. No demographic or causal inference.',
}


def segment_names(text):
    tokens = re.findall(WORD_PATTERN, text.replace('\u2019', "'").lower())
    n = len(tokens)
    length = LENGTH_GROUPS[0 if n <= 5 else 1 if n <= 10 else 2 if n <= 20 else 3]
    flags = ['?' in text, bool(re.search(r'[0-9]', text)),
             bool(set(tokens) & set(NEGATIONS)), bool(set(tokens) & {'i', 'my', 'me'})]
    return [length] + [f'{name}_{"yes" if flag else "no"}' for name, flag in zip(STYLES, flags)]


def _group_metrics(labels, outputs, categories, routes, threshold):
    n = len(labels)
    errors = sum(g != p.label for g, p in zip(labels, outputs))
    accepted = [(g, p) for g, p in zip(labels, outputs) if p.score >= threshold]
    wrong = sum(g != p.label for g, p in accepted)
    wrong_route = sum(routes[g] != routes[p.label] for g, p in accepted)
    metrics = evaluate_predictions(labels, outputs, categories=categories, routes=routes) if n else {}
    return {'n': n, 'errors': errors, 'accuracy': metrics.get('accuracy'), 'ece': metrics.get('ece'),
            'route_accuracy': metrics.get('routing_accuracy'), 'accepted': len(accepted),
            'reviewed': n - len(accepted), 'accepted_wrong': wrong,
            'accepted_wrong_route': wrong_route, 'coverage': len(accepted) / n if n else None,
            'selective_error': wrong / len(accepted) if accepted else None,
            'selective_route_error': wrong_route / len(accepted) if accepted else None}


def summarize(rows, outputs, categories, routes, threshold):
    if not rows or len(rows) != len(outputs) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Invalid aligned analysis input')
    labels = [row['category'] for row in rows]
    metrics = evaluate_predictions(labels, outputs, categories=categories, routes=routes)
    confusions = []
    matrix = metrics['confusion_matrix']['matrix']
    for i, truth in enumerate(categories):
        support = sum(matrix[i])
        for j, predicted in enumerate(categories):
            if i != j and matrix[i][j]:
                confusions.append(dict(truth=truth, predicted=predicted, count=matrix[i][j],
                    truth_support=support, rate_given_truth=matrix[i][j] / support,
                    same_route=routes[truth] == routes[predicted]))
    confusions.sort(key=lambda row: (-row['count'], row['truth'], row['predicted']))
    worst = [dict(category=c, n=v['support'], recall=v['recall'], errors=v['fn'])
             for c, v in metrics['per_class'].items() if v['support']]
    worst.sort(key=lambda row: (row['recall'], -row['n'], row['category']))
    groups = {group: [] for group in GROUPS}
    for index, row in enumerate(rows):
        for group in segment_names(row['text']):
            groups[group].append(index)
    return {'overall': {**_group_metrics(labels, outputs, categories, routes, threshold),
                        'macro_f1': metrics['macro_f1'], 'macro_f1_catalog_size': len(categories),
                        'confusion_matrix': metrics['confusion_matrix']},
            'top_confusions': confusions[:15], 'worst_classes': worst[:15],
            'error_routes': {name: sum(r['count'] for r in confusions if r['same_route'] == same)
                             for name, same in [('same_route', True), ('different_route', False)]},
            'segments': {group: _group_metrics([labels[i] for i in indexes],
                [outputs[i] for i in indexes], categories, routes, threshold)
                for group, indexes in groups.items()}}


def analyze(data, artifacts, evaluation_dir, routes, out, *, device='cpu'):
    data, evaluation_dir, routes, out = map(Path, (data, evaluation_dir, routes, out))
    artifacts = list(map(Path, artifacts))
    _output_location(out, data, [*artifacts, evaluation_dir], routes, routes)
    if out.exists():
        raise FileExistsError('Output already exists')
    freeze_bytes = _read(evaluation_dir / 'freeze.json')
    freeze = json.loads(freeze_bytes)
    if _source()['python_sha256'] != freeze['source']['python_sha256']:
        raise ValueError('Source code changed since freeze')
    if _hash(_read(data / 'manifest.json')) != freeze['data_manifest_sha256']:
        raise ValueError('Frozen data hash mismatch')
    if _hash(_read(routes)) != freeze['routes_sha256']:
        raise ValueError('Frozen routes hash mismatch')
    data_hash, hashes, catalog = _data(data)
    categories = sorted(catalog, key=catalog.get)
    route_map = load_routes(routes, categories).intent_to_route
    verified = {}
    for artifact in artifacts:
        manifest_bytes = _read(artifact / 'manifest.json')
        manifest = read_manifest(artifact)  # verify files without loading executable model
        version = manifest['model_version']
        entry = freeze['models'].get(version)
        if entry is None or version in verified or _hash(manifest_bytes) != entry['manifest_sha256']:
            raise ValueError('Frozen model hash mismatch or duplicate model')
        if manifest['data_manifest_sha256'] != data_hash or _json(artifact / 'label_to_id.json') != catalog:
            raise ValueError('Model data/catalog mismatch')
        folder = evaluation_dir / version
        for name in ['decision', 'calibration']:
            if _hash(_read(folder / f'{name}.json')) != entry[f'{name}_sha256']:
                raise ValueError('Frozen decision/calibration hash mismatch')
        decision = _json(folder / 'decision.json')
        calibration = _json(folder / 'calibration.json')
        if any(decision.get(k) != freeze[k] for k in ['data_manifest_sha256', 'routes_sha256']):
            raise ValueError('Decision data/routes mismatch')
        if decision.get('economic_config_sha256') != freeze['value_config_sha256'] or calibration.get('data_manifest_sha256') != data_hash:
            raise ValueError('Decision/calibration binding mismatch')
        bound = bind_decision(decision, model_version=version, artifact_sha256=entry['manifest_sha256'])
        temperature = load_calibration(folder / 'calibration.json', calibration=bound.calibration,
                                      model_version=version, artifact_sha256=entry['manifest_sha256'])
        verified[version] = (artifact, bound, temperature)
    if not verified:
        raise ValueError('At least one model is required')
    evaluation_bytes = _read(evaluation_dir / 'evaluation.json')
    evaluation = json.loads(evaluation_bytes)
    # The current evaluation includes the recorded wording-only freeze revision.
    provenance = evaluation['provenance']
    if evaluation['freeze_sha256'] != _hash(freeze_bytes) or provenance != freeze:
        raise ValueError('Evaluation provenance mismatch')
    # All frozen non-test identities verified before reading held-out split bytes.
    rows = _rows(data, 'test_unique', hashes, catalog)
    if len({r['group_id'] for r in rows}) != len(rows):
        raise ValueError('test_unique must contain unique groups')
    result = {'schema_version': 1, 'split_role': 'test_unique', 'definitions': DEFINITIONS,
              'metadata': {'script_sha256': _hash(_read(Path(__file__))),
                'freeze_sha256': _hash(freeze_bytes), 'evaluation_sha256': _hash(evaluation_bytes),
                'data_manifest_sha256': data_hash, 'split_sha256': hashes['test_unique.csv'],
                'routes_sha256': freeze['routes_sha256'], 'models': {v: freeze['models'][v] for v in verified},
                'metadata_revision': freeze.get('metadata_revision')}, 'models': {}}
    for version, (artifact, bound, temperature) in verified.items():
        predictor = Predictor.load(artifact, device=device)
        if predictor.categories != categories or predictor.model_version != version:
            raise ValueError('Loaded model catalog/version mismatch')
        predictions = predictor.with_calibration(temperature, bound.calibration_sha256).predict([r['text'] for r in rows])
        summary = summarize(rows, predictions, categories, route_map, bound.threshold)
        reference = evaluation['splits']['test_unique']['models'][version]['metrics']
        for key in ['accuracy', 'macro_f1']:
            if not math.isclose(summary['overall'][key], reference[key], rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError('Metrics differ from frozen evaluation')
        if summary['overall']['confusion_matrix'] != reference['confusion_matrix']:
            raise ValueError('Confusion matrix differs from frozen evaluation')
        summary.update(threshold=bound.threshold, reproduces_frozen_metrics=True)
        result['models'][version] = summary
    out.parent.mkdir(parents=True, exist_ok=True)
    _write(out, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['data', 'evaluation-dir', 'routes', 'out']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--artifacts', type=Path, nargs='+', required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args(argv)
    try:
        analyze(args.data, args.artifacts, args.evaluation_dir, args.routes, args.out, device=args.device)
    except (OSError, ValueError, KeyError, TypeError):
        print(json.dumps({'error': 'error_analysis_failed'}), file=sys.stderr)
        return 2
    print(json.dumps({'status': 'complete'}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
