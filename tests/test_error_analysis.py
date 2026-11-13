"""Aggregate analysis checks, including pre-inference freeze barriers."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_finalize import inputs, install_predictor


def module():
    path = Path(__file__).parents[1] / 'scripts' / 'analyze_errors.py'
    assert path.exists(), 'error analysis CLI missing'
    spec = importlib.util.spec_from_file_location('error_analysis', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def prediction(label, score):
    return SimpleNamespace(label=label, score=score, top3=[], model_version='v1', elapsed_ms=0.)


def test_fixed_word_boundaries_and_independent_styles():
    m = module()
    for count, expected in [(5, 'words_0_5'), (6, 'words_6_10'), (10, 'words_6_10'),
                            (11, 'words_11_20'), (20, 'words_11_20'), (21, 'words_21_plus')]:
        assert expected in m.segment_names(' '.join(['word'] * count))
    names = m.segment_names("I CAN'T see my 2 payments?")
    assert {'question_yes', 'digit_yes', 'negation_yes', 'first_person_yes'} <= set(names)
    assert 'words_6_10' in names  # apostrophe contraction counts as one token
    assert 'negation_yes' in m.segment_names('It doesn’t work')
    assert 'negation_no' in m.segment_names('notebook nothing cannotary')
    assert 'first_person_no' in m.segment_names('mine timely')


def test_hand_calculated_confusions_segments_and_empty_rates():
    m = module()
    rows = [dict(text='I paid?', category='a'), dict(text='No 2', category='a'),
            dict(text='paid', category='b')]
    outputs = [prediction('a', .8), prediction('b', .6), prediction('a', .9)]
    result = m.summarize(rows, outputs, ['a', 'b'], {'a': 'r1', 'b': 'r2'}, .7)
    assert result['overall']['errors'] == 2
    assert result['overall']['accuracy'] == pytest.approx(1 / 3)
    assert result['overall']['ece'] == pytest.approx((.2 + .6 + .9) / 3)
    assert result['overall']['accepted'] == 2
    assert result['overall']['accepted_wrong'] == 1
    assert result['overall']['selective_error'] == .5
    assert result['top_confusions'][0] == dict(truth='a', predicted='b', count=1,
        truth_support=2, rate_given_truth=.5, same_route=False)
    assert result['worst_classes'][0]['category'] == 'b'
    assert result['segments']['question_yes']['n'] == 1
    assert result['segments']['words_21_plus']['accuracy'] is None
    assert result['error_routes'] == {'same_route': 0, 'different_route': 2}
    with pytest.raises(ValueError):
        m.summarize(rows, outputs[:1], ['a', 'b'], {'a': 'r1', 'b': 'r2'}, .7)


def ready(inputs, monkeypatch):
    final = install_predictor(monkeypatch)
    data, artifacts, value, routes, folder = inputs
    final.prepare(data, artifacts, value, routes, folder)
    final.evaluate(data, artifacts, value, routes, folder, bootstrap=2)
    return module(), data, artifacts, routes, folder, folder.parent / 'analysis.json'


def test_frozen_analysis_private_and_exclusive(inputs, monkeypatch):
    m, data, artifacts, routes, folder, out = ready(inputs, monkeypatch)
    result = m.analyze(data, artifacts, folder, routes, out)
    assert result['split_role'] == 'test_unique'
    assert result['models']['v1']['overall']['accuracy'] == 1.
    assert result['models']['v1']['reproduces_frozen_metrics'] is True
    assert 'PRIVATE' not in out.read_text()
    assert 'test-0' not in out.read_text()
    with pytest.raises(FileExistsError):
        m.analyze(data, artifacts, folder, routes, out)


@pytest.mark.parametrize('target', ['model', 'decision', 'calibration', 'data', 'routes', 'source', 'test'])
def test_tampering_rejected_before_inference(inputs, monkeypatch, target):
    m, data, artifacts, routes, folder, out = ready(inputs, monkeypatch)
    paths = {'model': artifacts[0] / 'label_to_id.json', 'decision': folder / 'v1' / 'decision.json',
             'calibration': folder / 'v1' / 'calibration.json', 'data': data / 'manifest.json',
             'routes': routes, 'test': data / 'test_unique.csv'}
    if target == 'source':
        freeze = json.loads((folder / 'freeze.json').read_text())
        freeze['source']['python_sha256']['data.py'] = '0' * 64
        (folder / 'freeze.json').write_text(json.dumps(freeze))
    else:
        path = paths[target]
        path.write_bytes(path.read_bytes() + b' ')
    monkeypatch.setattr(m.Predictor, 'load', lambda *a, **kw: pytest.fail('loaded before verification'))
    with pytest.raises(ValueError):
        m.analyze(data, artifacts, folder, routes, out)
    assert not out.exists()


def test_unrelated_evaluation_rejected_before_inference(inputs, monkeypatch):
    m, data, artifacts, routes, folder, out = ready(inputs, monkeypatch)
    path = folder / 'evaluation.json'
    report = json.loads(path.read_text())
    report['freeze_sha256'] = '0' * 64
    path.write_text(json.dumps(report))
    monkeypatch.setattr(m.Predictor, 'load', lambda *a, **kw: pytest.fail('loaded unrelated evaluation'))
    with pytest.raises(ValueError):
        m.analyze(data, artifacts, folder, routes, out)

@pytest.mark.parametrize('key', ['economic_config_sha256', 'data_manifest_sha256', 'routes_sha256'])
def test_resealed_decision_still_requires_frozen_bindings(inputs, monkeypatch, key):
    import hashlib
    m, data, artifacts, routes, folder, out = ready(inputs, monkeypatch)
    decision_path = folder / 'v1' / 'decision.json'
    decision = json.loads(decision_path.read_text())
    decision[key] = '0' * 64
    decision_path.write_text(json.dumps(decision))
    freeze_path = folder / 'freeze.json'
    freeze = json.loads(freeze_path.read_text())
    freeze['models']['v1']['decision_sha256'] = hashlib.sha256(decision_path.read_bytes()).hexdigest()
    freeze_path.write_text(json.dumps(freeze))
    evaluation_path = folder / 'evaluation.json'
    evaluation = json.loads(evaluation_path.read_text())
    evaluation['provenance'] = freeze
    evaluation['freeze_sha256'] = hashlib.sha256(freeze_path.read_bytes()).hexdigest()
    evaluation_path.write_text(json.dumps(evaluation))
    monkeypatch.setattr(m.Predictor, 'load', lambda *a, **kw: pytest.fail('loaded divergent decision'))
    with pytest.raises(ValueError, match='mismatch'):
        m.analyze(data, artifacts, folder, routes, out)
