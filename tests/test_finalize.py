"""Freeze validation choices before test access; keep only aggregate outputs."""
import csv
import hashlib
import importlib
import json
from pathlib import Path

import pytest

from reclamoruta.predictor import Predictor, write_manifest
from reclamoruta.value import ASSUMPTION_PARAMETERS


def module():
    assert importlib.util.find_spec('reclamoruta.finalize') is not None, 'finalization CLI missing'
    return importlib.import_module('reclamoruta.finalize')


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    data = tmp_path / 'data'
    data.mkdir()
    catalog = {'a': 0, 'b': 1}
    (data / 'label_to_id.json').write_text(json.dumps(catalog))
    roles = ['validation_calibration', 'validation_threshold', 'test_official', 'test_unique']
    for role in roles:
        with (data / f'{role}.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['id', 'text', 'category', 'group_id', 'split'])
            writer.writeheader()
            for index, label in enumerate(catalog):
                group = f'{role}-{index}' if not role.startswith('test') else f'test-{index}'
                writer.writerow(dict(id=group, text=f'PRIVATE {group}', category=label,
                                     group_id=group, split=role))
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in data.iterdir()}
    (data / 'manifest.json').write_text(json.dumps({'output_sha256': hashes}))
    data_hash = hashlib.sha256((data / 'manifest.json').read_bytes()).hexdigest()
    artifacts = []
    for name in ['v1', 'v2']:
        artifact = tmp_path / name
        artifact.mkdir()
        (artifact / 'label_to_id.json').write_text(json.dumps(catalog))
        write_manifest(artifact, name, 'fake', data_hash)
        artifacts.append(artifact)
    assumptions = dict.fromkeys(ASSUMPTION_PARAMETERS, 1.)
    assumptions.update(t_manual=30, t_aceptar=5, t_corregir=60, t_revision=30, costo_hora=36)
    config = {'status': 'validated', 'assumptions_version': 'estimated-v1',
              'units': {'time': 'seconds', 'cost_scope': 'batch', 'currency': 'GTQ'},
              'assumptions': assumptions,
              'evidence': {k: {'estimated': True, 'source': 'academic scenario'} for k in assumptions}}
    value = tmp_path / 'value.json'
    value.write_text(json.dumps(config))
    routes = tmp_path / 'routes.json'
    routes.write_text(json.dumps({'version': '1', 'status': 'validated',
        'routes': {'r': {'name': 'Route', 'description': 'Academic route'}},
        'intents': {'a': 'r', 'b': 'r'}}))
    return data, artifacts, value, routes, tmp_path / 'out'


def install_predictor(monkeypatch):
    final = module()
    def load(path, device='cpu'):
        from reclamoruta.predictor import read_manifest
        manifest = read_manifest(path)
        return Predictor(manifest['model_version'], ['a', 'b'],
                         lambda texts: [[.9, .1] if text.endswith('0') else [.1, .9] for text in texts])
    monkeypatch.setattr(final.Predictor, 'load', load)
    monkeypatch.setattr(final, 'fit_temperature', lambda rows, gold: {'temperature': 1., 'n_examples': len(gold)})
    return final


def test_prepare_freezes_without_reading_test_then_evaluate(inputs, monkeypatch):
    final = install_predictor(monkeypatch)
    data, artifacts, value, routes, out = inputs
    original = Path.read_bytes
    def forbid_test(path):
        assert not path.name.startswith('test_'), 'prepare read held-out test'
        return original(path)
    with monkeypatch.context() as guard:
        guard.setattr(Path, 'read_bytes', forbid_test)
        final.prepare(data, artifacts, value, routes, out)
    freeze_bytes = (out / 'freeze.json').read_bytes()
    monkeypatch.setattr(final, 'fit_temperature', lambda *a: pytest.fail('test refitted calibration'))
    monkeypatch.setattr(final, 'select_threshold', lambda *a, **kw: pytest.fail('test selected threshold'))
    report = final.evaluate(data, artifacts, value, routes, out, bootstrap=5)
    assert report['splits']['test_official']['models']['v1']['metrics']['accuracy'] == 1.
    assert report['splits']['test_unique']['paired_v2_minus_v1']['n_boot'] == 5
    assert report['splits']['test_unique']['paired_estimated_value_v2_minus_v1']['mean_per_message']['delta'] == 0
    assert (out / 'freeze.json').read_bytes() == freeze_bytes
    assert 'PRIVATE' not in ''.join(p.read_text() for p in out.rglob('*.json'))
    with pytest.raises(FileExistsError):
        final.evaluate(data, artifacts, value, routes, out, bootstrap=5)
    with pytest.raises(FileExistsError):
        final.prepare(data, artifacts, value, routes, out)


@pytest.mark.parametrize('target', ['value', 'routes', 'model', 'decision', 'calibration', 'data'])
def test_tampering_rejected_before_test_read(inputs, monkeypatch, target):
    final = install_predictor(monkeypatch)
    data, artifacts, value, routes, out = inputs
    final.prepare(data, artifacts, value, routes, out)
    path = {'value': value, 'routes': routes, 'model': artifacts[0] / 'label_to_id.json',
            'decision': out / 'v1' / 'decision.json', 'calibration': out / 'v1' / 'calibration.json',
            'data': data / 'manifest.json'}[target]
    path.write_bytes(path.read_bytes() + b' ')
    original = Path.read_bytes
    def forbid_test(path):
        assert not path.name.startswith('test_')
        return original(path)
    monkeypatch.setattr(Path, 'read_bytes', forbid_test)
    with pytest.raises(ValueError):
        final.evaluate(data, artifacts, value, routes, out, bootstrap=5)


def test_wrong_split_role_rejected_even_with_updated_hash(inputs, monkeypatch):
    final = install_predictor(monkeypatch)
    data, artifacts, value, routes, out = inputs
    path = data / 'validation_threshold.csv'
    path.write_bytes(path.read_bytes().replace(b'validation_threshold', b'test_official'))
    manifest = json.loads((data / 'manifest.json').read_text())
    manifest['output_sha256'][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (data / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='split'):
        final.prepare(data, artifacts, value, routes, out)
    assert not out.exists()


@pytest.mark.parametrize('target', ['validation_calibration.csv', 'validation_threshold.csv', 'label_to_id.json'])
def test_prepare_rejects_modified_consumed_bytes(inputs, monkeypatch, target):
    final = install_predictor(monkeypatch)
    data, artifacts, value, routes, out = inputs
    path = data / target
    path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises(ValueError, match='hash'):
        final.prepare(data, artifacts, value, routes, out)
    assert not out.exists()


def test_prepare_rejects_pending_economic_template(inputs, monkeypatch):
    final = install_predictor(monkeypatch)
    data, artifacts, value, routes, out = inputs
    config = json.loads(value.read_text())
    config['status'] = 'pending'
    value.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='status'):
        final.prepare(data, artifacts, value, routes, out)
    assert not out.exists()


def test_rejects_model_from_another_data_manifest(inputs, monkeypatch):
    final = install_predictor(monkeypatch)
    data, artifacts, value, routes, out = inputs
    write_manifest(artifacts[0], 'v1', 'fake', 'f' * 64)
    with pytest.raises(ValueError, match='another data'):
        final.prepare(data, artifacts, value, routes, out)
    assert not out.exists()


def test_test_data_hash_is_checked_after_freeze(inputs, monkeypatch):
    final = install_predictor(monkeypatch)
    data, artifacts, value, routes, out = inputs
    final.prepare(data, artifacts, value, routes, out)
    path = data / 'test_official.csv'
    path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises(ValueError, match='hash'):
        final.evaluate(data, artifacts, value, routes, out, bootstrap=5)
    assert not (out / 'evaluation.json').exists()


def test_evaluate_requires_prepare_and_cli_reports_safe_failure(inputs, monkeypatch, capsys):
    final = install_predictor(monkeypatch)
    data, artifacts, value, routes, out = inputs
    with pytest.raises(FileNotFoundError):
        final.evaluate(data, artifacts, value, routes, out, bootstrap=5)
    args = ['--data', str(data), '--artifacts', *map(str, artifacts), '--value-config', str(value),
            '--routes', str(routes), '--out', str(out)]
    assert final.main(['--stage', 'evaluate', *args]) == 2
    assert json.loads(capsys.readouterr().err)['error'] == 'finalization_failed'
    assert final.main(['--stage', 'prepare', *args]) == 0
    assert final.main(['--stage', 'evaluate', '--bootstrap', '5', *args]) == 0


def test_source_changes_rejected_before_test(inputs, monkeypatch):
    final = install_predictor(monkeypatch)
    data, artifacts, value, routes, out = inputs
    final.prepare(data, artifacts, value, routes, out)
    source = final._source()
    source['python_sha256']['finalize.py'] = '0' * 64
    monkeypatch.setattr(final, '_source', lambda: source)
    original = Path.read_bytes
    def forbid_test(path):
        assert not path.name.startswith('test_')
        return original(path)
    monkeypatch.setattr(Path, 'read_bytes', forbid_test)
    with pytest.raises(ValueError, match='Source'):
        final.evaluate(data, artifacts, value, routes, out, bootstrap=5)


@pytest.mark.parametrize('parent', ['data', 'model'])
def test_output_cannot_mutate_immutable_inputs(inputs, monkeypatch, parent):
    final = install_predictor(monkeypatch)
    data, artifacts, value, routes, _ = inputs
    root = data if parent == 'data' else artifacts[0]
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    out = root / 'new-results'
    with pytest.raises(ValueError, match='overlap'):
        final.prepare(data, artifacts, value, routes, out)
    assert not out.exists()
    assert before == {p.name: p.read_bytes() for p in root.iterdir()}
