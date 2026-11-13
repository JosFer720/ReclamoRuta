"""Temperature fitting and provenance-bound calibration CLI."""
import csv
import hashlib
import json
import math

import pytest

from reclamoruta import calibration
from reclamoruta.contracts import ContractError, load_calibration
from reclamoruta.predictor import write_manifest


def test_fits_known_temperature_and_never_worsens_nll():
    # 75% correct at confidence .9: calibrated probability should be .75.
    result = calibration.fit_temperature([[.9, .1]] * 4, [0, 0, 0, 1])
    assert result['temperature'] == pytest.approx(2., rel=1e-6)
    assert result['nll_before'] == pytest.approx(-(.75 * math.log(.9) + .25 * math.log(.1)))
    assert result['nll_after'] < result['nll_before']
    assert result['n_examples'] == 4
    assert result == calibration.fit_temperature([[.9, .1]] * 4, [0, 0, 0, 1])


@pytest.mark.parametrize('rows,gold', [([], []), ([[]], [0]), ([[.5, .5], [1]], [0, 0]),
    ([[.4, .4]], [0]), ([[float('nan'), 1]], [0]), ([[float('inf'), 0]], [0]),
    ([[-.1, 1.1]], [0]), ([[True, 0]], [0]), ([[.5, .5]], [True]),
    ([[.5, .5]], [1.0]), ([[.5, .5]], [2]), ([[.5, .5]], []),
    ([['.5', .5]], [0])])
def test_rejects_invalid_inputs(rows, gold):
    with pytest.raises(ValueError):
        calibration.fit_temperature(rows, gold)


def test_zero_and_uniform_probabilities_are_stable():
    result = calibration.fit_temperature([[0., 1.], [.5, .5]], [0, 1])
    assert math.isfinite(result['nll_after'])
    assert result['nll_after'] <= result['nll_before']
    assert calibration.fit_temperature([[.5, .5]], [0])['temperature'] == 1.


@pytest.fixture
def calibration_inputs(tmp_path):
    data, artifact = tmp_path / 'data', tmp_path / 'artifact'
    data.mkdir()
    artifact.mkdir()
    catalog = {'a': 0, 'b': 1}
    for directory in (data, artifact):
        (directory / 'label_to_id.json').write_text(json.dumps(catalog))
    with (data / 'validation_calibration.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['text', 'category', 'split'])
        writer.writerows([['private sample', 'a', 'validation_calibration']] * 33)
    def refresh():
        hashes = {name: hashlib.sha256((data / name).read_bytes()).hexdigest()
                  for name in ['validation_calibration.csv', 'label_to_id.json']}
        (data / 'manifest.json').write_text(json.dumps({'output_sha256': hashes}))
        digest = hashlib.sha256((data / 'manifest.json').read_bytes()).hexdigest()
        write_manifest(artifact, 'fixture', 'baseline', digest)
    refresh()
    return data, artifact, refresh


def test_cli_fits_only_verified_calibration_split(calibration_inputs, monkeypatch, tmp_path):
    data, artifact, _ = calibration_inputs
    batches = []
    class Scorer:
        model_version = 'fixture'
        categories = ['a', 'b']
        def _probabilities(self, texts):
            batches.append(len(texts))
            return [[.6, .4] for _ in texts]
    monkeypatch.setattr(calibration.Predictor, 'load', lambda *a, **kw: Scorer())
    out = tmp_path / 'nested' / 'calibration.json'
    calibration.main(['--artifact', str(artifact), '--data', str(data), '--out', str(out)])
    result = json.loads(out.read_bytes())
    assert result['split_role'] == 'validation_calibration'
    assert result['n_examples'] == 33
    assert batches == [32, 1]
    assert 'private sample' not in out.read_text()
    assert load_calibration(out, calibration={'method': 'temperature', 'artifact_sha256':
        hashlib.sha256(out.read_bytes()).hexdigest()}, model_version='fixture',
        artifact_sha256=hashlib.sha256((artifact / 'manifest.json').read_bytes()).hexdigest()) > 0
    with pytest.raises(FileExistsError):
        calibration.main(['--artifact', str(artifact), '--data', str(data), '--out', str(out)])


@pytest.mark.parametrize('change', ['split', 'unknown', 'tampered', 'manifest', 'catalog'])
def test_rejects_bad_provenance_before_model_load(calibration_inputs, monkeypatch, tmp_path, change):
    data, artifact, refresh = calibration_inputs
    path = data / 'validation_calibration.csv'
    if change == 'split':
        path.write_text(path.read_text().replace('validation_calibration', 'test_official'))
        refresh()
    elif change == 'unknown':
        path.write_text(path.read_text().replace(',a,', ',missing,'))
        refresh()
    elif change == 'tampered':
        path.write_text(path.read_text() + '\n')
    elif change == 'manifest':
        with (data / 'manifest.json').open('a') as stream:
            stream.write(' ')
    else:
        (artifact / 'label_to_id.json').write_text(json.dumps({'b': 0, 'a': 1}))
        digest = hashlib.sha256((data / 'manifest.json').read_bytes()).hexdigest()
        write_manifest(artifact, 'fixture', 'baseline', digest)
    def forbid(*args, **kwargs):
        pytest.fail('invalid data must be rejected before loading model')
    monkeypatch.setattr(calibration.Predictor, 'load', forbid)
    with pytest.raises(ContractError):
        calibration.main(['--artifact', str(artifact), '--data', str(data),
                          '--out', str(tmp_path / 'result.json')])


def test_fitted_nll_matches_predictor_scaling():
    from reclamoruta.predictor import Predictor
    rows = [[.95, .04, .01], [0., .2, .8], [.8, .1, .1]]
    gold = [1, 1, 0]
    result = calibration.fit_temperature(rows, gold)
    predictor = Predictor('fixture', ['a', 'b', 'c'], None, result['temperature'])
    scaled = [predictor._scale(row) for row in rows]
    assert result['nll_after'] == pytest.approx(
        -sum(math.log(row[target]) for row, target in zip(scaled, gold)) / 3)
    assert [max(range(3), key=row.__getitem__) for row in scaled] == [0, 2, 0]


def test_rejects_output_inside_immutable_inputs(calibration_inputs):
    data, artifact, _ = calibration_inputs
    with pytest.raises(ContractError):
        calibration.calibrate(artifact, data, artifact / 'calibration.json')
    assert not (artifact / 'calibration.json').exists()
