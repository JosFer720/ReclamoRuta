import json
from pathlib import Path

import pytest

from scripts.analyze_economics import calculate_case, validate_cost_config, write_new_json, verify_inputs


COSTS = dict(electricity_gtq_kwh=1.51, cpu_nominal_watts=65, gpu_nominal_watts=300,
             training_host_other_watts=0, inference_host_other_watts=0,
             inference_capital_gtq=10000, training_capital_gtq=10000,
             lifetime_months=36, training_hours_per_month=160,
             messages_per_month=10000, non_compute_monthly_gtq=0)
VALUES = dict(t_manual=30, t_aceptar=5, t_corregir=20, t_revision=35,
              costo_hora=40, q_detectado=.8, costo_error_ruta_no_detectado=25,
              costo_inferencia=0, costo_operacion=0)
COUNTS = dict(accepted_correct=8, accepted_wrong=2, accepted_wrong_route=1, reviewed=2)


@pytest.mark.parametrize('q', [.5, .8, 1])
def test_detection_reuses_counts_and_hand_calculation(q):
    row = calculate_case(COUNTS, VALUES, COSTS, q=q, mean_ms=10, train_seconds=3600, training_device='gpu')
    assert row['riesgo_residual'] == pytest.approx((1-q)*25)
    assert row['T_assist'] == 120 + q*40
    assert row['labor_value_gtq'] == pytest.approx((360-row['T_assist'])/3600*40)
    assert row['accepted_wrong_route'] == 1


def test_time_units_capital_and_training_charged_once():
    row = calculate_case(COUNTS, VALUES, COSTS, q=.8, mean_ms=300000, train_seconds=3600, training_device='gpu')
    assert row['inference_hours'] == 1
    assert row['inference_energy_equivalent_nominal_kwh'] == .065
    assert row['training_energy_equivalent_nominal_kwh'] == .3
    assert row['inference_capital_gtq'] == pytest.approx(10000/36/10000*12)
    assert row['training_capital_gtq'] == pytest.approx(10000/36/160)
    assert row['recurring_value_gtq']-row['first_batch_value_gtq'] == pytest.approx(row['training_total_gtq'])
    assert row['monthly_recurring_value_gtq'] == pytest.approx(row['recurring_value_gtq']*10000/12)


def test_negative_recurring_value_is_preserved():
    row = calculate_case(COUNTS, VALUES, {**COSTS, 'non_compute_monthly_gtq':1e9}, q=1, mean_ms=1, train_seconds=1, training_device='cpu')
    assert row['recurring_value_gtq'] < 0


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -1, True])
def test_reject_invalid_costs(bad):
    with pytest.raises(ValueError):
        validate_cost_config({**COSTS, 'electricity_gtq_kwh':bad})


def test_never_overwrites_output(tmp_path):
    path = tmp_path/'out.json'
    path.write_text('original')
    with pytest.raises(FileExistsError):
        write_new_json(path, {})
    assert path.read_text() == 'original'


def test_modified_freeze_is_rejected(tmp_path):
    (tmp_path/'freeze.json').write_text('{}')
    (tmp_path/'evaluation.json').write_text(json.dumps({'freeze_sha256':'wrong'}))
    with pytest.raises(ValueError, match='freeze'):
        verify_inputs(tmp_path, tmp_path/'config.json', [], [])


def test_fractional_monthly_projection_keeps_case_mix():
    costs = {**COSTS, 'messages_per_month':1000}
    row = calculate_case(COUNTS, VALUES, costs, q=.5, mean_ms=10, train_seconds=3600, training_device='gpu')
    assert row['monthly_recurring_value_gtq'] == pytest.approx(row['recurring_value_gtq']*1000/12)


def test_extra_training_runs_do_not_change_recurring_cost():
    once = calculate_case(COUNTS, VALUES, COSTS, q=.8, mean_ms=10, train_seconds=3600, training_device='gpu')
    twice = calculate_case(COUNTS, VALUES, {**COSTS, 'training_runs':2}, q=.8, mean_ms=10, train_seconds=3600, training_device='gpu')
    assert twice['training_total_gtq'] == pytest.approx(2*once['training_total_gtq'])
    assert twice['recurring_value_gtq'] == once['recurring_value_gtq']


def test_training_break_even_uses_incremental_recurring_benefit():
    from scripts.analyze_economics import comparison
    new = dict(N=100, recurring_value_gtq=30, first_batch_value_gtq=10,
               training_total_gtq=20, labor_value_gtq=40, riesgo_residual=2,
               monthly_recurring_value_gtq=3000)
    old = dict(N=100, recurring_value_gtq=20, first_batch_value_gtq=15,
               training_total_gtq=5, labor_value_gtq=30, riesgo_residual=4,
               monthly_recurring_value_gtq=2000)
    delta = comparison(new, old)
    assert delta['requests_to_recover_extra_training'] == 150
    assert delta['maximum_extra_recurring_cost_gtq'] == 12
    assert comparison(old, new)['requests_to_recover_extra_training'] is None


@pytest.fixture
def real_inputs(tmp_path):
    """Small synthetic aggregate files with the production SHA-256 bindings."""
    from scripts.analyze_economics import digest
    from reclamoruta.value import compute_value
    evaluation = tmp_path/'evaluation'
    evaluation.mkdir()
    config = tmp_path/'value.json'
    config.write_text(json.dumps(dict(status='validated', assumptions_version='test',
        units=dict(time='seconds', currency='GTQ', cost_scope='batch'), assumptions=VALUES,
        evidence={k:dict(estimated=True, source='Conditional calculation parameter') for k in VALUES})))
    artifacts, benchmarks = [], []
    freeze = dict(models={}, value_config_sha256=digest(config))
    models = {}
    for version in ('baseline-tfidf', 'distilbert-v1', 'distilbert-v2'):
        artifact = tmp_path/version
        artifact.mkdir()
        metrics = dict(model_version=version, fit_seconds=1, train_seconds=227.1692948,
                       device='NVIDIA GeForce RTX 5070 Ti')
        (artifact/'metrics.json').write_text(json.dumps(metrics))
        manifest = dict(model_version=version, kind='baseline' if version=='baseline-tfidf' else 'distilbert',
                        files={'metrics.json':digest(artifact/'metrics.json')})
        (artifact/'manifest.json').write_text(json.dumps(manifest))
        artifacts.append(artifact)
        frozen_model = evaluation/version
        frozen_model.mkdir()
        (frozen_model/'decision.json').write_text(json.dumps(dict(threshold=.8)))
        (frozen_model/'calibration.json').write_text('{}')
        freeze['models'][version] = dict(manifest_sha256=digest(artifact/'manifest.json'),
            decision_sha256=digest(frozen_model/'decision.json'),
            calibration_sha256=digest(frozen_model/'calibration.json'))
        benchmark = tmp_path/f'{version}-benchmark.json'
        benchmark.write_text(json.dumps(dict(model_version=version, device='cpu', batch_size=1,
            n_requests=200, latency_ms=dict(mean=10), startup_ms=100, warmup_requests=10)))
        benchmarks.append(benchmark)
        models[version] = dict(threshold=.8, value=compute_value(**COUNTS, assumptions=VALUES))
    (evaluation/'freeze.json').write_text(json.dumps(freeze))
    report = dict(freeze_sha256=digest(evaluation/'freeze.json'), provenance=freeze,
                  splits={name:dict(n_examples=12, models=models) for name in ('test_official','test_unique')})
    (evaluation/'evaluation.json').write_text(json.dumps(report))
    return evaluation, config, artifacts, benchmarks


def test_real_frozen_inputs_pass_without_loading_weights(real_inputs):
    report, _, _, artifacts, benchmarks = verify_inputs(*real_inputs)
    assert artifacts['distilbert-v2']['train_seconds'] == 227.1692948
    assert benchmarks['baseline-tfidf']['mean_ms'] > 0
    assert len(report['splits']) == 2


def test_modified_evaluation_counts_are_rejected(real_inputs, tmp_path):
    import shutil
    evaluation, config, artifacts, benchmarks = real_inputs
    shutil.copytree(evaluation, tmp_path/'changed-evaluation')
    path = tmp_path/'changed-evaluation/evaluation.json'
    report = json.loads(path.read_text())
    report['splits']['test_official']['models']['distilbert-v2']['value']['accepted_correct'] += 1
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='counts|inconsistency'):
        verify_inputs(tmp_path/'changed-evaluation', config, artifacts, benchmarks)


def test_modified_training_metrics_are_rejected(real_inputs, tmp_path):
    import shutil
    evaluation, config, artifacts, benchmarks = real_inputs
    baseline = tmp_path/'baseline'
    baseline.mkdir()
    for name in ('manifest.json', 'metrics.json'):
        shutil.copy(artifacts[0]/name, baseline/name)
    (baseline/'metrics.json').write_text('{}')
    with pytest.raises(ValueError, match='metrics integrity'):
        verify_inputs(evaluation, config, [baseline, *artifacts[1:]], benchmarks)


def test_output_cannot_mutate_frozen_inputs(tmp_path):
    from scripts.analyze_economics import validate_output_location
    artifact = tmp_path/'artifact'
    artifact.mkdir()
    with pytest.raises(ValueError, match='Output'):
        validate_output_location(artifact/'new.json', [artifact], [])
    assert not (artifact/'new.json').exists()


def test_capital_volume_sensitivity_has_nine_continuous_cases(real_inputs):
    from scripts.analyze_economics import capital_volume_sensitivity
    rows = capital_volume_sensitivity(COUNTS, VALUES, COSTS, q=.8, mean_ms=10, train_seconds=3600, training_device='gpu')
    assert len(rows) == 9
    assert {(r['equipment_capital_gtq'],r['messages_per_month']) for r in rows} == {(c,n) for c in (5000,10000,15000) for n in (1000,10000,100000)}
    assert rows[0]['monthly_recurring_value_gtq'] == pytest.approx(rows[0]['recurring_value_gtq']*1000/12)
