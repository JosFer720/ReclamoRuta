"""Conditional economics over frozen aggregate counts; never selects thresholds.

Nominal component power is a calculation reference, not measured wall power or
an electrical upper bound. Startup and warmup are excluded from recurring time.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from numbers import Real
from pathlib import Path

from reclamoruta.value import COUNT_PARAMETERS, compute_value, validate_value_config


COST_KEYS = ('electricity_gtq_kwh', 'cpu_nominal_watts', 'gpu_nominal_watts',
             'inference_capital_gtq', 'training_capital_gtq', 'lifetime_months',
             'training_hours_per_month', 'messages_per_month', 'non_compute_monthly_gtq')


def number(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f'{name} must be numeric')
    value = float(value)
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f'{name} must be finite and nonnegative (positive for divisors)')
    return value


def validate_cost_config(config):
    result = {key: number(config[key], key, positive=key in (
        'lifetime_months', 'training_hours_per_month', 'messages_per_month')) for key in COST_KEYS}
    for key in ('training_host_other_watts', 'inference_host_other_watts'):
        result[key] = number(config.get(key, 0), key)
    result['training_runs'] = number(config.get('training_runs', 1), 'training_runs', positive=True)
    if not result['training_runs'].is_integer():
        raise ValueError('training_runs must be an integer')
    return result


def calculate_case(counts, values, cost_config, *, q, mean_ms, train_seconds, training_device):
    costs = validate_cost_config(cost_config)
    q = number(q, 'q')
    if q > 1:
        raise ValueError('q must be in [0,1]')
    mean_ms = number(mean_ms, 'mean_ms', positive=True)
    train_seconds = number(train_seconds, 'train_seconds')
    if training_device not in ('cpu', 'gpu'):
        raise ValueError('training_device must be cpu or gpu')
    row = compute_value(**counts, assumptions={**values, 'q_detectado':q,
                                              'costo_inferencia':0, 'costo_operacion':0})
    if row['N'] <= 0:
        raise ValueError('At least one case is required')
    train_hours = train_seconds / 3600 * costs['training_runs']
    inference_hours = row['N'] * mean_ms / 3_600_000
    inference_kwh = (costs['cpu_nominal_watts'] + costs['inference_host_other_watts']) / 1000 * inference_hours
    training_kwh = (costs[f'{training_device}_nominal_watts'] + costs['training_host_other_watts']) / 1000 * train_hours
    inference_energy = inference_kwh * costs['electricity_gtq_kwh']
    training_energy = training_kwh * costs['electricity_gtq_kwh']
    inference_capital = costs['inference_capital_gtq'] / costs['lifetime_months'] / costs['messages_per_month'] * row['N']
    training_capital = costs['training_capital_gtq'] / costs['lifetime_months'] / costs['training_hours_per_month'] * train_hours
    noncompute = costs['non_compute_monthly_gtq'] / costs['messages_per_month'] * row['N']
    legacy_inference = number(values['costo_inferencia'], 'costo_inferencia')
    legacy_operation = number(values['costo_operacion'], 'costo_operacion')
    recurring_cost = inference_energy + inference_capital + noncompute + legacy_inference + legacy_operation
    labor = row['ahorro_horas'] * values['costo_hora']
    recurring_value = labor - row['riesgo_residual'] - recurring_cost
    training_total = training_energy + training_capital
    row.update(q_detectado=q, labor_value_gtq=labor, inference_hours=inference_hours,
               training_hours=train_hours, training_runs=costs['training_runs'],
               inference_energy_equivalent_nominal_kwh=inference_kwh,
               training_energy_equivalent_nominal_kwh=training_kwh,
               inference_energy_gtq=inference_energy, training_energy_gtq=training_energy,
               inference_capital_gtq=inference_capital, training_capital_gtq=training_capital,
               non_compute_gtq=noncompute, legacy_inference_gtq=legacy_inference,
               legacy_operation_gtq=legacy_operation, recurring_cost_gtq=recurring_cost,
               training_total_gtq=training_total, recurring_value_gtq=recurring_value,
               first_batch_value_gtq=recurring_value-training_total,
               monthly_recurring_value_gtq=recurring_value*costs['messages_per_month']/row['N'],
               monthly_first_batch_value_gtq=recurring_value*costs['messages_per_month']/row['N']-training_total,
               costo_inferencia=inference_energy+inference_capital+legacy_inference,
               costo_operacion=noncompute+legacy_operation, valor_neto=recurring_value)
    for key, value in row.items():
        if isinstance(value, (float, int)) and not math.isfinite(value):
            raise ValueError(f'Nonfinite result: {key}')
    return row


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def verify_inputs(evaluation_dir, value_config, artifact_dirs, benchmark_paths):
    folder = Path(evaluation_dir)
    freeze = read_json(folder/'freeze.json')
    report = read_json(folder/'evaluation.json')
    if report.get('freeze_sha256') != digest(folder/'freeze.json'):
        raise ValueError('Evaluation freeze fingerprint mismatch')
    if report.get('provenance') != freeze:
        raise ValueError('Evaluation provenance differs from freeze')
    if digest(value_config) != freeze['value_config_sha256']:
        raise ValueError('Value configuration differs from freeze')
    config = read_json(value_config)
    values = validate_value_config(config)
    if config['units']['currency'] != 'GTQ':
        raise ValueError('Compute costs require GTQ')
    artifacts = {}
    for path in map(Path, artifact_dirs):
        manifest = read_json(path/'manifest.json')
        version = manifest['model_version']
        if version in artifacts:
            raise ValueError('Duplicate artifact model version')
        if digest(path/'manifest.json') != freeze['models'][version]['manifest_sha256']:
            raise ValueError('Artifact manifest differs from freeze')
        if digest(path/'metrics.json') != manifest['files']['metrics.json']:
            raise ValueError('Artifact metrics integrity mismatch')
        metrics = read_json(path/'metrics.json')
        if metrics.get('model_version', version) != version:
            raise ValueError('Training metrics model version mismatch')
        training_device = 'cpu' if manifest['kind'] == 'baseline' else 'gpu'
        if training_device == 'gpu' and '5070 Ti' not in metrics.get('device', ''):
            raise ValueError('GPU nominal reference requires RTX 5070 Ti training')
        seconds = metrics['fit_seconds' if training_device == 'cpu' else 'train_seconds']
        number(seconds, 'training duration')
        artifacts[version] = {'train_seconds':seconds, 'training_device':training_device,
                              'manifest_sha256':digest(path/'manifest.json'),
                              'metrics_sha256':digest(path/'metrics.json')}
    benchmarks = {}
    for path in map(Path, benchmark_paths):
        bench = read_json(path)
        version = bench['model_version']
        if version in benchmarks or version not in artifacts:
            raise ValueError('Duplicate or unknown benchmark model version')
        if bench.get('device') != 'cpu' or bench.get('batch_size') != 1:
            raise ValueError('Expected CPU batch-one benchmark')
        requests = number(bench.get('n_requests'), 'n_requests', positive=True)
        if not requests.is_integer():
            raise ValueError('n_requests must be integer')
        mean = number(bench['latency_ms']['mean'], 'latency_ms.mean', positive=True)
        benchmarks[version] = {'mean_ms':mean, 'benchmark_sha256':digest(path),
                               'startup_ms_excluded':bench.get('startup_ms'),
                               'warmup_requests_excluded':bench.get('warmup_requests')}
    if set(artifacts) != set(freeze['models']) or set(benchmarks) != set(artifacts):
        raise ValueError('Artifacts and benchmarks must match all frozen models')
    for version, binding in freeze['models'].items():
        for name in ('decision', 'calibration'):
            if digest(folder/version/f'{name}.json') != binding[f'{name}_sha256']:
                raise ValueError(f'Frozen {name} integrity mismatch')
        decision = read_json(folder/version/'decision.json')
        for split in report['splits'].values():
            model = split['models'][version]
            if model['threshold'] != decision['threshold']:
                raise ValueError('Evaluation threshold differs from frozen decision')
            counts = {key:model['value'][key] for key in COUNT_PARAMETERS}
            recomputed = compute_value(**counts, assumptions=values)
            if recomputed['N'] != split['n_examples']:
                raise ValueError('Evaluation counts disagree with split size')
            for key, expected in recomputed.items():
                actual = model['value'].get(key)
                if isinstance(expected, float):
                    if not isinstance(actual, Real) or not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-10):
                        raise ValueError(f'Evaluation value inconsistency: {key}')
                elif actual != expected:
                    raise ValueError(f'Evaluation value inconsistency: {key}')
    return report, config, values, artifacts, benchmarks


def comparison(new, old):
    benefit = new['recurring_value_gtq']-old['recurring_value_gtq']
    incremental_training = new['training_total_gtq']-old['training_total_gtq']
    gross = (new['labor_value_gtq']-new['riesgo_residual'])-(old['labor_value_gtq']-old['riesgo_residual'])
    return dict(recurring_delta_gtq=benefit,
                first_batch_delta_gtq=new['first_batch_value_gtq']-old['first_batch_value_gtq'],
                maximum_extra_recurring_cost_gtq=gross,
                remaining_extra_recurring_cost_gtq=benefit,
                incremental_training_gtq=incremental_training,
                requests_to_recover_extra_training=(max(0, incremental_training)/(benefit/new['N']) if benefit > 0 else None),
                months_to_recover_extra_training=(max(0, incremental_training)/(new['monthly_recurring_value_gtq']-old['monthly_recurring_value_gtq']) if benefit > 0 else None))


def capital_volume_sensitivity(counts, values, costs, **kwargs):
    rows = []
    for capital in (5000, 10000, 15000):
        for volume in (1000, 10000, 100000):
            row = calculate_case(counts, values, {**costs,
                'inference_capital_gtq':capital, 'training_capital_gtq':capital,
                'messages_per_month':volume}, **kwargs)
            rows.append(dict(equipment_capital_gtq=capital, messages_per_month=volume, **row))
    return rows


def analyze(evaluation_dir, value_config, cost_config, artifact_dirs, benchmark_paths):
    report, config, values, artifacts, benchmarks = verify_inputs(evaluation_dir, value_config, artifact_dirs, benchmark_paths)
    costs_raw = read_json(cost_config)
    costs = validate_cost_config(costs_raw)
    result = dict(schema_version=1, estimated=True, currency='GTQ',
                  calculation_basis='Conditional on supplied human-time, risk, tariff and equipment parameters; no financial outcome or wall-power measurement is claimed.',
                  energy_scope='Nominal CPU/GPU component reference plus configured overhead. TDP/TGP is neither measured wall power nor a hard electrical upper bound.',
                  duration_scope='Final selected training run only; inference uses warm batch-one CPU mean; startup and warmup excluded.',
                  value_parameters=config['assumptions'], assumptions_version=config['assumptions_version'],
                  cost_config=costs_raw,
                  source_sha256=dict(evaluation=digest(Path(evaluation_dir)/'evaluation.json'), freeze=digest(Path(evaluation_dir)/'freeze.json'), value_config=digest(value_config), cost_config=digest(cost_config), analysis_script=digest(__file__)),
                  sources={v:{**artifacts[v], **benchmarks[v]} for v in artifacts}, splits={})
    for split, entry in report['splits'].items():
        scenarios = []
        for q in (.5, .8, 1):
            rows = {}
            for version, model in entry['models'].items():
                counts = {key:model['value'][key] for key in COUNT_PARAMETERS}
                source = artifacts[version]
                rows[version] = dict(threshold=model['threshold'], **calculate_case(counts, values, costs, q=q,
                        mean_ms=benchmarks[version]['mean_ms'], train_seconds=source['train_seconds'], training_device=source['training_device']))
                rows[version]['capital_volume_sensitivity'] = capital_volume_sensitivity(
                    counts, values, costs, q=q, mean_ms=benchmarks[version]['mean_ms'],
                    train_seconds=source['train_seconds'], training_device=source['training_device'])
            comparisons = {f'distilbert-v2_minus_{old}':comparison(rows['distilbert-v2'], rows[old]) for old in ('distilbert-v1', 'baseline-tfidf')}
            scenarios.append(dict(q_detectado=q, models=rows, comparisons=comparisons))
        result['splits'][split] = scenarios
    return result


def write_new_json(path, value):
    encoded = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
    with Path(path).open('x', encoding='utf-8') as handle:
        handle.write(encoded)


def validate_output_location(path, immutable_dirs, immutable_files):
    destination = Path(path).resolve()
    if any(destination.is_relative_to(Path(folder).resolve()) for folder in immutable_dirs):
        raise ValueError('Output must be outside immutable input directories')
    if destination in {Path(item).resolve() for item in immutable_files}:
        raise ValueError('Output cannot replace an input file')
    if Path(path).exists() or Path(path).is_symlink():
        raise FileExistsError(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('evaluation-dir', 'value-config', 'cost-config', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--artifacts', nargs='+', type=Path, required=True)
    parser.add_argument('--benchmarks', nargs='+', type=Path, required=True)
    args = parser.parse_args()
    validate_output_location(args.out, [args.evaluation_dir, *args.artifacts],
                             [args.value_config, args.cost_config, *args.benchmarks])
    result = analyze(args.evaluation_dir, args.value_config, args.cost_config, args.artifacts, args.benchmarks)
    write_new_json(args.out, result)
    print(args.out)


if __name__ == '__main__':
    main()
