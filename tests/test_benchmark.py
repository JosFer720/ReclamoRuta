"""Latency benchmark: warm-up, batch 1, p50/p95, throughput, startup separated."""

import csv
import json

import pytest


def benchmark():
    from reclamoruta import benchmark as module

    return module


class FakeClock:
    """Deterministic monotonic clock; each predict advances it by a fixed step."""

    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class SteppingPredictor:
    model_version = "fake-v1"

    def __init__(self, clock, steps):
        self.clock = clock
        self.steps = list(steps)
        self.batches = []

    def predict(self, texts):
        from reclamoruta.contracts import Prediction

        self.batches.append(list(texts))
        self.clock.now += self.steps.pop(0)
        return [Prediction("a", 0.9, [], self.model_version, 0.0) for _ in texts]


def run(steps, texts, warmup=2, load_seconds=1.5):
    clock = FakeClock()
    predictor = SteppingPredictor(clock, steps)

    def load():
        clock.now += load_seconds
        return predictor

    result = benchmark().run_benchmark(load, texts, warmup=warmup, clock=clock, device="cpu")
    return result, predictor


def test_startup_warmup_and_measured_latencies_are_separated():
    # 2 warm-up calls of 9 s must not appear in the measured latencies.
    result, predictor = run([9, 9, 0.010, 0.020, 0.030, 0.040], ["a", "b", "c", "d"])
    assert result["startup_ms"] == pytest.approx(1500)
    assert result["warmup_requests"] == 2
    assert result["n_requests"] == 4
    assert result["batch_size"] == 1
    assert all(len(batch) == 1 for batch in predictor.batches)
    latency = result["latency_ms"]
    assert latency["p50"] == pytest.approx(25.0)
    assert latency["p95"] == pytest.approx(38.5)  # type-7 interpolation
    assert latency["max"] == pytest.approx(40.0)
    assert result["throughput_rps"] == pytest.approx(4 / 0.100)


def test_result_identifies_model_device_environment_and_is_json_without_texts():
    result, _ = run([0.01] * 4, ["secret message one", "secret message two"])
    assert result["model_version"] == "fake-v1"
    assert result["device"] == "cpu"
    env = result["environment"]
    for key in ("platform", "python", "machine", "cpu_count"):
        assert key in env
    assert result["clock"] == "time.perf_counter (monotónico)"
    dumped = json.dumps(result, allow_nan=False)
    assert "secret" not in dumped


def test_single_request_has_defined_percentiles():
    result, _ = run([0.005], ["only"], warmup=0)
    assert result["latency_ms"]["p50"] == result["latency_ms"]["p95"] == pytest.approx(5.0)


@pytest.mark.parametrize("texts", [[], ["ok", "   "], ["x" * 2001]])
def test_invalid_inputs_rejected(texts):
    from reclamoruta.contracts import ContractError

    with pytest.raises(ContractError):
        run([0.01] * 5, texts)


def test_negative_warmup_and_non_monotonic_clock_rejected():
    from reclamoruta.contracts import ContractError

    with pytest.raises(ContractError):
        run([0.01] * 3, ["a"], warmup=-1)
    with pytest.raises(ContractError, match="monot"):
        run([0.01, -5.0], ["a", "b"], warmup=0)  # clock goes backwards during measurement


def test_predictor_returning_wrong_count_is_rejected():
    from reclamoruta.contracts import ContractError

    class Bad:
        model_version = "x"

        def predict(self, texts):
            return []

    with pytest.raises(ContractError):
        benchmark().run_benchmark(lambda: Bad(), ["a"], warmup=0)


def write_split(path, split="validation_model"):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "text", "category", "group_id", "split"])
        for i in range(5):
            writer.writerow([f"train.csv:{i}", f"text {i}", "a", "g", split])


def test_read_texts_refuses_test_split_and_limits(tmp_path):
    from reclamoruta.contracts import ContractError

    ok = tmp_path / "validation_model.csv"
    write_split(ok)
    assert benchmark().read_texts(ok, limit=3) == ["text 0", "text 1", "text 2"]
    bad = tmp_path / "test_official.csv"
    write_split(bad, split="test_official")
    with pytest.raises(ContractError, match="test"):
        benchmark().read_texts(bad)


def test_write_result_refuses_to_overwrite(tmp_path):
    from reclamoruta.contracts import ContractError

    out = tmp_path / "bench.json"
    benchmark().write_result({"a": 1}, out)
    assert json.loads(out.read_text(encoding="utf-8")) == {"a": 1}
    with pytest.raises(ContractError):
        benchmark().write_result({"a": 2}, out)
