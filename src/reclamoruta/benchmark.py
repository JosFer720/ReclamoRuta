"""Inference latency benchmark: startup vs. inference, warm-up, batch 1, p50/p95.

Example (after Fabián's artifact exists)::

    python -m reclamoruta.benchmark --artifact artifacts/distilbert-v2 \
        --texts data/processed/banking77/validation_model.csv --n 200 --warmup 10 \
        --device cpu --out reports/benchmark_distilbert-v2_cpu.json

Results identify the model, device and machine; message texts are never saved.
Test splits are refused: latency is measured on validation text only.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

from reclamoruta.contracts import ContractError, PredictorLike, validate_message

CLOCK_NAME = "time.perf_counter (monotónico)"


def _environment(device: str) -> dict:
    env = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "cpu_count": os.cpu_count(),
        "device": device,
    }
    torch = sys.modules.get("torch")  # only report it if the predictor already loaded it
    if torch is not None:
        env["torch"] = getattr(torch, "__version__", None)
        try:
            if device.startswith("cuda") and torch.cuda.is_available():
                env["gpu"] = torch.cuda.get_device_name(0)
        except Exception:  # noqa: BLE001 - informative only
            env["gpu"] = None
    return env


def _quantile(ordered: Sequence[float], fraction: float) -> float:
    """Type-7 (linear) quantile, defined for a single value."""
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _timed(predictor: PredictorLike, text: str, clock: Callable[[], float]) -> float:
    start = clock()
    outputs = predictor.predict([text])
    end = clock()
    if not isinstance(outputs, (list, tuple)) or len(outputs) != 1:
        raise ContractError("El predictor debe devolver exactamente una predicción por texto")
    if end < start:
        raise ContractError("El reloj no es monotónico: el tiempo retrocedió")
    return end - start


def run_benchmark(
    load_predictor: Callable[[], PredictorLike],
    texts: Sequence[str],
    *,
    warmup: int = 10,
    clock: Callable[[], float] = time.perf_counter,
    device: str = "cpu",
) -> dict:
    """Load once (timed as startup), warm up, then time each text with batch size 1.

    Warm-up requests cycle through ``texts`` and are excluded from latencies.
    Throughput is measured requests divided by summed measured inference time.
    """
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ContractError("warmup debe ser un entero no negativo")
    messages = [validate_message(text) for text in texts]
    if not messages:
        raise ContractError("Se necesita al menos un texto para medir")

    start = clock()
    predictor = load_predictor()
    startup = clock() - start
    if startup < 0:
        raise ContractError("El reloj no es monotónico: el tiempo retrocedió")

    for index in range(warmup):
        _timed(predictor, messages[index % len(messages)], clock)
    latencies = [_timed(predictor, message, clock) for message in messages]
    ordered = sorted(latencies)
    total = sum(latencies)
    return {
        "model_version": getattr(predictor, "model_version", None),
        "device": device,
        "batch_size": 1,
        "warmup_requests": warmup,
        "n_requests": len(latencies),
        "startup_ms": startup * 1000,
        "latency_ms": {
            "p50": _quantile(ordered, 0.50) * 1000,
            "p95": _quantile(ordered, 0.95) * 1000,
            "mean": statistics.fmean(latencies) * 1000,
            "min": ordered[0] * 1000,
            "max": ordered[-1] * 1000,
        },
        "throughput_rps": len(latencies) / total if total > 0 else None,
        "clock": CLOCK_NAME,
        "measured_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": _environment(device),
    }


def read_texts(path: Path, limit: int | None = None) -> list[str]:
    """Read the ``text`` column of a processed split; test splits are refused."""
    path = Path(path)
    if path.name.startswith("test"):
        raise ContractError("No se mide latencia con el test: use un split de validación")
    texts = []
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if str(row.get("split", "")).startswith("test"):
                raise ContractError("El archivo contiene filas de test; use validación")
            texts.append(row["text"])
            if limit is not None and len(texts) >= limit:
                break
    return texts


def write_result(result: dict, out: Path) -> Path:
    """Write JSON to a new file; existing results are never overwritten."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with out.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    except FileExistsError as exc:
        raise ContractError(f"{out} ya existe; elija otro nombre") from exc
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark de latencia del predictor (batch 1)")
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--texts", type=Path, required=True, help="CSV procesado de validación")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    from reclamoruta.predictor import Predictor  # Fabián's module

    texts = read_texts(args.texts, limit=args.n)
    result = run_benchmark(
        lambda: Predictor.load(args.artifact, device=args.device), texts,
        warmup=args.warmup, device=args.device,
    )
    write_result(result, args.out)
    print(json.dumps({k: result[k] for k in ("model_version", "startup_ms", "latency_ms", "throughput_rps")},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
