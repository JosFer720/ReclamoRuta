"""Estimated economics for assisted routing, with explicit, caller-supplied inputs.

Times are seconds. All monetary inputs share one currency and period.
``costo_inferencia`` and ``costo_operacion`` are totals for the evaluated batch;
the former varies with volume, the latter stays fixed in volume projections.
These calculations are scenarios, never evidence of measured savings.
"""

from collections.abc import Mapping, Sequence
import math
from numbers import Integral, Real
import re


TIME_PARAMETERS = ("t_manual", "t_aceptar", "t_corregir", "t_revision")
ASSUMPTION_PARAMETERS = TIME_PARAMETERS + (
    "costo_hora", "costo_error_ruta_no_detectado", "costo_inferencia",
    "costo_operacion", "q_detectado",
)
COUNT_PARAMETERS = (
    "accepted_correct", "accepted_wrong", "accepted_wrong_route", "reviewed",
)


def _number(value: object, name: str, *, upper: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite nonnegative number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result) or result < 0 or (upper is not None and result > upper):
        raise ValueError(f"{name} outside its finite permitted range")
    return result


def _assumptions(assumptions: dict) -> dict[str, float]:
    if not isinstance(assumptions, Mapping):
        raise ValueError("assumptions must be a mapping")
    missing = set(ASSUMPTION_PARAMETERS) - assumptions.keys()
    if missing:
        raise ValueError(f"Missing required assumptions: {', '.join(sorted(missing))}")
    return {
        name: _number(assumptions[name], name, upper=1 if name == "q_detectado" else None)
        for name in ASSUMPTION_PARAMETERS
    }


def _counts(counts: dict) -> dict[str, int]:
    if not isinstance(counts, Mapping) or set(counts) != set(COUNT_PARAMETERS):
        raise ValueError("counts must contain exactly the four accepted/reviewed counts")
    for name, count in counts.items():
        if isinstance(count, bool) or not isinstance(count, Integral) or count < 0:
            raise ValueError(f"{name} must be a nonnegative integer, not a boolean")
    if counts["accepted_wrong_route"] > counts["accepted_wrong"]:
        raise ValueError("accepted_wrong_route cannot exceed accepted_wrong")
    return {name: int(counts[name]) for name in COUNT_PARAMETERS}


def _finite_results(result: dict) -> dict:
    for name, value in result.items():
        if isinstance(value, (int, float)):
            try:
                finite = math.isfinite(value)
            except OverflowError as exc:
                raise ValueError(f"{name} exceeds finite arithmetic range") from exc
            if not finite:
                raise ValueError(f"{name} exceeds finite arithmetic range")
    return result


def compute_value(
    accepted_correct: int,
    accepted_wrong: int,
    accepted_wrong_route: int,
    reviewed: int,
    assumptions: dict,
) -> dict:
    """Calculate a batch scenario; ``accepted_wrong`` counts intention errors.

    Same-route intention errors still incur correction time, but residual risk
    applies only to ``accepted_wrong_route``. Missing economics never become zero.
    Plain numeric assumptions are allowed here for isolated scenario calculations.
    """
    counts = _counts(dict(zip(COUNT_PARAMETERS, (
        accepted_correct, accepted_wrong, accepted_wrong_route, reviewed,
    ), strict=True)))
    values = _assumptions(assumptions)
    accepted = counts["accepted_correct"] + counts["accepted_wrong"]
    total = accepted + counts["reviewed"]
    try:
        baseline = total * values["t_manual"]
        assisted = (
            accepted * values["t_aceptar"]
            + values["q_detectado"] * counts["accepted_wrong"] * values["t_corregir"]
            + counts["reviewed"] * values["t_revision"]
        )
        hours = (baseline - assisted) / 3600
        residual = (
            (1 - values["q_detectado"]) * counts["accepted_wrong_route"]
            * values["costo_error_ruta_no_detectado"]
        )
        result = {
            **counts, "N": total, "accepted": accepted,
            "T_base": baseline, "T_assist": assisted, "ahorro_horas": hours,
            "riesgo_residual": residual,
            "costo_inferencia": values["costo_inferencia"],
            "costo_operacion": values["costo_operacion"],
            "valor_neto": hours * values["costo_hora"] - residual
            - values["costo_inferencia"] - values["costo_operacion"],
            "coverage": accepted / total if total else 0.0,
            "selective_error": counts["accepted_wrong"] / accepted if accepted else None,
            "estimated": True,
        }
    except OverflowError as exc:
        raise ValueError("Inputs exceed finite arithmetic range") from exc
    return _finite_results(result)


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def validate_value_config(config: dict) -> dict:
    """Reject pending templates and return a validated, detached assumptions map.

    ``validated`` records the caller's review, not independent verification of
    sources. Estimated evidence stays explicitly estimated. Monetary parameters
    must share ``units.currency`` and the batch period.
    """
    if not isinstance(config, Mapping) or config.get("status") != "validated":
        raise ValueError("Economic configuration must have status='validated'")
    _text(config.get("assumptions_version"), "assumptions_version")
    units = config.get("units")
    if not isinstance(units, Mapping):
        raise ValueError("Explicit units are required")
    if units.get("time") != "seconds" or units.get("cost_scope") != "batch":
        raise ValueError("Expected time in seconds and monetary totals per batch")
    currency = _text(units.get("currency"), "currency")
    if re.fullmatch(r"[A-Z]{3}", currency) is None:
        raise ValueError("currency must be a consistent three-letter currency code")
    assumptions = _assumptions(config.get("assumptions"))
    evidence = config.get("evidence")
    if not isinstance(evidence, Mapping):
        raise ValueError("Evidence is required for every assumption")
    for name in ASSUMPTION_PARAMETERS:
        item = evidence.get(name)
        if not isinstance(item, Mapping) or not isinstance(item.get("estimated"), bool):
            raise ValueError(f"Explicit estimated flag is required for {name}")
        _text(item.get("source"), f"source for {name}")
    return assumptions


def _calibration(calibration: object) -> dict:
    if not isinstance(calibration, Mapping):
        raise ValueError("Explicit calibration metadata is required")
    method = _text(calibration.get("method"), "calibration method")
    fitted = calibration.get("fitted")
    if not isinstance(fitted, bool):
        raise ValueError("calibration.fitted must be a boolean")
    if method == "none":
        if fitted:
            raise ValueError("Uncalibrated scores cannot claim fitted calibration")
        return {"method": method, "fitted": False}
    if not fitted or calibration.get("split_role") != "validation_calibration":
        raise ValueError("Fitted calibration requires validation_calibration metadata")
    artifact = calibration.get("artifact_sha256")
    _sha256(artifact)
    return {
        "method": method, "fitted": True, "split_role": "validation_calibration",
        "artifact_sha256": artifact,
    }


def _sha256(value: object) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("artifact_sha256 must be a lowercase hexadecimal SHA-256")
    return value


def validate_decision(decision: dict) -> dict:
    """Validate binding metadata before consuming a persisted decision.

    This checks metadata structure; callers must verify artifact content hashes
    and the provenance of splits against their frozen manifests separately.
    """
    if not isinstance(decision, Mapping) or decision.get("status") != "validated":
        raise ValueError("Decision must have status='validated'")
    if decision.get("split_role") != "validation_threshold":
        raise ValueError("Decision requires validation_threshold selection metadata")
    return {
        "status": "validated",
        "split_role": "validation_threshold",
        "model_version": _text(decision.get("model_version"), "model_version"),
        "artifact_sha256": _sha256(decision.get("artifact_sha256")),
        "threshold": _number(decision.get("threshold"), "threshold", upper=1),
        "calibration": _calibration(decision.get("calibration")),
        "assumptions_version": _text(decision.get("assumptions_version"), "assumptions_version"),
    }


def _sequence(value: object, name: str) -> Sequence:
    if (
        isinstance(value, (str, bytes, Mapping))
        or not hasattr(value, "__len__")
        or not hasattr(value, "__getitem__")
    ):
        raise ValueError(f"{name} must be a sequence, not text")
    return value


def threshold_curve(
    labels: Sequence[str],
    scores: Sequence[float],
    predicted_labels: Sequence[str],
    routes: Mapping[str, str],
    assumptions: dict,
    *,
    thresholds: Sequence[float] | None = None,
    split_role: str = "validation_threshold",
) -> list[dict]:
    """Return estimated value for each tau; ``score >= tau`` accepts a case.

    The default grid is 0, .05, ... 1. This low-level scenario primitive accepts
    numeric assumptions, or a full validated config. A role check guards obvious
    misuse; it does NOT prove provenance or detect relabeled test predictions.
    Only threshold-validation predictions may be used; fitting is never done.
    ``routes`` is the complete supplied intention catalog, all entries validated.
    """
    if split_role != "validation_threshold":
        raise ValueError("Threshold selection requires validation_threshold predictions")
    labels = _sequence(labels, "labels")
    scores = _sequence(scores, "scores")
    predicted_labels = _sequence(predicted_labels, "predicted_labels")
    if not len(labels) == len(scores) == len(predicted_labels):
        raise ValueError("labels, scores and predicted_labels must have equal lengths")
    if not isinstance(routes, Mapping) or not routes:
        raise ValueError("A complete nonempty intention-to-route catalog is required")
    for label, route in routes.items():
        _text(label, "catalog intention")
        _text(route, "catalog route")
    for label in (*labels, *predicted_labels):
        _text(label, "intention")
        if label not in routes:
            raise ValueError("An intention is missing from the route catalog")
    scores = [_number(score, "score", upper=1) for score in scores]
    grid = [step / 20 for step in range(21)] if thresholds is None else _sequence(thresholds, "thresholds")
    if len(grid) == 0:
        raise ValueError("At least one threshold is required")
    grid = sorted({_number(tau, "threshold", upper=1) for tau in grid})
    if isinstance(assumptions, Mapping) and "status" in assumptions:
        values = validate_value_config(assumptions)
    else:
        values = _assumptions(assumptions)
    rows = []
    for tau in grid:
        counts = dict.fromkeys(COUNT_PARAMETERS, 0)
        for truth, score, prediction in zip(labels, scores, predicted_labels, strict=True):
            if score < tau:
                counts["reviewed"] += 1
            elif truth == prediction:
                counts["accepted_correct"] += 1
            else:
                counts["accepted_wrong"] += 1
                counts["accepted_wrong_route"] += routes[truth] != routes[prediction]
        rows.append({"threshold": tau, **compute_value(**counts, assumptions=values)})
    return rows


def select_threshold(
    labels: Sequence[str],
    scores: Sequence[float],
    predicted_labels: Sequence[str],
    routes: Mapping[str, str],
    assumptions: dict,
    *,
    model_version: str,
    artifact_sha256: str,
    calibration: dict,
    thresholds: Sequence[float] | None = None,
    split_role: str = "validation_threshold",
) -> dict:
    """Choose maximum estimated net value; exact ties select the highest tau.

    Requires a full validated economic config and observed validation cases.
    Explicitly reports no abstention even when tau=0 is absent from the requested
    grid. Caller supplies the model artifact hash; this pure function neither
    reads an artifact nor verifies it. Selecting a threshold does not imply the
    system should be deployed.
    """
    values = validate_value_config(assumptions)
    binding = validate_decision({
        "status": "validated", "model_version": model_version,
        "artifact_sha256": artifact_sha256, "threshold": 0,
        "split_role": split_role,
        "calibration": calibration, "assumptions_version": assumptions["assumptions_version"],
    })
    curve = threshold_curve(labels, scores, predicted_labels, routes, values,
                            thresholds=thresholds, split_role=split_role)
    if curve[0]["N"] == 0:
        raise ValueError("Threshold selection requires validation observations")
    selected = max(curve, key=lambda row: (row["valor_neto"], row["threshold"]))
    no_abstention = threshold_curve(labels, scores, predicted_labels, routes, values,
                                   thresholds=[0], split_role=split_role)[0]
    return {
        **binding, "threshold": selected["threshold"], "split_role": split_role,
        "tie_break": "highest_threshold", "estimated": True,
        "currency": assumptions["units"]["currency"],
        "selected": selected, "no_abstention": no_abstention, "curve": curve,
    }


def sensitivity_analysis(counts: dict, assumptions: dict) -> list[dict]:
    """One-at-a-time scenarios of -50%, -20%, +20%, +50%.

    Vary each time parameter, hourly labor cost, and volume independently. Each
    baseline varied parameter and N must be finite and positive. Volume keeps
    case mix, per-case time/risk, and inference cost per case constant, scaling
    their totals continuously while holding operating cost fixed. Fractional
    projected counts are expectations, never truncated observations. No fit or
    measured financial outcome is claimed. Use ``validate_value_config`` first
    when consuming a persisted configuration.
    """
    counts = _counts(counts)
    values = _assumptions(assumptions)
    baseline = compute_value(**counts, assumptions=values)
    parameters = TIME_PARAMETERS + ("costo_hora",)
    if not baseline["N"] or any(values[name] <= 0 for name in parameters):
        raise ValueError("Relative sensitivity requires positive time/cost and volume baselines")
    scenarios = []
    for parameter in (*parameters, "volume"):
        for change in (-0.5, -0.2, 0.2, 0.5):
            multiplier = 1 + change
            if parameter == "volume":
                result = dict(baseline)
                for name in (*COUNT_PARAMETERS, "N", "accepted", "T_base", "T_assist",
                             "ahorro_horas", "riesgo_residual", "costo_inferencia"):
                    result[name] *= multiplier
                result["valor_neto"] = (
                    result["ahorro_horas"] * values["costo_hora"]
                    - result["riesgo_residual"] - result["costo_inferencia"]
                    - result["costo_operacion"]
                )
            else:
                varied = {**values, parameter: values[parameter] * multiplier}
                result = compute_value(**counts, assumptions=varied)
            scenarios.append(_finite_results({
                **result, "parameter": parameter, "relative_change": change,
                "multiplier": multiplier, "projected_N": float(result["N"]),
                "projection": parameter == "volume",
            }))
    return scenarios
