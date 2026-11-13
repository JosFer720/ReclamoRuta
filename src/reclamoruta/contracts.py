"""Shared contracts between predictor, application and evaluation.

Changing a field here changes the predictor and evaluation interfaces;
coordinate changes before merging.

Flow: text -> ``validate_message`` -> ``Predictor.predict`` -> ``Prediction`` ->
``route_message`` (threshold from a bound decision, route from ``routes.json``)
-> human ``confirm``. Nothing here trains, writes files or persists text.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
from typing import Protocol

from reclamoruta.value import validate_decision

MAX_MESSAGE_CHARS = 2000
MANIFEST_NAME = "manifest.json"


class ContractError(ValueError):
    """Input, configuration or predictor output violates the shared contract."""


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{name} debe ser un texto no vacío")
    return value


def _unit_score(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ContractError(f"{name} debe ser un número en [0, 1]")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ContractError(f"{name} debe ser un número en [0, 1]")
    return number


# --- Prediction -----------------------------------------------------------

@dataclass(frozen=True)
class Prediction:
    """One classifier output. ``score`` is the top label's score in [0, 1].

    ``top3`` holds up to three ``(label, score)`` pairs in descending score; when
    present its first entry must equal ``(label, score)``. ``elapsed_ms`` is the
    inference time attributed to this text.
    """

    label: str
    score: float
    top3: list[tuple[str, float]]
    model_version: str
    elapsed_ms: float

    def __post_init__(self) -> None:
        _text(self.label, "label")
        _text(self.model_version, "model_version")
        score = _unit_score(self.score, "score")
        if isinstance(self.elapsed_ms, bool) or not isinstance(self.elapsed_ms, Real) \
                or not math.isfinite(float(self.elapsed_ms)) or self.elapsed_ms < 0:
            raise ContractError("elapsed_ms debe ser un número finito no negativo")
        if not isinstance(self.top3, (list, tuple)) or len(self.top3) > 3:
            raise ContractError("top3 debe tener como máximo 3 pares (label, score)")
        ranking = []
        for item in self.top3:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ContractError("top3 debe contener pares (label, score)")
            ranking.append((_text(item[0], "label de top3"), _unit_score(item[1], "score de top3")))
        if len({label for label, _ in ranking}) != len(ranking):
            raise ContractError("top3 no puede repetir etiquetas")
        if any(a[1] < b[1] for a, b in zip(ranking, ranking[1:])):
            raise ContractError("top3 debe estar en orden descendente de score")
        if ranking and (ranking[0][0] != self.label or ranking[0][1] != score):
            raise ContractError("top3[0] debe coincidir con label y score")
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "elapsed_ms", float(self.elapsed_ms))
        object.__setattr__(self, "top3", ranking)


class PredictorLike(Protocol):
    """What the app needs from ``reclamoruta.predictor.Predictor``."""

    model_version: str

    def predict(self, texts: list[str]) -> list[Prediction]: ...


# --- Messages -------------------------------------------------------------

def validate_message(text: object) -> str:
    """Return the stripped message, or reject it before any model is invoked."""
    if not isinstance(text, str):
        raise ContractError("El mensaje debe ser texto")
    stripped = text.strip()
    if not stripped:
        raise ContractError("El mensaje está vacío")
    if len(stripped) > MAX_MESSAGE_CHARS:
        raise ContractError(f"El mensaje supera {MAX_MESSAGE_CHARS} caracteres")
    return stripped


# --- Routes ---------------------------------------------------------------

@dataclass(frozen=True)
class RouteMap:
    version: str
    status: str
    intent_to_route: dict[str, str]
    route_names: dict[str, str]
    route_descriptions: dict[str, str]

    @property
    def intents(self) -> list[str]:
        return sorted(self.intent_to_route)


def validate_routes(config: object, categories: Sequence[str]) -> RouteMap:
    """Check that every catalog intention maps to exactly one declared route.

    Routes are a group design, not dataset labels. Every declared route must be
    used, and the intention set must equal ``categories`` exactly.
    """
    if not isinstance(config, Mapping):
        raise ContractError("La configuración de rutas debe ser un objeto JSON")
    version = _text(config.get("version"), "version de rutas")
    status = _text(config.get("status"), "status de rutas")
    routes = config.get("routes")
    intents = config.get("intents")
    if not isinstance(routes, Mapping) or not routes or not isinstance(intents, Mapping):
        raise ContractError("Faltan 'routes' o 'intents' en la configuración de rutas")
    names, descriptions = {}, {}
    for route_id, info in routes.items():
        _text(route_id, "id de ruta")
        if not isinstance(info, Mapping):
            raise ContractError(f"La ruta {route_id} debe ser un objeto")
        names[route_id] = _text(info.get("name"), f"nombre de la ruta {route_id}")
        descriptions[route_id] = _text(info.get("description"), f"descripción de la ruta {route_id}")
    expected = set(categories)
    if len(expected) != len(categories) or not expected:
        raise ContractError("El catálogo de intenciones debe ser no vacío y sin duplicados")
    missing = expected - set(intents)
    extra = set(intents) - expected
    if missing or extra:
        raise ContractError(
            f"El mapa de rutas no coincide con el catálogo (faltan {len(missing)}, sobran {len(extra)})"
        )
    mapping = {}
    for intent, route_id in intents.items():
        if route_id not in routes:
            raise ContractError(f"La intención {intent} apunta a una ruta inexistente")
        mapping[intent] = route_id
    unused = set(routes) - set(mapping.values())
    if unused:
        raise ContractError(f"Rutas declaradas sin intenciones: {', '.join(sorted(unused))}")
    return RouteMap(version, status, mapping, names, descriptions)


def load_routes(path: Path, categories: Sequence[str]) -> RouteMap:
    try:
        config = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"No se pudo leer el mapa de rutas: {path}") from exc
    return validate_routes(config, categories)


# --- Decision binding -----------------------------------------------------

def artifact_digest(artifact_dir: Path) -> str:
    """SHA-256 of ``manifest.json``; the manifest lists hashes of every artifact file.

    ``Predictor.load`` must verify those per-file hashes; this digest binds a
    decision to one exact manifest.
    """
    manifest = Path(artifact_dir) / MANIFEST_NAME
    if manifest.is_symlink() or not manifest.is_file():
        raise ContractError(f"Artefacto incompleto: falta {MANIFEST_NAME} en {artifact_dir}")
    return hashlib.sha256(manifest.read_bytes()).hexdigest()


@dataclass(frozen=True)
class BoundDecision:
    model_version: str
    artifact_sha256: str
    threshold: float
    calibration: dict
    assumptions_version: str

    @property
    def calibrated(self) -> bool:
        return bool(self.calibration.get("fitted"))


def bind_decision(decision: object, *, model_version: str, artifact_sha256: str) -> BoundDecision:
    """Validate a decision and reject it unless it belongs to this exact model."""
    try:
        checked = validate_decision(decision)  # type: ignore[arg-type]
    except ValueError as exc:
        raise ContractError(f"Decisión no válida: {exc}") from exc
    if checked["model_version"] != model_version:
        raise ContractError(
            f"La decisión es de otro modelo ({checked['model_version']}), no de {model_version}"
        )
    if checked["artifact_sha256"] != artifact_sha256:
        raise ContractError("La decisión no corresponde a este artefacto (hash distinto)")
    return BoundDecision(
        checked["model_version"], checked["artifact_sha256"], checked["threshold"],
        checked["calibration"], checked["assumptions_version"],
    )


# --- Routing and human confirmation ---------------------------------------

@dataclass(frozen=True)
class RoutingResult:
    text: str
    prediction: Prediction
    intent: str
    route: str
    route_name: str
    score: float
    threshold: float
    needs_review: bool
    calibrated: bool


@dataclass(frozen=True)
class Confirmation:
    suggested_intent: str
    final_intent: str
    route: str
    route_name: str
    corrected: bool


def route_message(
    text: object, predictor: PredictorLike, decision: BoundDecision, routes: RouteMap,
) -> RoutingResult:
    """Validate, classify one message and attach the configured route.

    ``score >= threshold`` -> suggestion may be accepted (review still offered);
    ``score < threshold`` -> human review required.
    """
    message = validate_message(text)
    if getattr(predictor, "model_version", None) != decision.model_version:
        raise ContractError("El predictor cargado es de otro modelo que la decisión")
    outputs = predictor.predict([message])
    if not isinstance(outputs, (list, tuple)) or len(outputs) != 1:
        raise ContractError("El predictor debe devolver exactamente una predicción por texto")
    prediction = outputs[0]
    if not isinstance(prediction, Prediction):
        raise ContractError("El predictor debe devolver objetos Prediction")
    if prediction.model_version != decision.model_version:
        raise ContractError("La predicción proviene de otro modelo")
    if prediction.label not in routes.intent_to_route:
        raise ContractError(f"Intención desconocida para el catálogo: {prediction.label}")
    route = routes.intent_to_route[prediction.label]
    return RoutingResult(
        text=message, prediction=prediction, intent=prediction.label, route=route,
        route_name=routes.route_names[route], score=prediction.score,
        threshold=decision.threshold, needs_review=prediction.score < decision.threshold,
        calibrated=decision.calibrated,
    )


def confirm(result: RoutingResult, final_intent: str, routes: RouteMap) -> Confirmation:
    """Record the human's choice; correcting never retrains or changes weights."""
    if final_intent not in routes.intent_to_route:
        raise ContractError(f"Intención desconocida: {final_intent}")
    route = routes.intent_to_route[final_intent]
    return Confirmation(
        suggested_intent=result.intent, final_intent=final_intent, route=route,
        route_name=routes.route_names[route], corrected=final_intent != result.intent,
    )
