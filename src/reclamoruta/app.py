"""ReclamoRuta Streamlit app: suggest intent and route, always under human review.

Run (after Fabián's artifact and a validated decision exist)::

    RECLAMORUTA_ARTIFACT_DIR=artifacts/distilbert-v2 streamlit run src/reclamoruta/app.py

Environment variables (paths relative to the working directory):

- ``RECLAMORUTA_ARTIFACT_DIR`` (required): predictor artifact with ``manifest.json``.
- ``RECLAMORUTA_DECISION`` (default ``configs/decision.json``): validated decision.
- ``RECLAMORUTA_ROUTES`` (default ``configs/routes.json``).
- ``RECLAMORUTA_CATALOG`` (default ``reports/BANKING77_LABEL_TO_ID.json``).
- ``RECLAMORUTA_DEVICE`` (default ``cpu``).

The app never stores message text, never retrains on corrections and never
invents a fallback answer when the model is missing or fails.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
import os
from pathlib import Path

from reclamoruta.contracts import (
    BoundDecision, ContractError, MAX_MESSAGE_CHARS, PredictorLike, RouteMap,
    artifact_digest, bind_decision, confirm, load_routes, route_message,
)
from reclamoruta.value import validate_decision

SCOPE_WARNING = (
    "Prototipo académico. Clasifica consultas bancarias **solo en inglés** y sugiere una ruta "
    "de atención diseñada por el grupo. **No ejecuta operaciones bancarias**, no detecta fraude "
    "ni da asesoría financiera. No escriba datos personales, números de cuenta o de tarjeta. "
    "Una persona revisa y confirma cada sugerencia."
)

Loader = Callable[[Path, str], PredictorLike]


class AppConfigError(ContractError):
    """The app cannot start safely; the message is shown to the operator."""


@dataclass(frozen=True)
class Runtime:
    predictor: PredictorLike
    decision: BoundDecision
    routes: RouteMap
    test_double: bool = False


def _default_loader(artifact_dir: Path, device: str) -> PredictorLike:
    try:
        from reclamoruta.predictor import Predictor
    except ImportError as exc:
        raise AppConfigError(
            "El predictor (reclamoruta.predictor) no está disponible en esta instalación"
        ) from exc
    return Predictor.load(artifact_dir, device=device)


def _read_json(path: Path, what: str) -> object:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AppConfigError(f"No se pudo leer {what}: {path}") from exc


def load_runtime(
    *, artifact_dir: Path, decision_path: Path, routes_path: Path, catalog_path: Path,
    device: str = "cpu", loader: Loader | None = None,
) -> Runtime:
    """Load and cross-check catalog, routes, decision and predictor, cheapest first.

    The decision's artifact hash is checked against ``manifest.json`` before any
    weights are loaded. Every failure raises ``AppConfigError``; nothing falls back.
    """
    catalog = _read_json(catalog_path, "el catálogo de intenciones")
    if not isinstance(catalog, dict) or not catalog:
        raise AppConfigError("El catálogo de intenciones debe ser un objeto label→id no vacío")
    categories = list(catalog)
    try:
        routes = load_routes(routes_path, categories)
        digest = artifact_digest(Path(artifact_dir))
    except ContractError as exc:
        raise AppConfigError(str(exc)) from exc

    decision = _read_json(decision_path, "la decisión")
    try:
        checked = validate_decision(decision)  # type: ignore[arg-type]
    except ValueError as exc:
        raise AppConfigError(f"Decisión no válida o pendiente: {exc}") from exc
    if checked["artifact_sha256"] != digest:
        raise AppConfigError("La decisión no corresponde a este artefacto (hash de manifest distinto)")

    try:
        predictor = (loader or _default_loader)(Path(artifact_dir), device)
    except AppConfigError:
        raise
    except Exception as exc:  # noqa: BLE001 - any load failure must stop the app
        raise AppConfigError(
            f"No se pudo cargar el modelo desde {artifact_dir} ({type(exc).__name__}). "
            "Verifique que el artefacto esté completo y sin modificar."
        ) from exc

    predictor_categories = getattr(predictor, "categories", None)
    if predictor_categories is not None and list(predictor_categories) != categories \
            and set(predictor_categories) != set(categories):
        raise AppConfigError("El catálogo del modelo no coincide con el catálogo de intenciones")
    try:
        bound = bind_decision(
            decision, model_version=getattr(predictor, "model_version", ""), artifact_sha256=digest,
        )
    except ContractError as exc:
        raise AppConfigError(str(exc)) from exc
    return Runtime(predictor, bound, routes)


def _score_label(calibrated: bool) -> str:
    if calibrated:
        return "Probabilidad calibrada (estimada en validación)"
    return "Puntuación del modelo (sin calibrar; no es probabilidad de acierto)"


def render(runtime: Runtime) -> None:
    """Draw the UI for an already loaded runtime."""
    import streamlit as st

    st.title("ReclamoRuta")
    st.info(SCOPE_WARNING)
    if runtime.test_double:
        st.warning("DOBLE DE PRUEBA: predictor falso para pruebas automáticas. No es la demo.")
    st.caption(
        f"Modelo `{runtime.decision.model_version}` · umbral {runtime.decision.threshold:.2f} · "
        f"rutas `{runtime.routes.version}` ({runtime.routes.status})"
    )

    with st.form("classify_form"):
        text = st.text_area(
            "Consulta del cliente (en inglés)", key="message", max_chars=MAX_MESSAGE_CHARS,
            placeholder="e.g. My card still hasn't arrived, what should I do?",
        )
        submitted = st.form_submit_button("Clasificar", key="classify")

    if submitted:
        st.session_state.pop("result", None)
        st.session_state.pop("confirmation", None)
        try:
            st.session_state["result"] = route_message(
                text, runtime.predictor, runtime.decision, runtime.routes,
            )
        except ContractError as exc:
            st.warning(str(exc))
        except Exception:  # noqa: BLE001 - never show a made-up answer
            st.error("No se pudo clasificar el mensaje. Intente de nuevo o enrute manualmente.")

    result = st.session_state.get("result")
    if result is None:
        return

    st.subheader("Sugerencia del modelo")
    col1, col2 = st.columns(2)
    col1.metric("Intención", result.intent)
    col2.metric("Ruta sugerida", result.route_name)
    st.metric(_score_label(result.calibrated), f"{result.score:.3f}")
    st.caption("Una puntuación alta no garantiza acierto; el umbral no detecta todo mensaje fuera de tema.")
    if result.needs_review:
        st.warning(
            f"Revisión requerida: la puntuación ({result.score:.3f}) es menor que el umbral "
            f"({result.threshold:.2f}). Elija la intención correcta."
        )
    else:
        st.success("Sugerencia aceptable. Puede confirmarla o corregirla.")
    if len(result.prediction.top3) > 1:
        st.markdown("Alternativas: " + ", ".join(
            f"`{label}` ({score:.3f})" for label, score in result.prediction.top3[1:]
        ))

    intents = runtime.routes.intents
    final = st.selectbox(
        "Intención final (revisión humana)", intents, index=intents.index(result.intent),
        key="final_intent",
    )
    if st.button("Confirmar ruta", key="confirm"):
        st.session_state["confirmation"] = confirm(result, final, runtime.routes)

    confirmation = st.session_state.get("confirmation")
    if confirmation is not None:
        verb = "corregida" if confirmation.corrected else "confirmada sin cambios"
        st.success(f"Ruta {verb}: **{confirmation.route_name}** (intención `{confirmation.final_intent}`).")
        st.caption("La corrección no reentrena el modelo ni se guarda el texto del mensaje.")


@dataclass(frozen=True)
class _Settings:
    artifact_dir: str
    decision_path: str
    routes_path: str
    catalog_path: str
    device: str


def _settings() -> _Settings:
    artifact = os.environ.get("RECLAMORUTA_ARTIFACT_DIR", "").strip()
    if not artifact:
        raise AppConfigError("Falta RECLAMORUTA_ARTIFACT_DIR: indique el directorio del artefacto del modelo")
    return _Settings(
        artifact_dir=artifact,
        decision_path=os.environ.get("RECLAMORUTA_DECISION", "configs/decision.json"),
        routes_path=os.environ.get("RECLAMORUTA_ROUTES", "configs/routes.json"),
        catalog_path=os.environ.get("RECLAMORUTA_CATALOG", "reports/BANKING77_LABEL_TO_ID.json"),
        device=os.environ.get("RECLAMORUTA_DEVICE", "cpu"),
    )


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="ReclamoRuta", page_icon="📨")

    @st.cache_resource(show_spinner="Cargando modelo…")
    def cached(settings: _Settings) -> Runtime:  # failures are not cached
        return load_runtime(
            artifact_dir=Path(settings.artifact_dir), decision_path=Path(settings.decision_path),
            routes_path=Path(settings.routes_path), catalog_path=Path(settings.catalog_path),
            device=settings.device,
        )

    try:
        runtime = cached(_settings())
    except AppConfigError as exc:
        st.title("ReclamoRuta")
        st.error(f"La aplicación no puede iniciar: {exc}")
        st.stop()
        return
    render(runtime)


if __name__ == "__main__":
    main()
