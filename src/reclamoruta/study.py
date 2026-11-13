"""Human routing-time study: manual vs. assisted, anonymous and text-free by default.

Each trial records an anonymous participant ID, a message ID, the condition,
elapsed seconds (monotonic clock), the suggestion (assisted only) and the
final human choice. Message text is NOT kept unless ``store_text=True``.
Nothing here retrains a model or touches weights or test data; export writes
one new CSV, explicitly requested, with spreadsheet formulas neutralised.

``summarize`` turns trials into the time inputs of ``value.compute_value``
(``t_manual``, ``t_aceptar``, ``t_corregir``, ``t_revision``). They remain
estimates from a small study and must be documented as such.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
import csv
from dataclasses import asdict, dataclass, fields
import hashlib
from pathlib import Path
import random
import re
import secrets
import statistics
import time

CONDITIONS = ("manual", "asistida")
_PARTICIPANT = re.compile(r"P-[0-9a-f]{8}")
_MESSAGE = re.compile(r"[A-Za-z0-9_.:\-]{1,64}")
_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


class StudyError(ValueError):
    """Invalid study input or unsafe export."""


def new_participant_id() -> str:
    """Random anonymous ID; never derive it from a name or email."""
    return f"P-{secrets.token_hex(4)}"


@dataclass(frozen=True)
class Trial:
    participant_id: str
    message_id: str
    condition: str
    elapsed_s: float
    suggested_intent: str | None
    suggested_route: str | None
    final_intent: str
    final_route: str
    corrected: bool
    needs_review: bool | None
    model_version: str | None
    text: str | None = None


class StudySession:
    """One participant's trials, in memory. ``start`` then ``finish`` per message."""

    def __init__(
        self, *, participant_id: str, intent_to_route: Mapping[str, str],
        clock: Callable[[], float] = time.perf_counter, store_text: bool = False,
    ) -> None:
        if not isinstance(participant_id, str) or not _PARTICIPANT.fullmatch(participant_id):
            raise StudyError("participant_id debe ser anónimo con formato P-xxxxxxxx (use new_participant_id)")
        if not intent_to_route:
            raise StudyError("Se requiere el mapa intención→ruta")
        self.participant_id = participant_id
        self.routes = dict(intent_to_route)
        self.clock = clock
        self.store_text = store_text
        self.trials: list[Trial] = []
        self._open: dict | None = None
        self._seen: set[str] = set()

    def start(
        self, message_id: str, condition: str, *, text: str | None = None,
        suggested_intent: str | None = None, needs_review: bool | None = None,
        model_version: str | None = None,
    ) -> None:
        if self._open is not None:
            raise StudyError("Hay un ensayo abierto; termínelo antes de iniciar otro")
        if not isinstance(message_id, str) or not _MESSAGE.fullmatch(message_id):
            raise StudyError("message_id inválido (solo letras, números y _.:-)")
        if message_id in self._seen:
            raise StudyError("Este participante ya respondió ese mensaje")
        if condition not in CONDITIONS:
            raise StudyError(f"condition debe ser una de {CONDITIONS}")
        if condition == "manual" and suggested_intent is not None:
            raise StudyError("La condición manual no muestra sugerencia")
        if condition == "asistida":
            if suggested_intent not in self.routes:
                raise StudyError("La condición asistida requiere una sugerencia del catálogo")
            if not model_version:
                raise StudyError("La condición asistida requiere model_version")
        self._open = {
            "message_id": message_id, "condition": condition, "start": self.clock(),
            "text": text if self.store_text else None, "suggested_intent": suggested_intent,
            "needs_review": needs_review if condition == "asistida" else None,
            "model_version": model_version if condition == "asistida" else None,
        }

    def finish(self, final_intent: str) -> Trial:
        if self._open is None:
            raise StudyError("No hay un ensayo iniciado")
        if final_intent not in self.routes:
            raise StudyError(f"Intención final desconocida: {final_intent}")
        elapsed = self.clock() - self._open["start"]
        if elapsed < 0:
            raise StudyError("El reloj no es monotónico")
        opened, self._open = self._open, None
        suggested = opened["suggested_intent"]
        trial = Trial(
            participant_id=self.participant_id, message_id=opened["message_id"],
            condition=opened["condition"], elapsed_s=elapsed,
            suggested_intent=suggested, suggested_route=self.routes.get(suggested) if suggested else None,
            final_intent=final_intent, final_route=self.routes[final_intent],
            corrected=suggested is not None and final_intent != suggested,
            needs_review=opened["needs_review"], model_version=opened["model_version"],
            text=opened["text"],
        )
        self._seen.add(trial.message_id)
        self.trials.append(trial)
        return trial


def assign_conditions(message_ids: Sequence[str], participant_id: str, seed: int) -> dict[str, str]:
    """Balanced, reproducible manual/asistida assignment per participant."""
    ids = list(message_ids)
    if len(set(ids)) != len(ids):
        raise StudyError("message_ids repetidos")
    key = hashlib.sha256(f"{seed}:{participant_id}".encode()).hexdigest()
    shuffled = ids[:]
    random.Random(key).shuffle(shuffled)
    half = len(shuffled) // 2
    assignment = {m: "manual" for m in shuffled[:half]}
    assignment.update({m: "asistida" for m in shuffled[half:]})
    return {m: assignment[m] for m in sorted(ids)}


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def summarize(trials: Iterable[Trial]) -> dict:
    """Per-condition counts/times and medians usable as ``value.py`` time inputs.

    ``t_corregir`` estimates additional correction time as the corrected-total
    median minus the uncorrected acceptance median, excluding reviewed trials.
    This difference of group medians is not a measured correction phase and
    need not reconstruct total time for an arbitrary population. The raw
    corrected-total median remains in ``asistida.corrected_total_median_s``.
    ``correction_time_estimate`` documents the method, status and diagnostic.
    Missing groups or a negative estimate yield ``None``; equal medians yield
    a valid zero estimate.
    """
    trials = list(trials)
    manual = [t.elapsed_s for t in trials if t.condition == "manual"]
    assisted = [t for t in trials if t.condition == "asistida"]
    accepted = [t.elapsed_s for t in assisted if not t.corrected and not t.needs_review]
    corrected = [t.elapsed_s for t in assisted if t.corrected and not t.needs_review]
    reviewed = [t.elapsed_s for t in assisted if t.needs_review]
    accepted_median = _median(accepted)
    corrected_total_median = _median(corrected)
    correction_extra = None
    if accepted_median is None or corrected_total_median is None:
        correction_status = "missing_groups"
        correction_diagnostic = "Both nonreviewed accepted and corrected groups are required."
    elif corrected_total_median < accepted_median:
        correction_status = "negative_estimate"
        correction_diagnostic = (
            "Corrected-total median is below acceptance median; these aggregate "
            "samples cannot infer nonnegative additional correction time."
        )
    else:
        correction_extra = corrected_total_median - accepted_median
        correction_status = "estimated"
        correction_diagnostic = "Difference of group medians; not a measured correction phase."
    return {
        "manual": {"n": len(manual), "median_s": _median(manual),
                   "mean_s": statistics.fmean(manual) if manual else None},
        "asistida": {
            "n": len(assisted), "median_s": _median([t.elapsed_s for t in assisted]),
            "corrected_total_median_s": corrected_total_median,
            "correction_rate": sum(t.corrected for t in assisted) / len(assisted) if assisted else None,
        },
        "value_inputs": {
            "t_manual": _median(manual), "t_aceptar": accepted_median,
            "t_corregir": correction_extra, "t_revision": _median(reviewed),
        },
        "correction_time_estimate": {
            "method": "difference_of_group_medians", "status": correction_status,
            "diagnostic": correction_diagnostic,
        },
        "n_participants": len({t.participant_id for t in trials}),
        "estimated": True,
    }


def _safe_cell(value: object) -> object:
    if isinstance(value, str) and value.startswith(_FORMULA_PREFIXES):
        return "'" + value
    return value


def export_trials(trials: Iterable[Trial], out: Path) -> Path:
    """Write trials to a NEW CSV; the ``text`` column appears only if text was stored."""
    trials = list(trials)
    out = Path(out)
    if out.exists() or out.is_symlink():
        raise StudyError(f"{out} ya existe; la exportación nunca sobrescribe")
    names = [f.name for f in fields(Trial)]
    if all(t.text is None for t in trials):
        names.remove("text")
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with out.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=names, extrasaction="ignore")
            writer.writeheader()
            for trial in trials:
                writer.writerow({k: _safe_cell(v) for k, v in asdict(trial).items()})
    except FileExistsError as exc:
        raise StudyError(f"{out} ya existe") from exc
    return out
