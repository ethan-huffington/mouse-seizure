"""Event-level and sample-level scoring, and seizure-burden statistics.

Event scoring uses ``timescoring`` (the SzCORE reference implementation) rather
than hand-written overlap matching, so results are comparable to published work
and the many-to-one accounting -- one detection spanning two reference events,
two detections hitting one -- is settled the way the community settles it.

**Its defaults are scaled for human clinical seizures** (30 s / 60 s tolerance,
90 s merge gap) and would silently inflate rodent numbers. Every call here goes
through :data:`mouse_seizure.config.EVENT_SCORING`; nothing takes scoring
parameters as an argument, so the defaults cannot be reached by accident.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import average_precision_score, cohen_kappa_score
from timescoring import scoring
from timescoring.annotations import Annotation

from mouse_seizure.config import EVENT_SCORING, TARGET_FS


@dataclass(frozen=True)
class Scores:
    """Event-level and sample-level metrics for one recording."""

    event_sensitivity: float
    event_precision: float
    event_f1: float
    false_alarms_per_hour: float
    sample_sensitivity: float
    sample_precision: float
    sample_f1: float
    cohen_kappa: float
    n_reference_events: int
    n_detected_events: int
    hours: float

    def as_dict(self) -> dict:
        return asdict(self)


def _params() -> scoring.EventScoring.Parameters:
    return scoring.EventScoring.Parameters(**EVENT_SCORING.__dict__)


SCORING_DECIMATION = 20
"""Decimate masks before scoring.

Events last seconds and the tolerance collar is 0.5 s, so 400/20 = 20 Hz resolution
(0.05 s) is far finer than anything the scoring depends on. Verified to leave every
metric unchanged to five decimal places while running ~20x faster, which is what
makes a full threshold sweep affordable.
"""


def score(
    reference: np.ndarray,
    hypothesis: np.ndarray,
    fs: float = TARGET_FS,
    decimation: int = SCORING_DECIMATION,
) -> Scores:
    """Score a predicted mask against a reference mask, both per-sample binary."""
    if reference.shape != hypothesis.shape:
        raise ValueError(f"shape mismatch: {reference.shape} vs {hypothesis.shape}")

    if decimation > 1:
        reference = reference[::decimation]
        hypothesis = hypothesis[::decimation]
        fs = fs / decimation

    fs_int = int(round(fs))
    ref = Annotation(reference.astype(bool), fs_int)
    hyp = Annotation(hypothesis.astype(bool), fs_int)

    ev = scoring.EventScoring(ref, hyp, _params())
    sa = scoring.SampleScoring(ref, hyp, fs_int)
    hours = len(reference) / fs / 3600

    return Scores(
        event_sensitivity=float(ev.sensitivity),
        event_precision=float(ev.precision),
        event_f1=float(ev.f1),
        false_alarms_per_hour=float(ev.fpRate) / 24.0,  # timescoring reports per 24 h
        sample_sensitivity=float(sa.sensitivity),
        sample_precision=float(sa.precision),
        sample_f1=float(sa.f1),
        cohen_kappa=float(
            cohen_kappa_score(reference.astype(bool), hypothesis.astype(bool))
        ),
        n_reference_events=len(ev.ref.events),
        n_detected_events=len(ev.hyp.events),
        hours=hours,
    )


def window_auprc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Window-level AUPRC, measured *before* post-processing.

    Separates "is the classifier good" from "is the post-processing good", which
    matter separately because they need different fixes.
    """
    return float(average_precision_score(y_true, y_prob))


@dataclass(frozen=True)
class Burden:
    """Seizure burden for one session -- the actual scientific readout."""

    n_events: int
    events_per_hour: float
    time_in_swd_pct: float
    median_duration_s: float
    total_swd_s: float
    hours: float


def burden(events: np.ndarray, n_samples: int, fs: float = TARGET_FS) -> Burden:
    """Burden statistics from an event list.

    Only meaningful on full sessions. The official test recordings are cropped at
    a fixed seizure count, so their rates describe the benchmark's construction
    rather than the animal (see ``project_brief.md`` section 2).
    """
    hours = n_samples / fs / 3600
    durations = events[:, 1] - events[:, 0] if len(events) else np.array([])
    total = float(durations.sum()) if len(durations) else 0.0
    return Burden(
        n_events=len(events),
        events_per_hour=len(events) / hours if hours else 0.0,
        time_in_swd_pct=100 * total / (n_samples / fs) if n_samples else 0.0,
        median_duration_s=float(np.median(durations))
        if len(durations)
        else float("nan"),
        total_swd_s=total,
        hours=hours,
    )
