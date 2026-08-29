"""Central parameter set. Every value here is derived from measurement.

This module exists so no parameter is ever set at a call site. The
``timescoring`` defaults in particular are scaled for human clinical seizures
and produce meaningless, inflated numbers on rodent data *silently* -- there is
no error, just implausibly good results. Routing every score through
``EVENT_SCORING`` makes that failure impossible.

Provenance for each value is in ``project_brief.md``; the measurements behind
them are reproduced by ``experiments/audit_dataset.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def project_root() -> Path:
    """Directory containing ``pyproject.toml``.

    Resolved by walking up from this file rather than from the working
    directory, so a notebook in ``notebooks/`` and a script at the root both
    find the data without either one caring where it was launched from.
    """
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError("pyproject.toml not found above " + str(__file__))


RAW_DATA = project_root() / "data" / "raw"
"""Where the Zenodo ``.mat`` files live."""

# --- Signal ---------------------------------------------------------------

TARGET_FS = 400.0
"""Common sampling rate, Hz.

The test split's native rate, so evaluation data is never resampled; training
decimates 1000 -> 400 by a clean 2/5 polyphase ratio. The dataset paper uses
100 Hz, which smooths away the sharp spike component the morphology features
depend on.
"""

BANDPASS = (1.0, 100.0)
"""Band-pass corners, Hz. The upper corner is the common bandwidth ceiling
applied to both splits, so they match spectrally despite different
acquisition-side anti-alias filtering."""

# --- Windowing ------------------------------------------------------------

WINDOW_S = 2.0
"""Window length, seconds. Sits inside a typical 5.6 s event while giving
~0.5 Hz spectral resolution -- enough to separate the 5.9 Hz fundamental from
its neighbourhood."""

STRIDE_S = 0.5
"""Hop between windows, seconds (75% overlap at a 2 s window)."""

# --- Event formation ------------------------------------------------------

MERGE_GAP_S = 0.5
"""Fuse detected events separated by less than this.

Measured: 0.86% of genuinely-separate annotated event pairs are closer than
0.5 s, versus 1.75% below 1.0 s. Annotators do separate events across
sub-second gaps, so merging harder destroys real structure.
"""

MIN_DURATION_S = 1.0
"""Discard detected events shorter than this. Discards 0.02% of true events and
matches the annotators' own effective floor."""

# --- Scoring --------------------------------------------------------------

SWD_FUNDAMENTAL_HZ = 5.9
"""Measured SWD fundamental. Note that baseline theta peaks here too, so peak
frequency alone does not discriminate."""


@dataclass(frozen=True)
class EventScoringParams:
    """Rodent-scale overrides for every ``timescoring`` default.

    Defaults shown for contrast: toleranceStart 30 s, toleranceEnd 60 s,
    minDurationBetweenEvents 90 s, maxEventDuration 300 s.
    """

    toleranceStart: float = 0.5
    toleranceEnd: float = 0.5
    minOverlap: float = 0.0
    maxEventDuration: float = 60.0
    minDurationBetweenEvents: float = MERGE_GAP_S


EVENT_SCORING = EventScoringParams()
"""The only event-scoring parameter set. Never call timescoring without it."""
