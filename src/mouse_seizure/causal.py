"""Causal detection: the same pipeline restricted to past signal only.

The offline pipeline is free to look forward. It filters with ``sosfiltfilt``,
which runs the filter forwards and backwards so phase shifts cancel, and it
smooths a probability trace that already contains the whole recording. Both are
legitimate when the session is on disk and both are impossible in real time.

Three substitutions make the chain causal, and each one costs something:

* **One-directional filtering.** A forward-only Butterworth introduces a
  frequency-dependent group delay instead of cancelling it, so event boundaries
  shift later.
* **Trailing-window overlap-add.** A sample can only be informed by windows that
  have already closed, so the probability at time t uses strictly less evidence
  than the offline trace does.
* **A persistence counter instead of smoothing.** Declaring an event requires N
  consecutive positive steps. This is the real-time analogue of a median filter:
  it suppresses isolated blips, and it delays every detection by exactly N steps.

The point of this module is not to beat the offline detector -- it cannot. It is
to measure what the constraint costs, in both accuracy and latency, which is the
number that matters for a closed-loop stimulation context.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from mouse_seizure.config import BANDPASS, MIN_DURATION_S, STRIDE_S, TARGET_FS


def causal_bandpass(
    x: np.ndarray,
    fs: float = TARGET_FS,
    corners: tuple[float, float] = BANDPASS,
    order: int = 4,
) -> np.ndarray:
    """Forward-only Butterworth band-pass.

    ``sosfilt`` rather than ``sosfiltfilt``: no backward pass, so the filter uses
    only past samples and leaves a group delay in the output.
    """
    lo, hi = corners
    hi = min(hi, fs / 2 * 0.99)
    sos = signal.butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")
    return signal.sosfilt(sos, x)


def trailing_probability(
    window_probs: np.ndarray, n_steps: int | None = None
) -> np.ndarray:
    """Probability per stride step using only windows that have already closed.

    The offline overlap-add averages every window covering a sample, including
    ones extending into the future. Here a step takes the most recent completed
    window's score, which is the most information a causal system can have.
    """
    if n_steps is None:
        n_steps = len(window_probs)
    out = np.zeros(n_steps, dtype=np.float64)
    usable = min(n_steps, len(window_probs))
    out[:usable] = window_probs[:usable]
    return out


@dataclass(frozen=True)
class CausalDetection:
    """Events plus the latency the persistence rule imposed."""

    events: np.ndarray
    declaration_delay_s: float


def persistence_detect(
    window_probs: np.ndarray,
    threshold: float = 0.5,
    persistence: int = 3,
    stride_s: float = STRIDE_S,
    min_duration_s: float = MIN_DURATION_S,
    release: int = 2,
) -> CausalDetection:
    """Declare an event after ``persistence`` consecutive positive steps.

    ``release`` consecutive negative steps end it, which is the causal analogue
    of hysteresis: entering is deliberately harder than staying.

    Onsets are reported at the step where evidence *began*, not where the counter
    fired, because that is the true event boundary; the delay between the two is
    returned separately as the cost of the rule rather than hidden inside it.
    """
    if persistence < 1 or release < 1:
        raise ValueError("persistence and release must be >= 1")

    positive = window_probs >= threshold
    events: list[tuple[float, float]] = []

    active = False
    run_pos = run_neg = 0
    candidate_start = 0

    for i, is_pos in enumerate(positive):
        if is_pos:
            if run_pos == 0:
                candidate_start = i
            run_pos += 1
            run_neg = 0
            if not active and run_pos >= persistence:
                active = True
                start_step = candidate_start
        else:
            run_neg += 1
            run_pos = 0
            if active and run_neg >= release:
                active = False
                events.append((start_step * stride_s, (i - run_neg + 1) * stride_s))

    if active:
        events.append((start_step * stride_s, len(positive) * stride_s))

    if not events:
        return CausalDetection(np.empty((0, 2)), persistence * stride_s)

    array = np.array(events)
    keep = (array[:, 1] - array[:, 0]) >= min_duration_s
    return CausalDetection(array[keep], persistence * stride_s)
