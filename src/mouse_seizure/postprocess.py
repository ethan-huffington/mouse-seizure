"""Turn per-window probabilities into an event list.

This is the stage that decides whether one seizure is counted once, three times,
or not at all, and it drives event-level performance far more than the choice of
classifier. Five coupled steps:

1. reconcile overlapping windows into a per-sample probability trace
2. threshold with hysteresis
3. smooth
4. group into events, then merge short gaps and drop short events
5. (aggregation lives in ``evaluate``)

No library does this well, so it is ours. It is about fifty lines of run-length
logic and every one of them matters.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from mouse_seizure.config import (
    MERGE_GAP_S,
    MIN_DURATION_S,
    STRIDE_S,
    TARGET_FS,
    WINDOW_S,
)


def probability_trace(
    window_probs: np.ndarray,
    n_samples: int,
    fs: float = TARGET_FS,
    window_s: float = WINDOW_S,
    stride_s: float = STRIDE_S,
) -> np.ndarray:
    """Overlap-add window probabilities into one probability per sample.

    Each sample is covered by several overlapping windows; averaging their votes
    turns redundant predictions into a single decision rather than double counts.
    Implemented as two cumulative sums (weighted total and coverage count) so the
    cost does not depend on the overlap factor.
    """
    n_win = int(round(window_s * fs))
    step = int(round(stride_s * fs))

    total = np.zeros(n_samples + 1, dtype=np.float64)
    count = np.zeros(n_samples + 1, dtype=np.float64)
    starts = np.arange(len(window_probs)) * step
    ends = np.minimum(starts + n_win, n_samples)

    # Difference-array trick: add at the start index, subtract past the end, then
    # cumulative-sum once. Avoids touching every sample of every window.
    np.add.at(total, starts, window_probs)
    np.add.at(total, ends, -window_probs)
    np.add.at(count, starts, 1.0)
    np.add.at(count, ends, -1.0)

    trace = np.cumsum(total)[:n_samples]
    cover = np.cumsum(count)[:n_samples]
    return np.divide(trace, cover, out=np.zeros_like(trace), where=cover > 0)


def hysteresis_threshold(trace: np.ndarray, high: float, low: float) -> np.ndarray:
    """Dual-threshold binarisation.

    A region is positive if it ever exceeds ``high`` and stays positive while
    above ``low``. Using one threshold makes boundaries flicker; this is the
    same idea as a Schmitt trigger.
    """
    if low > high:
        raise ValueError(f"low ({low}) must not exceed high ({high})")
    above_low = trace >= low
    labels, n = ndimage.label(above_low)
    if n == 0:
        return np.zeros_like(trace, dtype=bool)
    # Keep only the runs that contain at least one sample above `high`.
    seeded = np.zeros(n + 1, dtype=bool)
    seeded[1:] = ndimage.maximum(trace, labels, index=np.arange(1, n + 1)) >= high
    return seeded[labels]


def smooth(
    trace: np.ndarray, fs: float = TARGET_FS, seconds: float = 0.5
) -> np.ndarray:
    """Median filter the probability trace to enforce temporal coherence."""
    size = max(1, int(round(seconds * fs)) | 1)  # odd size
    return ndimage.median_filter(trace, size=size, mode="nearest")


def form_events(
    mask: np.ndarray,
    fs: float = TARGET_FS,
    merge_gap_s: float = MERGE_GAP_S,
    min_duration_s: float = MIN_DURATION_S,
) -> np.ndarray:
    """Run-length group a binary mask into an ``(n, 2)`` array of second offsets.

    Merging happens before the duration filter: two fragments of one event should
    be fused and then judged on their combined length, not discarded separately.
    """
    padded = np.concatenate(([0], mask.astype(np.int8), [0]))
    diff = np.diff(padded)
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    if len(starts) == 0:
        return np.empty((0, 2))

    merge_gap = merge_gap_s * fs
    merged_starts, merged_ends = [starts[0]], [ends[0]]
    for s, e in zip(starts[1:], ends[1:]):
        if s - merged_ends[-1] <= merge_gap:
            merged_ends[-1] = e
        else:
            merged_starts.append(s)
            merged_ends.append(e)

    events = np.column_stack([merged_starts, merged_ends]) / fs
    keep = (events[:, 1] - events[:, 0]) >= min_duration_s
    return events[keep]


def events_to_mask(
    events: np.ndarray, n_samples: int, fs: float = TARGET_FS
) -> np.ndarray:
    """Rasterise an event list back to a per-sample binary mask."""
    mask = np.zeros(n_samples, dtype=bool)
    for start, end in events:
        mask[int(round(start * fs)) : int(round(end * fs))] = True
    return mask


def detect(
    window_probs: np.ndarray,
    n_samples: int,
    high: float = 0.6,
    low: float = 0.3,
    fs: float = TARGET_FS,
    smooth_s: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Full chain: window probabilities in, ``(events, probability_trace)`` out."""
    trace = probability_trace(window_probs, n_samples, fs)
    smoothed = smooth(trace, fs, smooth_s)
    mask = hysteresis_threshold(smoothed, high, low)
    return form_events(mask, fs), trace
