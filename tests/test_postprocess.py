"""Event formation is where a silent bug is most costly, so it is pinned here."""

import numpy as np

from mouse_seizure import postprocess as pp

FS = 400.0


def _mask(spans: list[tuple[float, float]], seconds: float = 120.0) -> np.ndarray:
    mask = np.zeros(int(seconds * FS), dtype=bool)
    for start, end in spans:
        mask[int(start * FS) : int(end * FS)] = True
    return mask


def test_merge_gap_heals_a_dropout() -> None:
    """A brief dropout must not split one event into two."""
    mask = _mask([(10.0, 12.0), (12.2, 15.0)])  # 0.2 s gap, below the 0.5 s rule
    events = pp.form_events(mask, FS)
    assert len(events) == 1
    assert np.allclose(events[0], [10.0, 15.0])


def test_merge_gap_respects_genuinely_separate_events() -> None:
    """A gap wider than the rule must leave two events standing."""
    mask = _mask([(10.0, 12.0), (13.0, 15.0)])  # 1.0 s gap, above the 0.5 s rule
    assert len(pp.form_events(mask, FS)) == 2


def test_min_duration_discards_a_blip() -> None:
    mask = _mask([(10.0, 15.0), (30.0, 30.4)])  # 0.4 s blip, below the 1.0 s floor
    events = pp.form_events(mask, FS)
    assert len(events) == 1
    assert events[0, 0] == 10.0


def test_merge_precedes_duration_filter() -> None:
    """Two sub-threshold fragments of one event survive by being merged first.

    Filtering before merging would discard both, losing a real seizure.
    """
    mask = _mask([(10.0, 10.6), (10.8, 11.4)])  # each < 1.0 s, merged = 1.4 s
    events = pp.form_events(mask, FS)
    assert len(events) == 1
    assert events[0, 1] - events[0, 0] >= 1.0


def test_events_touching_the_boundary_are_paired() -> None:
    """A recording starting and ending mid-event still yields matched spans."""
    mask = np.ones(int(10 * FS), dtype=bool)
    events = pp.form_events(mask, FS)
    assert len(events) == 1
    assert events[0, 1] > events[0, 0]


def test_overlap_add_reconstructs_a_constant() -> None:
    """Averaging overlapping windows of probability 1.0 must return 1.0."""
    n = int(60 * FS)
    n_windows = int((60 - 2.0) / 0.5) + 1
    trace = pp.probability_trace(np.ones(n_windows), n, FS)
    assert np.allclose(trace, 1.0)


def test_hysteresis_needs_a_seed_above_the_high_threshold() -> None:
    """A run that never exceeds `high` is rejected however long it stays above `low`."""
    trace = np.full(int(10 * FS), 0.4)
    assert not pp.hysteresis_threshold(trace, high=0.6, low=0.3).any()
    trace[int(5 * FS)] = 0.7
    assert pp.hysteresis_threshold(trace, high=0.6, low=0.3).all()


def test_hysteresis_rejects_inverted_thresholds() -> None:
    import pytest

    with pytest.raises(ValueError):
        pp.hysteresis_threshold(np.zeros(10), high=0.3, low=0.6)
