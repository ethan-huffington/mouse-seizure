"""The causal path must never use information from the future."""

import numpy as np
import pytest

from mouse_seizure.causal import causal_bandpass, persistence_detect
from mouse_seizure.preprocess import bandpass


def test_causal_filter_ignores_the_future() -> None:
    """Changing a later sample must not change an earlier output.

    This is the property that separates a causal filter from an offline one, and
    it is the whole reason the real-time variant exists.
    """
    rng = np.random.default_rng(0)
    x = rng.standard_normal(4000)
    y = x.copy()
    y[3000:] += 50.0  # a large disturbance, strictly in the future of sample 2000

    a = causal_bandpass(x)[:2000]
    b = causal_bandpass(y)[:2000]
    assert np.allclose(a, b)


def test_offline_filter_does_use_the_future() -> None:
    """The contrast: filtfilt propagates a later disturbance backwards."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(4000)
    y = x.copy()
    y[3000:] += 50.0

    assert not np.allclose(bandpass(x)[:2000], bandpass(y)[:2000])


def test_persistence_suppresses_isolated_blips() -> None:
    probs = np.zeros(200)
    probs[50] = 0.9  # a single positive step
    assert len(persistence_detect(probs, persistence=3).events) == 0


def test_persistence_declares_a_sustained_run() -> None:
    probs = np.zeros(200)
    probs[50:70] = 0.9
    out = persistence_detect(probs, persistence=3, stride_s=0.5)
    assert len(out.events) == 1
    # Onset is reported where evidence began, not where the counter fired.
    assert out.events[0, 0] == pytest.approx(25.0)
    assert out.declaration_delay_s == pytest.approx(1.5)


def test_release_tolerates_a_brief_dropout() -> None:
    """One negative step must not end an event when release requires two."""
    probs = np.zeros(200)
    probs[50:70] = 0.9
    probs[60] = 0.0
    out = persistence_detect(probs, persistence=3, release=2)
    assert len(out.events) == 1


def test_rejects_invalid_counters() -> None:
    with pytest.raises(ValueError):
        persistence_detect(np.zeros(10), persistence=0)
