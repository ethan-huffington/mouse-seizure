"""Signal conditioning: resampling, band-pass, normalization.

Kept deliberately thin -- these wrap ``scipy.signal`` so that the sampling-rate
mismatch between the training (1000 Hz) and test (400 Hz) splits is handled in
exactly one place.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
from scipy import signal

from mouse_seizure.config import BANDPASS, STRIDE_S, TARGET_FS, WINDOW_S


def resample_to(x: np.ndarray, fs_in: float, fs_out: float = TARGET_FS) -> np.ndarray:
    """Resample by a rational factor, anti-aliasing on the way down.

    ``resample_poly`` is used rather than Fourier resampling because it does not
    assume the signal is periodic -- a 24-hour recording emphatically is not.
    """
    if fs_in == fs_out:
        return x
    ratio = Fraction(fs_out / fs_in).limit_denominator(1000)
    return signal.resample_poly(x, ratio.numerator, ratio.denominator)


def resample_labels(y: np.ndarray, fs_in: float, fs_out: float = TARGET_FS) -> np.ndarray:
    """Resample a binary label array by nearest-neighbour index mapping.

    Labels must not be filtered: interpolating a 0/1 mask invents fractional
    values at every event boundary. Index mapping keeps them binary and keeps
    onsets within one sample of their true position.
    """
    if fs_in == fs_out:
        return y
    n_out = int(round(len(y) * fs_out / fs_in))
    idx = (np.arange(n_out) * (fs_in / fs_out)).round().astype(int)
    return y[np.clip(idx, 0, len(y) - 1)]


def bandpass(
    x: np.ndarray,
    fs: float = TARGET_FS,
    corners: tuple[float, float] = BANDPASS,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth band-pass.

    ``sosfiltfilt`` runs the filter forwards and backwards so event timing is
    not shifted. That is only legitimate offline; the causal variant (see
    ``project_brief.md`` section 8) must use a one-directional filter instead.
    """
    lo, hi = corners
    nyquist = fs / 2
    hi = min(hi, nyquist * 0.99)
    sos = signal.butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x)


def zscore(x: np.ndarray) -> np.ndarray:
    """Per-session standardisation.

    Signal units are arbitrary and amplitude scale differs between animals and
    days, so raw amplitude must not leak into features.
    """
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else x - x.mean()


def frame(
    x: np.ndarray,
    fs: float = TARGET_FS,
    window_s: float = WINDOW_S,
    stride_s: float = STRIDE_S,
) -> np.ndarray:
    """Slice a signal into overlapping windows, shape ``(n_windows, n_samples)``.

    Returns a strided view rather than a copy, so windowing a 24-hour recording
    at 75% overlap costs no extra memory.
    """
    n = int(round(window_s * fs))
    step = int(round(stride_s * fs))
    if len(x) < n:
        return np.empty((0, n), dtype=x.dtype)
    return np.lib.stride_tricks.sliding_window_view(x, n)[::step]


def frame_labels(
    y: np.ndarray,
    fs: float = TARGET_FS,
    window_s: float = WINDOW_S,
    stride_s: float = STRIDE_S,
) -> np.ndarray:
    """Positive fraction of each window, aligned with :func:`frame`.

    Left as a fraction rather than thresholded to a class: the overlap fraction
    that makes a window positive is a modeling decision, not a loading one.
    """
    return frame(y.astype(np.float32), fs, window_s, stride_s).mean(axis=1)
