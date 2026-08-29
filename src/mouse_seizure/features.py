"""Per-window features.

Phase 1 keeps this deliberately small. The EDA established that two hand-computed
quantities -- relative power at the SWD fundamental and at its harmonics -- already
reach ~0.96 window-level AUC, so a large feature set is not what stands between us
and a working detector. The set below is the smallest one that covers the three
mechanisms the EDA identified as actually discriminative:

* **harmonics** -- a spike-and-wave complex is not a sinusoid, so it deposits energy
  well above its fundamental. Theta rhythm, the main confound, does not.
* **morphology** -- line length and amplitude capture the sharp spike directly.
* **rhythmicity** -- autocorrelation structure separates a regular spike train from
  an equally-powered but irregular oscillation.

Everything is computed on a ``(n_windows, n_samples)`` matrix at once rather than
per window, because a 24-hour recording is millions of windows.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sps

from mouse_seizure.config import TARGET_FS

BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "fundamental": (4.0, 9.0),
    "beta": (9.0, 14.0),
    "harmonic": (14.0, 30.0),
    "high": (30.0, 100.0),
}

FEATURE_NAMES: list[str] = [
    *(f"rel_{b}" for b in BANDS),
    "harmonic_ratio",
    "spectral_entropy",
    "peak_freq",
    "log_line_length",
    "log_variance",
    "kurtosis",
    "autocorr_peak",
    "autocorr_lag_hz",
]


def _band_powers(freqs: np.ndarray, psd: np.ndarray) -> dict[str, np.ndarray]:
    return {
        name: np.trapezoid(
            psd[:, (m := (freqs >= lo) & (freqs < hi))], freqs[m], axis=1
        )
        for name, (lo, hi) in BANDS.items()
    }


def _autocorrelation(windows: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Height and position of the strongest autocorrelation peak in 3-12 Hz.

    A rhythmic spike train repeats at a consistent interval, so its
    autocorrelation has a tall, sharply located secondary peak. Irregular
    activity of the same power does not.
    """
    x = windows - windows.mean(axis=1, keepdims=True)
    n = x.shape[1]
    # Autocorrelation via FFT: cheaper than np.correlate for millions of windows.
    size = int(2 ** np.ceil(np.log2(2 * n - 1)))
    spec = np.fft.rfft(x, size, axis=1)
    ac = np.fft.irfft(spec * np.conj(spec), size, axis=1)[:, :n]
    ac /= np.where(ac[:, :1] == 0, 1, ac[:, :1])

    lo_lag, hi_lag = int(fs / 12.0), int(fs / 3.0)
    window = ac[:, lo_lag:hi_lag]
    idx = np.argmax(window, axis=1)
    peak = window[np.arange(len(window)), idx]
    lag_hz = fs / (idx + lo_lag)
    return peak, lag_hz


def compute(windows: np.ndarray, fs: float = TARGET_FS) -> np.ndarray:
    """Feature matrix ``(n_windows, len(FEATURE_NAMES))`` for framed signal."""
    if len(windows) == 0:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)

    nperseg = min(windows.shape[1], 512)
    freqs, psd = sps.welch(windows, fs, nperseg=nperseg, axis=1)

    powers = _band_powers(freqs, psd)
    total = sum(powers.values()) + 1e-12
    rel = {name: p / total for name, p in powers.items()}

    # Normalised spectral entropy: flat spectra score high, peaked ones low.
    band = (freqs >= 1) & (freqs <= 100)
    p_norm = psd[:, band] / (psd[:, band].sum(axis=1, keepdims=True) + 1e-12)
    entropy = -(p_norm * np.log(p_norm + 1e-12)).sum(axis=1) / np.log(p_norm.shape[1])

    peak_freq = freqs[band][np.argmax(psd[:, band], axis=1)]

    line_length = np.abs(np.diff(windows, axis=1)).sum(axis=1)
    variance = windows.var(axis=1)
    centred = windows - windows.mean(axis=1, keepdims=True)
    kurtosis = (centred**4).mean(axis=1) / (variance**2 + 1e-12)

    ac_peak, ac_lag = _autocorrelation(windows, fs)

    return np.column_stack(
        [
            *(rel[b] for b in BANDS),
            rel["harmonic"] / (rel["fundamental"] + 1e-12),
            entropy,
            peak_freq,
            np.log1p(line_length),
            np.log1p(variance),
            kurtosis,
            ac_peak,
            ac_lag,
        ]
    ).astype(np.float32)
