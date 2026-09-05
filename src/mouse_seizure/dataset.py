"""Build and cache the per-window feature table.

Features are cached once per recording at the inference stride (0.5 s). Because
the window is 2.0 s, every fourth cached row is a *non-overlapping* window, so
``[::4]`` yields the training set without a second pass and without subsampling
negatives -- which would distort the class prior and, with it, the calibration
the hysteresis thresholds depend on.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from mouse_seizure import features
from mouse_seizure.causal import causal_bandpass
from mouse_seizure.config import STRIDE_S, TARGET_FS, WINDOW_S, project_root
from mouse_seizure.io import Recording, load_recording
from mouse_seizure.preprocess import (
    bandpass,
    frame,
    frame_labels,
    resample_labels,
    resample_to,
    zscore,
)

CACHE_DIR = project_root() / "data" / "interim" / "features"
CAUSAL_CACHE_DIR = project_root() / "data" / "interim" / "features_causal"
NON_OVERLAP_STEP = int(round(WINDOW_S / STRIDE_S))


def prepare(rec: Recording, *, causal: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Resample, filter and standardise one recording to the common rate.

    With ``causal=True`` the zero-phase filter is replaced by a forward-only one.
    That changes the signal, so causal features must be recomputed rather than
    reused: training on zero-phase data and predicting on causally filtered data
    would introduce a domain shift we would then misread as the cost of going
    real-time.
    """
    resampled = resample_to(rec.ecog, rec.fs)
    filtered = causal_bandpass(resampled) if causal else bandpass(resampled, TARGET_FS)
    return zscore(filtered), resample_labels(rec.labels, rec.fs)


def build(path: str | Path, *, causal: bool = False) -> pl.DataFrame:
    """Feature table for one recording, one row per window at the inference stride."""
    rec = load_recording(path)
    x, y = prepare(rec, causal=causal)
    windows = frame(x, TARGET_FS)
    matrix = features.compute(windows, TARGET_FS)
    overlap = frame_labels(y, TARGET_FS)

    return pl.DataFrame(
        {name: matrix[:, i] for i, name in enumerate(features.FEATURE_NAMES)}
        | {
            "label_fraction": overlap.astype(np.float32),
            "window_index": np.arange(len(matrix), dtype=np.int32),
        }
    ).with_columns(
        pl.lit(rec.name).alias("file"),
        pl.lit(rec.mouse).alias("mouse"),
        pl.lit(rec.split).alias("split"),
        pl.lit(len(x)).alias("n_samples"),
    )


def cached(
    path: str | Path, *, rebuild: bool = False, causal: bool = False
) -> pl.DataFrame:
    """Feature table for one recording, computed once and reused thereafter."""
    path = Path(path)
    directory = CAUSAL_CACHE_DIR if causal else CACHE_DIR
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{path.stem}.parquet"
    if target.exists() and not rebuild:
        return pl.read_parquet(target)
    table = build(path, causal=causal)
    table.write_parquet(target)
    return table


def training_rows(table: pl.DataFrame) -> pl.DataFrame:
    """Non-overlapping subset of a cached table (every ``WINDOW_S / STRIDE_S`` row)."""
    return table.filter(pl.col("window_index") % NON_OVERLAP_STEP == 0)


def label_windows(label_fraction: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Binarise window labels by overlap fraction.

    A majority rule is used rather than any-overlap: at 0.5 s stride a window
    clipping one end of an event by a single sample would otherwise be a positive
    example that looks almost entirely like baseline, which blurs the very
    boundaries the post-processing then has to recover.
    """
    return (label_fraction >= threshold).astype(np.int8)


SIGNAL_CACHE_DIR = project_root() / "data" / "interim" / "signal"


def cached_signal(
    path: str | Path, *, rebuild: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Preprocessed waveform and per-sample labels, memory-mapped.

    The U-Net consumes the signal itself rather than the feature table, and 962
    hours at 400 Hz is ~5 GB. Caching as float32 ``.npy`` and opening with
    ``mmap_mode`` lets a training loop sample random segments from any recording
    without holding the cohort in RAM.
    """
    path = Path(path)
    SIGNAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    sig_path = SIGNAL_CACHE_DIR / f"{path.stem}_x.npy"
    lab_path = SIGNAL_CACHE_DIR / f"{path.stem}_y.npy"

    if rebuild or not (sig_path.exists() and lab_path.exists()):
        rec = load_recording(path)
        x, y = prepare(rec)
        np.save(sig_path, x.astype(np.float32))
        np.save(lab_path, y.astype(np.uint8))

    return (
        np.load(sig_path, mmap_mode="r"),
        np.load(lab_path, mmap_mode="r"),
    )
