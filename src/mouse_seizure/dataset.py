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
NON_OVERLAP_STEP = int(round(WINDOW_S / STRIDE_S))


def prepare(rec: Recording) -> tuple[np.ndarray, np.ndarray]:
    """Resample, filter and standardise one recording to the common rate."""
    x = zscore(bandpass(resample_to(rec.ecog, rec.fs), TARGET_FS))
    y = resample_labels(rec.labels, rec.fs)
    return x, y


def build(path: str | Path) -> pl.DataFrame:
    """Feature table for one recording, one row per window at the inference stride."""
    rec = load_recording(path)
    x, y = prepare(rec)
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


def cached(path: str | Path, *, rebuild: bool = False) -> pl.DataFrame:
    """Feature table for one recording, computed once and reused thereafter."""
    path = Path(path)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    target = CACHE_DIR / f"{path.stem}.parquet"
    if target.exists() and not rebuild:
        return pl.read_parquet(target)
    table = build(path)
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
