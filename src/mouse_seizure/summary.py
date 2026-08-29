"""Dataset-level summary tables, shared by the audit script and the EDA notebook."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from mouse_seizure.io import events, find_recordings, load_recording


def recording_summary(path: str | Path) -> dict:
    """Metadata and label statistics for one file, without reading the signal."""
    rec = load_recording(path, signal=False)
    ev = events(rec.labels, rec.fs)
    dur = ev[:, 1] - ev[:, 0]
    return {
        "file": rec.name,
        "split": rec.split,
        "mouse": rec.mouse,
        "day": rec.day,
        "fs": rec.fs,
        "hours": rec.duration_s / 3600,
        "mid_field": rec.mid_field,
        "mid_ok": int(rec.mouse.split("_M")[1]) == int(rec.mid_field or -1),
        "positive_pct": float(rec.labels.mean() * 100),
        "n_events": len(ev),
        "events_per_hour": len(ev) / (rec.duration_s / 3600),
        "median_duration_s": float(np.median(dur)) if len(ev) else float("nan"),
        "starts_in_event": rec.starts_in_event,
        "ends_in_event": rec.ends_in_event,
    }


def dataset_table(root: str | Path | None = None) -> pl.DataFrame:
    """One row per recording, sorted by split then file name."""
    rows = [recording_summary(p) for p in find_recordings(root)]
    return pl.DataFrame(rows).sort(["split", "file"])


def all_event_durations(root: str | Path | None = None) -> np.ndarray:
    """Pooled event durations in seconds across every recording."""
    out = []
    for p in find_recordings(root):
        rec = load_recording(p, signal=False)
        ev = events(rec.labels, rec.fs)
        out.append(ev[:, 1] - ev[:, 0])
    return np.concatenate(out)


def all_event_gaps(root: str | Path | None = None) -> np.ndarray:
    """Pooled inter-event gaps in seconds. Sets the merge-gap rule."""
    out = []
    for p in find_recordings(root):
        rec = load_recording(p, signal=False)
        ev = events(rec.labels, rec.fs)
        if len(ev) > 1:
            out.append(ev[1:, 0] - ev[:-1, 1])
    return np.concatenate(out)
