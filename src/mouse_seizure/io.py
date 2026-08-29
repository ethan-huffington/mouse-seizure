"""Load recordings from the Zenodo C3H/HeJ mouse EEG dataset.

The files are MATLAB v7.3, which is HDF5 underneath -- ``scipy.io.loadmat`` cannot
read them, so this module uses ``h5py``. Each file holds one struct ``rec`` with
fields ``mID``, ``SWDlabel``, ``ecog`` and ``fs``.

Three properties of the raw files are handled here rather than by callers:

* Array orientation varies between files -- some are ``(n, 1)``, some ``(1, n)``.
* Sampling rate varies between the training (1000 Hz) and test (400 Hz) splits.
* The ``mID`` field is unreliable: several test files report the same value. The
  filename is the authoritative animal identifier and is what ``mouse`` reports;
  ``mid_field`` preserves the stored value so the discrepancy stays auditable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from mouse_seizure.config import RAW_DATA

TRAIN_RE = re.compile(r"^Day(?P<day>\d+)_M(?P<mouse>\d+)$")
TEST_RE = re.compile(r"^(?P<date>\d{8})_M(?P<mouse>\d+)_MAT$")


@dataclass(frozen=True)
class Recording:
    """One continuous session: signal, per-sample labels, and provenance."""

    name: str
    mouse: str
    split: str
    fs: float
    ecog: np.ndarray
    labels: np.ndarray
    day: str | None = None
    mid_field: float | None = None

    @property
    def duration_s(self) -> float:
        # Derived from labels, not ecog: labels are always loaded, whereas the
        # signal is skipped when a caller only needs metadata.
        return len(self.labels) / self.fs

    @property
    def starts_in_event(self) -> bool:
        """True if the recording begins mid-seizure (label boundary is censored)."""
        return bool(self.labels[0])

    @property
    def ends_in_event(self) -> bool:
        """True if the recording ends mid-seizure (label boundary is censored)."""
        return bool(self.labels[-1])


def parse_name(stem: str) -> tuple[str, str, str | None]:
    """Return ``(mouse, split, day)`` from a file stem.

    The mouse identifier is namespaced by split because the two splits use
    disjoint cohorts: training ``M3`` and test ``M3`` are different animals.
    """
    if m := TRAIN_RE.match(stem):
        return f"train_M{m['mouse']}", "train", m["day"]
    if m := TEST_RE.match(stem):
        return f"test_M{m['mouse']}", "test", None
    raise ValueError(f"unrecognised file name: {stem!r}")


def load_recording(path: str | Path, *, signal: bool = True) -> Recording:
    """Read one ``.mat`` file. Pass ``signal=False`` to skip the ECoG array."""
    path = Path(path)
    mouse, split, day = parse_name(path.stem)

    with h5py.File(path, "r") as f:
        rec = f["rec"]
        fs = float(np.asarray(rec["fs"]).ravel()[0])
        labels = np.asarray(rec["SWDlabel"]).ravel()
        ecog = np.asarray(rec["ecog"]).ravel() if signal else np.empty(0)
        mid = float(np.asarray(rec["mID"]).ravel()[0])

    if signal and len(ecog) != len(labels):
        raise ValueError(f"{path.name}: {len(ecog)} samples vs {len(labels)} labels")

    return Recording(
        name=path.stem,
        mouse=mouse,
        split=split,
        fs=fs,
        ecog=ecog,
        labels=labels,
        day=day,
        mid_field=mid,
    )


def events(labels: np.ndarray, fs: float) -> np.ndarray:
    """Convert a per-sample binary mask to an ``(n, 2)`` array of second offsets.

    Zero-padding both ends makes runs that touch a boundary produce matched
    transitions; without it a recording starting or ending mid-event yields
    unpaired indices and nonsensical (even negative) durations.
    """
    padded = np.concatenate(([0], labels.astype(np.int8), [0]))
    diff = np.diff(padded)
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    return np.column_stack([starts, ends]) / fs


def find_recordings(root: str | Path | None = None) -> list[Path]:
    """All dataset files under ``root`` (default: the project's data/raw)."""
    return sorted(Path(root or RAW_DATA).glob("*.mat"))
