"""Cache preprocessed waveform + labels for the U-Net. Run once (~10 min, ~6 GB).

uv run python experiments/build_signal_cache.py
"""

from __future__ import annotations

import time

from mouse_seizure.dataset import cached_signal
from mouse_seizure.io import find_recordings, parse_name

paths = [p for p in find_recordings() if parse_name(p.stem)[1] == "train"]
print(f"{len(paths)} training recordings")
for i, path in enumerate(paths, 1):
    start = time.perf_counter()
    x, _ = cached_signal(path)
    print(
        f"[{i:2d}/{len(paths)}] {path.stem:12s} {len(x):>10,} samples "
        f"{time.perf_counter() - start:5.1f}s",
        flush=True,
    )
print("done")
