"""Featurize every training recording once, into data/interim/features/.

Roughly 14 s per recording. Re-running is cheap: cached files are skipped.

    uv run python experiments/build_feature_cache.py
"""

from __future__ import annotations

import time

from mouse_seizure.dataset import CACHE_DIR, cached
from mouse_seizure.io import find_recordings, parse_name

paths = [p for p in find_recordings() if parse_name(p.stem)[1] == "train"]
print(f"{len(paths)} training recordings -> {CACHE_DIR}", flush=True)

for i, path in enumerate(paths, 1):
    start = time.perf_counter()
    hit = (CACHE_DIR / f"{path.stem}.parquet").exists()
    table = cached(path)
    print(
        f"[{i:2d}/{len(paths)}] {path.stem:12s} {table.height:>7,} windows  "
        f"{'cached' if hit else f'{time.perf_counter() - start:5.1f}s'}",
        flush=True,
    )

print("done", flush=True)
