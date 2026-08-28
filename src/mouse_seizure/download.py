"""Fetch the C3H/HeJ mouse EEG dataset from Zenodo.

Record 10.5281/zenodo.17982390 (CC-BY 4.0): single-channel EEG from 18 mice with
spontaneous spike-wave discharges, ~19.5 GB across 50 MATLAB v7.3 files.

Downloads are resumable and MD5-verified, so re-running is cheap and safe: files
already present with the right checksum are skipped. A partially written file is
resumed with a Range request rather than restarted.

    uv run python -m mouse_seizure.download
    uv run python -m mouse_seizure.download --dest data/raw --only 20240209
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import requests

RECORD_ID = "17982390"
API_URL = f"https://zenodo.org/api/records/{RECORD_ID}"
CHUNK = 1 << 20


def fetch_manifest() -> list[dict]:
    """Return the record's file entries: name, size, md5 and download URL."""
    r = requests.get(API_URL, timeout=60)
    r.raise_for_status()
    return [
        {
            "name": f["key"],
            "size": f["size"],
            "md5": f["checksum"].removeprefix("md5:"),
            "url": f["links"]["self"],
        }
        for f in r.json()["files"]
    ]


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        while block := fh.read(CHUNK):
            h.update(block)
    return h.hexdigest()


def download_file(entry: dict, dest: Path) -> str:
    """Download one file, resuming a partial transfer. Returns a status word."""
    path = dest / entry["name"]

    if path.exists() and path.stat().st_size == entry["size"]:
        return "ok" if _md5(path) == entry["md5"] else "corrupt"

    # Resume from wherever a previous run stopped.
    start = path.stat().st_size if path.exists() else 0
    headers = {"Range": f"bytes={start}-"} if start else {}
    mode = "ab" if start else "wb"

    with requests.get(entry["url"], headers=headers, stream=True, timeout=120) as r:
        # A server that ignores Range replies 200; restart rather than append.
        if start and r.status_code == 200:
            start, mode = 0, "wb"
        r.raise_for_status()
        with path.open(mode) as fh:
            for block in r.iter_content(CHUNK):
                fh.write(block)

    if path.stat().st_size != entry["size"] or _md5(path) != entry["md5"]:
        return "corrupt"
    return "resumed" if start else "downloaded"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dest", type=Path, default=Path("data/raw"))
    p.add_argument("--only", help="substring filter on the file name")
    args = p.parse_args(argv)

    args.dest.mkdir(parents=True, exist_ok=True)
    files = fetch_manifest()
    if args.only:
        files = [f for f in files if args.only in f["name"]]

    total = sum(f["size"] for f in files) / 1e9
    print(f"{len(files)} files, {total:.1f} GB -> {args.dest}", flush=True)

    failed = []
    for i, entry in enumerate(sorted(files, key=lambda f: f["name"]), 1):
        try:
            status = download_file(entry, args.dest)
        except requests.RequestException as exc:
            status = f"FAILED ({exc.__class__.__name__})"
        if status in {"corrupt"} or status.startswith("FAILED"):
            failed.append(entry["name"])
        print(
            f"[{i:2d}/{len(files)}] {entry['name']:22s} "
            f"{entry['size'] / 1e6:8.1f} MB  {status}",
            flush=True,
        )

    if failed:
        print(f"\n{len(failed)} file(s) failed: {', '.join(failed)}", flush=True)
        print("Re-run to retry; completed files are skipped.", flush=True)
        return 1
    print("\nAll files present and verified.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
