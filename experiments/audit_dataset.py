"""Audit every raw file for the quirks the loader has to absorb.

Phase 0 verification: sampling rates, mID reliability, censored event boundaries,
and the label statistics that set post-processing parameters. Run after download.

    uv run python experiments/audit_dataset.py
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from mouse_seizure.io import events, find_recordings, load_recording

paths = find_recordings()
print(f"{len(paths)} files\n")

rows = []
for p in paths:
    rec = load_recording(p, signal=False)
    ev = events(rec.labels, rec.fs)
    dur = ev[:, 1] - ev[:, 0]
    rows.append(
        {
            "name": rec.name,
            "split": rec.split,
            "mouse": rec.mouse,
            "fs": rec.fs,
            "hours": rec.duration_s / 3600,
            "mid": rec.mid_field,
            "pos": rec.labels.mean() * 100,
            "n_ev": len(ev),
            "med": np.median(dur) if len(ev) else np.nan,
            "min": dur.min() if len(ev) else np.nan,
            "max": dur.max() if len(ev) else np.nan,
            "starts": rec.starts_in_event,
            "ends": rec.ends_in_event,
        }
    )

hdr = f"{'file':<22}{'split':<7}{'fs':>6}{'hours':>7}{'mID':>5}{'pos%':>7}{'ev':>6}{'medS':>7}{'minS':>7}{'maxS':>7}  bounds"
print(hdr)
print("-" * len(hdr))
for r in rows:
    b = ("S" if r["starts"] else "-") + ("E" if r["ends"] else "-")
    print(
        f"{r['name']:<22}{r['split']:<7}{r['fs']:>6.0f}{r['hours']:>7.2f}"
        f"{r['mid']:>5.0f}{r['pos']:>7.2f}{r['n_ev']:>6d}"
        f"{r['med']:>7.2f}{r['min']:>7.2f}{r['max']:>7.2f}  {b}"
    )

print("\n=== sampling rate by split ===")
by_split = defaultdict(set)
for r in rows:
    by_split[r["split"]].add(r["fs"])
for split, rates in sorted(by_split.items()):
    print(f"  {split:<6} {sorted(rates)}")

print("\n=== mID reliability (filename mouse number vs stored mID) ===")
bad = [r for r in rows if int(r["mouse"].split("_M")[1]) != int(r["mid"])]
print(f"  {len(bad)}/{len(rows)} files disagree")
for r in bad:
    print(
        f"    {r['name']:<22} filename says M{r['mouse'].split('_M')[1]}, mID says {r['mid']:.0f}"
    )
collide = defaultdict(list)
for r in rows:
    collide[(r["split"], int(r["mid"]))].append(r["name"])
for (split, mid), names in sorted(collide.items()):
    if len({n for n in names}) > 1 and split == "test":
        print(f"    test mID={mid} claimed by {len(names)} files: {', '.join(names)}")

print("\n=== censored event boundaries ===")
s = [r["name"] for r in rows if r["starts"]]
e = [r["name"] for r in rows if r["ends"]]
print(f"  start mid-event: {len(s)} -> {', '.join(s) if s else 'none'}")
print(f"  end mid-event:   {len(e)} -> {', '.join(e) if e else 'none'}")

print("\n=== cohort ===")
for split in ("train", "test"):
    sub = [r for r in rows if r["split"] == split]
    mice = sorted({r["mouse"] for r in sub})
    hours = sum(r["hours"] for r in sub)
    evs = sum(r["n_ev"] for r in sub)
    pos = np.average([r["pos"] for r in sub], weights=[r["hours"] for r in sub])
    print(
        f"  {split:<6} {len(sub):>2} files  {len(mice):>2} mice  "
        f"{hours:7.1f} h  {evs:>5} events  {pos:5.2f}% positive"
    )

print("\n=== event duration pooled ===")
alld = []
for p in paths:
    rec = load_recording(p, signal=False)
    ev = events(rec.labels, rec.fs)
    alld.append(ev[:, 1] - ev[:, 0])
alld = np.concatenate(alld)
qs = np.percentile(alld, [1, 5, 25, 50, 75, 95, 99])
print(f"  n={len(alld)}  min={alld.min():.2f}s  max={alld.max():.2f}s")
print("  percentiles 1/5/25/50/75/95/99: " + " ".join(f"{q:.2f}" for q in qs))
