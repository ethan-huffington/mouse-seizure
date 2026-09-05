"""Phase 3: the official test split. One evaluation, run once.

These ten mice have never been loaded by a training script. That is what makes
this number worth anything, and it is also why the protocol below is fixed before
the script is run rather than adjusted after seeing a result:

* **Both models train on all eight training animals.** No animal is held out here
  -- the test split *is* the held-out set.
* **Thresholds come from the leave-one-animal-out folds**, as the median of the
  eight values each procedure already chose on validation animals. Nothing about
  the operating point is derived from these ten recordings.
* **Both models are evaluated in the same pass.** Two models scored once is not
  the same thing as tuning against the split; what would invalidate it is looking,
  adjusting, and looking again.

Two properties of this split shape what may be reported from it:

* Recordings are **cropped at exactly 100 seizures and end mid-event**, so the
  trailing partial event is dropped before scoring -- it has no true offset.
* Because the crop fixes the seizure count, **events per hour and seizure burden
  describe the benchmark's construction rather than the animals**, and are not
  reported.

    uv run python experiments/phase3_testsplit.py
"""

from __future__ import annotations

import json
import time

import numpy as np
import polars as pl
import torch
from catboost import CatBoostClassifier

from mouse_seizure import evaluate
from mouse_seizure import postprocess as pp
from mouse_seizure.config import TARGET_FS, project_root
from mouse_seizure.dataset import cached, cached_signal, label_windows, training_rows
from mouse_seizure.features import FEATURE_GROUPS, FEATURE_NAMES
from mouse_seizure.io import find_recordings, parse_name
from mouse_seizure.unet import (
    SegmentSampler,
    UNet1D,
    best_device,
    dice_bce_loss,
    predict_trace,
)

ARTIFACTS = project_root() / "artifacts" / "phase3"
PHASE2 = project_root() / "artifacts" / "phase2"

COLUMNS = [c for c in FEATURE_NAMES if c not in FEATURE_GROUPS["artifact"]]
DECISION_FS = pp.DECISION_FS
DECIMATE = int(TARGET_FS // DECISION_FS)
STEPS, BATCH, SEGMENT_S = 2500, 16, 20.0


def split_paths(split: str) -> list:
    return [p for p in find_recordings() if parse_name(p.stem)[1] == split]


def reference_events(labels: np.ndarray, *, drop_trailing: bool) -> np.ndarray:
    """Annotated events at the decision rate.

    ``drop_trailing`` removes an event still running at the end of the recording.
    On this split that event is truncated by the 100-seizure crop, so it has no
    real offset; scoring against it would penalise a correct detection.
    """
    coarse = np.asarray(labels[::DECIMATE]).astype(bool)
    events = pp.form_events(coarse, DECISION_FS, merge_gap_s=0.0, min_duration_s=0.0)
    if drop_trailing and len(events) and coarse[-1]:
        events = events[:-1]
    return events


def score_one(
    labels: np.ndarray, events: np.ndarray, *, drop_trailing: bool
) -> evaluate.Scores:
    n = len(labels) // DECIMATE
    return evaluate.score(
        pp.events_to_mask(
            reference_events(labels, drop_trailing=drop_trailing), n, DECISION_FS
        ),
        pp.events_to_mask(events, n, DECISION_FS),
        DECISION_FS,
        decimation=1,
    )


def median_threshold(parquet: str) -> tuple[float, float]:
    """Median operating point across the eight leave-one-animal-out folds."""
    frame = pl.read_parquet(PHASE2 / parquet) if parquet.startswith("results") else None
    if frame is None:
        raise ValueError(parquet)
    per_fold = frame.group_by("mouse").agg(
        pl.col("high").first(), pl.col("low").first()
    )
    return (
        float(np.median(per_fold["high"].to_numpy())),
        float(np.median(per_fold["low"].to_numpy())),
    )


def run_features() -> tuple[list[dict], tuple[float, float]]:
    high, low = median_threshold("results_without_artifact.parquet")
    print(
        f"  features: threshold {high:.2f}/{low:.2f} (median of 8 LOAO folds)",
        flush=True,
    )

    train = pl.concat([training_rows(cached(p)) for p in split_paths("train")])
    model = CatBoostClassifier(
        iterations=400,
        learning_rate=0.1,
        depth=6,
        loss_function="Logloss",
        verbose=False,
        random_seed=0,
        thread_count=-1,
    )
    model.fit(
        train.select(COLUMNS).to_numpy(),
        label_windows(train["label_fraction"].to_numpy()),
    )

    rows = []
    for path in split_paths("test"):
        table = cached(path)
        probs = model.predict_proba(table.select(COLUMNS).to_numpy())[:, 1]
        duration = int(table["n_samples"][0]) / TARGET_FS
        events = pp.detect_events(probs, duration, high=high, low=low)
        _, labels = cached_signal(path)
        s = score_one(labels, events, drop_trailing=True)
        rows.append({"file": path.stem, "model": "features", **s.as_dict()})
    return rows, (high, low)


def run_unet() -> tuple[list[dict], tuple[float, float]]:
    unet_results = pl.read_parquet(ARTIFACTS / "results_unet.parquet")
    per_fold = unet_results.group_by("mouse").agg(
        pl.col("high").first(), pl.col("low").first()
    )
    high = float(np.median(per_fold["high"].to_numpy()))
    low = float(np.median(per_fold["low"].to_numpy()))
    print(
        f"  unet:     threshold {high:.2f}/{low:.2f} (median of 8 LOAO folds)",
        flush=True,
    )

    device = best_device()
    recordings = [cached_signal(p) for p in split_paths("train")]
    model = UNet1D().to(device)
    sampler = SegmentSampler(recordings, segment_s=SEGMENT_S, seed=0)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=STEPS)

    model.train()
    for step in range(STEPS):
        x, y = sampler.batch(BATCH)
        opt.zero_grad()
        loss = dice_bce_loss(model(x.to(device)), y.to(device))
        loss.backward()
        opt.step()
        sched.step()
        if step % 500 == 0:
            print(f"      step {step:4d}  loss {loss.item():.4f}", flush=True)

    rows = []
    for path in split_paths("test"):
        x, labels = cached_signal(path)
        trace = predict_trace(model, np.asarray(x), device)
        coarse = pp.smooth(trace[::DECIMATE], DECISION_FS, 0.5)
        events = pp.form_events(pp.hysteresis_threshold(coarse, high, low), DECISION_FS)
        s = score_one(labels, events, drop_trailing=True)
        rows.append({"file": path.stem, "model": "unet", **s.as_dict()})
    return rows, (high, low)


def main() -> int:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    print("=== OFFICIAL TEST SPLIT -- 10 unseen mice, one evaluation ===", flush=True)

    start = time.perf_counter()
    feature_rows, feature_thr = run_features()
    unet_rows, unet_thr = run_unet()
    results = pl.DataFrame(feature_rows + unet_rows)
    results.write_parquet(ARTIFACTS / "results_testsplit.parquet")

    summary: dict = {
        "thresholds": {"features": feature_thr, "unet": unet_thr},
        "note": "burden not reported: recordings are cropped at 100 seizures",
    }
    for model in ("features", "unet"):
        sub = results.filter(pl.col("model") == model)
        print(f"\n=== {model} on the official test split ===")
        print(
            sub.select(
                "file",
                "event_f1",
                "event_sensitivity",
                "event_precision",
                "false_alarms_per_hour",
                "sample_f1",
            ).sort("file")
        )
        entry = {}
        for metric in (
            "event_f1",
            "event_sensitivity",
            "event_precision",
            "false_alarms_per_hour",
            "sample_f1",
            "sample_sensitivity",
            "sample_precision",
            "cohen_kappa",
        ):
            v = sub[metric].to_numpy()
            v = v[np.isfinite(v)]
            entry[metric] = float(v.mean())
            print(f"  {metric:24s} {v.mean():7.3f} +/- {v.std():.3f}")
        summary[model] = entry

    print("\n=== comparison ===")
    print("  metric                    features     unet")
    for metric in ("event_f1", "false_alarms_per_hour", "sample_f1"):
        print(
            f"  {metric:24s} {summary['features'][metric]:8.3f} "
            f"{summary['unet'][metric]:8.3f}"
        )
    print(
        "\n  published (AugUNet1D, sample-level): precision 0.91  recall 0.90\n"
        "  published (Twin Peaks,  sample-level): precision 0.67  recall 0.81"
    )
    print(
        f"\n  ours, sample-level: features P {summary['features']['sample_precision']:.3f} "
        f"R {summary['features']['sample_sensitivity']:.3f}   "
        f"unet P {summary['unet']['sample_precision']:.3f} "
        f"R {summary['unet']['sample_sensitivity']:.3f}"
    )

    (ARTIFACTS / "summary_testsplit.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {ARTIFACTS}  ({time.perf_counter() - start:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
