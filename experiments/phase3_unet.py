"""Phase 3: the dense U-Net against the feature pipeline, same harness.

The feature path classifies a 2 s window at a time and then reconstructs time
resolution in post-processing. This path emits one probability per *sample*, so
windowing and overlap-add do not exist. Everything downstream is deliberately
identical -- the same hysteresis, merge-gap and minimum-duration rules, the same
rodent-scale event scoring, the same folds and animals -- so what the comparison
measures is the architecture and not the metric.

Training is a fixed step budget per fold rather than epochs: at ~2,500 s of
signal processed per second, an epoch over 720 hours would take 17 minutes and
eight folds of those is not worth the information.

    uv run python experiments/phase3_unet.py
"""

from __future__ import annotations

import json
import time

import numpy as np
import polars as pl
import torch

from mouse_seizure import evaluate
from mouse_seizure import postprocess as pp
from mouse_seizure.config import TARGET_FS, project_root
from mouse_seizure.dataset import cached_signal
from mouse_seizure.io import find_recordings, parse_name
from mouse_seizure.unet import (
    SegmentSampler,
    UNet1D,
    best_device,
    dice_bce_loss,
    predict_trace,
)

ARTIFACTS = project_root() / "artifacts" / "phase3"
STEPS = 2500
BATCH = 16
SEGMENT_S = 20.0
THRESHOLDS = [(round(h, 2), round(h - 0.15, 2)) for h in np.arange(0.20, 1.00, 0.05)]
DECISION_FS = pp.DECISION_FS
DECIMATE = int(TARGET_FS // DECISION_FS)


def load_by_mouse() -> dict[str, list[tuple[str, np.ndarray, np.ndarray]]]:
    out: dict[str, list[tuple[str, np.ndarray, np.ndarray]]] = {}
    for path in find_recordings():
        mouse, split, _ = parse_name(path.stem)
        if split != "train":
            continue
        x, y = cached_signal(path)
        out.setdefault(mouse, []).append((path.stem, x, y))
    return out


def train_fold(recordings: list[tuple[str, np.ndarray, np.ndarray]], device) -> UNet1D:
    model = UNet1D().to(device)
    sampler = SegmentSampler(
        [(x, y) for _, x, y in recordings], segment_s=SEGMENT_S, seed=0
    )
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=STEPS)

    model.train()
    for step in range(STEPS):
        x, y = sampler.batch(BATCH)
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = dice_bce_loss(model(x), y)
        loss.backward()
        opt.step()
        sched.step()
        if step % 500 == 0:
            print(f"      step {step:4d}  loss {loss.item():.4f}", flush=True)
    return model


def events_from_trace(trace: np.ndarray, high: float, low: float) -> np.ndarray:
    """Threshold a per-sample probability trace into events.

    The trace is decimated to the shared 40 Hz decision rate first, so the U-Net
    and the feature pipeline are thresholded at identical resolution. No
    overlap-add is applied -- the model already produces one value per sample,
    which is the architectural difference under test.
    """
    coarse = trace[::DECIMATE]
    smoothed = pp.smooth(coarse, DECISION_FS, 0.5)
    mask = pp.hysteresis_threshold(smoothed, high, low)
    return pp.form_events(mask, DECISION_FS)


def reference_events(labels: np.ndarray) -> np.ndarray:
    coarse = np.asarray(labels[::DECIMATE]).astype(bool)
    return pp.form_events(coarse, DECISION_FS, merge_gap_s=0.0, min_duration_s=0.0)


def score_one(labels: np.ndarray, events: np.ndarray) -> evaluate.Scores:
    n = len(labels) // DECIMATE
    return evaluate.score(
        pp.events_to_mask(reference_events(labels), n, DECISION_FS),
        pp.events_to_mask(events, n, DECISION_FS),
        DECISION_FS,
        decimation=1,
    )


def main() -> int:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    device = best_device()
    print(f"=== 1-D U-Net, leave-one-animal-out on {device} ===", flush=True)

    by_mouse = load_by_mouse()
    mice = sorted(by_mouse)
    rows: list[dict] = []

    for i, test_mouse in enumerate(mice):
        start = time.perf_counter()
        val_mouse = mice[(i + 1) % len(mice)]
        train_mice = [m for m in mice if m not in (test_mouse, val_mouse)]
        print(f"  fold {test_mouse} (val {val_mouse})", flush=True)

        model = train_fold([r for m in train_mice for r in by_mouse[m]], device)

        # Threshold chosen on the validation animal only.
        name, x, y = by_mouse[val_mouse][0]
        trace = predict_trace(model, np.asarray(x), device)
        best, best_f1 = THRESHOLDS[len(THRESHOLDS) // 2], -1.0
        for high, low in THRESHOLDS:
            s = score_one(y, events_from_trace(trace, high, low))
            if np.isfinite(s.event_f1) and s.event_f1 > best_f1:
                best_f1, best = s.event_f1, (high, low)
        high, low = best

        for name, x, y in by_mouse[test_mouse]:
            trace = predict_trace(model, np.asarray(x), device)
            events = events_from_trace(trace, high, low)
            s = score_one(y, events)
            n_samples = len(x)
            rows.append(
                {
                    "file": name,
                    "mouse": test_mouse,
                    "high": high,
                    "low": low,
                    **s.as_dict(),
                    "burden_pred": evaluate.burden(
                        events, n_samples, TARGET_FS
                    ).events_per_hour,
                    "burden_true": evaluate.burden(
                        reference_events(y), n_samples, TARGET_FS
                    ).events_per_hour,
                }
            )

        f1 = np.nanmean([r["event_f1"] for r in rows if r["mouse"] == test_mouse])
        print(
            f"    thr={high:.2f}/{low:.2f}  F1={f1:.3f}  "
            f"({time.perf_counter() - start:.0f}s)",
            flush=True,
        )

    results = pl.DataFrame(rows)
    results.write_parquet(ARTIFACTS / "results_unet.parquet")

    print("\n=== U-Net vs feature pipeline (same folds, same harness) ===")
    feature = pl.read_parquet(
        project_root() / "artifacts" / "phase2" / "results_without_artifact.parquet"
    ).sort("file")
    unet = results.sort("file")

    summary: dict = {}
    for metric in (
        "event_f1",
        "event_sensitivity",
        "event_precision",
        "false_alarms_per_hour",
        "sample_f1",
    ):
        a = feature[metric].to_numpy()
        b = unet[metric].to_numpy()
        ok = np.isfinite(a) & np.isfinite(b)
        summary[metric] = {"features": float(a[ok].mean()), "unet": float(b[ok].mean())}
        print(
            f"  {metric:24s} features {a[ok].mean():7.3f}   unet {b[ok].mean():7.3f}   "
            f"delta {b[ok].mean() - a[ok].mean():+.3f}"
        )

    t, p = unet["burden_true"].to_numpy(), unet["burden_pred"].to_numpy()
    ok = np.isfinite(t) & np.isfinite(p)
    summary["burden"] = {
        "bias": float((p[ok] - t[ok]).mean()),
        "corr": float(np.corrcoef(t[ok], p[ok])[0, 1]),
    }
    print(
        f"  burden events/h          true {t[ok].mean():.1f}  pred {p[ok].mean():.1f}  "
        f"bias {summary['burden']['bias']:+.1f}  corr {summary['burden']['corr']:.3f}"
    )

    (ARTIFACTS / "summary_unet.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {ARTIFACTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
