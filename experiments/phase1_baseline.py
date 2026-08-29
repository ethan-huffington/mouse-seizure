"""Phase 1: the thin end-to-end skeleton.

A deliberately minimal model carried through the *complete* chain -- features,
classifier, overlap-add, hysteresis, event formation, event-level scoring and
burden statistics. The point is not the score. The point is that an honest
event-level number exists, against which every later change is measured.

Validation is leave-one-animal-out over the 8 training mice. The official test
split is not touched until Phase 2 is finished.

    uv run python experiments/phase1_baseline.py
    uv run python experiments/phase1_baseline.py --quick   # 2 folds, Day1 only
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import polars as pl
from catboost import CatBoostClassifier

from mouse_seizure import evaluate
from mouse_seizure import postprocess as pp
from mouse_seizure.config import TARGET_FS, project_root
from mouse_seizure.dataset import cached, label_windows, training_rows
from mouse_seizure.features import FEATURE_NAMES
from mouse_seizure.io import find_recordings, parse_name

ARTIFACTS = project_root() / "artifacts" / "phase1"
HIGH, LOW = 0.60, 0.30


def load_tables(quick: bool) -> dict[str, list[pl.DataFrame]]:
    """Cached feature tables grouped by animal."""
    paths = [p for p in find_recordings() if parse_name(p.stem)[1] == "train"]
    if quick:
        paths = [p for p in paths if p.stem.startswith("Day1")]
    by_mouse: dict[str, list[pl.DataFrame]] = {}
    for path in paths:
        table = cached(path)
        by_mouse.setdefault(parse_name(path.stem)[0], []).append(table)
    return by_mouse


def fit_fold(train_tables: list[pl.DataFrame]) -> CatBoostClassifier:
    """CatBoost on non-overlapping windows, defaults plus class balancing."""
    train = pl.concat([training_rows(t) for t in train_tables])
    x = train.select(FEATURE_NAMES).to_numpy()
    y = label_windows(train["label_fraction"].to_numpy())
    model = CatBoostClassifier(
        iterations=400,
        learning_rate=0.1,
        depth=6,
        loss_function="Logloss",
        auto_class_weights="Balanced",
        verbose=False,
        random_seed=0,
        thread_count=-1,
    )
    model.fit(x, y)
    return model


def _reference_mask(table: pl.DataFrame, n_samples: int) -> np.ndarray:
    """Ground truth rebuilt on the same window grid as the predictions.

    Both sides go through the identical overlap-add path, so the comparison
    measures the detector rather than a mismatch in how the two were rasterised.
    """
    trace = pp.probability_trace(
        table["label_fraction"].to_numpy(), n_samples, TARGET_FS
    )
    return pp.events_to_mask(
        pp.form_events(trace > 0.5, TARGET_FS, merge_gap_s=0.0, min_duration_s=0.0),
        n_samples,
        TARGET_FS,
    )


def evaluate_recording(model: CatBoostClassifier, table: pl.DataFrame) -> dict:
    """Score one held-out recording end to end."""
    probs = model.predict_proba(table.select(FEATURE_NAMES).to_numpy())[:, 1]
    n_samples = int(table["n_samples"][0])

    events, _ = pp.detect(probs, n_samples, high=HIGH, low=LOW, fs=TARGET_FS)
    predicted = pp.events_to_mask(events, n_samples, TARGET_FS)

    reference = _reference_mask(table, n_samples)

    scores = evaluate.score(reference, predicted, TARGET_FS)
    truth = label_windows(table["label_fraction"].to_numpy())
    return {
        "file": table["file"][0],
        "mouse": table["mouse"][0],
        **scores.as_dict(),
        "window_auprc": evaluate.window_auprc(truth, probs)
        if truth.any()
        else float("nan"),
        "burden_pred": evaluate.burden(events, n_samples, TARGET_FS).events_per_hour,
        "burden_true": evaluate.burden(
            pp.form_events(reference, TARGET_FS, 0.0, 0.0), n_samples, TARGET_FS
        ).events_per_hour,
    }


SWEEP_THRESHOLDS = [
    (0.60, 0.30),
    (0.80, 0.50),
    (0.90, 0.70),
    (0.95, 0.80),
    (0.98, 0.90),
]


def sweep_recording(model: CatBoostClassifier, table: pl.DataFrame) -> list[dict]:
    """Score one recording at several operating points.

    Reported rather than tuned. Phase 1's headline stays at the fixed default so
    the baseline is honest; this curve exists to show where the threshold should
    move in Phase 2, and to make the cost of that move explicit.
    """
    probs = model.predict_proba(table.select(FEATURE_NAMES).to_numpy())[:, 1]
    n_samples = int(table["n_samples"][0])
    reference = _reference_mask(table, n_samples)

    out = []
    for high, low in SWEEP_THRESHOLDS:
        events, _ = pp.detect(probs, n_samples, high=high, low=low, fs=TARGET_FS)
        s = evaluate.score(
            reference, pp.events_to_mask(events, n_samples, TARGET_FS), TARGET_FS
        )
        out.append(
            {
                "mouse": table["mouse"][0],
                "high": high,
                "low": low,
                "event_f1": s.event_f1,
                "event_sensitivity": s.event_sensitivity,
                "event_precision": s.event_precision,
                "false_alarms_per_hour": s.false_alarms_per_hour,
            }
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="Day 1 only, 2 folds")
    ap.add_argument(
        "--sweep",
        action="store_true",
        help="also report the sensitivity-vs-false-alarm operating curve",
    )
    args = ap.parse_args()

    by_mouse = load_tables(args.quick)
    mice = sorted(by_mouse)
    folds = mice[:2] if args.quick else mice
    print(
        f"{len(mice)} animals, {sum(len(v) for v in by_mouse.values())} recordings; "
        f"{len(folds)} folds\n",
        flush=True,
    )

    rows: list[dict] = []
    sweep_rows: list[dict] = []
    for held_out in folds:
        start = time.perf_counter()
        train_tables = [t for m in mice if m != held_out for t in by_mouse[m]]
        model = fit_fold(train_tables)
        for table in by_mouse[held_out]:
            rows.append(evaluate_recording(model, table))
            if args.sweep:
                sweep_rows.extend(sweep_recording(model, table))
        f1 = np.mean([r["event_f1"] for r in rows if r["mouse"] == held_out])
        print(
            f"  held out {held_out:9s} event F1 {f1:.3f}   "
            f"({time.perf_counter() - start:.0f}s)",
            flush=True,
        )

    results = pl.DataFrame(rows)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    results.write_parquet(ARTIFACTS / "loao_results.parquet")

    per_mouse = (
        results.group_by("mouse")
        .agg(
            pl.col("event_f1").mean().round(3),
            pl.col("event_sensitivity").mean().round(3),
            pl.col("event_precision").mean().round(3),
            pl.col("false_alarms_per_hour").mean().round(2),
            pl.col("sample_f1").mean().round(3),
            pl.col("window_auprc").mean().round(3),
        )
        .sort("mouse")
    )
    print("\n=== per animal (leave-one-animal-out) ===")
    print(per_mouse)

    print("\n=== headline ===")
    for name in (
        "event_f1",
        "event_sensitivity",
        "event_precision",
        "false_alarms_per_hour",
        "sample_f1",
        "window_auprc",
    ):
        v = results[name].to_numpy()
        v = v[~np.isnan(v)]
        print(
            f"  {name:24s} {v.mean():6.3f} +/- {v.std():.3f}   "
            f"[{v.min():.3f}, {v.max():.3f}]"
        )

    if sweep_rows:
        curve = (
            pl.DataFrame(sweep_rows)
            .group_by("high", "low")
            .agg(
                pl.col("event_sensitivity").drop_nans().mean().round(3),
                pl.col("event_precision").drop_nans().mean().round(3),
                pl.col("event_f1").drop_nans().mean().round(3),
                pl.col("false_alarms_per_hour").drop_nans().mean().round(2),
                pl.col("event_precision").is_nan().sum().alias("n_no_detections"),
            )
            .sort("high")
        )
        print("\n=== operating curve (reported, not tuned) ===")
        print(curve)
        curve.write_parquet(ARTIFACTS / "operating_curve.parquet")
        # Keep the per-recording rows too, so the curve can be re-aggregated
        # without repeating the hour-long sweep.
        pl.DataFrame(sweep_rows).write_parquet(
            ARTIFACTS / "operating_curve_rows.parquet"
        )

    summary = {
        "n_folds": len(folds),
        "n_recordings": len(rows),
        "thresholds": {"high": HIGH, "low": LOW},
        "headline": {
            name: float(np.nanmean(results[name].to_numpy()))
            for name in (
                "event_f1",
                "event_sensitivity",
                "event_precision",
                "false_alarms_per_hour",
                "sample_f1",
                "window_auprc",
            )
        },
    }
    (ARTIFACTS / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {ARTIFACTS}/loao_results.parquet and summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
