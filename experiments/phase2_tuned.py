"""Phase 2: fix the operating point, then measure what the features are worth.

Phase 1 showed the classifier was not the bottleneck -- moving one threshold took
event F1 from 0.504 to 0.796 with an unchanged model. This script does that
properly rather than by reading a curve:

* **Calibrated probabilities.** Phase 1 used ``auto_class_weights="Balanced"``,
  which reweights training to a 50/50 prior against a true ~4%. The resulting
  probabilities are systematically inflated, which is *why* 0.60/0.30 was so
  permissive. Training on the natural prior keeps them meaningful.
* **Nested threshold selection.** Within each leave-one-animal-out fold, one
  further animal is held out to choose the threshold. The test animal is never
  involved in selecting it -- tuning on the test fold is the easiest way to
  report a number that will not survive contact with new data.
* **Ablations** by feature group, to show what each family actually buys.

    uv run python experiments/phase2_tuned.py
    uv run python experiments/phase2_tuned.py --ablate
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
from mouse_seizure.features import FEATURE_GROUPS, FEATURE_NAMES
from mouse_seizure.io import find_recordings, parse_name

ARTIFACTS = project_root() / "artifacts" / "phase2"

# Candidate operating points, coarse->fine. `low` trails `high` by a fixed margin:
# the hysteresis gap matters less than where the pair sits.
THRESHOLDS = [(h, round(h - 0.15, 2)) for h in np.arange(0.20, 1.00, 0.05)]
FA_BUDGET = 2.0


def load_by_mouse() -> dict[str, list[pl.DataFrame]]:
    by_mouse: dict[str, list[pl.DataFrame]] = {}
    for path in find_recordings():
        if parse_name(path.stem)[1] != "train":
            continue
        by_mouse.setdefault(parse_name(path.stem)[0], []).append(cached(path))
    return by_mouse


def fit(tables: list[pl.DataFrame], columns: list[str]) -> CatBoostClassifier:
    """CatBoost on the natural class prior, so probabilities stay calibrated."""
    train = pl.concat([training_rows(t) for t in tables])
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
        train.select(columns).to_numpy(),
        label_windows(train["label_fraction"].to_numpy()),
    )
    return model


DECISION_FS = pp.DECISION_FS


def duration_of(table: pl.DataFrame) -> float:
    return int(table["n_samples"][0]) / TARGET_FS


def reference_events(table: pl.DataFrame) -> np.ndarray:
    """Ground truth rebuilt on the prediction's own window grid.

    Both sides travel the identical overlap-add path, so the comparison measures
    the detector rather than a mismatch in how the two were rasterised.
    """
    n = round(duration_of(table) * DECISION_FS)
    trace = pp.probability_trace(table["label_fraction"].to_numpy(), n, DECISION_FS)
    return pp.form_events(trace > 0.5, DECISION_FS, merge_gap_s=0.0, min_duration_s=0.0)


def score_at(
    probs: np.ndarray, table: pl.DataFrame, high: float, low: float
) -> evaluate.Scores:
    duration = duration_of(table)
    n = round(duration * DECISION_FS)
    events = pp.detect_events(probs, duration, high=high, low=low)
    return evaluate.score(
        pp.events_to_mask(reference_events(table), n, DECISION_FS),
        pp.events_to_mask(events, n, DECISION_FS),
        DECISION_FS,
        decimation=1,
    )


def curve_for(
    model: CatBoostClassifier, tables: list[pl.DataFrame], columns: list[str]
) -> list[dict]:
    rows = []
    for table in tables:
        probs = model.predict_proba(table.select(columns).to_numpy())[:, 1]
        for high, low in THRESHOLDS:
            s = score_at(probs, table, high, low)
            rows.append(
                {
                    "file": table["file"][0],
                    "mouse": table["mouse"][0],
                    "high": high,
                    "low": low,
                    **s.as_dict(),
                }
            )
    return rows


def pick_threshold(rows: list[dict]) -> tuple[float, float]:
    """Threshold maximising mean event F1 on the validation animal."""
    frame = (
        pl.DataFrame(rows)
        .group_by("high", "low")
        .agg(pl.col("event_f1").drop_nans().mean().alias("f1"))
        .sort("f1", descending=True)
    )
    best = frame.row(0, named=True)
    return best["high"], best["low"]


def run(columns: list[str], label: str, *, verbose: bool = True) -> pl.DataFrame:
    """Leave-one-animal-out over the training mice, resumable.

    A completed run is reloaded from its parquet rather than recomputed. Each
    configuration costs ~9 minutes, so without this a single interruption throws
    away the whole sweep.
    """
    done = ARTIFACTS / f"results_{label}.parquet"
    if done.exists():
        if verbose:
            print(f"  [cached] {label}", flush=True)
        return pl.read_parquet(done)

    by_mouse = load_by_mouse()
    mice = sorted(by_mouse)
    rows, curve_rows = [], []

    for i, test_mouse in enumerate(mice):
        start = time.perf_counter()
        # Rotate the validation animal so no single mouse shapes every threshold.
        val_mouse = mice[(i + 1) % len(mice)]
        train_mice = [m for m in mice if m not in (test_mouse, val_mouse)]

        model = fit([t for m in train_mice for t in by_mouse[m]], columns)
        high, low = pick_threshold(curve_for(model, by_mouse[val_mouse], columns))

        for table in by_mouse[test_mouse]:
            probs = model.predict_proba(table.select(columns).to_numpy())[:, 1]
            s = score_at(probs, table, high, low)
            truth = label_windows(table["label_fraction"].to_numpy())
            n_samples = int(table["n_samples"][0])
            events = pp.detect_events(probs, duration_of(table), high=high, low=low)
            rows.append(
                {
                    "file": table["file"][0],
                    "mouse": test_mouse,
                    "high": high,
                    "low": low,
                    **s.as_dict(),
                    "window_auprc": evaluate.window_auprc(truth, probs)
                    if truth.any()
                    else float("nan"),
                    "burden_pred": evaluate.burden(
                        events, n_samples, TARGET_FS
                    ).events_per_hour,
                    "burden_true": evaluate.burden(
                        reference_events(table), n_samples, TARGET_FS
                    ).events_per_hour,
                }
            )
            # Full curve on the *test* animal, for reporting only.
            for h, lo in THRESHOLDS:
                cs = score_at(probs, table, h, lo)
                curve_rows.append(
                    {"mouse": test_mouse, "high": h, "low": lo, **cs.as_dict()}
                )

        if verbose:
            f1 = np.mean([r["event_f1"] for r in rows if r["mouse"] == test_mouse])
            print(
                f"  {test_mouse:9s} val={val_mouse:9s} thr={high:.2f}/{low:.2f} "
                f"F1={f1:.3f}  ({time.perf_counter() - start:.0f}s)",
                flush=True,
            )

    results = pl.DataFrame(rows)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    results.write_parquet(ARTIFACTS / f"results_{label}.parquet")
    if curve_rows:
        pl.DataFrame(curve_rows).write_parquet(ARTIFACTS / f"curve_{label}.parquet")
    return results


def report(results: pl.DataFrame, label: str) -> dict:
    print(f"\n=== per animal ({label}) ===")
    print(
        results.group_by("mouse")
        .agg(
            pl.col("event_f1").drop_nans().mean().round(3),
            pl.col("event_sensitivity").drop_nans().mean().round(3),
            pl.col("event_precision").drop_nans().mean().round(3),
            pl.col("false_alarms_per_hour").drop_nans().mean().round(2),
            pl.col("window_auprc").drop_nans().mean().round(3),
        )
        .sort("mouse")
    )

    print(f"\n=== headline ({label}) ===")
    summary = {}
    for name in (
        "event_f1",
        "event_sensitivity",
        "event_precision",
        "false_alarms_per_hour",
        "sample_f1",
        "window_auprc",
        "cohen_kappa",
    ):
        v = results[name].to_numpy()
        v = v[np.isfinite(v)]
        summary[name] = float(v.mean())
        print(f"  {name:24s} {v.mean():7.3f} +/- {v.std():.3f}")

    true_b = results["burden_true"].to_numpy()
    pred_b = results["burden_pred"].to_numpy()
    ok = np.isfinite(true_b) & np.isfinite(pred_b)
    summary["burden_bias"] = float((pred_b[ok] - true_b[ok]).mean())
    summary["burden_corr"] = float(np.corrcoef(true_b[ok], pred_b[ok])[0, 1])
    print(
        f"  burden events/h          true {true_b[ok].mean():.1f}  "
        f"pred {pred_b[ok].mean():.1f}  bias {summary['burden_bias']:+.1f}  "
        f"corr {summary['burden_corr']:.3f}"
    )
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--ablate", action="store_true", help="also run feature-group ablations"
    )
    ap.add_argument(
        "--combos",
        action="store_true",
        help="run multi-group removals that test redundancy between families",
    )
    args = ap.parse_args()

    print("=== full feature set, nested threshold selection ===", flush=True)
    results = run(FEATURE_NAMES, "full")
    summary = {"full": report(results, "full")}

    curve = pl.read_parquet(ARTIFACTS / "curve_full.parquet")
    agg = (
        curve.group_by("high", "low")
        .agg(
            pl.col("event_sensitivity").drop_nans().mean().round(3),
            pl.col("event_precision").drop_nans().mean().round(3),
            pl.col("event_f1").drop_nans().mean().round(3),
            pl.col("false_alarms_per_hour").drop_nans().mean().round(2),
        )
        .sort("high")
    )
    print("\n=== operating curve (test animals, reported) ===")
    print(agg)

    within = agg.filter(pl.col("false_alarms_per_hour") <= FA_BUDGET)
    if within.height:
        best = within.sort("event_sensitivity", descending=True).row(0, named=True)
        summary["sensitivity_at_fa_budget"] = {
            "fa_budget_per_hour": FA_BUDGET,
            "threshold": [best["high"], best["low"]],
            "sensitivity": best["event_sensitivity"],
            "precision": best["event_precision"],
            "event_f1": best["event_f1"],
            "false_alarms_per_hour": best["false_alarms_per_hour"],
        }
        print(
            f"\noperational point (<= {FA_BUDGET} FA/h): thr {best['high']:.2f}/{best['low']:.2f}"
            f"  sensitivity {best['event_sensitivity']:.3f}"
            f"  precision {best['event_precision']:.3f}"
            f"  FA/h {best['false_alarms_per_hour']:.2f}"
        )

    if args.combos:
        # Single-group ablation cannot distinguish "useless" from "redundant with
        # another group". Removing two families at once does: if dropping both
        # costs far more than either alone, they were carrying the same
        # information and only looked expendable individually.
        combos = {
            "spectral+artifact": ["spectral", "artifact"],
            "spectral+wavelet": ["spectral", "wavelet"],
            "wavelet+artifact": ["wavelet", "artifact"],
        }
        print("\n=== multi-group removals ===", flush=True)
        for name, groups in combos.items():
            drop = {c for g in groups for c in FEATURE_GROUPS[g]}
            columns = [c for c in FEATURE_NAMES if c not in drop]
            res = run(columns, f"without_{name}", verbose=False)
            f1 = float(np.nanmean(res["event_f1"].to_numpy()))
            delta = f1 - summary["full"]["event_f1"]
            summary[f"without_{name}"] = {
                "event_f1": f1,
                "delta": delta,
                "n_features": len(columns),
            }
            print(
                f"  without {name:20s} {len(columns):2d} features  "
                f"F1 {f1:.3f}  ({delta:+.3f})",
                flush=True,
            )

    if args.ablate:
        print("\n=== ablations (leave one feature group out) ===", flush=True)
        for group in FEATURE_GROUPS:
            columns = [c for c in FEATURE_NAMES if c not in FEATURE_GROUPS[group]]
            res = run(columns, f"without_{group}", verbose=False)
            f1 = float(np.nanmean(res["event_f1"].to_numpy()))
            delta = f1 - summary["full"]["event_f1"]
            summary[f"without_{group}"] = {"event_f1": f1, "delta": delta}
            print(f"  without {group:12s} F1 {f1:.3f}  ({delta:+.3f})", flush=True)

    (ARTIFACTS / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {ARTIFACTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
