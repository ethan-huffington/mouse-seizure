"""Phase 3: what does going real-time cost?

The offline detector looks forward. It filters with ``sosfiltfilt``, and it
averages every window covering a sample including ones that have not finished
yet. A closed-loop system can do neither. This script rebuilds the whole chain
under a strict past-only constraint and measures the difference in two currencies
that trade against each other:

* **accuracy** -- event F1, sensitivity, precision, false alarms per hour
* **latency** -- how long after a seizure actually starts the detector says so

Both sides use the same 19-feature frozen set, the same animals, and the same
leave-one-animal-out folds, so the only variable is causality. Features are
recomputed from causally filtered signal rather than reused, because a
forward-only filter changes the waveform and reusing offline features would
measure a domain shift instead of the constraint.

    uv run python experiments/phase3_causal.py
"""

from __future__ import annotations

import json
import time

import numpy as np
import polars as pl
from catboost import CatBoostClassifier

from mouse_seizure import evaluate
from mouse_seizure import postprocess as pp
from mouse_seizure.causal import persistence_detect
from mouse_seizure.config import STRIDE_S, TARGET_FS, project_root
from mouse_seizure.dataset import cached, label_windows, training_rows
from mouse_seizure.features import FEATURE_GROUPS, FEATURE_NAMES
from mouse_seizure.io import find_recordings, parse_name

ARTIFACTS = project_root() / "artifacts" / "phase3"

# Frozen set: the artifact group measured as dead weight (see CLAUDE.md).
COLUMNS = [c for c in FEATURE_NAMES if c not in FEATURE_GROUPS["artifact"]]

THRESHOLDS = [round(t, 2) for t in np.arange(0.30, 0.90, 0.05)]
PERSISTENCE = [2, 3, 4, 6]
DECISION_FS = pp.DECISION_FS


def load_by_mouse(*, causal: bool) -> dict[str, list[pl.DataFrame]]:
    by_mouse: dict[str, list[pl.DataFrame]] = {}
    for path in find_recordings():
        if parse_name(path.stem)[1] != "train":
            continue
        by_mouse.setdefault(parse_name(path.stem)[0], []).append(
            cached(path, causal=causal)
        )
    return by_mouse


def fit(tables: list[pl.DataFrame]) -> CatBoostClassifier:
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
        train.select(COLUMNS).to_numpy(),
        label_windows(train["label_fraction"].to_numpy()),
    )
    return model


def duration_of(table: pl.DataFrame) -> float:
    return int(table["n_samples"][0]) / TARGET_FS


def reference_events(table: pl.DataFrame) -> np.ndarray:
    n = round(duration_of(table) * DECISION_FS)
    trace = pp.probability_trace(table["label_fraction"].to_numpy(), n, DECISION_FS)
    return pp.form_events(trace > 0.5, DECISION_FS, merge_gap_s=0.0, min_duration_s=0.0)


def score_events(
    reference: np.ndarray, hypothesis: np.ndarray, duration: float
) -> evaluate.Scores:
    n = round(duration * DECISION_FS)
    return evaluate.score(
        pp.events_to_mask(reference, n, DECISION_FS),
        pp.events_to_mask(hypothesis, n, DECISION_FS),
        DECISION_FS,
        decimation=1,
    )


def causal_at(
    probs: np.ndarray, table: pl.DataFrame, threshold: float, persistence: int
) -> tuple[evaluate.Scores, np.ndarray, float]:
    reference = reference_events(table)
    detection = persistence_detect(
        probs, threshold=threshold, persistence=persistence, stride_s=STRIDE_S
    )
    scores = score_events(reference, detection.events, duration_of(table))
    latency = evaluate.onset_latency(reference, detection.events)
    return scores, latency, detection.declaration_delay_s


def pick_operating_point(
    model: CatBoostClassifier, tables: list[pl.DataFrame]
) -> tuple[float, int]:
    """Threshold and persistence maximising event F1 on the validation animal."""
    best, best_f1 = (0.5, 3), -1.0
    for table in tables[:1]:  # one session is enough to set two scalars
        probs = model.predict_proba(table.select(COLUMNS).to_numpy())[:, 1]
        for threshold in THRESHOLDS:
            for persistence in PERSISTENCE:
                scores, _, _ = causal_at(probs, table, threshold, persistence)
                if np.isfinite(scores.event_f1) and scores.event_f1 > best_f1:
                    best_f1, best = scores.event_f1, (threshold, persistence)
    return best


def main() -> int:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    print("=== causal pipeline, leave-one-animal-out ===", flush=True)
    print("building causal feature cache (first run only)...", flush=True)

    by_mouse = load_by_mouse(causal=True)
    mice = sorted(by_mouse)
    rows: list[dict] = []
    latencies: list[np.ndarray] = []

    for i, test_mouse in enumerate(mice):
        start = time.perf_counter()
        val_mouse = mice[(i + 1) % len(mice)]
        train_mice = [m for m in mice if m not in (test_mouse, val_mouse)]

        model = fit([t for m in train_mice for t in by_mouse[m]])
        threshold, persistence = pick_operating_point(model, by_mouse[val_mouse])

        for table in by_mouse[test_mouse]:
            probs = model.predict_proba(table.select(COLUMNS).to_numpy())[:, 1]
            scores, latency, declared = causal_at(probs, table, threshold, persistence)
            latencies.append(latency)
            rows.append(
                {
                    "file": table["file"][0],
                    "mouse": test_mouse,
                    "threshold": threshold,
                    "persistence": persistence,
                    "declaration_delay_s": declared,
                    "median_onset_latency_s": float(np.median(latency))
                    if len(latency)
                    else float("nan"),
                    **scores.as_dict(),
                }
            )

        f1 = np.nanmean([r["event_f1"] for r in rows if r["mouse"] == test_mouse])
        print(
            f"  {test_mouse:9s} thr={threshold:.2f} N={persistence} "
            f"F1={f1:.3f}  ({time.perf_counter() - start:.0f}s)",
            flush=True,
        )

    results = pl.DataFrame(rows)
    results.write_parquet(ARTIFACTS / "results_causal.parquet")

    print("\n=== causal vs offline (both 19 features, same folds) ===")
    offline = pl.read_parquet(
        project_root() / "artifacts" / "phase2" / "results_without_artifact.parquet"
    ).sort("file")
    causal = results.sort("file")

    summary = {}
    for name in (
        "event_f1",
        "event_sensitivity",
        "event_precision",
        "false_alarms_per_hour",
    ):
        off = offline[name].to_numpy()
        cau = causal[name].to_numpy()
        off, cau = off[np.isfinite(off)], cau[np.isfinite(cau)]
        summary[name] = {"offline": float(off.mean()), "causal": float(cau.mean())}
        print(
            f"  {name:24s} offline {off.mean():7.3f}   causal {cau.mean():7.3f}   "
            f"delta {cau.mean() - off.mean():+.3f}"
        )

    all_latency = np.concatenate([latency for latency in latencies if len(latency)])
    q = np.percentile(all_latency, [25, 50, 75, 90])
    summary["onset_latency_s"] = {
        "n_matched_events": len(all_latency),
        "p25": float(q[0]),
        "median": float(q[1]),
        "p75": float(q[2]),
        "p90": float(q[3]),
    }
    print(
        f"\n  onset latency over {len(all_latency)} matched events: "
        f"median {q[1]:.2f} s   IQR {q[0]:.2f}-{q[2]:.2f} s   p90 {q[3]:.2f} s"
    )
    print(
        f"  chosen persistence: {sorted(results['persistence'].unique().to_list())} "
        f"steps of {STRIDE_S} s"
    )

    (ARTIFACTS / "summary_causal.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {ARTIFACTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
