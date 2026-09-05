# mouse-seizure

## Commands

- Run:   `uv run mouse-seizure`
- Test:  `uv run pytest`
- Lint:  `ruff check --fix && ruff format`

Dependencies are managed with uv: `uv add <pkg>`, `uv add --dev <pkg>`.
Never use pip, and never activate the venv by hand — prefix commands with
`uv run`.

## Layout

- `src/mouse_seizure/` — the importable package; all reusable code lives here
- `experiments/` — training entry points and one-off runs
- `notebooks/` — exploration
- `configs/` — hyperparameters and run configuration
- `data/`, `artifacts/` — gitignored contents; structure tracked via .gitkeep

## Status

**Phase 0 — data verified, dataset chosen.** Nothing is built yet. `project_brief.md` is
the next artifact to write.

## Decisions locked

- **Modeling path: hand-crafted features + gradient-boosted trees (CatBoost).** Chosen
  deliberately over a deep model. The signal-processing and feature work is the point of the
  project, not the classifier. A dense 1-D U-Net (per-sample segmentation, no windowing) is a
  planned later extension, kept as a comparison against this baseline — not a replacement.
- **The classifier is not the bottleneck.** Post-processing and event formation drive
  event-level performance far more than model choice. Do not spend effort tuning CatBoost
  before the post-processing chain is right.
- **Evaluation uses `timescoring`** (the SzCORE reference implementation) rather than
  hand-written overlap matching, so results are comparable to published numbers.
  Its defaults are scaled for human clinical seizures (30 s / 60 s tolerance collars, 90 s
  merge gap). Rodent SWDs last ~1–2 s. **Every parameter must be overridden with
  rodent-scale values in one central config, and that choice documented in the README.**
  Running with defaults silently produces meaningless, inflated numbers.
- **Split by animal, always.** Never mix windows from one animal across train and test.
  Windows from one recording are highly correlated; splitting by window inflates results.
- **Public repo.** Assume everything committed here is world-readable.

## Dataset — Zenodo C3H/HeJ mice (decided)

`10.5281/zenodo.17982390`, CC-BY 4.0, 19.5 GB, 50 files, MATLAB v7.3 (HDF5 — use `h5py`,
`scipy.io.loadmat` will not open these). Each file holds one struct `rec` with fields
`mID`, `SWDlabel` (per-sample uint8), `ecog` (float64, arbitrary units), `fs`.

GAERS (IEEE DataPort) was evaluated and **rejected**: only 3 rats, all simultaneously
recorded, and only Day 1 was labelled (~50 h). Two usable EEG channels was not worth
n=3 for by-animal validation.

**Verified quirks the loader must handle — every one of these was found empirically,
not documented:**

- **Sampling rate differs between splits: training files are 1000 Hz, test files 400 Hz.**
  Resample to a common rate or nothing transfers. This is the highest-risk trap here.
- **Array orientation varies per file** — some `(n, 1)`, some `(1, n)`. Always ravel.
- **`mID` is unreliable.** Test files M6, M7 and M8 all report `mID = 5`. Group animals by
  **filename**, never by the `mID` field. Audit all 40 training files before splitting.
- **Test files are cropped at the 100th seizure** and end mid-event (99 complete + 1
  truncated). Their duration is "time to accumulate 100 SWDs", so events/hour and burden
  statistics are meaningless on the test split — report burden on training-set sessions only.
  Drop the trailing partial event before event-level scoring.
- **Some recordings start mid-event** (e.g. `Day2_M8`). Pad the label array before
  run-length detection or durations come out negative.
- Annotators applied a **~1 s minimum duration** floor; use that for the min-duration rule.

**Signal facts:** SWD fundamental stable at 5.5–5.9 Hz across every recording; events
median 5.6 s pooled (much longer than the ~1 s the source PDF assumed); positive fraction
1–7%; event rate 8.6–35.6/h and genuinely animal-dependent.

**The SWD band is not exclusive to SWDs — stated carefully, because the naive version is
wrong.** Averaged over a whole session the baseline spectrum peaks at 1.6–2.0 Hz, because
sleep dominates the average; it looks nothing like an SWD. But the classifier sees single
windows, and per window **22–32% of baseline windows have their spectral peak in 5–7 Hz**
(rodent theta), while **44–75% of baseline windows exceed the 5th-percentile SWD window**
on 4–9 Hz relative band power. Band power at the fundamental is therefore a weak
discriminator on its own; harmonics, spike morphology and rhythmicity carry the
separation. This is the central feature-engineering problem.

## Published benchmark (AugUNet1D, arXiv 2601.00459)

The dataset's own paper reports **sample-level** precision 0.91 / recall 0.90, per-mouse
F1 0.87-0.93, versus the "Twin Peaks" algorithmic method at precision 0.67 / recall 0.81.
Their audit numbers (961 h, 22,637 SWDs) match ours (962.2 h, 22,570 events), confirming
the loader.

**Their headline metric is per-timepoint, not event-level.** Our event-level numbers are
therefore not comparable to theirs. Report sample-level precision/recall/F1 as a secondary
metric purely so one directly comparable number exists.

They resample everything to **100 Hz**. We resample to **400 Hz** instead: the test split
is then untouched, and the sharp spike morphology that the line-length and spike-rate
features depend on survives. Low-pass both splits to a common ceiling after resampling so
train and test have identical bandwidth despite different acquisition filters.

## Open

- Labels are single-scorer, so Cohen's κ against a human–human agreement ceiling is not
  measurable. Document as an explicit departure from the source specification.
- Choose the common resample rate (400 Hz is the floor set by the test split).

## Results (recorded, do not lose)

Leave-one-animal-out over the 8 training mice, 40 recordings.

| | Phase 1 (untuned) | Phase 2 (tuned) |
|---|---|---|
| event F1 | 0.504 ± 0.192 | **0.791 ± 0.125** |
| event sensitivity | 0.980 | 0.798 |
| event precision | 0.365 | 0.834 |
| false alarms / hour | 44.6 | **4.18** |
| Cohen's κ | — | 0.624 |
| burden bias (events/h) | +43.9 | **+0.6** |
| burden correlation | 0.08 | **0.759** |

**Operational point at ≤2 false alarms/hour:** threshold 0.80/0.65, sensitivity 0.716,
precision 0.888.

**Window AUPRC barely moved (0.712 → 0.721) while event F1 rose 0.29.** The entire gain
came from calibration and threshold selection, not from the classifier. This is the
project's central claim, measured twice. Do not tune CatBoost.

### Ablations (leave one feature group out, event F1 vs 0.791 full)

| group removed | F1 | delta |
|---|---|---|
| spectral | 0.793 | **+0.003** |
| artifact | 0.796 | **+0.005** |
| wavelet | 0.781 | −0.009 |
| rhythmicity | 0.772 | −0.018 |
| morphology | 0.757 | **−0.033** |

**Morphology carries the most.** That inverts the source specification, which leads with
spectral band power.

Mean deltas are misleading at this scale -- fold-to-fold spread is 0.125, far larger than
most of these effects. Paired Wilcoxon over the 40 recordings (uncorrected; Bonferroni at
8 tests would be 0.006):

| removed | features | F1 | mean delta | paired p |
|---|---|---|---|---|
| morphology | 18 | 0.757 | −0.033 | **0.001** |
| rhythmicity | 19 | 0.772 | −0.018 | **0.001** |
| spectral+artifact | 11 | 0.786 | −0.005 | **0.002** |
| spectral+wavelet | 7 | 0.766 | −0.025 | 0.405 |
| wavelet | 15 | 0.781 | −0.009 | 0.563 |
| wavelet+artifact | 13 | 0.784 | −0.007 | 0.646 |
| spectral | 13 | 0.793 | +0.003 | 0.289 |
| artifact | 19 | 0.796 | +0.005 | 0.383 |

**Only morphology and rhythmicity are clearly load-bearing.** Everything else sits inside
the noise on a paired test, including the large-looking spectral+wavelet drop.

**The spectral/wavelet redundancy hypothesis is NOT confirmed.** Additive prediction for
removing both was −0.007; observed was −0.025, about 3.5x worse, which is the signature of
redundancy -- but at p=0.405 that is not distinguishable from zero. A handful of recordings
drive it. Record it as suggestive, not established, and do not repeat it as a finding.

**Frozen feature set: 19 -- the artifact group is dropped.** Its removal is the best
measured mean (0.796) though within noise, and the honest reading is that amplitude-only
artifact features cannot do the job the specification assigns to an EMG channel we do not
have.

### Still open

- `train_M3` runs at 13.9 FA/h against a cohort mean of 4.2 -- its fold chose a
  permissive 0.40/0.25 threshold. Per-animal threshold sensitivity is the largest
  remaining source of variance.
- `train_M5` has the weakest classifier (window AUPRC 0.527) yet reaches F1 0.781, another
  sign that post-processing dominates.
- Sample-level F1 is 0.633, well below the published 0.90 (which is sample-level on the
  official test split with a U-Net). Not comparable yet: different split, different task
  framing. Phase 3 should run the official test split for a like-for-like number.

## Phase 3 -- causal / real-time variant (recorded)

Same 19 features, same folds, same animals; the only change is that nothing may look
forward. `sosfilt` instead of `sosfiltfilt`, trailing-window probabilities, and a
persistence counter instead of median smoothing. Features are rebuilt from causally
filtered signal (separate cache in `data/interim/features_causal`) -- reusing offline
features would measure a domain shift, not the constraint.

| | offline | causal | paired p |
|---|---|---|---|
| event F1 | 0.796 | 0.786 | 0.368 (within noise) |
| event sensitivity | 0.805 | 0.836 | 0.451 (within noise) |
| event precision | 0.828 | 0.797 | **0.003** |
| false alarms / hour | 4.31 | 5.63 | **0.008** |

**Going real-time costs almost nothing in F1.** The cost is specifically in precision and
false-alarm rate, not in detection ability -- sensitivity actually rises. Do not report
this as "no cost"; report it as "the cost lands on precision".

**Two different latency numbers, do not conflate them:**

- **Onset localisation error** (how accurate the reported boundary is): median 0.00 s,
  IQR -0.50 to +1.00 s, p90 2.00 s over 18,441 matched events. Quantised to the 0.5 s
  stride, so this is "within one step". Negative values are real: the detector sometimes
  starts firing before the annotated onset.
- **Declaration delay** (how long before the system *knows*, the number a closed-loop
  device cares about): 1.0-3.0 s, median 1.8 s, set entirely by the persistence counter
  N x 0.5 s.

Onsets are reported where evidence began, not where the counter fired, so the declaration
delay is carried separately rather than hidden inside the onset. That is why the two
numbers differ.

**Hardware note:** Apple M4, 10-core GPU, 17 GB unified, Metal 4. PyTorch MPS is available,
so the U-Net does not need cloud compute.

## Phase 3 -- U-Net vs feature pipeline (recorded)

1.9M-parameter 1-D residual U-Net, per-sample output, no windowing and no overlap-add.
Same folds, same animals, same post-processing and scoring code. 2500 steps/fold on MPS,
~7 min/fold.

| metric | features | U-Net | delta | paired p |
|---|---|---|---|---|
| event F1 | 0.796 | 0.823 | +0.028 | 0.242 (**within noise**) |
| event sensitivity | 0.805 | 0.963 | +0.158 | <0.0001 |
| event precision | 0.828 | 0.737 | −0.091 | <0.0001 |
| false alarms / hour | 4.31 | 8.21 | +3.90 | <0.0001 |
| sample F1 | 0.639 | 0.763 | +0.124 | <0.0001 |
| burden bias (events/h) | +0.6 | +8.0 | | |
| burden correlation | 0.759 | 0.862 | | |

**The headline is that the two metrics disagree, and that is the whole point.**

- **At sample level the U-Net wins decisively** (+0.124, p<0.0001). Expected and
  architecturally meaningful: dense per-sample segmentation localises boundaries better
  than window classification followed by post-hoc reconstruction.
- **At event level it does not win** (p=0.242). Per animal it is 4 better / 4 worse.
  Better boundaries do not translate into better event *counting*; the gain is spent
  trading precision for sensitivity.
- **For the actual deliverable it is worse.** Burden bias goes +0.6 -> +8.0 because it
  raises twice the false alarms, even though burden correlation improves.

So which architecture "wins" depends entirely on which metric is the headline -- which is
the same argument this project has been making since Phase 1. Do not report the U-Net as
better without saying at what.

Threshold instability is worse for the U-Net: 6 of 8 folds chose 0.95/0.80, but two chose
0.50/0.35 and 0.30/0.15.

## Phases

0. **Data gate** (now) — verify montage, sampling rate, label format; plot real SWDs. EDA notebook.
1. **Thin end-to-end skeleton** — a deliberately dumb model, but the *complete* chain through
   event formation, event-level metrics, and burden stats. Produces an honest baseline number
   on day one that every later change is measured against.
2. **Make it good** — full feature set, artifact handling, post-processing tuning, ablations.
3. **Comparison and writeup** — U-Net variant against the same evaluation harness; README.

## Library spine

Prefer well-known libraries over hand-rolled implementations. `mne` (I/O, filtering,
epoching), `scipy`, `pywavelets`, `catboost`, `scikit-learn`, `timescoring`, `polars`.
Event formation from a probability trace (~50 lines of run-length logic) is ours to write;
no library does it well.

Keep the codebase small. Do not add abstraction layers for swappable backends until there
are actually two working backends.

## Notebooks

`notebooks/` holds narrative deliverables (EDA, results) and is part of what a reader sees.
Notebooks **import from `src/`** and never define reusable logic. Commit outputs for these
so they render on GitHub — the rendered figures are the point.

## Data hygiene

Never commit recordings, checkpoints, or credentials. Dataset access notes must not
contain passwords. Raw `.mat` files go in `data/raw/`.

<!--
Keep this file focused on standing instructions and non-obvious context — not a
changelog, and not anything readable from the source.
-->
