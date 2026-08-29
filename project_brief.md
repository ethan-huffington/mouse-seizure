# Automated spike-wave discharge detection in C3H/HeJ mice

**What this is.** A detector that finds spike-wave discharge (SWD) events in continuous
mouse EEG and converts them into per-session seizure-burden statistics, replacing manual
visual scoring. This document translates the source specification
(`mouse seizure detection project.pdf`) into what will actually be built against the
dataset we have, and records every point where the two diverge.

Every parameter below is derived from measurements on the real data, not from the
source PDF's assumptions. Where the two disagree, the measurement wins and the
disagreement is noted.

---

## 1. Objective

Locate every SWD event in a continuous single-channel EEG recording, and report seizure
burden per session: event count, events per hour, total time in SWD, and the duration
distribution.

The scientific readout is the burden statistics, not per-window labels. A model that
scores well per-window but miscounts events is a failure at the task.

## 2. Data

Zenodo record [10.5281/zenodo.17982390](https://doi.org/10.5281/zenodo.17982390),
CC-BY 4.0. Single-channel epidural EEG from C3H/HeJ mice, a line with spontaneous SWDs.
50 files, 18 GB, MATLAB v7.3 (HDF5).

| | training | test |
|---|---|---|
| files | 40 (`DayN_MZ.mat`) | 10 (`20240209_MZ_MAT.mat`) |
| mice | 8, five days each | 10 different mice, one day |
| duration | 962 h total, 22–26 h per session | 61.5 h total, 2.2–6.5 h per session |
| sampling rate | **1000 Hz** | **400 Hz** |
| events | 22,570 | 1,890 |
| positive fraction | 3.81 % | 4.34 % |

### Verified quirks

Every one of these was found empirically; none is documented by the dataset.

- **Sampling rate differs between splits.** Handled by resampling (§4.1).
- **Array orientation varies per file** — some `(n, 1)`, some `(1, n)`.
- **The `mID` field is unreliable.** Test files M5, M6, M7 and M8 all report `mID = 5`.
  Animals are identified by filename. All 40 training files are correct.
- **Mouse numbers overlap across splits but are different animals** — training `M3` and
  test `M3` are unrelated. Identifiers are namespaced `train_M3` / `test_M3`.
- **Test recordings are cropped at exactly 100 seizures** and end mid-event. Their
  duration is "time to accumulate 100 SWDs", so **events/hour and burden are meaningless
  on the test split** (§9).
- **Two training files start mid-event** (`Day1_M4`, `Day2_M8`), so boundary censoring is
  not test-only. Label arrays are zero-padded before run-length detection.

### The train/test split confounds three variables

The two splits differ in animal, recording session, **and acquisition hardware**
(1000 vs 400 Hz, different anti-alias filtering). A performance drop on the official test
split therefore cannot be attributed to failure to generalize across animals alone.

**Consequence:** primary validation is leave-one-animal-out *within* the training split,
where only the animal varies. The official test split is reported as a secondary, harder
number that additionally measures robustness to acquisition shift.

### Signal facts

- SWD fundamental **~5.9 Hz**, with strong harmonic content to ~20 Hz from the sharp
  spike component.
- Event durations (n = 24,460): median **5.6 s**, IQR 4.1–7.1 s, 1st–99th percentile
  1.6–12.9 s, extremes 0.67 s and 31.2 s. **This is 5× longer than the "~1 s or longer"
  the source PDF assumes**, and it changes window length, merge-gap and minimum-duration.
- Inter-event gaps: median 29.7 s, but 1.75 % fall below 1 s and the minimum is 0.019 s.
  Annotators do separate events across sub-second gaps, so aggressive merging destroys
  real structure.
- Burden is strongly animal-dependent and stable within animal: `train_M2` sits at
  1.1–1.6 % positive across all five days, `train_M3` at 5.3–7.3 %.

### The central feature-engineering problem

The SWD fundamental is stable at **5.5–5.9 Hz** in every recording checked. The background
is not so clean, and the claim needs stating carefully because the obvious version is wrong.

Averaged across a whole session, the baseline spectrum peaks at **1.6–2.0 Hz** — sleep
dominates the average, and it looks nothing like an SWD. But the classifier never sees a
session average; it sees one 2 s window. Per window:

- **22–32% of baseline windows have their spectral peak in 5–7 Hz** — rodent theta rhythm,
  entirely normal during exploration and REM sleep, sitting exactly on the SWD fundamental.
- **44–75% of baseline windows exceed the 5th-percentile SWD window** on 4–9 Hz relative
  band power.

So band power at the fundamental is a weak discriminator on its own. The separation has to
come from **harmonics** (a spike-and-wave complex is not a sinusoid and deposits energy at
12/18/24 Hz; theta does not), **morphology** (the sharp spike), and **rhythmicity**
(regular spike train vs. irregular oscillation). This is the substance of the feature work.

## 3. Ground truth

Per-sample binary labels (`SWDlabel`, uint8), one per EEG sample — denser than the
onset/offset event lists the source PDF assumes, which is strictly better.

**Departure from spec:** the PDF specifies two independent scorers with a third
adjudicating (§3), yielding a consensus label set and a measurable inter-scorer
agreement. These labels are **single-scorer**. Consequences:

- **Cohen's κ against a human–human agreement ceiling is not measurable.** The PDF's
  framing of model performance "relative to human–human agreement rather than a
  hypothetical perfect label" cannot be reproduced. κ against the single scorer is still
  reported, but it is agreement with one annotator, not with a consensus.
- Label noise cannot be estimated, so no confidence weighting.

Annotators applied an effective **~1 s minimum duration floor**: only 0.02 % of events
are shorter than 1 s. That empirical floor sets the minimum-duration rule (§7).

## 4. Pipeline

### 4.1 Preprocess

- **Resample to a common 400 Hz.** Test data is untouched; training decimates 1000 → 400
  by polyphase resampling (ratio 2/5). The source PDF's 256 Hz and the dataset paper's
  100 Hz are both rejected: 100 Hz smooths away the sharp spike component that the
  morphology features depend on, and there is no reason to go below the test split's
  native rate.
- **Low-pass both splits to a common ceiling** after resampling, so train and test have
  identical bandwidth despite different acquisition-side anti-alias filters. Without
  this, the splits differ spectrally by construction.
- **Band-pass** to the physiological range. **No notch filter** — this is US-recorded
  data and mains contamination will be checked rather than assumed.
- **Per-session z-scoring.** Units are arbitrary and amplitude scale differs between
  animals and days, so raw amplitude must not leak into features.
- **Artifact handling.** The PDF leans on EMG channels for this (§4.1); we have none
  (§10). Amplitude and clipping heuristics only.

### 4.2 Windowing

**Window 2.0 s, stride 0.5 s.** The PDF specifies ~1 s windows because it assumes ~1 s
events; measured events are 5.6 s median, so a 2 s window still sits inside a typical
event while giving ~0.5 Hz spectral resolution — enough to separate the 5.9 Hz
fundamental from its neighbourhood. Stride 0.5 s preserves the PDF's detection cadence.

### 4.3 Featurize

One row per window. Feature groups, in the order they will be ablated:

| group | features | rationale |
|---|---|---|
| spectral | band power at the fundamental and harmonics, spectral entropy, band ratios | SWDs concentrate energy at ~5.9 Hz and multiples |
| wavelet | coefficient energies at SWD scales | resolves transient spike-and-wave morphology better than a plain FFT |
| morphology | line length, amplitude, variance, Hjorth parameters, spike rate | encodes spike-and-wave shape in the time domain |
| rhythmicity | autocorrelation peak height and lag, regularity of inter-spike intervals | separates rhythmic SWD from equally-powered but irregular theta |

**Departure from spec:** the PDF's **cross-channel synchrony group (§5) is impossible** —
the recording is single-channel. The PDF calls this "often among the most discriminative"
for generalized events. The `rhythmicity` group above is added to partially compensate,
targeting the theta confound the PDF does not anticipate.

### 4.4 Model

CatBoost on the per-window feature table, outputting **P(SWD) per window** — a
probability, not a hard class. Everything downstream needs a graded score.

Class weighting for the ~4 % positive rate; threshold selection is a separate decision
made on validation folds (§8).

## 5. Post-processing and event formation

The five coupled steps from PDF §7, with parameters set from measurement:

1. **Reconcile overlap.** Average overlapping window predictions into a per-sample
   probability trace (overlap-add).
2. **Threshold with hysteresis.** Dual threshold, enter high / stay low, tuned on
   validation.
3. **Smooth.** Median filter over the probability trace.
4. **Form events.** Zero-pad, run-length group, then apply:
   - **merge gap 0.5 s** — merges only 0.86 % of genuinely-separate annotated event
     pairs, while still closing model dropout flicker. A 1 s gap would wrongly merge
     1.75 %.
   - **minimum duration 1.0 s** — discards 0.02 % of true events. Matches the annotators'
     own floor.
5. **Aggregate** into the event list and burden statistics.

## 6. Validation and metrics

**Primary protocol: leave-one-animal-out over the 8 training mice** (8 folds). Never mix
windows from one animal across train and test — windows within a recording are highly
correlated and splitting by window inflates results. Reported as a **per-animal table plus
mean ± spread**, never a bare mean: with n = 8 the spread is part of the result.

**Secondary: the official 10-mouse test split**, which additionally measures robustness to
acquisition shift.

### Metrics, in tiers

- **Headline — event-level**, via `timescoring` (the SzCORE reference implementation):
  sensitivity, precision, F1, and **false alarms per hour**. FA/h is the metric that
  governs operational trust.
- **Comparable — sample-level** precision/recall/F1. Computed for one reason: the
  dataset's own paper reports *per-timepoint* scores (precision 0.91, recall 0.90, per-mouse
  F1 0.87–0.93; "Twin Peaks" at 0.67/0.81). Without a sample-level number we have nothing
  directly comparable to published work. Event-level and sample-level numbers are always
  labelled as such and never mixed.
- **Diagnostic — window-level AUPRC**, before post-processing, to separate classifier
  quality from post-processing quality.
- **Scientific — burden agreement.** Predicted vs. true events/hour and % time in SWD per
  session, as a Bland–Altman plot. A detector can have good F1 and still bias burden.
- **Operational — detection latency**, the distribution of onset and offset errors.
- **Agreement — Cohen's κ**, with the single-scorer caveat from §3.

### ⚠️ `timescoring` parameters must be overridden

Its defaults are scaled for human clinical seizures. Running with defaults on rodent data
produces meaningless, inflated numbers **silently** — no error, just implausibly good
results. All values live in one config module:

| parameter | default | ours | why |
|---|---|---|---|
| `toleranceStart` | 30 s | **0.5 s** | ~9 % of the 5.6 s median event |
| `toleranceEnd` | 60 s | **0.5 s** | same |
| `minDurationBetweenEvents` | 90 s | **0.5 s** | matches the merge-gap rule |
| `maxEventDuration` | 300 s | **60 s** | above the 31.2 s longest real event, so genuine events are never split |
| `minOverlap` | 0 | **0** | any-overlap matching, per SzCORE convention |

### Reliability

- **Bootstrap confidence intervals resampled over animals, not windows.** Windows within a
  recording are not independent; bootstrapping over them yields absurdly tight intervals.
- **A sensitivity-vs-FA/h operating curve** rather than a single F1, since different labs
  want different operating points.
- **Calibration**: reliability diagram. Hysteresis thresholds only mean something if the
  model's 0.8 corresponds to 80 %; boosted trees are typically miscalibrated.
- **Thresholds tuned on validation folds only**, never on test.
- **Baselines for context**: a trivial line-length threshold detector of ours, plus the
  two published reference points above.

## 7. Outputs

- **Per-session event list**: onset, offset, duration.
- **Seizure burden**: event count, events per hour, total time in SWD.
- **Duration distribution**: median and spread.
- **Cross-animal comparison**: burden differences between animals — the actual readout.

**Departure from spec:** two of the PDF's §9 outputs are dropped.
- **Circadian structure** is not reportable: sessions carry no wall-clock or light-cycle
  metadata, only relative day numbers.
- **Genotype / drug comparisons** are impossible: every animal is the same line with no
  experimental conditions. Burden is compared *across animals* instead.

Burden statistics are reported **from training-split sessions only**, because the test
split's crop-at-100-seizures construction makes its rates an artifact (§2).

## 8. Scope and variants

Core scope is **offline** detection, free to smooth bidirectionally. Two extensions, in
priority order:

1. **Causal real-time variant** — persistence counter instead of bidirectional smoothing,
   with a latency budget. Small addition, and the form relevant to closed-loop
   neurostimulation.
2. **Dense 1-D U-Net** — per-sample segmentation with no windowing and no overlap-add,
   evaluated through the identical harness. This is a genuine architectural comparison
   (hand-crafted + per-window vs. learned + per-sample), not a library swap. It is a
   comparison against the baseline, not a replacement for it.

**Departure from spec:** the PDF's multichannel extension (§10) is out of scope — the
data is single-channel.

---

## Departures from the source specification

| PDF says | reality | consequence |
|---|---|---|
| 2 EEG channels + EMG, bihemispheric | 1 EEG channel | cross-channel synchrony features impossible; no EMG-based artifact gating |
| 256 Hz | 1000 Hz train / 400 Hz test | resample to a common 400 Hz; acquisition shift confounds the official split |
| events ~1 s | median 5.6 s | longer windows, different merge-gap and minimum-duration |
| 2 scorers + adjudication | single scorer | no human–human agreement ceiling; κ is against one annotator |
| in-house lab recordings | public CC-BY dataset | no control over montage or protocol |
| circadian patterning | no clock metadata | dropped |
| genotype/drug comparisons | one line, no conditions | replaced by cross-animal comparison |

## Phases

0. **Data gate** — *complete*. Format verified, quirks catalogued, loader written and
   tested, dataset audited. Remaining: EDA notebook.
1. **Thin end-to-end skeleton** — deliberately minimal model, but the *complete* chain
   through event formation, event-level metrics and burden statistics. Produces an honest
   event-level baseline that every later change is measured against.
2. **Make it good** — full feature set, artifact handling, post-processing tuning,
   feature-group ablations, calibration.
3. **Comparison and writeup** — causal variant and U-Net through the same harness; README.

## Library spine

`mne` (I/O, filtering, epoching), `scipy`, `pywavelets`, `catboost`, `scikit-learn`,
`timescoring`, `polars`. Event formation from a probability trace is ours to write; no
library does it well.

Keep the codebase small. No abstraction layers for swappable backends until two backends
actually work.
