# Automated spike-wave discharge detection in mouse EEG

Detecting absence seizures in 962 hours of continuous single-channel mouse EEG, and
turning them into the seizure-burden statistics a lab actually uses. Two detectors —
hand-crafted features with gradient boosting, and a dense 1-D U-Net — are built and
compared through one evaluation harness.

The interesting results are not the headline scores. They are the three places where the
obvious conclusion turned out to be wrong.

---

## Results

**Leave-one-animal-out over 8 mice, 40 recordings, 962 hours.**

| | event F1 | sensitivity | precision | false alarms/h | sample F1 |
|---|---|---|---|---|---|
| Phase 1 baseline (untuned) | 0.504 ± 0.192 | 0.980 | 0.365 | 44.6 | 0.507 |
| **Feature pipeline** | **0.796** | 0.805 | 0.828 | **4.31** | 0.639 |
| Causal / real-time | 0.786 | 0.836 | 0.797 | 5.63 | — |
| 1-D U-Net | 0.823 | 0.963 | 0.737 | 8.21 | **0.763** |

At an operating point of ≤2 false alarms/hour, the feature pipeline reaches sensitivity
0.716 at precision 0.888.

**Official held-out test split — 10 unseen mice, evaluated once.**

| | event F1 | false alarms/h | sample F1 | sample P / R |
|---|---|---|---|---|
| Feature pipeline | 0.622 ± 0.159 | 36.3 | 0.668 | 0.551 / 0.910 |
| **1-D U-Net** | **0.912 ± 0.033** | **5.4** | **0.890** | **0.838 / 0.955** |
| *AugUNet1D (published)* | — | — | *~0.90* | *0.91 / 0.90* |
| *Twin Peaks (published)* | — | — | — | *0.67 / 0.81* |

The U-Net is competitive with the published benchmark and clearly beats the algorithmic
baseline. The feature pipeline collapsed — [see below](#3-hand-crafted-features-were-the-brittle-ones).

---

## Three findings

### 1. The classifier was never the bottleneck

Event F1 went from **0.504 to 0.796** between Phase 1 and Phase 2. Over the same period,
window-level AUPRC moved from **0.712 to 0.721**.

Essentially all of that gain came from calibrating the probabilities and choosing the
decision threshold on a held-out animal — not from the model. Holding the classifier
completely fixed and moving one threshold takes event F1 from 0.50 to 0.80 and false
alarms from 44.6/h to 3.5/h.

The consequence for how effort was spent: **no time was invested in tuning CatBoost.**
Post-processing and event formation are where event-level performance is decided.

### 2. Which architecture "wins" depends on the metric

On leave-one-animal-out, the U-Net beats the feature pipeline on sample-level F1 by
**+0.124 (p < 0.0001)** — dense per-sample segmentation localises boundaries better than
classifying windows and reconstructing time resolution afterwards, exactly as the
architecture predicts.

At **event** level the same comparison is **not significant** (+0.028, p = 0.242; four
animals better, four worse), and for seizure burden — the actual scientific deliverable —
the U-Net is **worse**: bias +8.0 events/hour against the feature pipeline's +0.8, because
it raises nearly twice the false alarms.

Reporting only sample-level F1, the metric the published paper leads with, would have
concluded the U-Net is straightforwardly better and shipped the detector that is worse at
the job.

### 3. Hand-crafted features were the brittle ones

On the official test split the feature pipeline fell from **0.796 to 0.622**, with false
alarms rising **4.3/h → 36.3/h**. Sensitivity held at 0.949, so it still *finds* seizures;
precision collapsed to 0.485. A transfer failure, not a detection failure.

Comparing baseline-window feature distributions between splits, in training standard
deviations:

| feature | train | test | shift |
|---|---|---|---|
| `harmonic_ratio` | 0.271 | 3.896 | **+16.6 SD** |
| `rel_high` (30–100 Hz) | 0.018 | 0.147 | **+8.2 SD** |
| `wavelet_l4` | 0.010 | 0.090 | **+7.6 SD** |

Median shift across all 19 features: 0.93 SD. The test domain sits outside the training
distribution, concentrated in the high-frequency features.

**Cause.** Training data was acquired at 1000 Hz — oversampled, so its 30–100 Hz band is
clean — then decimated. Test data was acquired natively at 400 Hz, where a weaker
acquisition-side anti-alias filter folds content above 200 Hz down into that band. The
common 1–100 Hz band-pass applied to both splits equalises *bandwidth* but cannot undo
aliasing that happened in the amplifier.

The U-Net saw the same shift and barely noticed, presumably because it learned
low-frequency waveform morphology rather than high-frequency band ratios.

**A plausible fix — low-pass both splits below the contaminated band and drop the two
worst-shifted features — is deliberately left untested.** Validating it would mean a
second look at a split that had been spent, and a number that improves after you look is
not a held-out number.

---

## The task

Spike-wave discharges are the electrographic signature of absence epilepsy. Quantifying
them across 24-hour recordings is the standard readout for genotype and drug studies, and
manual scoring takes a skilled human hours per record.

**The difficulty is not amplitude.** Ordinary rodent theta rhythm occupies the same
frequency band as the seizure: roughly a quarter of non-seizure windows have their spectral
peak in 5–7 Hz, the same place the SWD fundamental sits. Dominant frequency alone is
useless as a discriminator (AUC 0.53, and 0.42 against theta-like windows — *below* chance,
because theta actually sits slightly higher).

The separation comes from harmonics. A spike-and-wave complex is not a sinusoid, so it
deposits energy at 12/18/24 Hz where a smooth oscillation cannot: 10–25 Hz relative power
alone reaches AUC 0.91.

See [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) for the full analysis, and
[`docs/pipeline.html`](docs/pipeline.html) for stage-by-stage diagrams.

## Pipeline

```
50 .mat files                                  962 h, 8 mice, 22,570 events
      │
      ▼
1. acquire & audit      HDF5, animals keyed by filename, defects catalogued
      ▼
2. condition & window   resample → band-pass → z-score → 2.0 s windows every 0.5 s
      ▼
3. featurize & classify 19 features in 4 groups → CatBoost → P(seizure) per window
      ▼
4. form events          overlap-add → hysteresis → merge gaps → drop fragments
      ▼
5. evaluate & report    event-level scoring (SzCORE) + seizure burden
```

Stage 4 is where the result is won. Its parameters are derived from the annotation
distribution rather than convention: **merge gap 0.5 s** (0.86% of genuinely separate
annotated pairs fall below it, versus 1.75% below 1.0 s) and **minimum duration 1.0 s**
(discards 0.02% of true events, matching the annotators' own floor).

The U-Net replaces stages 2–3 with per-sample segmentation — no windowing, no overlap-add —
and runs through the identical stage 4 and 5 code, so the comparison measures architecture
rather than metric.

## Quickstart

```bash
uv sync
uv run python -m mouse_seizure.download          # 19.5 GB from Zenodo, resumable
uv run python experiments/audit_dataset.py       # verify the download
uv run python experiments/build_feature_cache.py # ~12 min
uv run python experiments/phase2_tuned.py        # leave-one-animal-out
uv run pytest
```

## Dataset

[Zenodo 10.5281/zenodo.17982390](https://doi.org/10.5281/zenodo.17982390), CC-BY 4.0.
Single-channel EEG from 18 C3H/HeJ mice with per-sample manual annotations.

**Six undocumented properties the loader has to absorb**, every one found empirically:

| | |
|---|---|
| Sampling rate differs by split | 1000 Hz train, 400 Hz test — the trap that drives finding 3 |
| `mID` is unreliable | test files M6, M7, M8 all report `mID = 5`; animals are keyed by filename |
| Mouse numbers collide across splits | training `M3` and test `M3` are different animals |
| Test recordings are cropped at 100 seizures | so events/hour and burden are meaningless there |
| Recordings can start or end mid-event | label arrays are zero-padded before run-length detection |
| Array orientation varies per file | some `(n, 1)`, some `(1, n)` |

## Evaluation

Event scoring uses [`timescoring`](https://github.com/esl-epfl/timescoring), the SzCORE
reference implementation, so results are comparable to published work and the many-to-one
matching is resolved the way the community resolves it.

**Its defaults are scaled for human clinical seizures** — a 30-second tolerance collar
around events that here last five — and would silently inflate every number without raising
an error. All five parameters are overridden at rodent scale in
[`config.py`](src/mouse_seizure/config.py), and nothing in the codebase accepts scoring
parameters as an argument, so the defaults cannot be reached by accident.

Validation is **leave-one-animal-out with nested threshold selection**: within each fold a
second animal is held out purely to choose the operating point, so the test animal never
influences it.

## Departures from the original specification

| specified | actual | consequence |
|---|---|---|
| 2 EEG channels + EMG | 1 EEG channel | no cross-channel synchrony features; no EMG artifact gating |
| 256 Hz | 1000 / 400 Hz | resampled to a common 400 Hz; drives finding 3 |
| events ~1 s | median 5.6 s | longer windows, different merge-gap and minimum-duration |
| 2 scorers + adjudication | single scorer | no human–human agreement ceiling; κ is against one annotator |
| circadian analysis | no clock metadata | dropped |
| genotype/drug comparison | one line, no conditions | replaced by cross-animal comparison |

Measured ablations also invert the specification's priority: it leads with spectral band
power, but only **morphology** (−0.033, p = 0.001) and **rhythmicity** (−0.018, p = 0.001)
are distinguishable from noise on a paired test. The artifact group measured as dead weight
and was removed.

## Layout

```
src/mouse_seizure/   io, preprocess, features, postprocess, evaluate, causal, unet
experiments/         audit, phase1–phase3 runners
notebooks/01_eda.ipynb
docs/pipeline.html   stage-by-stage diagrams
tests/               17 tests
```

~3,100 lines. `data/` and `artifacts/` are gitignored.

## Limitations

- **Burden agreement** on the frozen 19-feature set: bias +0.8 events/hour, r = 0.770.
- **The frozen feature set is a selected maximum.** It is the best of nine configurations
  scored on the same cross-validation data, so its 0.796 is mildly optimistic; the
  unselected 21-feature set scores 0.791, and the difference was not distinguishable from
  noise. The held-out test number is unaffected — that configuration was fixed first.
- **Eight training animals.** Leave-one-animal-out over n=8 has real fold-to-fold variance
  (±0.125 event F1); most ablation differences sit inside it, and are reported as such.
- **Single-scorer labels**, so no inter-rater ceiling and no estimate of label noise.
- **The official test split confounds animal, session and acquisition hardware**, so a
  drop across it cannot be attributed to any one cause — which is exactly what finding 3
  turned out to be about.
- **The causal variant's cost lands on precision**, not detection: sensitivity actually
  rises, false alarms go 4.3 → 5.6/h. Declaration delay is 1.0–3.0 s (median 1.8 s), set by
  the persistence counter.

## References

- Dataset: [Zenodo 10.5281/zenodo.17982390](https://doi.org/10.5281/zenodo.17982390) (CC-BY 4.0)
- Benchmark: *Combining Residual U-Net and Data Augmentation for Dense Temporal Segmentation
  of Spike Wave Discharges in Single-Channel EEG*, [arXiv:2601.00459](https://arxiv.org/abs/2601.00459)
- Scoring: [SzCORE](https://arxiv.org/abs/2402.13005) / [`timescoring`](https://github.com/esl-epfl/timescoring)
