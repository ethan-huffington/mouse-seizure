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

**Signal facts:** SWD fundamental ~5.9 Hz; events median 3.7–5.9 s (much longer than the
~1 s the source PDF assumed); positive fraction 1–7%; event rate 8.6–35.6/h and genuinely
animal-dependent. **Baseline theta sits at the same ~5.9 Hz peak as SWD** — peak frequency
alone does not separate them, so amplitude and harmonic structure carry the discrimination.
This is the central feature-engineering problem.

## Open

- Labels are single-scorer, so Cohen's κ against a human–human agreement ceiling is not
  measurable. Document as an explicit departure from the source specification.
- Choose the common resample rate (400 Hz is the floor set by the test split).

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
