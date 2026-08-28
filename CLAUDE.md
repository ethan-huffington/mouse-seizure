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

## Architecture

<!-- One paragraph, once the shape of the project is clear. -->

## Conventions & gotchas

<!-- Only things that cannot be discovered by reading the code. -->

<!--
Keep this file short. It is loaded into context on every turn of every Claude
Code session in this repo, so it should hold standing instructions and
non-obvious context — not a changelog, and not anything readable from the source.
-->
