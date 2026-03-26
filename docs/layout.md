# Layout (Benchmark)

This is the primary navigation layout for the benchmark repo.

## Main directories

- `pipeline/`: benchmark run-path code and runner entrypoints.
  - `pipeline/run_experiments.py`
  - `pipeline/scripts/`
  - `pipeline/scripts/diagnostics/`
  - `pipeline/tests/unit/`
  - `pipeline/tests/dryrun/`
- `paper/`: manuscript refresh scripts and paper-specific outputs.
  - `paper/scripts/`
- `utilities/`: active non-pipeline helpers.
  - `utilities/scripts/`
  - `utilities/qc/`
- `deprecated/`: retired scripts, notes, launchers, and data snapshots.
- `docs/`: structural docs, inventories, and migration notes.

## Entry policy

- Use `pipeline/run_experiments.py` for job generation.
- Use `pipeline/scripts/`, `paper/scripts/`, and `utilities/scripts/` directly.

## Stability guarantees

- `slurm/` remains the stable launcher surface.
- Generated jobfiles under `slurm/generated/` are disposable local byproducts, not long-lived code surfaces.
