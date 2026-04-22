# Layout (benchmark)

This file is a quick map of the benchmark repository.

## Main directories

- `pipeline/`: benchmark workflow code and runner entrypoints.
  - `pipeline/run_experiments.py`
  - `pipeline/scripts/`
  - `pipeline/scripts/diagnostics/`
  - `pipeline/tests/unit/`
  - `pipeline/tests/dryrun/`
- `paper/`: manuscript table and figure scripts plus paper-specific outputs.
  - `paper/scripts/`
- `utilities/`: active helper scripts outside the main workflow.
  - `utilities/scripts/`
  - `utilities/qc/`
- `deprecated/`: retired scripts, notes, launchers, and data snapshots.
- `docs/`: layout notes, file lists, and migration notes.

## Entry policy

- Use `pipeline/run_experiments.py` for job generation.
- Use `pipeline/scripts/`, `paper/scripts/`, and `utilities/scripts/`
  directly.

## Stability notes

- `slurm/` remains the stable launcher path.
- Generated jobfiles under `slurm/generated/` are disposable local byproducts,
  not long-lived source files.
