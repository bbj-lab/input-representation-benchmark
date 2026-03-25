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
- `tests/`: compatibility wrappers that forward to `pipeline/tests/unit/`.
- `deprecated/`: retired scripts, notes, launchers, and data snapshots.
- `docs/`: structural docs, inventories, and migration notes.

## Compatibility policy

- Keep old entry paths during migration:
  - `run_experiments.py` (root)
  - `scripts/*`
  - `scripts/diagnostics/*`
- Old entry paths are wrappers that execute the main files in `pipeline/`, `paper/`, or `utilities/`.

## Stability guarantees

- `slurm/` remains stable during the current rerun chain.
- Existing queued jobs should resolve old script paths unchanged.
