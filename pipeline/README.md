# Pipeline Surface

`pipeline/` contains the main benchmark orchestration code.

- `run_experiments.py`: jobfile generator and experiment matrix control.
- `scripts/`: benchmark-side stage scripts and stats refresh logic.
- `scripts/diagnostics/`: diagnostics consumed by manuscript figure builders.
- `tests/unit/`: unit and contract checks for each pipeline script.
- `tests/dryrun/`: one dry-run wrapper per pipeline script.

Compatibility wrappers are kept at old paths (`run_experiments.py`, `scripts/*`) so active jobs and older operator commands keep working during migration.
