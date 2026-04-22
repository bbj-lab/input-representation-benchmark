# Pipeline overview

`pipeline/` contains the main benchmark orchestration code.

- `run_experiments.py`: jobfile generator and experiment matrix control.
- `scripts/`: benchmark-side stage scripts and statistics logic.
- `scripts/diagnostics/`: diagnostics consumed by manuscript figure builders.
- `tests/unit/`: unit and contract checks for each pipeline script.
- `tests/dryrun/`: one dry-run wrapper per pipeline script.
