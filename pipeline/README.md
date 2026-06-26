# Pipeline overview

Orchestration code for the benchmark. Navigation lives in
[`../README.md`](../README.md) and the stage-by-stage walkthrough in
[`../PIPELINE.md`](../PIPELINE.md).

| Path | Role |
| --- | --- |
| `run_experiments.py` | experiment matrix and jobfile generation |
| `scripts/` | benchmark-side stage scripts and statistics logic |
| `scripts/diagnostics/` | paper-side check scripts for manuscript figures |
| `tests/unit/` | script contract tests |
| `tests/dryrun/` | one compile-only wrapper per pipeline script |

Tests: [`tests/README.md`](tests/README.md).
