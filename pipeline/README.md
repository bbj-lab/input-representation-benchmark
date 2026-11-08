# Pipeline Overview

Orchestration, outcome extraction, statistical inference, and mechanistic evaluation code for the ML4H 2026 benchmark. See [`../README.md`](../README.md) and [`../PIPELINE.md`](../PIPELINE.md) for the full stage execution order.

| Path | Purpose |
| --- | --- |
| `run_experiments.py` | Define the experiment matrix and generate Stage 0 through Stage 3 commands |
| `task_registry.py` | Declare the 17 binary and 22 continuous clinical probe tasks |
| `provisional_24to48_task_registry.py` | Define the 37-task 24-to-48-hour follow-up contract (`(24h, 48h]`) |
| `scripts/` | Cohort alignment, outcome extraction, and hierarchical bootstrap entrypoints |
| `scripts/diagnostics/` | Mechanistic analysis scripts for centile PCA, boundary probes, attention washout, and xVal scaling |
| `tests/unit/` | Unit and contract tests for pipeline scripts, figure generators, and statistical routines |
| `tests/dryrun/` | One CLI dry-run wrapper for each script under `pipeline/` |

See [`tests/README.md`](tests/README.md) for test execution commands.
