# Pipeline

Experiment grid, outcome construction, and statistical tests for the benchmark. See [`../PIPELINE.md`](../PIPELINE.md) for the stage order.

| Path | Contents |
| :-- | :-- |
| `run_experiments.py` | Experiment grid and Stage 0–3 jobfile generation |
| `task_registry.py` | Binary and regression outcome task lists |
| `provisional_24to48_task_registry.py` | Outcome task lists for hours 24–48 in each experiment |
| `scripts/` | Cohort alignment, outcome extraction, and bootstrap and permutation tests |
| `scripts/diagnostics/` | Embedding PCA, clinical boundary probe, attention, and xVal analyses |
| `tests/unit/` | Unit tests for pipeline, statistics, and table scripts |
| `tests/dryrun/` | One dry-run wrapper for each script under `pipeline/` |

```bash
pytest pipeline/tests/unit
bash pipeline/tests/dryrun/run_all.sh
```
