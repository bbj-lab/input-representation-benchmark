# Pipeline Tests

Unit and CLI contract checks for the benchmark pipeline.

```bash
conda activate input-rep
pytest pipeline/tests/unit
bash pipeline/tests/dryrun/run_all.sh
```

- `unit/`: Tests for cohort alignment, CLIF-to-MEDS conversion, 24-to-48-hour outcome extraction, batch geometry calibration, bootstrap and permutation routines, table and figure builders, and script contracts.
- `dryrun/`: Compile-only and `--help` execution wrappers for every script listed in `dryrun/manifest.json`. Set `IRB_DRYRUN_EXECUTE_MODE=1` to run execute-mode checks.
