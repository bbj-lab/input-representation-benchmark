# Pipeline tests

Contract checks for every script under `pipeline/`.

```bash
conda activate input-rep
pytest pipeline/tests/unit
bash pipeline/tests/dryrun/run_all.sh
```

- `unit/`: `main()` guards, dry-run wrapper coverage, manifest parity.
- `dryrun/`: compile-only by default; set `IRB_DRYRUN_EXECUTE_MODE=1` for safe
  execute-mode wrappers listed in `dryrun/manifest.json`.
