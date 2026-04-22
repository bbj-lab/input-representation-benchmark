# Pipeline Tests

This directory is the main place to check pipeline scripts.

## Layout

- `unit/`: unit and contract checks for pipeline scripts.
- `dryrun/`: one dry-run wrapper per pipeline script.

## Dry-run usage

Compile-only mode (default):

```bash
bash pipeline/tests/dryrun/run_all.sh
```

Execute-mode (for scripts that support safe CLI dry-runs):

```bash
IRB_DRYRUN_EXECUTE_MODE=1 bash pipeline/tests/dryrun/run_all.sh
```
