# Run Artifact Tree

This directory is reserved for the main run-artifact tree used by the
benchmark workflow.

The intended live layout is:

```text
artifacts/
  runs/
    models/
    tokenized/
    exp3/
```

Policy:

- this tree holds the real non-deprecated run artifacts
- active benchmark workflows should point here directly
- temporary migration symlinks may exist in local run storage, but active workflows should point here directly
- deprecated or stale artifacts do **not** belong here; they belong under
  `deprecated/`

The migration is driven by:

- `pipeline/scripts/build_run_artifact_tree.py`

Use that script in dry-run mode first, then run it without `--dry-run` only
after the final go/no-go review.
