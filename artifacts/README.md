# Run Artifact Tree

This directory is reserved for the canonical run-artifact tree used by the
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

- the canonical tree holds the real non-deprecated run artifacts
- active benchmark workflows should point here directly
- legacy paths in the repo are preserved as symlinks for compatibility
- deprecated or stale artifacts do **not** belong here; they belong under
  `deprecated/`

The migration is driven by:

- `scripts/build_canonical_artifact_tree.py`

Use that script in dry-run mode first, then run it without `--dry-run` only
after the final go/no-go review.
