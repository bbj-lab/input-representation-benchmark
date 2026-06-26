# Layout

Start with [`../README.md`](../README.md) for the run path, output locations, and tests.

This file only records layout policy:

- `pipeline/`, `paper/`, `utilities/`: Python entrypoints called by `slurm/` launchers.
- `slurm/`: stable launcher path; `slurm/generated/` and `slurm/state/` are local
  job byproducts or resume markers, not source documentation.
- `artifacts/runs/`: active run outputs; see [`../artifacts/README.md`](../artifacts/README.md).
- `deprecated/`: retired scripts, launchers, and snapshots.

For stage-by-stage script names and hand-offs, use [`../PIPELINE.md`](../PIPELINE.md).
