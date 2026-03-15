# Deprecated Benchmark Artifacts

This directory holds scripts, logs, and result files that are no longer part of
the active benchmark workflow.

## What belongs here

- stale SLURM wrappers with placeholder checkpoint paths
- one-off helper scripts used for earlier exploratory analyses
- duplicate metrics dumps and verification text files
- logs from deprecated evaluation runs

## What does not belong here

- current stage scripts in `slurm/`
- active utility scripts in `scripts/`
- result files still referenced by the manuscript or current rerun workflow

## Current policy

- keep deprecated material here instead of deleting it outright
- do not use files in this directory as source of truth for the manuscript or
  for new reruns unless they are explicitly revalidated
- when replacing active workflow pieces, move clearly stale artifacts here under
  the existing `deprecated/` tree rather than creating new `legacy/` or similar
  directories
