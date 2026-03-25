# Benchmark SLURM Layout

This directory holds the live benchmark runners.

SLURM scripts still call compatibility paths under `scripts/` and `run_experiments.py`.
Main Python implementations now live under `../pipeline/`, `../paper/`, and `../utilities/`.

## Numbered stage runners

- `00_preamble.sh`: shared environment setup
- `01_phase0_extract_meds.sh`: MEDS extraction wrapper
- `02_*`: Stage 0 array runners
- `03_*` and `04_*`: Stage 0 work scripts
- `04_*`, `07_*`, `08_*`: Stage 1 work scripts
- `09_*`: Stage 2 array runner
- `10_*`, `11_*`: Stage 3 submission and array runners
- `12_*`: gated or diagnostic helpers
- `13_*`: extended outcome refresh
- `15_*`: aligned stats regeneration

## Generated jobfiles

- `generated/statistics/`: pinned stats rerun sheets
- `generated/outcome_reruns/`: pinned outcome rerun sheets

Mode-specific or one-off jobfile trees may also appear under other subdirectories when generating isolated reruns.

## Vendored Stage 2/3 wrappers

`ref_qse/` contains the benchmark-maintained copies of the borrowed Stage 2 and Stage 3 wrappers used by this repo.

## Archived material

Retired runners and stale submission helpers belong in `../deprecated/slurm/`, not here.
