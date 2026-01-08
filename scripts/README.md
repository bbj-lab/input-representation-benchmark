# `scripts/` (Utility CLI Tools)

This directory contains **small, single-purpose command-line tools** used by the benchmark.
They are intentionally kept separate from the SLURM orchestration scripts in `slurm/`.

## Environment setup (reproducibility)

- **`setup_conda_env_meds_extract.sh`**: create the `meds-extract` conda env (MEDS_transforms stack).
- **`setup_conda_env_input_rep.sh`**: create the `input-rep` conda env (fms-ehrs + benchmark deps; optional training deps).

## MEDS-specific (native MIMIC → MEDS arm)

- **`extract_outcomes_meds.py`**: compute outcomes directly from MEDS event timestamps (storetime semantics) and join onto tokenized timelines (writes `tokens_timelines_outcomes.parquet`).
- **`normalize_meds_tokenized_layout.py`**: normalize tokenized MEDS layout for downstream scripts (utility for interoperability).

## CLIF-specific (standardized vocabulary arm, Exp3)

- **`align_cohorts.py`**: derive ICU ≥24h cohort splits (patient IDs) for MEDS/CLIF parity in Experiment 3.
- **`split_clif_by_patient_splits.py`**: apply patient split lists to unsplit CLIF parquet tables to create `raw/{train,val,test}` directories.
- **`augment_clif_labs_with_ref_ranges.py`**: add `ref_range_lower`/`ref_range_upper` to CLIF lab tables (enables clinically anchored binning in Exp3).

## Validation / QC

- **`validate_cohort_parity.py`**: check MEDS vs CLIF cohort parity (counts, intersections, sanity checks).
- **`validate_imv_detection.py`**: cross-validate IMV label detection between MEDS timestamp-derived logic and CLIF token-derived logic.
- **`smoke_test_exp2.py`**: quick configuration-level checks for Exp2 representation mechanics pipeline.
- **`minimal_e2e_dryrun.py`**: minimal end-to-end dry run (useful when porting to a new cluster).

## Notes

- These scripts should remain **import-stable** (referenced by SLURM job scripts and documentation).
- If a script becomes obsolete, prefer **deprecation** (clear header comment + replacement pointer) over silent deletion.

