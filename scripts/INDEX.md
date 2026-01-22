# `scripts/` (Utility CLI tools)

This directory contains small, single-purpose command-line utilities used by the benchmark.

## Environment setup

- `setup_conda_env_meds_extract.sh`: create the `meds-extract` conda env (MEDS_transforms stack)
- `setup_conda_env_input_rep.sh`: create the `input-rep` conda env (fms-ehrs + benchmark deps; optional training deps)

## MEDS-specific

- `extract_outcomes_meds.py`: compute outcomes directly from MEDS timestamps (storetime semantics) and join onto tokenized timelines
- `normalize_meds_tokenized_layout.py`: normalize tokenized MEDS layout for interoperability

## CLIF-specific (Exp3)

- `align_cohorts.py`: derive ICU ≥24h cohort splits (patient IDs) for MEDS/CLIF parity
- `split_clif_by_patient_splits.py`: apply patient split lists to CLIF parquet tables to create `raw/{train,val,test}`
- `augment_clif_labs_with_ref_ranges.py`: add `ref_range_lower`/`ref_range_upper` to CLIF lab tables (enables anchored binning)

## Validation / QC

- `validate_cohort_parity.py`: check MEDS vs CLIF cohort parity (counts, intersections)
- `validate_imv_detection.py`: cross-validate IMV label detection (MEDS timestamp vs CLIF token)
- `smoke_test_exp2.py`: quick sanity checks for Exp2 representation mechanics
- `minimal_e2e_dryrun.py`: minimal end-to-end dry run (useful when porting to a new cluster)

