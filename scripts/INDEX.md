# `scripts/` (Utility CLI tools)

This directory contains small, single-purpose command-line utilities used by the benchmark.

## Environment setup

- `setup_conda_env_meds_extract.sh`: create the `meds-extract` conda env (MEDS_transforms stack)
- `setup_conda_env_input_rep.sh`: create the `input-rep` conda env (fms-ehrs + benchmark deps; optional training deps)

## MEDS-specific

- `extract_outcomes_meds.py`: compute outcomes directly from MEDS timestamps (storetime semantics) and join onto tokenized timelines
- `extract_extended_outcomes.py`: extract Tier 2 regression targets (peak/min lab values, time-to-ICU) and Tier 3 binary outcomes (hyperkalemia, anemia, hypoglycemia, vasopressor, hypotension) from MEDS events; joins onto existing outcomes parquet
- `normalize_meds_tokenized_layout.py`: normalize tokenized MEDS layout for interoperability

## Exp3 cohort + arms (MEDS-only)

- `align_cohorts.py`: derive ICU-hospitalization cohort \(H_{\mathrm{ICU}}\) split lists (patient-level split projected to admissions)
- `split_meds_by_hadm_splits.py`: filter MEDS events to \(H_{\mathrm{ICU}}\) and emit `{train,val,test}/meds.parquet`
- `build_exp3_meds_semantics_arms.py`: build the 3 derived arms (`meds_mapped`, `meds_randomized`, `meds_freqmatched`) by rewriting only identifiers (`code`) while holding rows/timestamps/values fixed

## Validation / QC

- `validate_imv_detection.py`: cross-validate IMV label detection (MEDS timestamp vs CLIF token)
- `smoke_test_exp2.py`: quick sanity checks for Exp2 representation mechanics
- `minimal_e2e_dryrun.py`: minimal end-to-end dry run (useful when porting to a new cluster)

## Optional CLIF-based validation (not required for the primary benchmark pipeline)

These scripts are kept for orthogonal QC when you have CLIF parquet tables available (e.g., from an external CLIF build):

- `extract_outcomes_clif.py`: compute outcomes from CLIF tables and join onto tokenized timelines
- `split_clif_by_patient_splits.py`: apply patient split lists to CLIF parquet tables to create `raw/{train,val,test}`
- `augment_clif_labs_with_ref_ranges.py`: add `ref_range_lower`/`ref_range_upper` to CLIF lab tables (enables anchored binning)
- `validate_cohort_parity.py`: check MEDS vs CLIF cohort parity (counts, intersections)

