# Script Compatibility Index

`scripts/` is a compatibility layer kept for old commands and queued jobs.

Main code locations are:

- `pipeline/scripts/` for run-path logic and diagnostics
- `paper/scripts/` for manuscript refresh scripts
- `utilities/scripts/` for active helper scripts

## Main pipeline scripts

- `pipeline/scripts/align_cohorts.py`
- `pipeline/scripts/build_run_artifact_tree.py`
- `pipeline/scripts/build_exp3_meds_semantics_arms.py`
- `pipeline/scripts/create_exp1_best_model_aliases.py`
- `pipeline/scripts/create_xval_rep_mechanics.py`
- `pipeline/scripts/extract_outcomes_meds.py`
- `pipeline/scripts/extract_extended_outcomes.py`
- `pipeline/scripts/minimal_e2e_dryrun.py`
- `pipeline/scripts/normalize_meds_tokenized_layout.py`
- `pipeline/scripts/regenerate_aligned_family_stats.py`
- `pipeline/scripts/split_meds_by_hadm_splits.py`
- `pipeline/scripts/write_reference_winner_files.py`
- `pipeline/scripts/diagnostics/*`

## Main paper scripts

- `paper/scripts/generate_mlhc_appendix_tables.py`
- `paper/scripts/generate_mlhc_paper_figures.py`

## Main utility scripts

- `utilities/scripts/analyze_meds_notation_ambiguity.py`
- `utilities/scripts/cleanup_artifacts.sh`
- `utilities/scripts/compute_prevalence.py`
- `utilities/scripts/download_literature.py`
- `utilities/scripts/generate_calibration_plot.py`
- `utilities/scripts/preflight_perf_knobs.py`
- `utilities/scripts/setup_conda_env_input_rep.sh`
- `utilities/scripts/setup_conda_env_meds_extract.sh`
- `utilities/scripts/verify_refrange_meds.py`
- `utilities/scripts/verify_refrange_meds_charttime.py`
- `utilities/scripts/verify_refrange_stats.py`
- `utilities/scripts/xgboost_baseline.py`

## Legacy archives

- deprecated CLIF/UCMC helpers: `deprecated/scripts/clif/`
- older ad-hoc scripts: `deprecated/scripts/legacy_misc/`
- archived paper-audit note: `deprecated/docs/paper_audit_trail.md`

If a script is not listed under `pipeline/`, `paper/`, or `utilities/`, treat it as legacy and check `deprecated/`.
