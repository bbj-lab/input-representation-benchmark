# File inventory (benchmark reorg)

This inventory classifies the active code and file layout for the benchmark-first setup.

## Pipeline-Critical

These paths are part of the Stage0 -> Stage3 -> stats -> paper build chain.

- `pipeline/run_experiments.py`
- `slurm/00_preamble.sh`
- `slurm/01_phase0_extract_meds.sh`
- `slurm/02_run_stage0_tier2q_tokenize.sh`
- `slurm/02_run_stage0e_tier2q_tokenize.sh`
- `slurm/03_stage0_tokenize_and_outcomes_meds.sh`
- `slurm/04_exp1_stage1_tune_packed.sh`
- `slurm/04_exp3_stage0_tokenize_and_outcomes.sh`
- `slurm/05_run_stage1_gpu4_train.sh`
- `slurm/06_run_stage1_gpu1_train.sh`
- `slurm/07_exp2_stage1_train_representation.sh`
- `slurm/08_exp3_stage1_train_representation.sh`
- `slurm/09_run_stage2_gpu2_extract.sh`
- `slurm/11_run_stage3_tier2q_lr.sh`
- `slurm/12_gate_submit_exp2_discrete_after_exp1_winner.sh`
- `slurm/13_refresh_all_extended_outcomes.sh`
- `slurm/13_refresh_exp3_extended_outcomes.sh`
- `slurm/15_run_stats_cpu_jobfile.sh`
- `slurm/15_submit_aligned_family_stats.sh`
- `slurm/ref_qse/09_extract_reps.sh`
- `slurm/ref_qse/10_xfer_rep_based_preds.sh`
- `pipeline/scripts/align_cohorts.py`
- `pipeline/scripts/build_exp3_meds_semantics_arms.py`
- `pipeline/scripts/split_meds_by_hadm_splits.py`
- `pipeline/scripts/extract_outcomes_meds.py`
- `pipeline/scripts/extract_extended_outcomes.py`
- `pipeline/scripts/regenerate_aligned_family_stats.py`
- `pipeline/scripts/write_reference_winner_files.py`
- `paper/scripts/generate_mlhc_appendix_tables.py`
- `paper/scripts/generate_mlhc_appendix_outcome_descriptives.py`
- `paper/scripts/generate_mlhc_paper_figures.py`
- `pipeline/tests/unit/`
- `pipeline/tests/dryrun/`
- `outputs/`
- `benchmarks/mimic-meds-extraction/`

## Active manuscript files

These paths are the current manuscript files for the MLHC paper and its
supporting statistics.

- `../MLHC2026/MLHC/paper.tex`
- `../MLHC2026/MLHC/generated/`
- `../MLHC2026/MLHC/figures/`
- `paper/scripts/generate_mlhc_appendix_tables.py`
- `paper/scripts/generate_mlhc_appendix_outcome_descriptives.py`
- `paper/scripts/generate_mlhc_paper_figures.py`
- `paper/scripts/recompute_baseline_pairwise_view.py`
- `outputs/runs/statistics/paper_stats_combined/` (combined CSVs read by the paper generator scripts: `all_family_metrics.csv`, `all_family_pairwise_baseline.csv`)
- `outputs/runs/statistics/paper_stats_run_outputs/` (per-family Exp1/2/3 statistics produced by `slurm/16_prepare_submit_paper_rerun.sh`; source of the Exp1 rows in the combined dir)
- `outputs/runs/statistics/paper_stats_exp23_discrete_none_baseline/` (per-family Exp2/3 statistics under the `discrete_none` Exp2 baseline; source of the Exp2/3 rows in the combined dir)

## Active Utilities

These paths are active but not on the mandatory run chain.

- `pipeline/scripts/build_run_output_tree.py`
- `pipeline/scripts/create_exp1_best_model_aliases.py`
- `pipeline/scripts/create_xval_rep_mechanics.py`
- `pipeline/scripts/minimal_e2e_dryrun.py`
- `pipeline/scripts/normalize_meds_tokenized_layout.py`
- `pipeline/scripts/diagnostics/`
- `utilities/scripts/analyze_meds_notation_ambiguity.py`
- `utilities/scripts/cleanup_outputs.sh`
- `utilities/scripts/compute_prevalence.py`
- `utilities/scripts/download_literature.py`
- `utilities/scripts/generate_calibration_plot.py`
- `utilities/scripts/preflight_perf_settings.py`
- `utilities/scripts/setup_conda_env_input_rep.sh`
- `utilities/scripts/setup_conda_env_meds_extract.sh`
- `utilities/scripts/verify_refrange_meds.py`
- `utilities/scripts/verify_refrange_meds_charttime.py`
- `utilities/scripts/verify_refrange_stats.py`
- `utilities/qc/`
- `slurm/12_run_diag_gpu1.sh`
- `slurm/14_run_token_ce_gpuq.sh`
- `slurm/ref_qse/11_diag_eval.sh`
- `slurm/90_stage0_tokenize_only_debug.sh`
- `slurm/91_meds_notation_ambiguity_qc.sh`
- `slurm/92_flashattn_install_and_preflight_gpuq.sh`
- `slurm/compute_prevalence.sh`
- `slurm/gen_calibration_plot.sh`
- `slurm/generate_exp1_preds.sh`

## Deprecated Archive

These paths are outside the active benchmark path and are retained only for audit
or backtracking.

- `deprecated/slurm/strict_parity_exp23/generated/demo/`
- `deprecated/slurm/xgboost_baseline.sh`
- `deprecated/scripts/legacy_misc/check_icu.py`
- `deprecated/scripts/legacy_misc/extract_metrics.py`
- `deprecated/scripts/legacy_misc/inspect_data.py`
- `deprecated/scripts/xgboost_baseline.py`
- `deprecated/outputs/xgboost_baseline/`
- `deprecated/docs/paper_audit_trail.md`
- `deprecated/figures/`

## Freeze notes

- Local submit-time jobfiles under `slurm/generated/` are disposable and should not be treated as versioned files.
- Archived rerun-specific jobfiles that still matter for later review belong under `deprecated/`, not under the live `slurm/` tree.
