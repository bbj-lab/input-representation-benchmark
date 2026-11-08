# Slurm launchers

Batch scripts for the Randi cluster at the University of Chicago. The launchers source [`00_preamble.sh`](00_preamble.sh), which activates the `input-rep` environment and sets the repository paths and the Hugging Face scratch cache. See [`../PIPELINE.md`](../PIPELINE.md) for the stage order.

## Launchers used for the paper

| Stage | Launcher | Worker |
| :-- | :-- | :-- |
| −1 · MEDS extraction | `01_phase0_extract_meds.sh` | `benchmarks/mimic-meds-extraction/scripts/01_extract_meds_full.sh` |
| −0.5 · Exp. 3 cohorts | `02_phase05_exp3_cohort_arms.sh` | — |
| 0 · Tokenization | `03_stage0_tokenize_and_outcomes_meds.sh`, `04_exp3_stage0_tokenize_and_outcomes.sh` | — |
| 0.5 · 24–48h outcomes | `21_submit_ml4h_24to48h_outcomes_cpu.sh` | `15_run_stats_cpu_jobfile.sh` |
| 1 · Training | `10_submit_ml4h_campaign_stage1.sh` | `06_run_stage1_gpu1_train.sh` |
| 2 · Extraction | `11_submit_ml4h_exp{1,2,3}_stage2_opportunistic.sh` | `09_run_stage2_gpu2_extract.sh` |
| 3 · Probes | `22_submit_ml4h_24to48h_probes.sh` | `11_run_stage3_tier2q_lr.sh` |
| Metric tables | `23_build_ml4h_metric_tables.sh` | — |
| Hospitalization bootstrap | `28_submit_ml4h_24to48h_bootstrap.sh` | `28_run_*`, `28_concat_*` |
| Admission-swap tests | `34_submit_ml4h_family_effect_bootstrap.sh` | `34_run_*`, `34_concat_*` |
| Patient-clustered bootstrap | `35_submit_ml4h_patient_clusters.sh` | `35_build_*`, `35_run_*`, `35_concat_*` |
| Tables and boundary probe figure | `33_rebuild_ml4h_review_tables_and_compile.sh` | — |
| Centile PCA figure | `31_rerun_ml4h_pca_probe.sh` | — |
| Length histograms | `34_rebuild_ml4h_length_histograms.sh` | — |
| Exp. 3 token counts | `31_count_exp3_train_tokens.sh` | — |

`ref_qse/` holds the single-task extraction and probe wrappers. `21_*` submits `22_*` after the outcome jobs complete, and `22_*` submits `23_*` after the probe jobs complete. The other scripts in this folder are earlier launchers, calibration runs, and debug helpers.
