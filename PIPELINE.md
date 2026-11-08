# Benchmark Pipeline

This document traces the end-to-end execution path from raw MIMIC-IV v3.1 tables to the ML4H 2026 manuscript figures and tables.

## 1. Data Preparation

### Raw MIMIC-IV to MEDS (Stage -1)

Run the extraction wrapper in `benchmarks/mimic-meds-extraction/` (`slurm/01_extract_mimic_meds.sh`) to build the split MEDS directories.

- `benchmarks/mimic-meds-extraction/data/meds/data/raw/train`.
- `benchmarks/mimic-meds-extraction/data/meds/data/raw/val`.
- `benchmarks/mimic-meds-extraction/data/meds/data/raw/test`.

### Experiment 3 Matched Cohorts (Stage -0.5)

Experiment 3 compares native MIMIC-IV MEDS timelines against CLIF 2.1.0 timelines on a matched ICU cohort (`83,842` admissions across `64,954` patients).

- `pipeline/scripts/align_cohorts.py`.
- `pipeline/scripts/split_meds_by_hadm_splits.py`.
- `slurm/02_phase05_exp3_cohort_arms.sh`.
- `../fms-ehrs/fms_ehrs/scripts/collate_clif_with_cocoa.py`.

The launcher freezes the native patient-level split (`70 / 10 / 20`). It then calls the Cocoa adapter to build matched CLIF 2.1.0 MEDS tables under `outputs/runs/exp3/`.

## 2. Stage 0: Tokenization and Base Outcomes

Stage 0 entrypoints.

- `slurm/03_stage0_tokenize_and_outcomes_meds.sh`.
- `slurm/04_exp3_stage0_tokenize_and_outcomes.sh`.
- `../fms-ehrs/fms_ehrs/scripts/tokenize_w_config.py`.
- `pipeline/scripts/extract_outcomes_meds.py`.

Stage 0 outputs.

- `<data_version>-tokenized/train/vocab.gzip`.
- `<data_version>-tokenized/train/numeric_stats.json`.
- `<data_version>-tokenized/<split>/tokens_timelines.parquet`.
- `<data_version>_first_24h-tokenized/<split>/tokens_timelines.parquet`.
- `<data_version>_first_24h-tokenized/<split>/tokens_timelines_outcomes_24to48h.parquet`.

## 3. Stage 0.5: 24-to-48-Hour Clinical Outcomes

Stage 0.5 entrypoints.

- `pipeline/scripts/extract_extended_outcomes.py`.
- `utilities/scripts/generate_ml4h_24to48h_outcome_jobfiles.py`.
- `slurm/21_submit_ml4h_24to48h_outcomes_cpu.sh`.

Stage 0.5 output.

- `<data_version>_first_24h-tokenized/<split>/tokens_timelines_extended_outcomes_24to48h.parquet`.

## 4. Stage 1: Model Pretraining

Stage 1 entrypoints.

- `utilities/scripts/generate_ml4h_campaign.py`.
- `slurm/10_submit_ml4h_campaign_stage1.sh`.
- `../fms-ehrs/fms_ehrs/scripts/tune_model.py` (Experiment 1 packed training).
- `../fms-ehrs/fms_ehrs/scripts/train_representation.py` (Experiments 2 and 3 windowed training).

Stage 1 outputs.

- Checkpoints under `outputs/runs/ml4h_2026/models/`.
- `representation_mechanics.pt` for numeric and temporal wrapper models.

## 5. Stage 2: 24-Hour Hidden-State Extraction

Stage 2 entrypoints.

- `utilities/scripts/generate_ml4h_exp1_stage2_jobfiles.py`.
- `utilities/scripts/generate_ml4h_exp2_stage2_jobfiles.py`.
- `utilities/scripts/generate_ml4h_exp3_stage2_jobfiles.py`.
- `slurm/11_submit_ml4h_exp1_stage2_opportunistic.sh`.
- `slurm/11_submit_ml4h_exp2_stage2_opportunistic.sh`.
- `slurm/11_submit_ml4h_exp3_stage2_opportunistic.sh`.
- `../fms-ehrs/fms_ehrs/scripts/extract_hidden_states.py`.

Stage 2 output.

- `<data_version>_first_24h-tokenized/<split>/features-<model_stem>.npy`.

## 6. Stage 3: Locked Linear Probes

Stage 3 entrypoints.

- `utilities/scripts/generate_ml4h_24to48h_stage3_jobfiles.py`.
- `slurm/22_prepare_submit_ml4h_24to48h_probes.sh`.
- `slurm/22_submit_ml4h_24to48h_exp2_exp3_probes.sh`.
- `slurm/ref_qse/10_xfer_rep_based_preds.sh`.
- `../fms-ehrs/fms_ehrs/scripts/transfer_rep_based_preds.py`.

Stage 3 output.

- `<data_version>_first_24h-tokenized/test/*-preds-*.pkl`.

## 7. Statistical Inference and Bootstrap Evaluation

Statistical entrypoints.

- `pipeline/scripts/bootstrap_ml4h_24to48h_metrics.py` (`slurm/28_submit_ml4h_24to48h_bootstrap.sh`).
- `pipeline/scripts/bootstrap_ml4h_family_effects.py` (`slurm/34_submit_ml4h_family_effect_bootstrap.sh`).
- `pipeline/scripts/bootstrap_ml4h_patient_clusters.py` (`slurm/35_submit_ml4h_patient_clusters.sh`).
- `paper/scripts/ml4h_model_level_tests.py`.

Statistical outputs.

- `outputs/runs/ml4h_2026/metrics/metrics_long.parquet` and `metrics_long_with_ci.parquet`.
- `outputs/runs/ml4h_2026/family_effect_bootstrap/family_effect_intervals.parquet`.
- `outputs/runs/ml4h_2026/patient_cluster_bootstrap/patient_cluster_intervals.parquet`.
- `outputs/runs/ml4h_2026/model_level_tests/family_tests.csv` and `outcome_tests.csv`.

## 8. Manuscript Figures and Tables

Paper build entrypoints.

- `paper/scripts/generate_ml4h_outcome_family_traces.py`.
- `paper/scripts/generate_ml4h_outcome_factor_traces.py`.
- `paper/scripts/generate_ml4h_appendix_result_tables.py`.
- `paper/scripts/generate_ml4h_boundary_probe_plot.py`.
- `pipeline/scripts/diagnostics/rebuild_ml4h_pca_and_probe.py`.
- `slurm/33_rebuild_ml4h_review_tables_and_compile.sh`.

Paper build outputs.

- `../ML4H2026/generated/*.tex`.
- `../ML4H2026/figures/*.pdf`.

## 9. Verification

Run unit and dry-run CLI checks before submitting cluster jobs.

```bash
pytest pipeline/tests/unit
bash pipeline/tests/dryrun/run_all.sh
```
