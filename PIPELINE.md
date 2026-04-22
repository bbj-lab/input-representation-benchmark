# Benchmark pipeline

This file maps the current benchmark path from raw MEDS data to the paper
files. It only covers the current workflow.

## 1. Data preparation

### Raw MIMIC -> MEDS

Use the extraction wrapper under `benchmarks/mimic-meds-extraction/` to populate:

- `benchmarks/mimic-meds-extraction/data/meds/data/raw/train`
- `benchmarks/mimic-meds-extraction/data/meds/data/raw/val`
- `benchmarks/mimic-meds-extraction/data/meds/data/raw/test`

### Exp3 cohort and arms

Exp3 needs a fixed ICU-hospitalization cohort plus three derived comparison
arms.

Live scripts:

- `pipeline/scripts/align_cohorts.py`
- `pipeline/scripts/split_meds_by_hadm_splits.py`
- `pipeline/scripts/build_exp3_meds_semantics_arms.py`

Outputs are written under `artifacts/runs/exp3/`.

## 2. Stage 0: tokenization + base outcomes

Entry points:

- `slurm/03_stage0_tokenize_and_outcomes_meds.sh`
- `slurm/04_exp3_stage0_tokenize_and_outcomes.sh`
- `fms_ehrs/scripts/tokenize_w_config.py`
- `pipeline/scripts/extract_outcomes_meds.py`

Main outputs:

- `<data_version>-tokenized/train/vocab.gzip`
- `<data_version>-tokenized/train/numeric_stats.json`
- `<data_version>-tokenized/<split>/tokens_timelines.parquet`
- `<data_version>_first_24h-tokenized/<split>/tokens_timelines.parquet`
- `<data_version>_first_24h-tokenized/<split>/tokens_timelines_outcomes.parquet`

## 3. Extended outcomes

Entry points:

- `pipeline/scripts/extract_extended_outcomes.py`
- `slurm/13_refresh_all_extended_outcomes.sh`
- `slurm/13_refresh_exp3_extended_outcomes.sh`

Main output:

- `<data_version>_first_24h-tokenized/<split>/tokens_timelines_extended_outcomes.parquet`

## 4. Stage 1: training

Entry points:

- Exp1: `slurm/04_exp1_stage1_tune_packed.sh`
- Exp2: `slurm/07_exp2_stage1_train_representation.sh`
- Exp3: `slurm/08_exp3_stage1_train_representation.sh`
- Model repo scripts: `fms_ehrs/scripts/tune_model.py`, `fms_ehrs/scripts/train_representation.py`

Main outputs:

- checkpoints under `artifacts/runs/models/`
- `representation_mechanics.pt` for wrapper models

## 5. Stage 2: hidden-state extraction

Entry points:

- `slurm/ref_qse/09_extract_reps.sh`
- `slurm/09_run_stage2_gpu2_extract.sh`
- `fms_ehrs/scripts/extract_hidden_states.py`

Main output:

- `<data_version>_first_24h-tokenized/<split>/features-<model>.npy`

## 6. Stage 3: downstream probes

Entry points:

- `slurm/ref_qse/10_xfer_rep_based_preds.sh`
- `slurm/11_run_stage3_tier2q_lr.sh`
- `fms_ehrs/scripts/transfer_rep_based_preds.py`

Main output:

- `<data_version>_first_24h-tokenized/test/*-preds-*.pkl`

## 7. Aligned family statistics

Entry points:

- `pipeline/scripts/regenerate_aligned_family_stats.py`
- `slurm/15_run_stats_cpu_jobfile.sh`
- `slurm/15_submit_aligned_family_stats.sh`
- local jobfiles written under `slurm/generated/statistics/`

Primary stats root:

- `artifacts/runs/statistics/paper_stats_run_artifacts/`

## 8. Table and figure build

Entry points:

- `paper/scripts/generate_mlhc_appendix_tables.py`
- `paper/scripts/generate_mlhc_appendix_outcome_descriptives.py`
- `paper/scripts/generate_mlhc_paper_figures.py`
- manuscript file `../MLHC2026/MLHC/paper.tex`

Outputs:

- `../MLHC2026/MLHC/generated/`
- `../MLHC2026/MLHC/figures/`
- `../MLHC2026/MLHC/figures/sources/`

## 9. Order of operations

1. Build or verify MEDS extraction outputs.
2. Build the Exp3 ICU cohort and control arms if needed.
3. Run Exp1.
4. Materialize winner files.
5. Run Exp2.
6. Run Exp3.
7. Build extended outcomes if needed.
8. Regenerate aligned family stats.
9. Rebuild tables, figures, and manuscript assets.

## 10. Verification checks

- `pipeline/tests/unit/`: unit and contract checks for pipeline scripts.
- `pipeline/tests/dryrun/`: one dry-run wrapper per pipeline script.

If this file drifts from the stage scripts or `pipeline/run_experiments.py`, update the docs and leave the code path unchanged.
