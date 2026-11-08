# Input Representation Benchmark

Benchmark orchestration, statistical evaluation, and figure generation codebase for *Representation Before Training: A Practical Benchmark for Generative Medical Event Model Tokenization*.

Core tokenization, model training, hidden-state extraction, and linear probe entrypoints live in the companion repository [`../fms-ehrs`](../fms-ehrs).

## Setup

Clone [`../fms-ehrs`](../fms-ehrs) next to this repository and create the shared Conda environment.

```bash
conda env create -f environment.yml
conda activate input-rep
```

## Study Design

We train 156 causal decoder models on full MIMIC-IV v3.1 hospitalizations across two architectures (`Llama-3.2`-style and `Qwen3`-style), three random seeds (`42`, `43`, `44`), and three experiments.

- **Experiment 1 (Quantization and fusion, 72 runs):** Crosses six quantization schemes (`deciles`, `ventiles`, `5-10-5` reference-anchored ventiles, `trentiles`, `10-10-10` reference-anchored trentiles, and `centiles`) with `unfused` and `fused` code-value layouts (12 configurations).
- **Experiment 2 (Numeric and temporal encoding, 72 runs):** Crosses four numeric encodings (`discrete`, `soft`, `xval`, `xval_affine`) with three temporal encodings (`time_tokens`, `event_order`, `time_rope`) (12 configurations).
- **Experiment 3 (Native MIMIC versus CLIF schema, 12 runs):** Compares native MIMIC-IV MEDS against Common Longitudinal Intensive Care Unit Data Format (CLIF 2.1.0) on a matched ICU cohort (`deciles + unfused + discrete + time_tokens`, 2 configurations).

Each model encodes the first 24 hours of an intensive care unit (ICU) or hospital admission (`[0h, 24h]`). Locked L2-regularized logistic regression (`C = 0.01`) and Ridge regression (`alpha = 100.0`) probes predict outcomes during hours 24 to 48 (`(24h, 48h]`, 37 tasks in Experiments 1 and 2, and 36 tasks in Experiment 3) across eight outcome families (four binary families scored by AUROC and four regression families scored by Spearman correlation).

## Pipeline Stages

Stage-by-stage file contracts and execution order are documented in [`PIPELINE.md`](PIPELINE.md).

| Stage | Benchmark Entrypoint | `fms-ehrs` Entrypoint | Purpose |
| --- | --- | --- | --- |
| -1 | `benchmarks/mimic-meds-extraction/`, `slurm/01_*` | — | Extract MEDS tables from raw MIMIC-IV v3.1 |
| -0.5 | `pipeline/scripts/align_cohorts.py`, `slurm/02_*` | `collate_clif_with_cocoa.py` | Build matched native MIMIC-IV and CLIF 2.1.0 ICU cohorts |
| 0 | `pipeline/run_experiments.py`, `slurm/03_*`, `slurm/04_*` | `tokenize_w_config.py` | Tokenize timelines and extract base outcomes |
| 0.5 | `pipeline/scripts/extract_extended_outcomes.py`, `slurm/21_*` | — | Materialize 24-to-48-hour clinical outcomes |
| 1 | `slurm/10_submit_ml4h_campaign_stage1.sh` | `tune_model.py`, `train_representation.py` | Pretrain 156 causal medical event models |
| 2 | `slurm/11_submit_ml4h_exp*_stage2_opportunistic.sh` | `extract_hidden_states.py` | Extract 24-hour prefix hidden states |
| 3 | `slurm/22_*`, `slurm/26_*`, `slurm/ref_qse/` | `transfer_rep_based_preds.py` | Fit locked linear probes on frozen features |
| Stats | `pipeline/scripts/bootstrap_ml4h_*.py`, `slurm/28_*`, `slurm/34_*`, `slurm/35_*` | — | Run hierarchical admission-swap, patient-cluster, and model-level tests |
| Paper | `paper/scripts/generate_ml4h_*.py`, `slurm/33_*` | — | Generate manuscript tables and figures |

## Primary Result Locations

| Output | Path |
| --- | --- |
| Per-run outcome metrics (`23,040` rows) | `outputs/runs/ml4h_2026/metrics/metrics_long.parquet` |
| Patient-bootstrap 95% confidence intervals | `outputs/runs/ml4h_2026/metrics/metrics_long_with_ci.parquet` |
| Paired admission-swap family contrasts (`136` tests) | `outputs/runs/ml4h_2026/family_effect_bootstrap/family_effect_intervals.parquet` |
| Patient-cluster bootstrap sensitivity intervals | `outputs/runs/ml4h_2026/patient_cluster_bootstrap/patient_cluster_intervals.parquet` |
| Model-level paired t-tests (`df = 5`) | `outputs/runs/ml4h_2026/model_level_tests/family_tests.csv` |
| Outcome prevalence and coverage tables | `outputs/runs/ml4h_2026/metrics/outcome_coverage_summary.csv` |
| Pretrained model checkpoints (`156` runs) | `outputs/runs/ml4h_2026/models/` |
| Tokenized MEDS timelines (Experiments 1 & 2) | `outputs/runs/tokenized/mimiciv-3.1_meds_70-10-20/` |
| Experiment 3 matched cohorts and timelines | `outputs/runs/exp3/` |

See [`outputs/README.md`](outputs/README.md) and [`outputs/runs/ml4h_2026/metrics/README.md`](outputs/runs/ml4h_2026/metrics/README.md) for table schemas.

## Mechanistic Analysis Scripts

| Analysis | Script |
| --- | --- |
| Centile embedding PCA and leave-one-out boundary probe | `pipeline/scripts/diagnostics/rebuild_ml4h_pca_and_probe.py` |
| Clinical boundary probe standalone runner | `pipeline/scripts/diagnostics/diag_clinical_boundary_probe.py` |
| Embedding geometry analysis | `pipeline/scripts/diagnostics/diag_embedding_geometry.py` |
| Soft versus discrete contextual cross-entropy | `pipeline/scripts/diagnostics/diag_attention_washout.py` |
| xVal near-zero scaling analysis | `pipeline/scripts/diagnostics/diag_xval_zero_out.py` |

## Repository Layout

| Path | Contents |
| --- | --- |
| `configs/` | Experiment configuration (`configs/experiment.yaml`) |
| `pipeline/` | Experiment matrix, outcome extractors, bootstrap runners, and tests |
| `paper/` | Manuscript figure and table generators |
| `slurm/` | Slurm batch scripts for Stages -1 through paper compilation |
| `utilities/` | Campaign jobfile generators, token counters, and quality checks |
| `benchmarks/mimic-meds-extraction/` | MIMIC-IV to MEDS extraction submodule wrapper |
| `outputs/runs/` | Tokenized datasets, checkpoints, and evaluation tables |

## Testing

Run the unit tests and dry-run CLI contract suite from the repository root.

```bash
conda activate input-rep
pytest pipeline/tests/unit
bash pipeline/tests/dryrun/run_all.sh
```
