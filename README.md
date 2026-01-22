# Input Representation Benchmark

**EHR Input Representation Benchmarking Framework**. Works with [fms-ehrs](https://github.com/your-org/fms-ehrs) as a sister repository.

## Overview

This repository evaluates input representation methods for EHR foundation models across three experimental axes:

| Experiment | Focus | Configurations | Training Runs (single-seed dev) |
|------------|-------|----------------|---------------|
| **Exp 1** | Granularity & Semantic Anchoring | Decile, Ventile, Trentile, Centile × Fused tokens | 12 (12 × 1 seed) |
| **Exp 2** | Representation Mechanics | Discrete, Soft, Continuous × Time tokens/Time2Vec (**requires Exp1 winner binning**) | 6 (6 × 1 seed) |
| **Exp 3** | Vocabulary Semantics | **4-arm design**: MEDS-native, CLIF, randomized-mapping control, frequency-matched collapse control (ICU-aligned cohort) | 4 (4 × 1 seed) |
| **Total (when Exp3 controls enabled)** | | 22 configurations | **22 training runs** |

Notes:
- Exp3 requires additional **data preparation** (CLIF build + semantic control-arm construction). Until those artifacts exist, Exp3 may be run as a 2-arm smoke test (MEDS vs CLIF) or skipped.

**Model**: **Scaled-down LLaMA 3.2** (config overrides: `hidden_size=1024`, `intermediate_size=2048`, `num_hidden_layers=8`, `num_attention_heads=8`; **≈87M parameters** for our vocab sizes; 128K context window in the base architecture).

### Current run status (live)

Snapshot time: **2026-01-15** (from SLURM at time of writing)

| Experiment | Stage 0 (tier2q: tokenize+outcomes) | Stage 1 (gpuq: train/HPO) | Stage 2 (gpuq: extract reps) | Stage 3 (tier2q: LR) |
|---|---|---|---|---|
| **Exp1 (granularity)** | **Running/queued**: `4924779` (12 tasks) | **Queued (afterok)**: `4924780` (12 tasks) | not submitted | not submitted |
| **Exp2 (rep mechanics)** | **Queued**: `4924781` (2 tasks; dedup by `data_version`) | **Queued (afterok)**: `4924782` (4 tasks) | not submitted | not submitted |
| **Exp3 (MEDS vs CLIF)** | **Pending**: CLIF build + cohort-parity prep | not submitted | not submitted | not submitted |

#### Tokenization artifact status (why Stage 0 is re-running)

We already have tokenized timelines + outcomes for the demo `data_version`s, but **`numeric_stats.json` is missing** for all of them. Since **continuous Exp2/Exp3** uses `numeric_stats.json` for quantizer/anchoring-independent scaling, Stage 0 is designed to re-run until it is present.

### Data directory conventions

- **`DATA_DIR`** (environment variable used by SLURM scripts): points to the **MEDS events directory**, i.e., the directory containing split subdirectories like `train/`, `tuning/`, `test/` and a `raw/` shim used by tokenization. On this repo it defaults to:
  - `benchmarks/mimic-meds-extraction/data/meds/data`
- Do **not** append `/data` to `DATA_DIR` in scripts; older versions of this repo did and it caused silent path ambiguity.

| data_version | tokenized timelines | 24h timelines + outcomes | numeric_stats.json |
|---|---:|---:|---:|
| `deciles_none_unfused_time_tokens` | yes | yes | **no (will regenerate)** |
| `deciles_none_fused_time_tokens` | yes | yes | **no (will regenerate)** |
| `ventiles_none_unfused_time_tokens` | yes | yes | **no (will regenerate)** |
| `ventiles_none_fused_time_tokens` | yes | yes | **no (will regenerate)** |
| `ventiles_5-10-5_unfused_time_tokens` | yes | yes | **no (will regenerate)** |
| `ventiles_5-10-5_fused_time_tokens` | yes | yes | **no (will regenerate)** |
| `ventiles_5-10-5_unfused_time2vec` | yes | yes | **no (will regenerate)** |
| `trentiles_none_unfused_time_tokens` | yes | yes | **no (will regenerate)** |
| `trentiles_none_fused_time_tokens` | yes | yes | **no (will regenerate)** |
| `trentiles_10-10-10_unfused_time_tokens` | yes | yes | **no (will regenerate)** |
| `trentiles_10-10-10_fused_time_tokens` | yes | yes | **no (will regenerate)** |
| `centiles_none_unfused_time_tokens` | yes | yes | **no (will regenerate)** |
| `centiles_none_fused_time_tokens` | yes | yes | **no (will regenerate)** |

### Reference implementation alignment (lab standard)

We treat `/gpfs/data/bbj-lab/users/daniel/Quantifying-Surprise-EHRs` as the reference implementation for:

- **Model architecture**: scaled-down Llama config (1024 hidden, 2048 MLP, 8 layers, 8 heads; ≈87M params for our vocab sizes)
- **FM training + HPO**: Optuna HPO executed under DDP (`torchrun`), with small `n_trials` (3) as in the reference
- **Downstream evaluation**: classification metrics logged by `fms-ehrs` (currently: AUROC, accuracy, balanced accuracy, precision, recall)

Important update (benchmark engineering choice):
- **Epoch budget**: We default to **single-epoch training** (`IRB_STAGE1_EPOCHS=1`) across Exp1–Exp3 to match modern pretraining practice (“more data, fewer epochs”) and to make 20–100+ run sweeps feasible under a 24h/job limit.
- **GPU packing**: On `gpuq` (8-GPU nodes), we often submit Stage 1 as **4 GPUs/job** so that **two jobs can share one node**, improving queue throughput. Use `slurm/05_run_stage1_gpu4_train.sh` for this mode.

Because our benchmark uses a larger cohort (MIMIC-HOSP + MIMIC-ICU) and more MEDS-derived columns, we follow
the reference DDP + iterable packed dataset pattern to keep per-trial walltime within the 24h constraint.

## Installation

```bash
# Clone both repos as siblings (required by SLURM scripts + default configs)
git clone https://github.com/bbj-lab/input-representation-benchmark.git
cd input-representation-benchmark
# IMPORTANT: this benchmark currently targets the `dev-input-rep` branch of fms-ehrs.
git clone --branch dev-input-rep --single-branch https://github.com/bbj-lab/fms-ehrs.git ../fms-ehrs

# Optional (Exp3 only): CLIF build depends on CLIF-MIMIC conversion rules.
# If you already have CLIF-2.1 parquet tables under data/clif/raw/{train,val,test}/,
# you do NOT need this repo.
git clone https://github.com/bbj-lab/CLIF-MIMIC.git ../CLIF-MIMIC

# MEDS extraction environment
bash scripts/setup_conda_env_meds_extract.sh

# Tokenization + training environment
bash scripts/setup_conda_env_input_rep.sh --with-training-deps
```

Dependency notes:
- **ETHOS-ARES is not required as a sibling repo**. We use its approach and cite it, but this repository contains the MEDS extraction adapter and uses the `MEDS_transforms` Python package during Phase 0.
- **`../fms-ehrs` is required** (tokenization + training + evaluation live there).
- **`../CLIF-MIMIC` is only required if you want to (re)build CLIF parquet tables for Exp3**.

**Note**: The MIMIC-IV dataset (`physionet.org/`) is excluded from git. You must download it separately and place it under this repository.
MIMIC-IV is a **single-site dataset** from Beth Israel Deaconess Medical Center (Boston, MA), which matters for external validity and motivates our explicit “vocabulary semantics” controls in Exp3.

## Quick Start

```bash
# 0. Environment Setup (see Installation above)

# 1. MEDS Extraction (converts MIMIC-IV to MEDS format)
cd benchmarks/mimic-meds-extraction
bash scripts/01_extract_meds_full.sh 3  # 3 = number of parallel workers

# 2. Generate job files for experiments
cd ../..  # Back to repo root
python run_experiments.py --mode demo  # Exp1–Exp3 (4-stage; jobfiles under slurm/generated/demo/)

# 3. Submit to SLURM (recommended: Stage0 tokenizes once; Stage1 trains per seed)
#
# Exp1
# Stage 0 (tier2q CPU): tokenize + outcomes ONCE per config (12 tasks)
sbatch --array=0-11%2 slurm/02_run_stage0_tier2q_tokenize.sh slurm/generated/demo/04_exp1_stage0_tokenize.jobfile
# Stage 1 (GPU): FM pretraining/HPO per config×seed (demo = 12 tasks)
# Recommended default: 1 epoch + 4 GPUs/job (packs 2 jobs/node on 8-GPU nodes).
TRAIN_JID=$(sbatch --parsable --export=ALL,IRB_STAGE1_EPOCHS=1 --array=0-11%2 slurm/05_run_stage1_gpu4_train.sh slurm/generated/demo/07a_exp1_stage1_train.jobfile)
# Optional: 8 GPUs/job (faster per job, but harder to schedule when the partition is busy)
# TRAIN_JID=$(sbatch --parsable --export=ALL,IRB_STAGE1_EPOCHS=1 --array=0-11%2 slurm/05_run_stage1_gpu8_train.sh slurm/generated/demo/07a_exp1_stage1_train.jobfile)
# Stage 2 (GPU; 2 GPUs/job): extract representations AFTER training
EXTRACT_JID=$(sbatch --parsable --dependency=afterok:"${TRAIN_JID}" --array=0-11%4 slurm/09_run_stage2_gpu2_extract.sh slurm/generated/demo/07b_exp1_stage2_extract_reps.jobfile)
# Stage 3 (CPU tier2q): logistic regression on extracted reps AFTER extraction
sbatch --dependency=afterok:"${EXTRACT_JID}" --array=0-11%8 slurm/11_run_stage3_tier2q_lr.sh slurm/generated/demo/07c_exp1_stage3_lr.jobfile

# Exp2
# Stage 0 (tier2q CPU): tokenize + outcomes (deduplicated by data_version; 2 tasks)
sbatch --array=0-1%2 slurm/02_run_stage0_tier2q_tokenize.sh slurm/generated/demo/08_exp2_stage0_tokenize.jobfile
# Stage 1 (GPU): train (6 configs × 1 seed = 6 tasks)
TRAIN2_JID=$(sbatch --parsable --export=ALL,IRB_STAGE1_EPOCHS=1 --array=0-5%2 slurm/05_run_stage1_gpu4_train.sh slurm/generated/demo/10a_exp2_stage1_train.jobfile)
# Stage 2 (2 GPUs/job): extract reps
EXTRACT2_JID=$(sbatch --parsable --dependency=afterok:"${TRAIN2_JID}" --array=0-5%4 slurm/09_run_stage2_gpu2_extract.sh slurm/generated/demo/10b_exp2_stage2_extract_reps.jobfile)
# Stage 3 (tier2q): logistic regression
sbatch --dependency=afterok:"${EXTRACT2_JID}" --array=0-5%8 slurm/11_run_stage3_tier2q_lr.sh slurm/generated/demo/10c_exp2_stage3_lr.jobfile

# Exp3 (requires CLIF preprocessing; see methods/manuscript.md)
# Stage 0 (CPU): tokenize + outcomes (2 tasks: MEDS vs CLIF)
sbatch --array=0-1%2 slurm/02_run_stage0_tier2q_tokenize.sh slurm/generated/demo/12_exp3_stage0_tokenize.jobfile
# Stage 1 (GPU): train (2 configs × 1 seed = 2 tasks)
TRAIN3_JID=$(sbatch --parsable --export=ALL,IRB_STAGE1_EPOCHS=1 --array=0-1%2 slurm/05_run_stage1_gpu4_train.sh slurm/generated/demo/14a_exp3_stage1_train.jobfile)
# Stage 2 (2 GPUs/job): extract reps
EXTRACT3_JID=$(sbatch --parsable --dependency=afterok:"${TRAIN3_JID}" --array=0-1%4 slurm/09_run_stage2_gpu2_extract.sh slurm/generated/demo/14b_exp3_stage2_extract_reps.jobfile)
# Stage 3 (tier2q): logistic regression
sbatch --dependency=afterok:"${EXTRACT3_JID}" --array=0-1%8 slurm/11_run_stage3_tier2q_lr.sh slurm/generated/demo/14c_exp3_stage3_lr.jobfile
```

## Architecture

This benchmark uses a **two-repository architecture**:

### `input-representation-benchmark/` (this repo)
- **Data**: MEDS extraction pipeline, MIMIC-IV data
- **Configuration**: Experiment parameters (`configs/experiment.yaml`)
- **Orchestration**: Job generation (`run_experiments.py`)
- **MEDS-specific scripts**: Outcome extraction, layout normalization
- **Documentation**: Methods and manuscript

### `fms-ehrs/` (sister repo)
- **Tokenization**: `fms_ehrs/scripts/tokenize_w_config.py`
- **Pretraining (Exp 1)**: `fms_ehrs/scripts/tune_model.py`
- **Representation Training (Exp 2)**: `fms_ehrs/scripts/train_representation.py`
- **Legacy (not used in benchmark)**: `fms_ehrs/scripts/fine_tune_classification.py`
- **Framework**: Vocabulary, dataset, model wrapper, encoders

## Project Structure

```
input-representation-benchmark/
├── README.md                      # This file
├── THIRD_PARTY_NOTICES.md         # Attribution for third-party code
├── pyproject.toml                 # Dependencies
├── run_experiments.py             # Job generation for all experiments
│
├── methods/
│   ├── manuscript.md              # Manuscript draft
│   ├── data-columns.md            # Data columns used in each experiment
│   └── overall-pipeline.mmd       # Pipeline diagram (Mermaid)
│
├── configs/
│   └── experiment.yaml            # Experiment parameter reference
│
├── scripts/                       # MEDS-specific utility scripts
│   ├── extract_outcomes_meds.py   # 4-outcome extraction from MEDS data
│   ├── normalize_meds_tokenized_layout.py  # Layout normalization for fms-ehrs
│   ├── align_cohorts.py           # Cohort alignment for Exp 3
│   ├── validate_cohort_parity.py  # Data parity validation
│   ├── validate_imv_detection.py  # IMV detection cross-validation
│   ├── minimal_e2e_dryrun.py      # Minimal end-to-end test
│   └── smoke_test_exp2.py         # Exp 2 configuration smoke test
│
├── slurm/                         # SLURM job submission
│   ├── 00_preamble.sh             # Environment setup for all jobs
│   ├── 01_phase0_extract_meds.sh  # Phase 0: MEDS extraction wrapper
│   ├── 02_run_stage0_tier2q_tokenize.sh  # Stage 0 runner (tier2q; CPU-only)
│   ├── 03_stage0_tokenize_and_outcomes_meds.sh  # Stage 0 (MEDS): tokenize + outcomes
│   ├── 04_exp3_stage0_tokenize_and_outcomes.sh  # Exp3 Stage 0 (MEDS/CLIF): tokenize + outcomes
│   ├── 05_run_stage1_gpu8_train.sh        # Stage 1 runner (8 GPUs; Exp1)
│   ├── 05_run_stage1_gpu4_train.sh        # Stage 1 runner (4 GPUs; packs 2 jobs/node on gpuq)
│   ├── 06_run_stage1_gpu1_train.sh        # Stage 1 runner (1 GPU; optional/debug)
│   ├── 07_exp2_stage1_train_representation.sh   # Exp2 Stage 1 implementation
│   ├── 08_exp3_stage1_train_representation.sh   # Exp3 Stage 1 implementation
│   ├── 09_run_stage2_gpu2_extract.sh      # Stage 2 runner (2 GPUs): extract reps
│   ├── 10_submit_stage2_3_after_train.sh  # Helper: submit Stage 2 -> 3 with afterok deps
│   ├── 11_run_stage3_tier2q_lr.sh         # Stage 3 runner (tier2q; CPU-only): LR
│   ├── 12_gate_submit_exp2_discrete_after_exp1_winner.sh  # Gate: submit Exp2 discrete-only after Exp1 winner
│   ├── 90_stage0_tokenize_only_debug.sh   # Manual tokenization helper (debug)
│   ├── generated/                 # Auto-generated jobfiles (gitignored)
│   ├── output/                    # SLURM stdout/stderr logs
│   └── ref_qse/                   # Vendored reference scripts (Option B evaluation)
│
├── tests/                         # Unit tests
│   ├── test_extract_outcomes_meds.py
│   └── test_validate_imv_detection.py
│
├── benchmarks/
│   └── mimic-meds-extraction/     # MEDS data pipeline
│       ├── configs/               # MEDS event configs
│       │   └── event_configs_v3.1_full.yaml  # With storetime semantics
│       ├── meds_pipeline/         # MEDS extraction code (from ETHOS-ARES)
│       ├── scripts/               # Pipeline scripts
│       └── data/                  # Output data (not in git)
│           ├── pre_meds/          # Pre-processed MIMIC data
│           └── meds/              # MEDS-formatted data
│
└── physionet.org/                 # MIMIC-IV data (not in git)
    └── files/mimiciv/3.1/
```

## Operational notes (single source of truth)

- **SLURM jobfiles are generated** (don’t hand-edit): `python run_experiments.py --mode {demo,full}` writes jobfiles under `slurm/generated/<mode>/`.
- **No UCMC transfer**: unlike the reference `Quantifying-Surprise-EHRs` transfer experiments, this benchmark evaluates within our MIMIC-derived cohorts; the same dataset is used for “orig” and “new” in LR evaluation.
- **Vendored reference scripts**: we vendor the reference SLURM launchers under `slurm/ref_qse/` with minimal path/resource adaptations; see `slurm/ref_qse/PROVENANCE.md`.
- **Utility CLIs**: see `scripts/INDEX.md` for what each helper script does (outcome extraction, cohort alignment, QC).

## Experiment Pipeline

Each experiment follows this pipeline (orchestrated by `run_experiments.py`; Option B evaluation):

```
┌─────────────────┐     ┌─────────────────┐     ┌──────────────────────┐     ┌─────────────────────┐
│   Tokenization  │ ──> │    Training     │ ──> │  Rep extraction (GPU) │ ──> │  LR eval (CPU tier2q)│
│  (fms-ehrs)     │     │  (fms-ehrs)     │     │  (fms-ehrs)           │     │  (fms-ehrs)          │
└─────────────────┘     └─────────────────┘     └──────────────────────┘     └─────────────────────┘
        │                       │                       │                           │
        v                       v                       v                           v
  tokenize_w_config.py    tune_model.py OR        extract_hidden_states.py     transfer_rep_based_preds.py
                          train_representation.py  (torchrun; 2 GPUs)          (--classifier logistic_regression)
```

### Experiment 1: Granularity
- **Tokenization**: Varies quantizer (deciles/ventiles/trentiles/centiles) and clinical anchoring
- **Training**: `tune_model.py` with packed collation
- **Evaluation**: 4 outcomes (mortality, LOS, ICU admission, IMV)

### Experiment 2: Representation Mechanics
- **Tokenization**: Uses Exp 1 winner's quantizer settings
- **Training**: `train_representation.py` with padded collation (required for soft/continuous encoders and Time2Vec)
- **Configurations**: 3 representations × 2 temporal encodings = 6 configs

### Experiment 3: Vocabulary Semantics
- **Comparison**: MEDS (native MIMIC) vs. CLIF (standardized) vocabularies
- **Training**: Uses Exp 2 winner's representation settings
- **Cohort**: Aligned ICU patients (≥24h stays) for data parity

## Dependencies and Attribution

### Sister Repository: fms-ehrs

All tokenization, model training, and evaluation is performed through **fms-ehrs**, which provides:

- `fms_ehrs/framework/tokenizer.py`: Configurable tokenizer with time-spacing, 24h truncation
- `fms_ehrs/framework/dataset.py`: Dataset loader with numeric_values and relative_times support
- `fms_ehrs/framework/model_wrapper.py`: Wrapper for soft/continuous encoders and Time2Vec
- `fms_ehrs/framework/soft_discretization.py`: ConSE-inspired convex combination encoder
- `fms_ehrs/framework/continuous_encoder.py`: xVal-inspired MLP encoder
- `fms_ehrs/framework/time2vec.py`: Learned temporal embeddings

**Setup**: Clone `fms-ehrs` as a sibling directory and install with `pip install -e ../fms-ehrs`.

### MEDS Extraction Pipeline: ETHOS-ARES

The MEDS extraction pipeline in `benchmarks/mimic-meds-extraction/meds_pipeline/`
is adapted from the **ETHOS-ARES** repository.

- **Repository**: https://github.com/ipolharvard/ethos-ares
- **License**: MIT (Copyright (c) 2024 Paweł Renc)
- **Citation**:

```bibtex
@article{10.1093/gigascience/giaf107,
    author = {Renc, Pawel and Grzeszczyk, Michal K and Oufattole, Nassim and Goode, Deirdre and Jia, Yugang and Bieganski, Szymon and McDermott, Matthew B A and Was, Jaroslaw and Samir, Anthony E and Cunningham, Jonathan W and Bates, David W and Sitek, Arkadiusz},
    title = {Foundation model of electronic medical records for adaptive risk estimation},
    journal = {GigaScience},
    volume = {14},
    pages = {giaf107},
    year = {2025},
    doi = {10.1093/gigascience/giaf107},
}
```

See `THIRD_PARTY_NOTICES.md` for complete attribution details.

### Model Architecture

The scaled-down LLaMA 3.2 configuration (≈87M parameters, depending on vocab size) is borrowed from:

- **Burkhart et al. (2025)**: "Quantifying surprise in clinical care" (arXiv:2507.22798)
  - Source of scaled-down configuration: hidden=1024, intermediate=2048, layers=8, heads=8
- **Grattafiori et al. (2024)**: "The Llama 3 Herd of Models" (arXiv:2407.21783)
  - Source of base LLaMA 3.2 architecture

## Weights & Biases Setup

Training runs are logged to Weights & Biases for real-time monitoring. Set up your API key:

```bash
# Option 1: Add to ~/.bashrc (persistent)
echo 'export WANDB_API_KEY="your-key-here"' >> ~/.bashrc
echo 'export WANDB_ENTITY="your-username"' >> ~/.bashrc
source ~/.bashrc

# Option 2: Interactive login
wandb login
```

Get your API key at: https://wandb.ai/authorize

The training scripts automatically use W&B when `WANDB_API_KEY` is set. If not set, runs proceed in offline mode (see `slurm/00_preamble.sh`).

## Contact

Daniel Lee (ihlee@uchicago.edu), University of Chicago

## Citation

If you use this benchmarking framework, please cite:

```bibtex
@article{
  TO-BE-ADDED
}
```

Please also cite the underlying frameworks:

```bibtex
% ETHOS-ARES (MEDS extraction pipeline)
@article{10.1093/gigascience/giaf107,
    author = {Renc, Pawel and others},
    title = {Foundation model of electronic medical records for adaptive risk estimation},
    journal = {GigaScience},
    volume = {14},
    pages = {giaf107},
    year = {2025},
    doi = {10.1093/gigascience/giaf107},
}

% Model architecture reference
@article{burkhart2025quantifying,
    author = {Burkhart, Michael C. and others},
    title = {Quantifying surprise in clinical care},
    journal = {arXiv preprint arXiv:2507.22798},
    year = {2025}
}
```
