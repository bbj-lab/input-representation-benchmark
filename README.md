# Input Representation Benchmark

**EHR Input Representation Benchmarking Framework**. Works with [fms-ehrs](https://github.com/bbj-lab/fms-ehrs) as a sister repository.

## Overview

This repository evaluates input representation methods for EHR foundation models across three experimental axes:

| Experiment | Focus | Configurations | Training Runs (single-seed dev) |
|------------|-------|----------------|---------------|
| **Exp 1** | Granularity & Semantic Anchoring | Decile, Ventile, Trentile, Centile × Fused tokens | 12 (12 × 1 seed) |
| **Exp 2** | Representation Mechanics | Discrete, Soft, xVal, xVal-affine × Time tokens/Time-Aware RoPE (**requires Exp1 winner binning**) | 8 (8 × 1 seed) |
| **Exp 3** | Vocabulary Semantics | **4-arm design**: MEDS-native, MEDS→CLIF mapped, randomized-mapping control, frequency-matched mapping control (Exp3 ICU-hospitalization cohort; see below) | 4 (4 × 1 seed) |
| **Total (when Exp3 controls enabled)** | | 24 configurations | **24 training runs** |

Notes:
- Exp3 requires additional **data preparation**: constructing the 3 derived arms (`meds_mapped`, `meds_randomized`, `meds_freqmatched`) from a fixed MEDS events directory. The transformation uses CLIF-MIMIC mapping CSVs, but does **not** require building CLIF parquet tables.

**Model**: **Scaled-down LLaMA 3.2** (config overrides: `hidden_size=1024`, `intermediate_size=2048`, `num_hidden_layers=8`, `num_attention_heads=8`; **≈87M parameters** for our vocab sizes; 128K context window in the base architecture).

### Current run status (live)

Snapshot time: **2026-02-26**

| Experiment | Stage 0 (tier2q: tokenize+outcomes) | Stage 1 (gpuq: train) | Stage 2 (gpuq: extract reps) | Stage 3 (tier2q: LR) |
|---|---|---|---|---|
| **Exp1 (granularity)** | **Complete** (12 configs) | **Complete** (12 configs; ctx=4096) | **Complete** (12 tasks) | **Complete** (12 tasks) |
| **Exp2 (rep mechanics)** | **Complete** (8 configs) | **Complete** (8 configs) | **Complete** (8 tasks) | **Complete** (8 tasks) |
| **Exp3 (vocab semantics; 4 MEDS-derived arms)** | **Complete** (4 arms) | **Complete** (4 arms) | **Complete** (4 tasks) | **Complete** (4 tasks) |

#### Exp1 results summary (context length 4096)

**Overall Exp1 winner**: deciles, population, fused (mortality AUROC 0.893, IMV AUROC 0.897).
**Best unfused**: deciles, population, unfused (mortality AUROC 0.861, IMV AUROC 0.885).

Exp2 settings derived from Exp1:
- **Binning**: deciles, population quantiles.
- **Tokenization (Exp2)**: all Exp2 value encoders use **unfused** tokenization (Discrete/Soft/xVal/xVal-affine) so the value-encoding comparison is architecture-matched; xVal/xVal-affine implement continuous scaling via a code+`[NUM]` slot.

#### Exp2 results summary

**Exp2 winner**: soft discretization + Time-Aware RoPE.

| Config | Mortality | Long LoS | ICU Adm | IMV |
|---|---|---|---|---|
| Discrete + time tokens | 0.862 | 0.740 | 0.774 | 0.893 |
| Discrete + Time-Aware RoPE | 0.861 | 0.744 | 0.770 | 0.894 |
| Soft + time tokens | 0.871 | 0.748 | 0.778 | 0.897 |
| **Soft + Time-Aware RoPE** | **0.869** | **0.749** | **0.780** | **0.898** |
| xVal + time tokens | 0.818 | 0.698 | 0.722 | 0.839 |
| xVal + Time-Aware RoPE | 0.828 | 0.698 | 0.726 | 0.837 |
| xVal-affine + time tokens | 0.841 | 0.717 | 0.740 | 0.849 |
| xVal-affine + Time-Aware RoPE | 0.837 | 0.710 | 0.738 | 0.842 |

Key findings:
- Soft discretization matches or exceeds discrete baselines on all outcomes (ΔAUROC ≤ 0.01)
- Canonical xVal substantially underperforms discrete/soft encodings (ΔAUROC 0.04–0.06 across outcomes)
- xVal-affine partially closes the xVal gap but remains below discrete/soft
- Time-Aware RoPE produces marginal, non-significant shifts vs. time tokens

#### Tokenization artifacts (`numeric_stats.json`; required for xVal)

Stage 0 writes per-code numeric summary statistics to `numeric_stats.json` inside each tokenized dataset (e.g., `.../<data_version>-tokenized/train/numeric_stats.json`). xVal (Exp2/Exp3) uses these statistics for quantizer/anchoring-independent scaling; if the file is missing for a given `data_version`, re-run Stage 0 for that configuration.

### Data directory conventions

- **`DATA_DIR`** (environment variable used by SLURM scripts): points to the **MEDS events directory**, i.e., the directory containing split subdirectories like `train/`, `tuning/`, `test/` and a `raw/` shim used by tokenization. On this repo it defaults to:
  - `benchmarks/mimic-meds-extraction/data/meds/data`
- Do **not** append `/data` to `DATA_DIR` in scripts; older versions of this repo did and it caused silent path ambiguity.

### Reference implementation alignment (lab standard)

We treat `/gpfs/data/bbj-lab/users/daniel/Quantifying-Surprise-EHRs` as the reference implementation for:

- **Model architecture**: scaled-down Llama config (1024 hidden, 2048 MLP, 8 layers, 8 heads; ≈87M params for our vocab sizes)
- **FM training + HPO**: Optuna HPO executed under DDP (`torchrun`), with small `n_trials` (3) as in the reference
- **Downstream evaluation**: representation-based prediction via `fms-ehrs`, including AUROC/AUPRC, calibration (Brier/ECE), bootstrap CIs, validation-tuned logistic regression regularization \(C\), and validation-selected decision thresholds for thresholded metrics

Important update (benchmark engineering choice):
- **Epoch budget / token-exposure fairness**: We default to **single-epoch training** for all three experiments:
  - Exp1: `IRB_EXP1_STAGE1_EPOCHS=1` (packed collation; implemented via a deterministic `max_steps` budget in `fms_ehrs/scripts/tune_model.py`)
  - Exp2–3: `IRB_EXP23_STAGE1_EPOCHS=1` (padded collation; deterministic 1-epoch pass over the expanded windowed dataset in `fms_ehrs/scripts/train_representation.py`)
  This enforces the fairness budget we care about: **each configuration trains through its entire one-epoch training dataset** (Exp2/3 do not use early stopping).
- **Windowed padded training (Exp2/3 Stage1)**: We train on full variable-length admission timelines by slicing each admission into fixed-length windows of length `IRB_MAX_SEQ_LENGTH` (default 4096). We use **non-overlapping windows** (no overlap between consecutive windows), and we prefix continuation windows with `TL_CONT`.
  - Stride control: `IRB_WINDOW_STRIDE` (default `IRB_MAX_SEQ_LENGTH`).
  - Extreme-length guardrail: `IRB_MAX_WINDOWS_PER_ADMISSION` (default 128). If an admission would emit more windows, we **subsample windows deterministically** (uniformly across the timeline, always including the first and last window) to bound compute for pathological outliers.
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

# Exp3 mapping uses CLIF-MIMIC mapping CSVs (labs/vitals categories).
# This is NOT a CLIF-parquet extraction dependency for the primary Exp3 arms.
git clone https://github.com/bbj-lab/CLIF-MIMIC.git ../CLIF-MIMIC

# MEDS extraction environment
bash scripts/setup_conda_env_meds_extract.sh

# Tokenization + training environment
bash scripts/setup_conda_env_input_rep.sh --with-training-deps
```

Dependency notes:
- **ETHOS-ARES is not required as a sibling repo**. We use its approach and cite it, but this repository contains the MEDS extraction adapter and uses the `MEDS_transforms` Python package during Phase 0.
- **`../fms-ehrs` is required** (tokenization + training + evaluation live there).
- **`../CLIF-MIMIC` is required for Exp3** because the primary mapped arm uses its mapping CSVs (`data/mappings/... labs.csv` and `... vitals.csv`).

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
sbatch --array=0-11 slurm/02_run_stage0_tier2q_tokenize.sh slurm/generated/demo/04_exp1_stage0_tokenize.jobfile
# Stage 1 (GPU): FM pretraining/HPO per config×seed (demo = 12 tasks)
# Recommended default: 1 epoch + 4 GPUs/job (packs 2 jobs/node on 8-GPU nodes).
TRAIN_JID=$(sbatch --parsable --export=ALL,IRB_EXP1_STAGE1_EPOCHS=1 --array=0-11 slurm/05_run_stage1_gpu4_train.sh slurm/generated/demo/07a_exp1_stage1_train.jobfile)
# Stage 2 (GPU; 1 GPU/job): extract representations AFTER training
EXTRACT_JID=$(sbatch --parsable --dependency=afterok:"${TRAIN_JID}" --array=0-11 slurm/09_run_stage2_gpu2_extract.sh slurm/generated/demo/07b_exp1_stage2_extract_reps.jobfile)
# Stage 3 (CPU tier2q; 8 CPUs/job): logistic regression on extracted reps AFTER extraction
sbatch --dependency=afterok:"${EXTRACT_JID}" --array=0-11 slurm/11_run_stage3_tier2q_lr.sh slurm/generated/demo/07c_exp1_stage3_lr.jobfile

# Exp2
# Stage 0 (tier2q CPU): tokenize + outcomes (deduplicated by data_version; 2 tasks)
sbatch --array=0-1 slurm/02_run_stage0_tier2q_tokenize.sh slurm/generated/demo/08_exp2_stage0_tokenize.jobfile
# Stage 1 (GPU): train (6 primary configs × 1 seed = 6 tasks)
TRAIN2_JID=$(sbatch --parsable --export=ALL,IRB_EXP23_STAGE1_EPOCHS=1 --array=0-5 slurm/05_run_stage1_gpu4_train.sh slurm/generated/demo/10a_exp2_stage1_train.jobfile)
# Stage 2 (1 GPU/job): extract reps
EXTRACT2_JID=$(sbatch --parsable --dependency=afterok:"${TRAIN2_JID}" --array=0-5 slurm/09_run_stage2_gpu2_extract.sh slurm/generated/demo/10b_exp2_stage2_extract_reps.jobfile)
# Stage 3 (tier2q; 8 CPUs/job): logistic regression
sbatch --dependency=afterok:"${EXTRACT2_JID}" --array=0-5 slurm/11_run_stage3_tier2q_lr.sh slurm/generated/demo/10c_exp2_stage3_lr.jobfile

# (Optional) Exp2 xVal-affine variants (2 tasks: time tokens, Time-Aware RoPE)
TRAIN2A_JID=$(sbatch --parsable --export=ALL,IRB_EXP23_STAGE1_EPOCHS=1 --array=0-1 slurm/05_run_stage1_gpu4_train.sh slurm/generated/demo/10a_exp2_stage1_train_xval_affine.jobfile)
EXTRACT2A_JID=$(sbatch --parsable --dependency=afterok:"${TRAIN2A_JID}" --array=0-1 slurm/09_run_stage2_gpu2_extract.sh slurm/generated/demo/10b_exp2_stage2_extract_reps_xval_affine.jobfile)
sbatch --dependency=afterok:"${EXTRACT2A_JID}" --array=0-1 slurm/11_run_stage3_tier2q_lr.sh slurm/generated/demo/10c_exp2_stage3_lr_xval_affine.jobfile

# Exp3 (vocabulary semantics; 4 MEDS-derived arms)
# Prereq: construct Exp3 cohort + arms (see below), then generate jobs.
# Stage 0 (CPU): tokenize + outcomes (4 tasks)
sbatch --array=0-3 slurm/02_run_stage0_tier2q_tokenize.sh slurm/generated/demo/12_exp3_stage0_tokenize.jobfile
# Stage 1 (GPU): train (4 configs × 1 seed = 4 tasks)
TRAIN3_JID=$(sbatch --parsable --export=ALL,IRB_EXP23_STAGE1_EPOCHS=1 --array=0-3 slurm/05_run_stage1_gpu4_train.sh slurm/generated/demo/14a_exp3_stage1_train.jobfile)
# Stage 2 (1 GPU/job): extract reps
EXTRACT3_JID=$(sbatch --parsable --dependency=afterok:"${TRAIN3_JID}" --array=0-3 slurm/09_run_stage2_gpu2_extract.sh slurm/generated/demo/14b_exp3_stage2_extract_reps.jobfile)
# Stage 3 (tier2q; 8 CPUs/job): logistic regression
sbatch --dependency=afterok:"${EXTRACT3_JID}" --array=0-3 slurm/11_run_stage3_tier2q_lr.sh slurm/generated/demo/14c_exp3_stage3_lr.jobfile
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
│   ├── 04_exp3_stage0_tokenize_and_outcomes.sh  # Exp3 Stage 0 (4 MEDS-derived arms): tokenize + outcomes
│   ├── 05_run_stage1_gpu4_train.sh        # Stage 1 runner (4 GPUs; packs 2 jobs/node on gpuq)
│   ├── 07_exp2_stage1_train_representation.sh   # Exp2 Stage 1 implementation
│   ├── 08_exp3_stage1_train_representation.sh   # Exp3 Stage 1 implementation
│   ├── 09_run_stage2_gpu2_extract.sh      # Stage 2 runner (1 GPU): extract reps
│   ├── 10_submit_stage2_3_after_train.sh  # Helper: submit Stage 2 -> 3 with afterok deps
│   ├── 11_run_stage3_tier2q_lr.sh         # Stage 3 runner (tier2q; 8 CPUs): LR
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

- **SLURM jobfiles are generated** (don’t hand-edit): `python run_experiments.py --mode demo` writes jobfiles under `slurm/generated/demo/`.
- **No UCMC transfer**: unlike the reference `Quantifying-Surprise-EHRs` transfer experiments, this benchmark evaluates within our MIMIC-derived cohorts; the same dataset is used for “orig” and “new” in LR evaluation.
- **Vendored reference scripts**: we vendor the reference SLURM launchers under `slurm/ref_qse/` with minimal path/resource adaptations; see `slurm/ref_qse/PROVENANCE.md`.
- **Utility CLIs**: see `scripts/INDEX.md` for what each helper script does (outcome extraction, cohort alignment, qc).
- **Pre-rerun cleanup**: use `scripts/cleanup_artifacts.sh` (dry-run by default). For Stage0 collisions due to stale `*-tokenized` symlinks under `${DATA_DIR}`, use `--delete-tokenized-symlinks`. To rerun Stage2/Stage3 without collisions, use `--delete-stage2-stage3-artifacts`.

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
                          train_representation.py  (torchrun; 4 GPUs)          (--classifier logistic_regression)
```

### Evaluation-time retokenization (Exp1 Stage0E; fused vs unfused fairness)

If Exp1 comparisons are intended to isolate **token semantics** (rather than *token budget / event coverage*),
then a fixed training-time `max_padded_len` can be unfair: **fused** tokenization may fit materially more
events into the same context window.

This repo supports an evaluation-only retokenization step that:
- **reuses the original training vocabulary** (`vocab.gzip`) so token IDs stay aligned with the pretrained model
- **retokenizes with a larger `max_padded_len`** to reduce truncation differences
- **does not require rerunning Exp1 Stage1** (only Stage0E + Stage2 + Stage3)
- **uses a fast-path when possible**: instead of re-parsing raw MEDS, Stage0E can derive eval artifacts directly
  from an existing `*_first_24h-tokenized/` dataset by rewriting parquet (drop `padded*`, truncate `tokens` to \(L\)
  with a trailing `TRUNC`, and keep aligned arrays like `times` / `numeric_values` aligned). This is much faster and
  avoids storing huge fixed-width padded arrays for evaluation.

Mechanics:
- `run_experiments.py --exp1_eval_max_padded_len <L>` generates an Exp1 Stage0E jobfile and points Exp1 Stage2/3
  at the resulting `*_evalL<L>_first_24h-tokenized/` dataset version.
- Choose \(L\) from the observed token-length distribution using `qc/eval_maxlen_analysis.py`. A practical first
  pass is **4096**, then bump to **8192** if unfused truncation remains material.

Empirical check (centiles, 24h; train split):
- Versions:
  - `centiles_none_unfused_time_tokens_first_24h-tokenized`
  - `centiles_none_fused_time_tokens_first_24h-tokenized`
- Results (from `outputs/eval_maxlen_centiles.json`):
  - At `L=2048`: truncation **3.53% (unfused)** vs **0.66% (fused)**
  - At `L=4096`: truncation **0.041% (unfused)** vs **0.0% (fused)**
  - At `L=8192`: truncation **0.0%** for both
- Recommendation (based on current length histograms): **`eval_max_padded_len=4096`** often makes fused vs unfused truncation
  essentially negligible for this cohort and tokenization pattern.
  - Note: these length statistics are effectively **quantizer-invariant** (deciles/centiles change token IDs, not
    the number of tokens emitted per event), so this conclusion transfers across Exp1 quantizers under the same
    fused/unfused + temporal-token settings.

**Limitation / compute note (train vs eval context)**:
- Exp1 Stage1 training uses `fms_ehrs/scripts/tune_model.py` with default `max_seq_length=4096` (or override via `IRB_MAX_SEQ_LENGTH`)
  overridden), for tractability under walltime and GPU-memory constraints.
- Increasing the training context from 1024→4096 would substantially increase compute and memory because
  transformer self-attention scales approximately as \(O(L^2)\). A 4× longer context implies \(\approx16\times\)
  attention FLOPs and \(\approx16\times\) attention activation memory, typically forcing smaller batch sizes and
  materially increasing GPU-hours. We therefore use longer context *only* for eval-time representation extraction
  (Stage2/Stage3), where it serves as a fairness control on truncation rather than a training-budget change.

### Clinical Prediction Targets

All outcomes are extracted from MEDS events and defined relative to the first 24h observation window.
Patients who experience the target event within 24h are excluded from that task to prevent temporal leakage.

#### Primary binary outcomes (`extract_outcomes_meds.py`)

| Outcome | Definition |
|---------|-----------|
| `same_admission_death` | In-hospital mortality during the admission |
| `long_length_of_stay` | Hospital LOS > 7 days |
| `icu_admission` | Any ICU admission during the stay (Exp 1–2) |
| `imv_event` | Invasive mechanical ventilation (itemids 224385, 225792) |
| `prolonged_icu_stay` | ICU LOS > 48h (used instead of icu_admission in Exp 3) |

#### Additional binary outcomes (`extract_extended_outcomes.py`)

| Outcome | Definition |
|---------|-----------|
| `hyperkalemia` | Peak potassium after 24h >= 6.5 mEq/L |
| `severe_hypokalemia` | Minimum potassium after 24h < 2.5 mEq/L |
| `severe_anemia` | Minimum hemoglobin after 24h < 7.0 g/dL |
| `hypoglycemia` | Minimum glucose after 24h < 54 mg/dL |
| `profound_hyponatremia` | Minimum sodium after 24h < 125 mEq/L |
| `severe_hypernatremia` | Maximum sodium after 24h >= 160 mEq/L |
| `tachycardia_hr130` | Maximum heart rate after 24h >= 130 bpm |
| `severe_hypertension` | Maximum SBP >= 180 mmHg or maximum DBP >= 120 mmHg after 24h |
| `vasopressor_initiation` | Any vasopressor infusion after 24h (norepinephrine, epinephrine, vasopressin, phenylephrine, dopamine, dobutamine) |
| `hypotension` | Any MAP < 65 mmHg or SBP < 90 mmHg after 24h |
| `crrt_initiation` | Any CRRT event after 24h |
| `hemodialysis_initiation` | Any hemodialysis event after 24h |

#### Regression outcomes (`extract_extended_outcomes.py`)

| Outcome | Definition |
|---------|-----------|
| `peak_creatinine` | Max creatinine after 24h (mg/dL) |
| `peak_troponin` | Max troponin T or I after 24h (ng/mL) |
| `min_hemoglobin` | Min hemoglobin after 24h (g/dL) |
| `peak_potassium` | Max potassium after 24h (mEq/L) |
| `min_potassium` | Min potassium after 24h (mEq/L) |
| `min_glucose` | Min glucose after 24h (mg/dL) |
| `min_sodium` | Min sodium after 24h (mEq/L) |
| `max_sodium` | Max sodium after 24h (mEq/L) |
| `peak_bnp` | Max BNP or NT-proBNP after 24h (pg/mL) |
| `max_heart_rate` | Max heart rate after 24h (bpm) |
| `max_sbp` | Max systolic blood pressure after 24h (mmHg) |
| `max_dbp` | Max diastolic blood pressure after 24h (mmHg) |

Regression outcomes are evaluated with Ridge regression (Spearman ρ).
The evaluation sample size varies per target due to post-24h measurement missingness.

### Experiment 1: Granularity
- **Tokenization**: Varies quantizer (deciles/ventiles/trentiles/centiles) and clinical anchoring
- **Training**: `tune_model.py` with packed collation
- **Evaluation**: primary binary outcomes (mortality, LOS, ICU admission, IMV), plus additional binary and regression outcomes via `13_extract_extended_outcomes.sh`

### Experiment 2: Representation Mechanics
- **Tokenization**: Uses Exp 1 winner's quantizer settings; for xVal, numeric events are tokenized as `(code_token, [NUM])` via `--numeric_encoding xval`
- **Training**: `train_representation.py` with padded collation (required for soft discretization, xVal, and Time-Aware RoPE). Soft discretization uses a two-point soft target at quantile-token positions; xVal adds an auxiliary numeric regression loss at `[NUM]` positions.
- **Configurations**: 3 representations × 2 temporal encodings = 6 primary configs (+ 2 xVal-affine variants)

### Experiment 3: Vocabulary Semantics
- **Comparison**: alternative identifier schemes applied to the *same MEDS event rows* (native vs. mapped vs. two null controls)
- **Training**: Uses Exp 2 winner's representation settings
- **Cohort (explicit; no shorthand)**:
  - **Hospitalization-level inclusion**: \(H_{\mathrm{ICU}}\) is the set of MIMIC-IV hospital admissions (`hadm_id`) with **hospital LOS ≥ 24h** (computed from `hosp/admissions.csv.gz` `admittime`/`dischtime`) and **≥ 1 linked ICU stay record** in `icu/icustays.csv.gz` for the same `hadm_id`.
  - **Splitting**: patients are split **at the patient level** by `subject_id` into 70/10/20 train/val/test using a fixed RNG seed; each split’s `hadm_id` list is then derived by taking all admissions for those patients and intersecting with \(H_{\mathrm{ICU}}\) (see `scripts/align_cohorts.py` outputs `*_hadm_ids.csv`).
  - **Full MEDS timeline**: All four arms train on the complete MEDS event timeline across all event families (labs, vitals, medications, infusions, transfers, procedures). The CLIF mapping selectively rewrites code identifiers for the four covered domains (LAB, VITAL, MEDICATION, INFUSION); all other codes are identical across arms.

### Exp3 data preparation (cohort + arms)

Exp3 requires one additional data-prep step (all in this repo; no CLIF-parquet arm):

- **1) Build split lists for \(H_{\mathrm{ICU}}\)**: run `scripts/align_cohorts.py` to emit `train_hadm_ids.csv`, `val_hadm_ids.csv`, `test_hadm_ids.csv` (patient-level split projected onto `hadm_id` and intersected with ICU-stay + LOS≥24h eligibility).
- **2) Filter MEDS events to \(H_{\mathrm{ICU}}\)**: run `scripts/split_meds_by_hadm_splits.py` to create an Exp3 MEDS events dir with `{train,val,test}/meds.parquet`.
- **3) Build the 3 derived arms from the same rows**: run `scripts/build_exp3_meds_semantics_arms.py` to create:
  - `meds_mapped` (MEDS→CLIF category identifiers, row-fixed),
  - `meds_randomized` (null: randomized mapping within domain),
  - `meds_freqmatched` (null: frequency-matched mapping within domain).

These directories are then passed to `run_experiments.py` via the `--exp3_meds_*_data_dir` flags.

### Data inclusion / leakage controls (merged from old methods docs)

- **Timestamp semantics**: whenever available, MEDS extraction uses **`storetime`** ordering to approximate information availability (see `benchmarks/mimic-meds-extraction/configs/event_configs_v3.1_full.yaml`).
- **Leakage-prone billing tables excluded**: we exclude `diagnoses_icd`, `procedures_icd`, `hcpcsevents`, `drgcodes` at extraction time.
- **Tokenization inclusion notes**:
  - **OMR excluded at tokenization** to avoid vocabulary inflation from heterogeneous names.
  - **`INFUSION_END` treated as categorical** (the numeric channel focuses on rates via `INFUSION_START`).

## Technical Implementation Details

### Optional performance knobs (objective-preserving)

We keep the training objective fixed (causal LM with the same loss definitions) but allow optional,
reproducible performance switches that change only numerical precision and attention kernel backend:

- **bf16 mixed precision**: set `IRB_USE_BF16=true` (default in `slurm/00_preamble.sh`).
- **Attention backend** (Transformers): set `IRB_ATTN_IMPL=sdpa` (default) or `IRB_ATTN_IMPL=flash_attention_2`.

Notes:
- `flash_attention_2` requires the optional `flash-attn` package and a compatible GPU/CUDA build.
- We do **not** use `accelerate launch`; we continue to use `torchrun` for distributed training.
- **Stage applicability**: `IRB_ATTN_IMPL` affects only **Stage1 (training)** and **Stage2 (representation extraction)** because those are the only stages that instantiate a transformer model. Stage0/Stage0E/Stage3 do not benefit from FlashAttention-2.

#### Installing `flash-attn` (optional; for `IRB_ATTN_IMPL=flash_attention_2`)

This environment currently has `torch==2.9.1+cu128` and `transformers==4.57.3`. Prebuilt `flash-attn` wheels
may not exist for every CUDA/PyTorch combination; if no wheel matches, installation requires compiling from source
on a node with CUDA build toolchain (e.g., `nvcc`).

Important cluster note:
- Some prebuilt `flash-attn` wheels distributed as `linux_x86_64` can require newer GLIBC than is available on
  RHEL8-like systems. If you see an error like `GLIBC_2.32 not found` when importing `flash_attn`, you must
  build from source on a GPU node with `nvcc`/`CUDA_HOME`.

Recommended workflow:
1. Use the official wheel finder: `https://flashattn.dev/install` (select Python/PyTorch/CUDA; copy the command).
2. Install into the active `input-rep` environment.
3. Verify:
   `python -c "import flash_attn; print('flash_attn ok')"`

Verified on our cluster (`input-rep` env):
- Built from source using `slurm/92_flashattn_install_and_preflight_gpuq.sh` (job `5647601`).
- Installed: `flash-attn==2.8.3`.
- Preflight (`scripts/preflight_perf_knobs.py`) confirmed:
  - `torch==2.9.1+cu128`, `transformers==4.57.3`, `accelerate==1.12.0`
  - `IRB_USE_BF16=true` (bf16 mixed precision) and `tf32=True` (Ampere matmul)
  - `IRB_ATTN_IMPL=flash_attention_2` and `flash_attn_present=True` on NVIDIA A100-PCIE-40GB.

### Hyperparameter Search Spaces (ML4H Paper Supplement)

This section provides complete technical specifications referenced in the ML4H 2025 paper to maintain implementation transparency while keeping the main text focused.

**Experiment 1 (Foundation Model Training)**:
- Training script: `fms_ehrs/scripts/tune_model.py`
- Hardware: 4 GPUs per job via `torchrun`
- HPO: Optuna with 3 trials
- Learning rate: $[5 \times 10^{-5}, 5 \times 10^{-4}]$ (log-uniform)
- Gradient accumulation: fixed to $2$ (removed from HPO for fairness)
- Collation: packed (iterable datasets)

**Experiments 2--3 (Representation Training)**:
- Training script: `fms_ehrs/scripts/train_representation.py`  
- Hardware: 2 GPUs per job (Exp2/3); 4 GPUs per job (Exp1)
- HPO: **none** (no Optuna / trial-based tuning in Exp2/3; hyperparameters are fixed)
- Batch size: 1 per device
- Gradient accumulation: fixed to $2$ (removed from HPO for fairness)
- Budget: **single epoch** (`IRB_EXP23_STAGE1_EPOCHS=1`) and **no early stopping** (Exp2/3 always consume the full 1-epoch training dataset)
- Collation: padded + **windowed padded** over full admission timelines (required for soft discretization, xVal, and Time-Aware RoPE)
- Windowing: window length \(L=\texttt{IRB\_MAX\_SEQ\_LENGTH}\) (default 4096), **non-overlapping windows** (\(\texttt{IRB\_WINDOW\_STRIDE}=L\)), and continuation marker `TL_CONT` on windows after the first.
- Extreme-length guardrail: cap windows per admission with `IRB_MAX_WINDOWS_PER_ADMISSION` (default 128); when exceeded, windows are subsampled deterministically (uniformly across the admission, always including first+last).

**Time-Aware RoPE Details**:
- Time-Aware RoPE adds **0 additional parameters**; it reuses existing RoPE weights and replaces token-index position_ids with continuous relative timestamps scaled by a frequency factor (default $s=60$).

**xVal Continuous Encoder Details**:
- Tokenization: numeric events emit `(code_token, [NUM])` and align the scalar to `[NUM]` positions (`--numeric_encoding xval`)
- Embedding: multiplicatively scale the `[NUM]` embedding by a per-code standardized scalar (train-split `numeric_stats.json`)
- Objective: token cross-entropy + auxiliary numeric regression (MSE) via a numeric head
- Parameter overhead: numeric head adds $H+1$ parameters (for $H=1024$, this is 1,025)

**Training Step Calculation**:
For Exp1 packed training (iterable datasets), we explicitly compute:
$$\texttt{max\_steps} = \frac{n_{\text{train}} \times n_{\text{epochs}}}{\texttt{per\_device\_train\_batch\_size} \times \texttt{num\_gpus} \times \texttt{gradient\_accumulation\_steps}}\,.$$

For Exp2/3 windowed padded training, we train for exactly **one epoch** over the expanded window dataset, i.e., each emitted window is consumed once (modulo the standard DDP sampler padding when \(n_{\text{train}}\) is not divisible by world size).

### Evaluation Pipeline Technical Details

**Stage2 (Representation Extraction)**:
- Script: `fms_ehrs/scripts/extract_hidden_states.py`
- Hardware: 1 GPU (torchrun `--nproc_per_node=1`)
- Output: `features-<model>.npy` per split

**Stage3 (Logistic Regression Evaluation)**:
- Script: `fms_ehrs/scripts/transfer_rep_based_preds.py`
- Hardware: 8 CPUs per job (tier2q)
- Regularization: $C$ tuned on validation set
- Threshold selection: Youden's $J$ (maximizes TPR - FPR)
- Bootstrap CIs: 1000 resamples with replacement

## Implementation appendix

### Clinically anchored quantization

Clinically anchored quantization partitions values relative to a reference interval \([L, U]\) and then performs quantile binning \emph{within each region} (below/within/above). The allocation scheme (e.g., 5-10-5 for ventiles) specifies the number of bins per region.

```python
def compute_anchored_breaks(values, ref_lower, ref_upper, allocation="5-10-5"):
    below = values[values < ref_lower]
    within = values[(values >= ref_lower) & (values <= ref_upper)]
    above = values[values > ref_upper]

    if allocation == "5-10-5":
        n_below, n_within, n_above = 5, 10, 5
    elif allocation == "10-10-10":
        n_below, n_within, n_above = 10, 10, 10
    else:
        raise ValueError(f"Unknown allocation: {allocation}")

    breaks = []
    breaks.extend(np.quantile(below, np.linspace(0, 1, n_below + 1)[1:-1]))
    breaks.append(ref_lower)
    breaks.extend(np.quantile(within, np.linspace(0, 1, n_within + 1)[1:-1]))
    breaks.append(ref_upper)
    breaks.extend(np.quantile(above, np.linspace(0, 1, n_above + 1)[1:-1]))
    return np.unique(breaks)
```

### Soft discretization

Soft discretization is implemented as a \emph{local} convex combination between the two neighboring bin embeddings based on within-bin position:

- Find \(i\) such that \(b_i \le v < b_{i+1}\)
- Compute \(\alpha = (v-b_i)/(b_{i+1}-b_i)\)
- Emit \((1-\alpha)\mathbf{E}[b_i] + \alpha\mathbf{E}[b_{i+1}]\)

The implementation is vectorized and uses per-code boundary tensors (indexed by integer code IDs) to avoid per-example branching.

### Tokenization validation snapshot (example run)

One Phase-1 validation run (Deciles, Unfused, Time Tokens) produced:
- Vocabulary size: 19,363
- Train/val/test timelines (\(\ge\)24h): 297,817 / 41,869 / 85,530
- Mean/median unpadded tokens: 2,041 / 296
- Peak RSS: \(\approx\) 58 GB
- Runtime (8 CPUs): \(\approx\) 41 min

### QC rules (tokenization/outcome extraction)

Deterministic QC rules applied before training/evaluation:
- **Schema validity**: drop rows missing `subject_id`, `time`, or `code`
- **Timeline integrity**: drop admissions with zero events after filtering; enforce chronological ordering (storetime where available)
- **Value sanity**: drop NaN/Inf numeric values; retain zeros
- **Reference ranges**: apply clinical anchoring only when both bounds exist; otherwise fall back to population quantiles

### Script inventory (single place to look)

Phase 0 (MEDS extraction):
- `benchmarks/mimic-meds-extraction/scripts/01_extract_meds_full.sh`
- `benchmarks/mimic-meds-extraction/configs/event_configs_v3.1_full.yaml`

Phase 1 (tokenize + outcomes):
- `slurm/03_stage0_tokenize_and_outcomes_meds.sh`
- `qc/eval_maxlen_analysis.py` (selecting evaluation truncation cap, e.g., 4096)

Phase 2 & 3 (train + evaluate):
- `slurm/04_exp1_stage1_tune_packed.sh` (Exp1 Stage1; packed FM training + small LR sweep)
- `slurm/07_exp2_stage1_train_representation.sh` (Exp2 Stage1; representation mechanics)
- `slurm/08_exp3_stage1_train_representation.sh` (Exp3 Stage1; vocabulary semantics arms)
- `slurm/05_run_stage1_gpu4_train.sh` (Stage1 runner; 4 GPUs/config)
- `slurm/09_run_stage2_gpu2_extract.sh` (Stage2 runner; 1 GPU)
- `slurm/11_run_stage3_tier2q_lr.sh` (Stage3 runner; CPU tier2q)
- `slurm/ref_qse/09_extract_reps.sh`
- `slurm/ref_qse/10_xfer_rep_based_preds.sh`

### Parameter overhead (representation modules)

Let hidden size \(H=1024\). Added parameter counts:

- Time-Aware RoPE: 0 (reuses existing RoPE weights; changes position_ids only)
- xVal: 1,025 (numeric head)
- Soft discretization: \(K \times H\) (learned bin-embedding table; e.g., ventiles \(K=20\Rightarrow 20{,}480\))

## Dependencies and Attribution

### Sister Repository: fms-ehrs

All tokenization, model training, and evaluation is performed through **fms-ehrs**, which provides:

- `fms_ehrs/framework/tokenizer.py`: Configurable tokenizer with time-spacing, 24h truncation
- `fms_ehrs/framework/dataset.py`: Dataset loader with numeric_values and relative_times support
- `fms_ehrs/framework/model_wrapper.py`: Wrapper for soft discretization and Time-Aware RoPE; routes xVal to `XValModelWrapper`
- `fms_ehrs/framework/soft_discretization.py`: Convex-combination encoder for numeric values
- `fms_ehrs/framework/xval.py`: xVal wrapper (`[NUM]` tokenization + multiplicative scaling + numeric head)

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

## Paper Audit Trail

This section maps every major verifiable claim in `paper.tex` to the exact file(s) in this
repository where the number can be confirmed. All values listed here were cross-checked against
actual outputs during the 2026-02-23 audit.

### Dataset statistics (Table: cohort\_summary)

| Claim | Value | Source |
|---|---|---|
| Patients | 364,627 | `physionet.org/files/mimiciv/3.1/hosp/patients.csv.gz` (row count) |
| Hospital admissions | 546,028 | `physionet.org/files/mimiciv/3.1/hosp/admissions.csv.gz` (row count) |
| ICU stays | 94,458 | `physionet.org/files/mimiciv/3.1/icu/icustays.csv.gz` (row count) |
| Lab events (MEDS-extracted) | 84,133,368 | SLURM 6158197: pyarrow scan of `benchmarks/mimic-meds-extraction/data/meds/shard_events/hosp/labevents/[0-158374764).parquet` with `hadm_id` AND `storetime` not-null filter (matching MEDS YAML `time: col(storetime)`) |
| Unique laboratory codes | 895 | Same run |
| Codes with reference ranges | 334 (37.3%) | Same run; either `ref_range_lower` or `ref_range_upper` not null |
| Events with reference ranges | 71,326,729 (84.8%) | Same run |

### Vocabulary sizes (§5.4 Computational Efficiency)

| Config | Vocab size | Source |
|---|---|---|
| Deciles, pop, unfused | 19,363 | `outputs/log_parsing/stage0_stats.md` |
| Deciles, pop, fused | 30,535 | `outputs/log_parsing/stage0_stats.md`; `outputs/diagnostics/sparsity_distance_results.json` → `Fused Deciles.vocab_size` |
| Centiles, pop, fused | 83,927 | `outputs/diagnostics/sparsity_distance_results.json` → `Fused Centiles.vocab_size` |

### Sequence length (§5.4)

| Claim | Status | Source |
|---|---|---|
| Fused reduces median length ~33% (198 vs 296 tokens, 24h) | ✅ | `artifacts/runs/tokenized/*/deciles_none_fused*` vs `*unfused*` parquets, `seq_len` median |
| RoPE saves ~11% vs time tokens (83 vs 93 tokens median, 24h) | ✅ | `artifacts/runs/tokenized/*/deciles_none_unfused_time_rope*` vs `*time_tokens*` parquets |

### Mechanistic analyses (§5.3)

| Claim | Value | Source |
|---|---|---|
| Centile probe accuracy | 93–98% across 7 analytes | `outputs/diagnostics/clinical_boundary_probe_results.json` → per-analyte `accuracy` |
| Decile/ventile probe accuracy | 70–90% | Same file |
| xVal norm ratio at layer 7 | 0.974–0.979 | `outputs/diagnostics/xval_zero_out_results.json` → `norm_ratio` at layer index 7 |
| % of [NUM] positions in trench (\|z\|<0.5) | 9.1% | Same file → `pct_near_zero` |
| CE after numeric (Discrete-RoPE) | 1.95 | `outputs/diagnostics/attention_washout_results.json` → `ce_after_numeric` for `Discrete-RoPE` |
| CE after numeric (Soft-RoPE) | 5.98 | Same file → `Soft-RoPE` entry |
| CE after categorical (Discrete-RoPE) | 7.54 | Same file → `ce_after_categorical` |
| Adjacent bin cosine similarity ~0.5 | ✅ | `outputs/diagnostics/embedding_geometry_results.json` → off-diagonal values in cosine similarity matrix |
| Fused Centiles frozen tokens | 10,866 (12.95%) | `outputs/diagnostics/sparsity_distance_results.json` → `Fused Centiles.n_frozen_tokens` / `pct_frozen` |

### XGBoost baseline (§5.2)

| Claim | Value | Source |
|---|---|---|
| Mortality AUROC | 0.742 [0.730, 0.753] | `outputs/xgboost_baseline/xgboost_baseline_results.json`; `slurm/logs/xgb_baseline_6039956.out` |
| Long LoS AUROC | 0.731 [0.723, 0.738] | Same |
| Feature count (37) | 37 | `outputs/xgboost_baseline/xgboost_baseline_results.json` → `n_features`; code: `scripts/xgboost_baseline.py` lines 112–119 (top-200) and 312–315 (set intersection) |

### CLIF mapping (§5.3 Exp3; Appendix)

| Claim | Status | Source |
|---|---|---|
| VITAL: 9 canonical categories, 26 mapped itemids | ✅ | `../CLIF-MIMIC/data/mappings/mimic-to-clif-mappings - vitals.csv`: 26 non-NO-MAPPING rows, 9 unique `vital_category` values |
| VITAL event coverage ~14% | From build script | `scripts/build_exp3_meds_semantics_arms.py` mapping stats |
| LAB: 51 CLIF target categories | ✅ | `../CLIF-MIMIC/data/mappings/mimic-to-clif-mappings - labs.csv`: 51 unique non-empty `lab_category` values loaded by `_load_lab_mapping` (all rows with non-null itemid/lab_category) |
| LAB coverage ~59% | From build script | Same |

### Exp3 test-set characteristics

| Stat | Value | Source |
|---|---|---|
| N admissions | 3,250 | `slurm/logs/prevalence_6039779.out` |
| Mortality prevalence (ICU cohort) | ~9.7% | Same |
| Long LoS prevalence (ICU cohort) | ~50.7% | Same |
| IMV prevalence (ICU cohort) | ~33.8% | Same |
| Prolonged ICU Stay prevalence | ~52.8% | Same |
| ICU Admission prevalence | 100% (all-positive; omitted from table) | Same |

> [!NOTE]
> To spot-check any value: `python3 -c "import json; d=json.load(open('outputs/diagnostics/<file>.json')); print(d)"`.
> Dataset row counts: `python3 -c "import gzip; print(sum(1 for _ in gzip.open('<file.csv.gz>','rb'))-1)"`.
