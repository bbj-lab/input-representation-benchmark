# Input Representation Benchmark

**EHR Input Representation Benchmarking Framework**. Works with [fms-ehrs](https://github.com/your-org/fms-ehrs) as a sister repository.

## Overview

This repository evaluates input representation methods for EHR foundation models across three experimental axes:

| Experiment | Focus | Configurations | Training Runs |
|------------|-------|----------------|---------------|
| **Exp 1** | Granularity & Semantic Anchoring | Decile, Ventile, Trentile, Centile × Fused tokens | 60 (12 × 5 seeds) |
| **Exp 2** | Representation Mechanics | Discrete, Soft, Continuous × Time tokens/Time2Vec | 30 (6 × 5 seeds) |
| **Exp 3** | Vocabulary Semantics | MIMIC-IV native vs. CLIF standardized | 10 (2 × 5 seeds) |
| **Total** | | 20 configurations | **100 training runs** |

**Model**: LLaMA 3.2 (67.3M parameters, 128K context window)

## Installation

```bash
# Clone both repos as siblings (required by SLURM scripts + default configs)
git clone https://github.com/bbj-lab/input-representation-benchmark.git
cd input-representation-benchmark
# IMPORTANT: this benchmark currently targets the `dev-input-rep` branch of fms-ehrs.
git clone --branch dev-input-rep --single-branch https://github.com/bbj-lab/fms-ehrs.git ../fms-ehrs

# MEDS extraction environment
bash scripts/setup_conda_env_meds_extract.sh

# Tokenization + training environment
bash scripts/setup_conda_env_input_rep.sh --with-training-deps
```

**Note**: The MIMIC-IV dataset (`physionet.org/`) is excluded from git. You must download it separately and place it under this repository.

## Quick Start

```bash
# 0. Environment Setup (see Installation above)

# 1. MEDS Extraction (converts MIMIC-IV to MEDS format)
cd benchmarks/mimic-meds-extraction
bash scripts/01_extract_meds_full.sh 3  # 3 = number of parallel workers

# 2. Generate job files for experiments
cd ../..  # Back to repo root
python run_experiments.py --mode demo --exp 1  # Exp 1 (single seed)
python run_experiments.py --mode demo --exp 2  # Exp 2 (single seed)
python run_experiments.py --mode demo --exp 3  # Exp 3 (single seed)

# 3. Run locally (for testing) or submit to SLURM
bash slurm/exp1_demo_jobs.sh                   # Local execution
# OR (with max 8 concurrent GPUs)
sbatch --array=0-11%8 slurm/run_from_jobfile.sh slurm/exp1_demo_jobs.sh  # SLURM
```

## Architecture

This benchmark uses a **two-repository architecture**:

### `input-representation-benchmark/` (this repo)
- **Data**: MEDS extraction pipeline, MIMIC-IV data
- **Configuration**: Experiment parameters (`configs/experiment.yaml`)
- **Orchestration**: Job generation (`run_experiments.py`)
- **MEDS-specific scripts**: Outcome extraction, layout normalization
- **Documentation**: Methods, proposal, paper

### `fms-ehrs/` (sister repo)
- **Tokenization**: `fms_ehrs/scripts/tokenize_w_config.py`
- **Pretraining (Exp 1)**: `fms_ehrs/scripts/tune_model.py`
- **Representation Training (Exp 2)**: `fms_ehrs/scripts/train_representation.py`
- **Classification Fine-tuning**: `fms_ehrs/scripts/fine_tune_classification.py`
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
│   ├── proposal.md                # Research proposal
│   ├── paper.md                   # Paper draft
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
│   ├── 00_extract_meds.sh         # Phase 0: MEDS extraction job
│   ├── 01_tokenize.sh             # Phase 1: Tokenization job
│   ├── preamble.sh                # Environment setup for jobs
│   └── run_from_jobfile.sh        # Array job runner for training
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

## Experiment Pipeline

Each experiment follows this pipeline (orchestrated by `run_experiments.py`):

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Tokenization  │ ──> │    Training     │ ──> │   Evaluation    │
│  (fms-ehrs)     │     │  (fms-ehrs)     │     │  (fms-ehrs)     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │                       │
        v                       v                       v
  tokenize_w_config.py    tune_model.py OR       extract_outcomes +
                          train_representation.py fine_tune_classification.py
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

The scaled-down LLaMA 3.2 configuration (67.3M parameters) is borrowed from:

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

The training scripts automatically use W&B when `WANDB_API_KEY` is set. If not set, runs proceed in offline mode (see `slurm/preamble.sh`).

## Technical Notes

### Ventile Quantization (Reference Range-Aware)

**Goal**: Incorporate clinical knowledge into tokenization by using reference ranges (`ref_range_lower`, `ref_range_upper`) to define bins (e.g., 5 bins below normal, 10 within, 5 above).

**Results (MIMIC-IV v3.1)**:
- **Dataset**: 364,627 patients, 142M lab events, 1,128 unique lab codes
- **Reference Ranges**: 392 codes (34.8%) have paired bounds, covering 114M events (80.3%)
- **Strict Filtering**: 155/203 candidate codes (76.4%) pass the strict 20-bin requirement
- **All passing codes achieve exactly 20 bins**: ✓
- **Glucose Example**: Verified 5 bins below [70], 10 within [70-105], 5 above [105]

### Time2Vec Temporal Encoding

Time2Vec uses **relative time** (hours since admission) rather than absolute timestamps, respecting MIMIC-IV's deidentification policy:

> "A single date shift was assigned to each subject_id. As a result, the data for a single patient are internally consistent... Conversely, distinct patients are not temporally comparable."

This preserves clinically meaningful temporal patterns while avoiding spurious cross-patient correlations.

### 4 Evaluation Outcomes

1. **same_admission_death**: In-hospital mortality
2. **long_length_of_stay**: Hospital stay > 7 days
3. **icu_admission**: ICU admission after first 24 hours
4. **imv_event**: Invasive mechanical ventilation after first 24 hours

All outcomes use `storetime` semantics to prevent look-ahead bias.

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
