# Input Representation Benchmark

**Systematic Evaluation of Input Representations for EHR Foundation Models**

This repository benchmarks input representation methods for transformer-based EHR foundation models, evaluating how discretization granularity, value encoding strategies, and vocabulary semantics affect clinical prediction performance.

## Quick Links

- **Documentation**: `methods/proposal.md` (detailed methodology), `methods/paper.md` (manuscript)
- **Sister Repository**: [fms-ehrs](../fms-ehrs) — tokenization, training, and evaluation framework
- **Data Pipeline**: `benchmarks/mimic-meds-extraction/` — MIMIC-IV to MEDS conversion

## Experiments Overview

| Experiment | Focus | Configurations | Training Runs |
|------------|-------|----------------|---------------|
| **Exp 1** | Granularity & Clinical Anchoring | 12 (decile/ventile/trentile/centile × fused/unfused) | 60 (× 5 seeds) |
| **Exp 2** | Representation Mechanics | 6 (discrete/soft/continuous × time_tokens/time2vec) | 30 (× 5 seeds) |
| **Exp 3** | Vocabulary Semantics | 2 (MEDS native vs CLIF standardized) | 10 (× 5 seeds) |
| **Total** | | **20 configurations** | **100 training runs** |

**Model**: LLaMA 3.2 (67.3M parameters, scaled-down configuration)

**Evaluation**: 4 clinical prediction tasks (in-hospital mortality, long LOS, ICU admission, IMV event)

## Project Structure

```
input-representation-benchmark/
├── README.md                      # This file
├── THIRD_PARTY_NOTICES.md         # Attribution for third-party code
├── pyproject.toml                 # Python dependencies
├── run_experiments.py             # Job generator (calls fms-ehrs scripts)
│
├── methods/
│   ├── proposal.md                # Research proposal & methodology
│   └── paper.md                   # Manuscript draft
│
├── benchmarks/
│   └── mimic-meds-extraction/     # MEDS data extraction pipeline
│       ├── configs/               # MEDS event configurations
│       ├── meds_pipeline/         # Extraction code (from ETHOS-ARES)
│       ├── scripts/               # Pipeline scripts
│       └── data/                  # Output (excluded from git)
│           ├── pre_meds/          # Pre-processed MIMIC tables
│           └── meds/              # MEDS-formatted events
│
├── scripts/                       # Utility scripts
│   ├── extract_outcomes_meds.py   # Outcome label extraction for MEDS
│   ├── normalize_meds_tokenized_layout.py  # MEDS → fms-ehrs layout adapter
│   ├── align_cohorts.py           # Cohort alignment for Exp 3
│   ├── validate_cohort_parity.py  # Data parity validation
│   ├── minimal_e2e_dryrun.py      # Local pipeline smoke test
│   └── smoke_test_exp2.py         # Exp2 representation smoke test
│
├── slurm/                         # SLURM job submission
│   ├── preamble.sh                # Common SLURM setup
│   └── run_from_jobfile.sh        # Array job runner
│
├── tests/                         # Unit tests
│
└── physionet.org/                 # Raw MIMIC-IV data (excluded from git)
```

## Installation

### Prerequisites

- Python 3.11+
- CUDA 11.8+ (for GPU training)
- MIMIC-IV v3.1 access (PhysioNet credentialed)

### Setup

```bash
# 1. Clone both repositories
git clone https://github.com/your-org/input-representation-benchmark
git clone https://github.com/your-org/fms-ehrs

# 2. Create environment
conda create -n ehr-benchmark python=3.12 -y
conda activate ehr-benchmark

# 3. Install fms-ehrs (sister repository)
cd fms-ehrs
pip install -e .

# 4. Install MEDS transforms (for data extraction)
pip install "MEDS_transforms[local_parallelism]"

# 5. Install benchmark dependencies
cd ../input-representation-benchmark
pip install -e .
```

### Data Setup

1. Download MIMIC-IV v3.1 from PhysioNet to `physionet.org/files/mimiciv/3.1/`
2. Run MEDS extraction (see Pipeline section below)

## Pipeline

### Step 1: MEDS Extraction

Convert MIMIC-IV to MEDS format with reference ranges for clinical anchoring:

```bash
cd benchmarks/mimic-meds-extraction
bash scripts/01_extract_meds_full.sh 3  # 3 = parallel workers
```

**Output**: `data/meds/data/{train,tuning,held_out}/*.parquet`

### Step 2: Generate Experiment Jobs

```bash
# Generate single-seed demo jobs for testing
python run_experiments.py --mode demo

# Generate full 5-seed jobs for statistical robustness
python run_experiments.py --mode full

# Generate for specific experiment
python run_experiments.py --mode demo --exp 1  # Exp 1 only
```

**Output**: `slurm/exp{1,2,3}_{demo,full}_jobs.sh`

### Step 3: Run Experiments

Each generated job file contains a complete pipeline:
1. **Tokenize** → `fms_ehrs/scripts/tokenize_w_config.py`
2. **Train** → `fms_ehrs/scripts/train_representation.py` (Exp 2) or `tune_model.py` (Exp 1)
3. **Extract Outcomes** → `scripts/extract_outcomes_meds.py`
4. **Fine-tune Classifier** → `fms_ehrs/scripts/fine_tune_classification.py`

```bash
# Local execution (for testing)
bash slurm/exp1_demo_jobs.sh

# SLURM submission
sbatch slurm/run_from_jobfile.sh slurm/exp1_demo_jobs.sh
```

### Step 4: Local Smoke Tests

```bash
# End-to-end pipeline test (tiny subset)
python scripts/minimal_e2e_dryrun.py --n_hadm 5

# Exp2 representation mechanics test
python scripts/smoke_test_exp2.py
```

## Key Implementation Details

### Experiment 1: Discretization Granularity

- **Quantizers**: Deciles (10), Ventiles (20), Trentiles (30), Centiles (100)
- **Clinical Anchoring**: 5-10-5 (ventiles), 10-10-10 (trentiles) using reference ranges
- **Token Layout**: Fused vs unfused `(code, quantile)` pairs

### Experiment 2: Representation Mechanics

- **Discrete**: Standard token embeddings (baseline)
- **Soft Discretization**: Convex combinations of adjacent bin embeddings (ConSE-inspired)
- **Continuous Encoder**: MLP projection of z-score normalized values (xVal-inspired)
- **Time Tokens**: Discrete temporal intervals (`T_5m-15m`, `T_1h-2h`, etc.)
- **Time2Vec**: Learned periodic + linear temporal encoding

**Note**: Soft and continuous encoders require **unfused tokenization** (`fused_category_values=false`).

### Experiment 3: Vocabulary Semantics

- **MEDS**: Native MIMIC-IV codes (e.g., `LAB//50912//CREATININE`)
- **CLIF**: Standardized clinical concepts (e.g., `creatinine`)
- **Cohort**: ICU patients with ≥24h stays (matched between formats)

### Temporal Encoding Policy

Time2Vec uses **relative time** (hours since admission) rather than absolute timestamps. This respects MIMIC-IV's deidentification policy: "A single date shift was assigned to each subject_id. As a result, the data for a single patient are internally consistent... Conversely, distinct patients are not temporally comparable."

### Outcome Extraction

Labels are computed from **storetime** semantics (when data entered EHR) to prevent look-ahead bias:
- `same_admission_death`: Discharge disposition = expired
- `long_length_of_stay`: Hospital stay > 7 days
- `icu_admission`: ICU transfer within hospitalization
- `imv_event`: Invasive mechanical ventilation during stay
- `icu_admission_24h`, `imv_event_24h`: Events within first 24 hours

## Dependencies

### Sister Repository: fms-ehrs

Provides tokenization framework, representation methods, and training scripts:
- `tokenizer.py`, `tokenizer_base.py`: Tokenization with quantization
- `soft_discretization.py`: Soft discretization encoder
- `continuous_encoder.py`: MLP-based continuous encoder
- `time2vec.py`: Learned temporal embeddings
- `model_wrapper.py`: Representation model wrapper
- `train_representation.py`: Unified training script

### MEDS Extraction: ETHOS-ARES

The MEDS extraction pipeline in `benchmarks/mimic-meds-extraction/meds_pipeline/` is adapted from ETHOS-ARES:

- **Repository**: https://github.com/ipolharvard/ethos-ares
- **License**: MIT (Copyright © 2024 Paweł Renc)
- **Citation**: Renc et al. (2025). GigaScience, 14, giaf107.

See `THIRD_PARTY_NOTICES.md` for complete attribution.

## Randi Cluster Deployment

### Post-SSH Setup

```bash
# 1. Activate environment
source ~/.bashrc
conda activate ehr-benchmark

# 2. Set environment variables
export MIMIC_RAW_DIR=/gpfs/data/bbj-lab/users/$USER/input-representation-benchmark/physionet.org/files/mimiciv/3.1
export MEDS_DATA_DIR=/gpfs/data/bbj-lab/users/$USER/input-representation-benchmark/benchmarks/mimic-meds-extraction/data/meds
export MODEL_DIR=/gpfs/data/bbj-lab/users/$USER/input-representation-benchmark/models

# 3. Verify fms-ehrs is accessible
python -c "from fms_ehrs.framework.tokenizer import Tokenizer21; print('fms-ehrs OK')"
```

### Running Experiments

```bash
# Generate job scripts
python run_experiments.py --mode demo --exp 1

# Submit to SLURM (24h limit per job)
sbatch --partition=gpuq --gres=gpu:1 --time=24:00:00 slurm/exp1_demo_jobs.sh

# Monitor
squeue -u $USER
tail -f slurm/output/*.out
```

## Citation

```bibtex
@article{lee2025input,
  title={Input Representation Benchmark for EHR Foundation Models},
  author={Lee, Daniel and others},
  journal={arXiv preprint},
  year={2025}
}
```

Please also cite the underlying frameworks:

```bibtex
@article{10.1093/gigascience/giaf107,
    author = {Renc, Pawel and others},
    title = {Foundation model of electronic medical records for adaptive risk estimation},
    journal = {GigaScience},
    volume = {14},
    pages = {giaf107},
    year = {2025},
    doi = {10.1093/gigascience/giaf107},
}

@article{burkhart2025quantifying,
    author = {Burkhart, Michael C. and others},
    title = {Quantifying surprise in clinical care},
    journal = {arXiv preprint arXiv:2507.22798},
    year = {2025}
}
```

## Contact

Inhyeok (Daniel) Lee (ihlee@uchicago.edu), University of Chicago
