# Data Columns and Event Types in Input Representation Benchmark

## Overview

This document specifies exactly which MIMIC-IV tables and columns are used in each experiment. The primary focus of Experiments 1 and 2 is on **laboratory values** (`labevents`), while additional clinical events provide context for the patient timeline.

## Primary Focus: Laboratory Values (Experiments 1 & 2)

### Table: `hosp/labevents`

This is the **central table** for Experiments 1 and 2. Laboratory values are the primary target for:
- **Experiment 1**: Testing different quantization granularities (deciles, ventiles, trentiles, centiles) and clinical anchoring strategies
- **Experiment 2**: Testing different representation mechanics (discrete, soft, continuous)

| Column | Usage | Description |
|--------|-------|-------------|
| `itemid` | Code token | Laboratory test identifier (e.g., 50931 = Glucose) |
| `valuenum` | **Primary target** | Numeric measurement value (the focus of our representation experiments) |
| `valueuom` | Code token | Unit of measurement |
| `ref_range_lower` | Clinical anchoring | Lower bound of reference range (Exp 1) |
| `ref_range_upper` | Clinical anchoring | Upper bound of reference range (Exp 1) |
| `storetime` | Event ordering | When the result was stored in EHR (prevents look-ahead bias) |

### Why `labevents` is the Focus

1. **Continuous measurements**: Unlike diagnoses or medications, lab values have meaningful numeric values that benefit from different representation strategies
2. **Clinical reference ranges**: Lab tests have established normal/abnormal thresholds that enable clinically-anchored binning
3. **High frequency**: 142M events across 364K patients provides sufficient data for robust evaluation
4. **Clinical significance**: Lab values are critical for detecting deterioration, making representation quality clinically important

### Reference Range Coverage

From our MIMIC-IV v3.1 analysis:
- **34.8%** of lab codes (392/1,128) have paired reference ranges
- **80.3%** of lab events (114M/142M) have reference range annotations
- The top 200 codes by frequency cover **97.06%** of all lab events

## Supporting Event Types

These tables provide clinical context in the patient timeline but are **not the focus** of the representation experiments:

### Hospital Module (`mimic-hosp`)

| Table | Events | Usage |
|-------|--------|-------|
| `admissions` | HOSPITAL_ADMISSION, HOSPITAL_DISCHARGE, ED_REGISTRATION, ED_OUT | Timeline boundaries |
| `emar` | MEDICATION | Medication administration context |
| `omr` | Outpatient measurements | Additional clinical measurements |
| `patients` | GENDER, MEDS_BIRTH, MEDS_DEATH | Demographics, mortality outcomes |
| `transfers` | TRANSFER_TO | Care unit transitions |

### ICU Module (`mimic-icu`)

| Table | Events | Primary Columns | Usage |
|-------|--------|-----------------|-------|
| `icustays` | ICU_ADMISSION, ICU_DISCHARGE | intime, outtime | ICU outcome labels |
| `chartevents` | VITAL | `valuenum`, `storetime` | Vital signs (numeric measurements) |
| `inputevents` | INFUSION_START, INFUSION_END | `rate`, `amount`, `storetime` | IV medications (numeric values) |
| `outputevents` | FLUID_OUTPUT | `value`, `storetime` | Fluid balance (numeric values) |
| `procedureevents` | PROCEDURE, PROCEDURE_END | `value`, `storetime` | ICU procedures (IMV detection) |

### Excluded Tables (Data Leakage Risk)

These tables are **excluded** because billing codes are assigned after discharge:

| Table | Reason |
|-------|--------|
| `hosp/diagnoses_icd` | ICD codes assigned by coders after discharge |
| `hosp/procedures_icd` | ICD procedure codes (billed post-discharge) |
| `hosp/hcpcsevents` | CPT/HCPCS billing codes |
| `hosp/drgcodes` | DRG reimbursement codes |

## Experiment-Specific Data Usage

### Experiment 1: Granularity and Semantic Anchoring

**Target data**: `labevents.valuenum`

**Important implementation detail**: The **quantizer setting** (deciles/ventiles/trentiles/centiles) is applied uniformly to *all* MEDS event types with a scalar `numeric_value` (e.g., LAB, VITAL, FLUID_OUTPUT, INFUSION_START). However, **clinical anchoring** requires reference intervals and therefore applies only to laboratory events that carry `ref_range_lower`/`ref_range_upper`.

| Condition | Data Columns Used |
|-----------|-------------------|
| Deciles (10 bins) | `valuenum` → population quantiles |
| Ventiles (20 bins) | `valuenum` → population quantiles |
| Ventiles 5-10-5 | `valuenum` + `ref_range_lower` + `ref_range_upper` |
| Trentiles (30 bins) | `valuenum` → population quantiles |
| Trentiles 10-10-10 | `valuenum` + `ref_range_lower` + `ref_range_upper` |
| Centiles (100 bins) | `valuenum` → population quantiles |

**Fused tokens option**: All configurations can use fused tokens where a numeric event’s code and its discretized bin are combined into a single token (e.g., `LAB//50931//Q7` instead of separate `LAB//50931` and `Q7` tokens). This reduces sequence length for lab- and vitals-dense timelines.

### Experiment 2: Representation Mechanics

**Target data**: `labevents.valuenum` (and other numeric values from chartevents, inputevents, outputevents)

| Condition | Data Columns Used | Model Input |
|-----------|-------------------|-------------|
| Discrete | `valuenum` → quantile token | Categorical embedding lookup |
| Soft discretization | `valuenum` (raw) | Convex combination of adjacent bin embeddings |
| Continuous encoder | `valuenum` (raw) | MLP projection of z-score normalized value |
| Time tokens | `storetime` differences | Categorical time interval tokens |
| Time2Vec | `storetime` (relative hours) | Learned sinusoidal time embedding |

**Numeric value alignment**: For soft/continuous representations, the raw `valuenum` is preserved and aligned to token positions via the `numeric_values` array in the tokenized dataset.

### Experiment 3: Data Format / Vocabulary Semantics

**Target data**: Same as Exp 2 winner, but comparing vocabulary mappings

| Format | Vocabulary Source | Example Code |
|--------|-------------------|--------------|
| MEDS | Native MIMIC itemid | `LAB//50931` |
| CLIF | Standardized clinical concepts | `LAB//glucose_serum` |

**Clinical anchoring in CLIF (reference ranges)**:

The CLIF 2.1 labs schema includes `reference_unit` but does not provide lower/upper reference interval bounds by default. To enable clinically anchored binning symmetrically in Experiment 3 when Experiment 1 selects a clinically anchored winner, we augment CLIF lab tables (`clif_labs.parquet`) with `ref_range_lower` and `ref_range_upper` derived from MIMIC-IV `labevents` grouped by `(lab_loinc_code, reference_unit)`. See: `scripts/augment_clif_labs_with_ref_ranges.py`.

## Timestamp Semantics

All tables with `storetime` use storage time (when data entered EHR) rather than event occurrence time:

```
Event occurrence (charttime): 08:00 - Lab drawn
Data entry (storetime):       10:00 - Results stored in EHR
Model sees event at:          10:00 (storetime)
```

This prevents look-ahead bias by ensuring the model sees information in the same temporal order as clinicians.

## Data Flow Summary

```
MIMIC-IV Raw Data
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│ Phase 0: MEDS Extraction (01_extract_meds_full.sh)           │
│   - event_configs_v3.1_full.yaml defines column mappings     │
│   - Preserves ref_range_lower/upper for clinical anchoring   │
│   - Uses storetime for event ordering                        │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│ Phase 1: Tokenization (fms-ehrs/tokenize_w_config.py)        │
│   - Exp 1: labevents.valuenum → quantile bins (configurable) │
│   - Exp 2: Preserves numeric_values array for soft/continuous│
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│ Phase 2: Training                                            │
│   - Exp 1: tune_model.py (discrete tokens)                   │
│   - Exp 2: train_representation.py (soft/continuous options) │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│ Phase 3: Evaluation                                          │
│   - 4 outcomes: mortality, LOS, ICU admission, IMV           │
│   - Metrics: AUROC, AUPRC, Brier, ECE                        │
└──────────────────────────────────────────────────────────────┘
```

## Configuration Files

| File | Purpose |
|------|---------|
| `benchmarks/mimic-meds-extraction/configs/event_configs_v3.1_full.yaml` | MEDS extraction column mappings |
| `fms-ehrs/fms_ehrs/config/mimic-meds-ed.yaml` | MEDS tokenization settings |
| `fms-ehrs/fms_ehrs/config/clif-21.yaml` | CLIF tokenization settings (Exp 3) |
| `configs/experiment.yaml` | Experiment parameter reference |

## Execution Provenance (Phase 0 → Phase 1): Commands + Validated Readouts

This section records the **exact commands executed** and the **falsifiable readouts** that were checked for:

- Phase 0: MEDS extraction (`slurm/00_extract_meds.sh`)
- Phase 1: Tokenization (`slurm/01_tokenize.sh`)

The goal is reproducibility: a third-party researcher should be able to clone the repos, run these commands, and verify the same output layout and basic statistics.

### Phase 0 (MEDS extraction): `slurm/00_extract_meds.sh`

- **Model (pipeline)**: MIMIC-IV v3.1 raw tables → pre-MEDS parquet materialization → MEDS_transforms extraction stages → finalized MEDS cohort split into `train/`, `tuning/`, `test/`.
- **Hypothesis (minimal assumptions)**:
  - The raw data exists at `physionet.org/files/mimiciv/3.1/`.
  - The event config excludes leakage-prone billing tables (e.g., ICD/DRG) and uses `storetime` ordering.
  - `MEDS_transforms` can shard large ICU tables (e.g., `icu/chartevents`) without OOM when `row_chunksize` is reduced.
- **Falsifiable consequences (what must be true if Phase 0 succeeded)**:
  - Output directories exist:
    - `benchmarks/mimic-meds-extraction/data/meds/data/{train,tuning,test}/*.parquet`
    - `benchmarks/mimic-meds-extraction/data/meds/metadata/{dataset.json,subject_splits.parquet,codes.parquet}`
  - The pipeline explicitly logs `row_chunksize: 100000000` during `stage=shard_events` (memory mitigation).
  - Subject split counts are non-zero and recorded in the metadata finalization stage.

- **Experiment (commands executed)**:

```bash
# Step 0: activate meds-extract env (see scripts/setup_conda_env_meds_extract.sh)
cd /home/ihlee/Desktop/input-representation-benchmark
source /home/ihlee/Desktop/miniconda3/etc/profile.d/conda.sh
conda activate meds-extract

# Step 1: run full extraction wrapper (pre_MEDS.py + MEDS_transforms pipeline)
LOG_DIR="/home/ihlee/Desktop/input-representation-benchmark/benchmarks/mimic-meds-extraction/data/.logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/01_extract_meds_full_$(date +%Y%m%d_%H%M%S).log"
echo "Starting MEDS extraction (N_WORKERS=1). Logging to: $LOG_FILE"
bash /home/ihlee/Desktop/input-representation-benchmark/benchmarks/mimic-meds-extraction/scripts/01_extract_meds_full.sh 1 >"$LOG_FILE" 2>&1
echo "MEDS extraction finished. Log: $LOG_FILE"
```

**Observed failure mode (config override via Hydra CLI)**:

```bash
source /home/ihlee/Desktop/miniconda3/etc/profile.d/conda.sh
conda activate meds-extract
export BENCHMARK_DIR=/home/ihlee/Desktop/input-representation-benchmark/benchmarks/mimic-meds-extraction
export MIMICIV_PRE_MEDS_DIR="$BENCHMARK_DIR/data/pre_meds"
export MIMICIV_MEDS_COHORT_DIR="$BENCHMARK_DIR/data/meds"
export EVENT_CONVERSION_CONFIG_FP="$BENCHMARK_DIR/configs/event_configs_v3.1_full.yaml"
export PIPELINE_CONFIG_FP="$BENCHMARK_DIR/meds_pipeline/configs/extract_MIMIC.yaml"
export N_WORKERS=1
cd "$BENCHMARK_DIR/meds_pipeline"

LOG_DIR="$BENCHMARK_DIR/data/.logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/02_meds_transform_runner_resume_$(date +%Y%m%d_%H%M%S).log"
MEDS_transform-runner "pipeline_config_fp=$PIPELINE_CONFIG_FP" \
  stage_runner_fp=configs/local_parallelism_runner.yaml \
  stage_configs.shard_events.row_chunksize=100000000 >"$LOG_FILE" 2>&1
```

Result (from `data/.logs/02_meds_transform_runner_resume_20260106_220417.log`):
- `Could not override 'stage_configs.shard_events.row_chunksize'` and `Key 'stage_configs' is not in struct`.

**Resolution (hardcode `row_chunksize` in a local pipeline config and re-run)**:

```bash
source /home/ihlee/Desktop/miniconda3/etc/profile.d/conda.sh
conda activate meds-extract
export BENCHMARK_DIR=/home/ihlee/Desktop/input-representation-benchmark/benchmarks/mimic-meds-extraction
export MIMICIV_PRE_MEDS_DIR="$BENCHMARK_DIR/data/pre_meds"
export MIMICIV_MEDS_COHORT_DIR="$BENCHMARK_DIR/data/meds"
export EVENT_CONVERSION_CONFIG_FP="$BENCHMARK_DIR/configs/event_configs_v3.1_full.yaml"
export PIPELINE_CONFIG_FP="$BENCHMARK_DIR/meds_pipeline/configs/extract_MIMIC_local.yaml"  # row_chunksize: 100000000
export N_WORKERS=1
export HYDRA_FULL_ERROR=1
cd "$BENCHMARK_DIR/meds_pipeline"

LOG_DIR="$BENCHMARK_DIR/data/.logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/03_meds_transform_runner_local_chunks_$(date +%Y%m%d_%H%M%S).log"
MEDS_transform-runner "pipeline_config_fp=$PIPELINE_CONFIG_FP" \
  stage_runner_fp=configs/local_parallelism_runner.yaml >"$LOG_FILE" 2>&1
```

**Observed failure mode (stale cache/lock during merge)**:

```bash
cd /home/ihlee/Desktop/input-representation-benchmark
rm -rf benchmarks/mimic-meds-extraction/data/meds/merge_to_MEDS_cohort/train/.1.parquet_cache

source /home/ihlee/Desktop/miniconda3/etc/profile.d/conda.sh
conda activate meds-extract
export BENCHMARK_DIR=/home/ihlee/Desktop/input-representation-benchmark/benchmarks/mimic-meds-extraction
export MIMICIV_PRE_MEDS_DIR="$BENCHMARK_DIR/data/pre_meds"
export MIMICIV_MEDS_COHORT_DIR="$BENCHMARK_DIR/data/meds"
export EVENT_CONVERSION_CONFIG_FP="$BENCHMARK_DIR/configs/event_configs_v3.1_full.yaml"
export PIPELINE_CONFIG_FP="$BENCHMARK_DIR/meds_pipeline/configs/extract_MIMIC_local.yaml"
export N_WORKERS=1
export HYDRA_FULL_ERROR=1
export POLARS_MAX_THREADS=1
export RAYON_NUM_THREADS=1
cd "$BENCHMARK_DIR/meds_pipeline"

LOG_DIR="$BENCHMARK_DIR/data/.logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/04_meds_transform_runner_resume_after_lock_$(date +%Y%m%d_%H%M%S).log"
MEDS_transform-runner "pipeline_config_fp=$PIPELINE_CONFIG_FP" \
  stage_runner_fp=configs/local_parallelism_runner.yaml >"$LOG_FILE" 2>&1
```

- **Validation (readouts confirmed)**:
  - **Memory mitigation applied**: `row_chunksize: 100000000` present in `data/.logs/03_meds_transform_runner_local_chunks_20260106_221144.log`.
  - **Shard stage runtime**: `Sub-sharding completed in 0:33:17.857889` (same log).
  - **Final split sizes (subjects)**: from `data/.logs/04_meds_transform_runner_resume_after_lock_20260106_231923.log`:
    - train: `255239` subjects
    - tuning: `36463` subjects
    - test: `72925` subjects
  - **Tokenization shim created automatically** (by `slurm/00_extract_meds.sh` post-step):
    - `.../data/raw/train -> ../train`
    - `.../data/raw/val   -> ../tuning` (or `../val` if present)
    - `.../data/raw/test  -> ../test`

### Phase 1 (tokenization): `slurm/01_tokenize.sh`

- **Model (pipeline)**: MEDS split parquet shards → `Tokenizer21` (fms-ehrs) → tokenized timelines written as Parquet + vocabulary saved.
- **Hypothesis (minimal assumptions)**:
  - MEDS splits exist at `${DATA_DIR}/data/{train,tuning,test}`.
  - A `raw/` shim exists at `${DATA_DIR}/data/raw/{train,val,test}` (created automatically by `slurm/01_tokenize.sh` if missing).
  - The tokenization config `fms-ehrs/fms_ehrs/config/mimic-meds-ed.yaml` matches the MEDS schema produced by our extraction.
- **Falsifiable consequences (what must be true if Phase 1 succeeded)**:
  - Tokenized artifacts exist:
    - `${DATA_DIR}/data/deciles_none_unfused-tokenized/{train,val,test}/tokens_timelines.parquet`
    - `${DATA_DIR}/data/deciles_none_unfused_first_24h-tokenized/{train,val,test}/tokens_timelines.parquet`
    - `vocab.gzip` exists at least for `train/` (and for `*_first_24h/train/`).
  - Non-zero timeline counts are logged per split.
  - Runtime can be computed from the log’s start/end timestamps.

- **Experiment (commands executed, local run)**:

```bash
# Step 0: activate input-rep env (see scripts/setup_conda_env_input_rep.sh)
source /home/ihlee/Desktop/miniconda3/etc/profile.d/conda.sh
conda activate input-rep

# Step 1: run tokenization with logging
cd /home/ihlee/Desktop/input-representation-benchmark
mkdir -p slurm/output
LOG_FILE="slurm/output/local_tokenize_$(date +%Y%m%d_%H%M%S).log"
(bash slurm/01_tokenize.sh >"${LOG_FILE}" 2>&1 & echo $! > slurm/output/local_tokenize.pid)
echo "Log: ${LOG_FILE}"
```

- **Validation (readouts confirmed)**:
  - **Log file**: `slurm/output/local_tokenize_20260107_032155.log`
  - **Runtime (wall-clock)**:
    - start: `[2026-01-07T03:21:56-0600] running ...tokenize_w_config.py`
    - end:   `[2026-01-07T04:03:24-0600] ---fin`
    - \(\Delta t \approx 41\,\mathrm{min}\,28\,\mathrm{s}\)
  - **Timelines generated** (from the same log):
    - train: `297817` (24h-cut: `296209`)
    - val:   `41869`  (24h-cut: `41652`)
    - test:  `85530`  (24h-cut: `85057`)
  - **Output files created (sizes as observed)**:
    - `deciles_none_unfused-tokenized/train/tokens_timelines.parquet` (~`1.4G`) + `vocab.gzip` (~`336K`)
    - `deciles_none_unfused-tokenized/val/tokens_timelines.parquet` (~`183M`)
    - `deciles_none_unfused-tokenized/test/tokens_timelines.parquet` (~`377M`)
    - `deciles_none_unfused_first_24h-tokenized/train/tokens_timelines.parquet` (~`324M`) + `vocab.gzip` (~`336K`)
    - `deciles_none_unfused_first_24h-tokenized/val/tokens_timelines.parquet` (~`45M`)
    - `deciles_none_unfused_first_24h-tokenized/test/tokens_timelines.parquet` (~`91M`)

#### Phase 1 sanity checks (schema-level, falsifiable)

We also verified directly from MEDS Parquet shards (train split):

- **Admission boundary events exist** (required by `mimic-meds-ed.yaml` reference frame):
  - `HOSPITAL_ADMISSION` rows: `381811`
  - `HOSPITAL_DISCHARGE` rows: `381811`
- **Leakage-prone ICD diagnosis tables are excluded**, consistent with the benchmark design:
  - `DIAGNOSIS*` code rows in MEDS: `0` (by event config exclusion)

These checks ensure that:
- the reference cohort is non-empty, and
- the absence of ICD diagnosis codes is intentional (leakage prevention), not a pipeline failure.
