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
