# Input Representation Benchmark

**EHR Input Representation Benchmarking Framework**. Extends [ETHOS-ARES](https://github.com/ipolharvard/ethos-ares).

> **Compatibility Warning**: This codebase is tested ONLY for compatibility with `ethos-ares` commit `2d54383` (https://github.com/ipolharvard/ethos-ares). Use with other versions may require adjustments.

This repository facilitates the comparison of different input representation strategies for Electronic Health Records (EHR), such as:
1.  Deciles (Baseline)
2.  **Ventile Quantization (Implemented)**: Reference range-aware binning.
3.  Standard Deviations
4.  Normal vs. Abnormal tokens
5.  Time gap tokens / Positional encodings

## 1. Ventile Quantization (Reference Range-Aware)

**Goal**: Incorporate clinical knowledge into tokenization by using reference ranges (`ref_range_lower`, `ref_range_upper`) to define bins (e.g., 5 bins below normal, 10 within, 5 above).

**Results (MIMIC-IV v3.1)**:
- **Dataset**: 364,627 patients, 142M lab events, 1,128 unique lab codes
- **Reference Ranges**: 392 codes (34.8%) have paired bounds, covering 114M events (80.3%)
- **Strict Filtering**: 155/203 candidate codes (76.4%) pass the strict 20-bin requirement
- **All passing codes achieve exactly 20 bins**: ✓
- **Glucose Example**: Verified 5 bins below [70], 10 within [70-105], 5 above [105]

**Details**: `benchmarks/quantization-ventile/SUMMARY.md`

## Edge Cases and Important Notes

### 1. Vitals (BMI, SBP, DBP) Are Not Lab Codes

**Finding**: The quantization pipeline processes 203 codes total:
- 200 lab codes (`LAB//...`)
- 3 vitals (`BMI//Q`, `VITAL//Q//SBP`, `VITAL//Q//DBP`)

**Source**:
- **BMI**: Calculated from patient demographics (weight/height), not from `labevents`.
- **SBP/DBP**: Systolic and Diastolic Blood Pressure from ICU monitoring (`chartevents`), not from `labevents`.

**Reference Ranges**:
- **None of the vitals have reference ranges** in the MEDS data.
- They are quantized using **Strategy B** (data-driven 20-bin ventile).
- This is expected because reference ranges for BMI and BP are patient-specific and not stored in the database schema.

**Implication**: When analyzing "lab codes with/without reference ranges", exclude these 3 vitals from the count or note them separately.

### 2. Non-Numeric Lab Codes

**Findings**:
- **392 out of 1,128 lab codes (34.8%) have 0% numeric values**
- **61 additional codes have <1% numeric values**
- Total: 453 codes are mostly/completely non-numeric

**Data Types** (from `text_value` column):
- **Specimen metadata**: "HOLD. DISCARD GREATER THAN 4 HOURS OLD", "HOLD. DISCARD GREATER THAN 24 HRS OLD"
- **Source indicators**: "ART." (arterial), "VEN." (venous)
- **Qualitative results**: "NONE", "___" (placeholder), "Pos/Neg"
- **NULL/Missing**: Most common - indicates no result recorded or test not applicable

**Top Non-Numeric Codes** (examples):
- `LAB//52033//UNK`: 757,259 events (specimen source: "ART." or "VEN.")
- `LAB//51506//UNK`: 515,825 events (NULL/None - test not run)
- `LAB//51487//UNK`: 515,813 events (NULL/None)
- `LAB//50920//UNK`: 1,496,577 events (NULL/None)
- `LAB//50933//UNK`: 351,982 events (specimen handling instructions)

**Handling in Pipeline**:
- These codes are **automatically filtered out** during preprocessing (`retain_only_test_with_numeric_result`).
- They do **not** appear in the top 200 quantized codes.
- The quantization process only considers codes with numeric values.

**Note**: Many codes have `UNK` (unknown) or `N/A` units, indicating they are metadata fields, test orders, or specimen tracking rather than actual lab measurements.

### 3. Reference Range Availability

**Key Finding**: Reference ranges in MIMIC-IV v3.1 are **always paired** (both or neither).

**Statistics**:
- 392 codes (34.8%) have both `ref_range_lower` AND `ref_range_upper`
- 736 codes (65.2%) have neither
- 0 codes have only one bound

**Implication**: The ventile quantization implementation was simplified from 4 strategies to 2:
- Strategy A: Both bounds (5 below + 10 within + 5 above)
- Strategy B: No bounds (standard 20-bin ventile)

### 4. Top 200 Filtering

**Ethos-ARES Implementation**: The standard tokenization pipeline filters to the top 200 most frequent lab codes.

**Source**: `ethos-ares/src/ethos/tokenize/mimic/preprocessors.py` line 556:
```python
known_lab_names = list(counts.keys())[:200]
```

**Coverage**: Top 200 codes cover 97.06% of all lab events (137,951,527 out of 142,131,243 events).

**Excluded**: 928 codes (rank 201-1,128) account for only 2.94% of events.

**Consideration**: When comparing ventile quantization results, note that:
- We only quantize the top 200 lab codes (not all 1,128).
- Rare labs are not discretized, even if they have reference ranges.

### 5. Filtering Rule

**Policy**: If the number of unique numeric values for a lab code is less than the specified number of bins, we **do not use** that lab code.

**Rationale**: To ensure statistical validity and avoid degenerate bins.

**Consideration**:
- Codes with sparse data (< specified bins) are excluded from the quantization map.

For full details, see: `benchmarks/quantization-ventile/SUMMARY.md`

## Installation

```bash
conda create -n ehr-benchmark python=3.12
conda activate ehr-benchmark
pip install "MEDS_transforms[local_parallelism]"
pip install -e /path/to/ethos-ares
pip install -e .
```

**Note**: The MIMIC-IV dataset (`physionet.org/`) is excluded from git. You must download it separately and place it under this repository.

## Quick Start: Ventile Benchmark

Run the complete pipeline from environment setup to analysis:

```bash
# 0. Environment Setup
conda create -n ehr-benchmark python=3.12 -y
conda activate ehr-benchmark
pip install "MEDS_transforms[local_parallelism]"
pip install -e ../ethos-ares  # Clone ethos-ares as a sibling directory
pip install -e .

# Navigate to benchmark directory
cd benchmarks/ventile-quantization

# 1. MEDS Extraction (ETL using ETHOS-ARES pipeline)
# This extracts MIMIC-IV data with our custom event config that includes ref_ranges
bash scripts/01_extract_meds.sh 3

# 2. Standard Tokenization (ETHOS-ARES preprocessing)
# This runs the standard ethos pipeline up to stage 14 (code counting)
ethos_tokenize -m worker='range(0,3)' \
    input_dir=data/meds/data/train \
    output_dir=data/tokenized \
    out_fn=train dataset=mimic num_quantiles=20

# 3. Apply Ventile Quantization (Our custom strategy)
# This reads stage 04 output and creates ventile_breaks.json with ref-range awareness
python scripts/02_run_ventile_quantization.py

# 4. Analyze Results
python scripts/03_analyze_ventiles.py

# Optional: Analyze non-numeric patterns
python scripts/99_analyze_non_numeric.py
```

**Pipeline Outputs**:
- `data/meds/data/`: MEDS format data with reference ranges
- `data/tokenized/train/`: Standard ethos preprocessing outputs (stages 01-14)
- `data/tokenized/train/ventile_breaks.json`: Reference range-aware quantile breaks
- `analysis/ventile_breaks_summary.csv`: Detailed bin analysis per code
- `analysis/reference_range_coverage.csv`: Reference range statistics per code

## Project Structure

```
input-representation-benchmark/
├── README.md                   # Project overview
├── pyproject.toml              # Dependencies
├── run_benchmark.sh            # Pipeline runner
│
├── src/input_representation/   # Core library
│   └── tokenize/common/
│       └── quantization.py             # QuantizationVentile, etc.
│
├── benchmarks/
│   └── quantization-ventile/
│       ├── SUMMARY.md          # Implementation & results
│       ├── test_ventile_pipeline.py    # Automated tests
│       ├── configs/            # MEDS event configs
│       ├── scripts/            # Pipeline scripts
│       ├── data/               # MEDS + tokenized data
│       └── analysis/           # Statistical analysis outputs
│
└── methods/                    # Methodological notes
```

## Dependence on ETHOS-ARES

This project is an extension of **[ETHOS-ARES](https://github.com/ipolharvard/ethos-ares)** (Renc et al., 2025). 

### What We Adopt from ETHOS-ARES

1.  **MEDS Extraction Pipeline**: Complete ETL using `MEDS_transform-runner`
2.  **Standard Tokenization**: Preprocessing stages 01-14 using `ethos_tokenize`
3.  **Infrastructure**: `LabData`, `DemographicData` classes and preprocessing logic

### Our Contributions

1.  **Custom Event Configurations**: Extended MEDS configs to include `ref_range_lower` and `ref_range_upper`
2.  **Quantization Strategies**: Novel binning methods (e.g., `QuantizationVentile` with 5-10-5 reference range-aware binning)
3.  **Strict Filtering**: Quality control ensuring all tokens have sufficient granularity
4.  **Analysis Tools**: Validation scripts and automated testing

### Integration Philosophy

**No modifications to `ethos-ares` repository**  
**Uses standard ETHOS-ARES commands**  
**Clean separation via custom configs and post-processing**  
**Full compatibility maintained**

All credit for the underlying MEDS and tokenization infrastructure goes to the ETHOS-ARES team.

## Contact

Daniel Lee (ihlee@uchicago.edu), University of Chicago

## Citation

If you use this benchmarking framework, please cite:

```bibtex
@article{
  TO-BE-ADDED
}
```

Please also cite the underlying ETHOS-ARES framework:

```bibtex
@article{renc2025ethos,
  title={Foundation model of electronic medical records for adaptive risk estimation},
  author={Renc, Pawel and others},
  journal={GigaScience},
  year={2025}
}
```