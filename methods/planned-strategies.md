# Overall Input Representation Plans

This framework is designed to benchmark various EHR input representation strategies. Below are the strategies planned for implementation and comparison.

## 1. Numeric Representation Strategies

### 1.1. Quantile-Based (Implemented)
- **Ventile (20-bins)**: Current implementation using reference ranges.
- **Decile (10-bins)**: Baseline implementation (standard in many models).
- **Percentile (100-bins)**: High-granularity option.

### 1.2. Distribution-Based
- **Standard Deviations**: Binning based on Z-scores (e.g., <-2σ, -2σ to -1σ, -1σ to +1σ, etc.).
- **Fixed Value Ranges**: predefined clinical cutoffs (e.g., hypotension < 90 mmHg).

### 1.3. Qualitative
- **Normal vs. Abnormal**: Binary tokenization based on reference ranges.
- **3-State**: Low / Normal / High.
- **5-State**: Critical Low / Low / Normal / High / Critical High.

## 2. Temporal Representation Strategies

### 2.1. Time Tokens
- **Time Gap Tokens**: Explicit tokens representing time elapsed since previous event (e.g., `TIME_1H`, `TIME_1D`).
- **Absolute Time**: Tokens for time of day, day of week, etc.

### 2.2. Encodings
- **Positional Encodings**: Transformer-style continuous positional embeddings.
- **Time-Aware Attention**: Modifying attention mechanisms to weigh by time difference.

## 3. Event Definitions

- **Hierarchical Codes**: Using ontology parents (e.g., ICD chapter vs specific code).
- **Unified Vocabularies**: Mapping local codes to standard ontologies (SNOMED, LOINC) before tokenization.

