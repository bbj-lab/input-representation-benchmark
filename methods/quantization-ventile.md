# Method: Reference Range-Aware Ventile Quantization

## 1. Overview

This strategy incorporates clinical knowledge into the discretization of continuous lab values. Instead of purely data-driven quantiles, we anchor the bins to the clinical reference range (normal range) defined by a lower bound $L$ and an upper bound $U$.

## 2. Formal Definition

Given a set of observed values $X = \{x_1, ..., x_n\}$ for a specific lab code, and a reference range $[L, U]$:

We partition the value space into 20 disjoint intervals (bins) $B_1, ..., B_{20}$. The allocation of bins is determined by the clinical regions:

1.  **Below Normal** ($x < L$): allocated 5 bins ($B_1 - B_5$)
2.  **Within Normal** ($L \le x \le U$): allocated 10 bins ($B_6 - B_{15}$)
3.  **Above Normal** ($x > U$): allocated 5 bins ($B_{16} - B_{20}$)

The breakpoints within each region are determined by local quantiles of the data distribution within that region.

### 2.1. Algorithm

Let $X_{below} = \{x \in X | x < L\}$
Let $X_{within} = \{x \in X | L \le x \le U\}$
Let $X_{above} = \{x \in X | x > U\}$

The set of break points $Q$ is defined as:

$$
Q = Q_{below} \cup \{L\} \cup Q_{within} \cup \{U\} \cup Q_{above}
$$

Where:
*   $Q_{below} = \{q_k(X_{below}) \mid k \in \{0.2, 0.4, 0.6, 0.8\}\}$ (4 breaks)
*   $Q_{within} = \{q_k(X_{within}) \mid k \in \{0.1, 0.2, ..., 0.9\}\}$ (9 breaks)
*   $Q_{above} = \{q_k(X_{above}) \mid k \in \{0.2, 0.4, 0.6, 0.8\}\}$ (4 breaks)

This results in exactly 19 break points, defining 20 bins.

## 3. Implementation Strategy

Based on empirical analysis of MIMIC-IV v3.1, we implement two distinct strategies:

### Strategy A: Paired Reference Ranges (Both $L$ and $U$ exist)
Applied when both `ref_range_lower` and `ref_range_upper` are present.
- **Allocation**: 5 (Low) + 10 (Normal) + 5 (High)
- **Clinical Rationale**: Provides higher granularity within the normal range (where most data lies) while maintaining sufficient resolution for abnormal values.

### Strategy B: Data-Driven Baseline (No reference range)
Applied when `ref_range_lower` and `ref_range_upper` are null.
- **Allocation**: 20 equiprobable bins over the entire distribution $X$.
- **Method**: Standard ventile quantization.

*Note: Analysis of 142M events confirmed that reference ranges in MIMIC-IV are always paired (or both missing), eliminating the need for handling partial ranges.*

## 4. Architecture

### 4.1. Class Structure

The `QuantizationVentile` class in `src/input_representation/tokenize/common/quantization.py` implements this logic.

```python
class QuantizationVentile:
    def agg(self, in_fps, out_fp, num_quantiles=20):
        # Aggregates values from all shards
        # Detects reference ranges
        # Computes breaks per code
        ...
        
    def _compute_ventile_breaks(self, values, ref_lower, ref_upper, num_bins):
        # Dispatches to Strategy A or B
        ...
```

### 4.2. Integration

This module integrates into the tokenization pipeline after the initial data aggregation stage. It replaces the standard `Quantizator` when reference-aware binning is required.


## Validation Results (MIMIC-IV v3.1)

Extracted from `benchmarks/ventile-quantization/REPORT.md`.

### Executive Summary

Successfully implemented and validated reference range-aware 20-bin ventile quantization on MIMIC-IV v3.1 (364,627 patients, 142 million lab events).

### Reference Range Coverage

**Dataset**: 1,128 unique lab codes, 142,131,243 lab events

**By Lab Code**:
- Both bounds: 392 codes (34.8%)
- No bounds: 736 codes (65.2%)
- Lower only: 0 codes (0%)
- Upper only: 0 codes (0%)

**By Lab Event**:
- Both bounds: 114,135,372 events (80.3%)
- No bounds: 27,995,871 events (19.7%)
- Lower/upper only: 0 events (0%)

**Critical Finding**: Reference ranges are **always paired** in MIMIC-IV v3.1 (verified across 100% of events).

### Ventile Quantization Results

**Implementation**: VentileQuantizator with reference range-aware binning

**Total codes quantized**: 200 lab codes (top 200 by frequency)

**Bin distribution**:
- 20 bins: 161 codes (80.5%) - achieved target
- 16 bins: 37 codes (18.5%) - insufficient data in one region
- <16 bins: 2 codes (1.0%)

**Improvement over standard**: Ventile achieves 80.5% codes with 20 bins vs 64% with data-driven approach (+16.5% improvement)

### Glucose Verification Example

**Lab**: Glucose (itemid 50931)  
**Reference Range**: [70.0, 105.0] mg/dL  
**Total bins**: 20

**Binning structure** (verified):
```
Below ref (<70):     [54, 61, 65, 67]                =  5 bins
Within ref [70-105]: [70, 79, 84, 87, 90, 92,        = 10 bins
                      95, 97, 100, 103, 105]
Above ref (>105):    [115, 127, 146, 184]            =  5 bins
                                             Total = 20 bins
```

**Confirmed**: 5 below + 10 within + 5 above = 20 bins

Reference bounds (70, 105) are present as break points, creating clinically meaningful categories for below-normal, normal, and above-normal glucose values.

### Evidence-Based Q&A

#### Q1: Are reference ranges always paired in MIMIC-IV v3.1?

**Answer**: Yes, verified across 100% of events.

**Evidence** - Analysis of all 142,131,243 lab events:
- Both NULL: 27,995,871 events (19.7%)
- Both NOT NULL: 114,135,372 events (80.3%)
- Lower only: 0 events (0%)
- Upper only: 0 events (0%)

Method: Queried all 17 training parquet files, checked all combinations. Zero instances of unpaired bounds.

#### Q5: Why only 200 lab codes quantized?

**Answer**: Hardcoded top 200 threshold in ethos-ares

**Evidence from source code** (`ethos-ares/src/ethos/tokenize/mimic/preprocessors.py` line 556):
```python
known_lab_names = list(counts.keys())[:200]
```

**Validation**: Top 200 codes cover 97.06% of all lab events (137,951,527 events), confirming they "cover most" as stated. The remaining 928 codes account for only 2.94% of events.
