# Ventile Quantization

**Strategy**: Reference range-aware 20-bin quantization  
**Dataset**: MIMIC-IV v3.1 (364,627 patients, 142M lab events)

## Implementation

### Pipeline
1. **MEDS Extraction**: ETHOS-ARES pipeline + custom `event_configs_v3.1.yaml` (adds `ref_range_lower`, `ref_range_upper`)
2. **Tokenization**: ETHOS-ARES `ethos_tokenize` (stages 01-14, unchanged)
3. **Quantization**: Custom `QuantizationVentile` with 5-10-5 binning
4. **Validation**: Automated tests + analysis scripts

### Quantization Strategy
- **With ref ranges** (80.3% of events): 5 bins below + 10 bins within + 5 bins above = 20 bins
- **Without ref ranges** (19.7% of events): Standard 20-bin ventile

### Strict Filtering
- **With ref ranges**: Requires ≥5, ≥10, ≥5 unique values in below/within/above regions
- **Without ref ranges**: Requires ≥20 unique values overall
- **Result**: All 155 passing codes have exactly 20 bins

## Results

### Reference Range Coverage
- **Codes**: 392/1,128 (34.8%) have paired ref ranges
- **Events**: 114M/142M (80.3%) have ref ranges
- **Key Finding**: Ref ranges always paired (both or neither)

### Quantization Results
- **Candidates**: 203 codes (top 200 labs + 3 vitals)
- **Passing**: 155 codes (76.4%)
- **Rejected**: 48 codes (23.6%, insufficient data for 20 bins)

### Glucose Validation
**Code**: `LAB//Q//50931//MG/DL`, **Ref Range**: [70, 105] mg/dL

```
Below (<70):     [54, 61, 65, 67]                                → 5 bins (V1-V5)
Within [70-105]: [79, 84, 87, 90, 92, 95, 97, 100, 103]        → 10 bins (V6-V15)
Above (>105):    [115, 127, 146, 184]                           → 5 bins (V16-V20)
```

## Non-Numeric Data

- **Non-numeric events**: 21.5M (13.57%)
- **Common types**: Null (16.1M), "NEG" (1.3M), "HOLD" (1.1M), "ART./VEN." (0.5M)
- **Handling**: Automatically filtered during preprocessing

## Files

### Input
- `configs/event_configs_v3.1.yaml`: MEDS config with ref_ranges
- Raw MIMIC-IV v3.1 data

### Output
- `data/tokenized/train/ventile_breaks.json`: 155 codes × 19 breaks (20 bins)
- `analysis/ventile_breaks_summary.csv`: Per-code statistics
- `analysis/reference_range_coverage.csv`: Ref range analysis (1,128 codes)

### Code
- `../../src/input_representation/tokenize/common/quantization.py`: Core logic
- `scripts/01_extract_meds.sh`: MEDS extraction wrapper
- `scripts/02_run_ventile_quantization.py`: Apply ventile strategy
- `scripts/03_analyze_ventiles.py`: Generate analysis
- `test_ventile_pipeline.py`: Automated validation

## Usage

```bash
cd benchmarks/quantization-ventile
bash scripts/01_extract_meds.sh 3
ethos_tokenize -m worker='range(0,3)' input_dir=data/meds/data/train output_dir=data/tokenized out_fn=train dataset=mimic num_quantiles=20
python scripts/02_run_ventile_quantization.py
python scripts/03_analyze_ventiles.py
python test_ventile_pipeline.py  # Verify
```

## Integration with ETHOS-ARES

**No modifications to ethos-ares repository**. We use:
- Standard commands: `MEDS_transform-runner`, `ethos_tokenize`
- Custom config: Environment variable points to our `event_configs_v3.1.yaml`
- Post-processing: Our `QuantizationVentile` reads ethos stage 04 output

All credit for MEDS/tokenization infrastructure: [ETHOS-ARES](https://github.com/ipolharvard/ethos-ares)
