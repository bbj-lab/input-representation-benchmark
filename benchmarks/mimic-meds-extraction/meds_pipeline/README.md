# MEDS Extraction Pipeline

This directory contains the MEDS (Medical Event Data Standard) extraction pipeline for converting MIMIC-IV data into a standardized format suitable for EHR foundation model training.

## Attribution

**This pipeline is adapted from the ETHOS-ARES project.**

- **Source Repository**: https://github.com/ipolharvard/ethos-ares
- **License**: MIT License (Copyright © 2024 Paweł Renc)
- **Original Authors**: Paweł Renc, Michał K. Grzeszczyk, Nassim Oufattole, Deirdre Goode, Yugang Jia, Szymon Bieganski, Matthew B.A. McDermott, Jarosław Was, Anthony E. Samir, Jonathan W. Cunningham, David W. Bates, Arkadiusz Sitek

**Citation**:
```bibtex
@article{renc2025ethos,
  title={Foundation Model of Electronic Medical Records for Adaptive Risk Estimation},
  author={Renc, Pawe{\l} and Grzeszczyk, Micha{\l} K. and Oufattole, Nassim and Goode, Deirdre and Jia, Yugang and Bieganski, Szymon and McDermott, Matthew B.A. and Was, Jaros{\l}aw and Samir, Anthony E. and Cunningham, Jonathan W. and Bates, David W. and Sitek, Arkadiusz},
  journal={GigaScience},
  volume={14},
  pages={giaf107},
  year={2025},
  doi={10.1093/gigascience/giaf107}
}
```

## Modifications from Original

We have adapted the original ETHOS-ARES MEDS extraction pipeline with the following modifications:

1. **Split Fractions**: Changed from 90/10 (train/test) to 70/10/20 (train/val/test) to match our benchmark requirements.

2. **Event Configuration**: Created custom `event_configs_v3.1_full.yaml` with:
   - Use of `storetime` instead of `charttime` for event ordering (prevents look-ahead bias)
   - Exclusion of post-discharge billing codes (ICD, CPT, DRG)
   - Inclusion of ICU module tables (chartevents, inputevents, outputevents, procedureevents)
   - Extended MEDS fields (`ref_range_lower`, `ref_range_upper`) for clinical anchoring

3. **Dataset Version**: Updated for MIMIC-IV v3.1 (original was v2.2)

## Directory Structure

```
meds_pipeline/
├── README.md                   # This file
├── pre_MEDS.py                 # Pre-MEDS data wrangling (from ETHOS-ARES)
└── configs/
    ├── extract_MIMIC.yaml      # Main extraction config (modified)
    ├── pre_MEDS.yaml           # Pre-MEDS config
    └── local_parallelism_runner.yaml  # Parallelism settings
```

## Usage

The pipeline is invoked via the extraction script:

```bash
cd input-representation-benchmark/benchmarks/mimic-meds-extraction
./scripts/01_extract_meds_full.sh [N_WORKERS]
```

This script:
1. Runs `pre_MEDS.py` to wrangle raw MIMIC-IV data
2. Runs `MEDS_transform-runner` with our custom event configuration

## Dependencies

- Python 3.10+
- `MEDS_transforms` package (pip install)
- `polars`, `hydra-core`, `loguru`

## Output

The pipeline produces MEDS-formatted parquet files in `data/meds/` with:
- Vocabulary size: ~20,882 tokens
- Average timeline length: ~1,450 tokens/patient
- Extended columns for clinical anchoring (`ref_range_lower`, `ref_range_upper`)

## License

This adapted code retains the MIT License from the original ETHOS-ARES repository.

```
MIT License

Copyright (c) 2024 Paweł Renc

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
