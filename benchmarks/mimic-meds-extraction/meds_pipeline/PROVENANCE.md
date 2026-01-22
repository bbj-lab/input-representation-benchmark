# MEDS extraction pipeline (provenance)

This directory contains the MEDS (Medical Event Data Standard) extraction pipeline used to convert MIMIC-IV
data into standardized parquet timelines.

## Attribution

This pipeline is adapted from the ETHOS-ARES project (MIT License):
- Source repository: `https://github.com/ipolharvard/ethos-ares`
- Citation: Renc, P., et al. (2025) *GigaScience*, 14, giaf107.

## Local modifications (high level)

- Split fractions: 70/10/20 (train/val/test)
- Event configuration: storetime semantics + ICU module tables + exclusion of post-discharge billing codes
- Extended MEDS fields: `ref_range_lower` / `ref_range_upper` for clinically anchored binning
- Updated for MIMIC-IV v3.1

See the top-level `README.md` and `THIRD_PARTY_NOTICES.md` for how we invoke this pipeline and for full attribution.

