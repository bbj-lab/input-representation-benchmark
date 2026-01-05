"""
Utility scripts for the Input Representation Benchmark.

This package contains scripts for data processing, cohort alignment,
and pipeline validation for Experiment 3.

Scripts:
    align_cohorts.py: Extract common ICU ≥24h cohort for MEDS/CLIF data parity
    validate_cohort_parity.py: Verify data equivalence between pipelines
    extract_outcomes_meds.py: MEDS-specific outcome extraction (4 outcomes + 24h flags)
    validate_imv_detection.py: MEDS vs CLIF label parity report (IMV-focused)

Attribution:
    The MEDS extraction pipeline used in this benchmark is adapted from ETHOS-ARES:
    https://github.com/ipolharvard/ethos-ares (MIT License, Copyright © 2024 Paweł Renc)
    Citation: Renc, P., et al. (2025). GigaScience, 14, giaf107.
"""
