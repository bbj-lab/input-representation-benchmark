#!/usr/bin/env python3
r"""
Cohort alignment script for Experiment 3.

This script extracts a shared cohort for Experiment 3 based on an explicit, reproducible
hospitalization-level inclusion rule:

\(H_{\mathrm{ICU}}\): MIMIC-IV hospital admissions (`hadm_id`) with hospital LOS \(\ge\)24h
(computed from `hosp/admissions.csv.gz` `admittime`/`dischtime`) AND \(\ge\)1 linked ICU stay
record in `icu/icustays.csv.gz` for the same `hadm_id`.

We additionally produce a patient-level train/val/test split and then derive split-specific
`hadm_id` lists by intersecting admissions for patients in each split with \(H_{\mathrm{ICU}}\).

from MIMIC-IV data and creates patient ID lists that can be applied to both
MEDS and CLIF pipelines to ensure data parity.

The cohort criteria match CLIF's natural scope:
- Patients with at least one ICU stay
- Hospital stay duration ≥24 hours
- Patient-level split: 70/10/20 (train/val/test)

Usage:
    python pipeline/scripts/align_cohorts.py \
        --mimic_dir /path/to/physionet.org/files/mimiciv/3.1 \
        --output_dir /path/to/output \
        --data_seed 42

Output:
    - cohort_patient_ids.csv: All patient IDs in the cohort
    - train_patient_ids.csv: Training set patient IDs
    - val_patient_ids.csv: Validation set patient IDs
    - test_patient_ids.csv: Test set patient IDs
    - cohort_hadm_ids.csv: All `hadm_id` in \(H_{\mathrm{ICU}}\) (LOS>=24h and >=1 ICU stay)
    - train_hadm_ids.csv: \(H_{\mathrm{ICU}}\) `hadm_id` for the training patient split
    - val_hadm_ids.csv: \(H_{\mathrm{ICU}}\) `hadm_id` for the validation patient split
    - test_hadm_ids.csv: \(H_{\mathrm{ICU}}\) `hadm_id` for the test patient split
    - cohort_stats.json: Summary statistics

Attribution:
    This script is part of the Input Representation Benchmark.
    The MEDS extraction pipeline used in this benchmark is adapted from ETHOS-ARES:
    https://github.com/ipolharvard/ethos-ares (MIT License, Copyright © 2024 Paweł Renc)
"""

import argparse
import csv
import gzip
import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_admissions(mimic_dir: Path) -> tuple[set[int], set[int], dict[int, set[int]]]:
    r"""
    Stream admissions.csv.gz and return:
      - long_stay_patients (subject_id) where any admission LOS>=24h
      - long_stay_hadm_ids (hadm_id) where admission LOS>=24h
      - subject_to_hadm mapping for all admissions (subject_id -> set(hadm_id))
    """
    admissions_path = mimic_dir / "hosp" / "admissions.csv.gz"
    logger.info(f"Loading admissions from {admissions_path}")
    
    long_stay_patients: set[int] = set()
    long_stay_hadm: set[int] = set()
    subject_to_hadm: dict[int, set[int]] = {}

    with gzip.open(admissions_path, "rt", newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            try:
                sid = int(row["subject_id"])
                hid = int(row["hadm_id"])
            except Exception:
                continue

            subject_to_hadm.setdefault(sid, set()).add(hid)

            admittime = row.get("admittime")
            dischtime = row.get("dischtime")
            if not admittime or not dischtime:
                continue
            try:
                a = datetime.strptime(admittime, "%Y-%m-%d %H:%M:%S")
                d = datetime.strptime(dischtime, "%Y-%m-%d %H:%M:%S")
            except Exception:
                continue
            los_hours = (d - a).total_seconds() / 3600.0
            if los_hours >= 24.0:
                long_stay_patients.add(sid)
                long_stay_hadm.add(hid)
    
    return long_stay_patients, long_stay_hadm, subject_to_hadm


def load_icustays(mimic_dir: Path) -> tuple[set[int], set[int]]:
    """Stream icustays.csv.gz and return (icu_patients, icu_hadm_ids)."""
    icustays_path = mimic_dir / "icu" / "icustays.csv.gz"
    logger.info(f"Loading ICU stays from {icustays_path}")
    icu_patients: set[int] = set()
    icu_hadm: set[int] = set()
    with gzip.open(icustays_path, "rt", newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            try:
                icu_patients.add(int(row["subject_id"]))
            except Exception:
                pass
            try:
                icu_hadm.add(int(row["hadm_id"]))
            except Exception:
                pass
    return icu_patients, icu_hadm


def extract_icu_cohort(
    long_stay_patients: set[int],
    long_stay_hadm: set[int],
    icu_patients: set[int],
    icu_hadm: set[int],
    min_los_hours: float = 24.0
) -> Tuple[Set[int], Set[int]]:
    r"""
    Extract Exp3 cohort objects based on the hospitalization-level ICU-admission cohort \(H_{\mathrm{ICU}}\):
      - patient cohort: `subject_id` with ≥1 ICU stay AND ≥24h hospital LOS
      - hospitalization cohort (H_ICU): `hadm_id` with ≥1 ICU stay AND ≥24h hospital LOS
    
    Args:
        admissions: Hospital admissions dataframe
        icustays: ICU stays dataframe
        min_los_hours: Minimum length of stay in hours
        
    Returns:
        (cohort_patient_ids, cohort_hadm_ids)
    """
    logger.info(f"Patients with ICU stays: {len(icu_patients):,}")
    logger.info(f"Hospitalizations with ICU stays: {len(icu_hadm):,}")
    logger.info(f"Patients with LOS ≥{min_los_hours}h: {len(long_stay_patients):,}")
    logger.info(f"Hospitalizations with LOS ≥{min_los_hours}h: {len(long_stay_hadm):,}")
    
    # Intersection: ICU patients with long stays
    cohort = icu_patients & long_stay_patients
    logger.info(f"H_ICU patient cohort (ICU stay + LOS≥{min_los_hours}h): {len(cohort):,} patients")
    
    # H_ICU hospitalizations (hadm_id with ICU stay and LOS>=24h)
    cohort_hadm = icu_hadm & long_stay_hadm
    logger.info(f"H_ICU hospitalizations (ICU stay + LOS≥{min_los_hours}h): {len(cohort_hadm):,} hadm_id")

    return cohort, cohort_hadm


def patient_level_split(
    patient_ids: List[int],
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[List[int], List[int], List[int]]:
    """
    Split patient IDs into train/val/test sets.
    
    Args:
        patient_ids: List of patient IDs
        train_ratio: Training set ratio
        val_ratio: Validation set ratio  
        test_ratio: Test set ratio
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_ids, val_ids, test_ids)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    
    # Shuffle with fixed seed (stdlib only; avoids numpy dependency)
    shuffled = list(patient_ids)
    random.Random(seed).shuffle(shuffled)
    
    n_total = len(shuffled)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_ids = shuffled[:n_train]
    val_ids = shuffled[n_train:n_train + n_val]
    test_ids = shuffled[n_train + n_val:]
    
    logger.info(f"Split: train={len(train_ids):,}, val={len(val_ids):,}, test={len(test_ids):,}")
    
    return train_ids, val_ids, test_ids


def save_cohort(
    cohort: Set[int],
    train_ids: List[int],
    val_ids: List[int],
    test_ids: List[int],
    cohort_hadm_ids: Set[int],
    subject_to_hadm: dict[int, set[int]],
    output_dir: Path,
    seed: int
) -> Dict:
    """
    Save cohort files and return statistics.
    
    Args:
        cohort: Full cohort patient IDs
        train_ids: Training patient IDs
        val_ids: Validation patient IDs
        test_ids: Test patient IDs
        output_dir: Output directory
        seed: Random seed used
        
    Returns:
        Dictionary of cohort statistics
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save patient ID lists
    def _write_csv(path: Path, header: str, rows: list[int]) -> None:
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([header])
            for x in rows:
                w.writerow([x])

    _write_csv(output_dir / "cohort_patient_ids.csv", "subject_id", sorted(cohort))
    _write_csv(output_dir / "train_patient_ids.csv", "subject_id", train_ids)
    _write_csv(output_dir / "val_patient_ids.csv", "subject_id", val_ids)
    _write_csv(output_dir / "test_patient_ids.csv", "subject_id", test_ids)

    # Derive H_ICU hadm_ids per patient split.
    # We use admissions to map subject_id -> hadm_id, then intersect with cohort_hadm_ids
    # (hadm-level ICU + LOS>=24 eligibility).
    def hadm_for(subject_ids: list[int]) -> list[int]:
        hadm: set[int] = set()
        for sid in subject_ids:
            hadm |= subject_to_hadm.get(int(sid), set())
        return sorted(hadm & cohort_hadm_ids)

    train_hadm = hadm_for(train_ids)
    val_hadm = hadm_for(val_ids)
    test_hadm = hadm_for(test_ids)

    _write_csv(output_dir / "cohort_hadm_ids.csv", "hadm_id", sorted(cohort_hadm_ids))
    _write_csv(output_dir / "train_hadm_ids.csv", "hadm_id", train_hadm)
    _write_csv(output_dir / "val_hadm_ids.csv", "hadm_id", val_hadm)
    _write_csv(output_dir / "test_hadm_ids.csv", "hadm_id", test_hadm)
    
    # Compute and save statistics
    stats = {
        "total_patients": len(cohort),
        "train_patients": len(train_ids),
        "val_patients": len(val_ids),
        "test_patients": len(test_ids),
        "total_icu_eligible_hospitalizations": len(cohort_hadm_ids),
        "train_icu_eligible_hospitalizations": len(train_hadm),
        "val_icu_eligible_hospitalizations": len(val_hadm),
        "test_icu_eligible_hospitalizations": len(test_hadm),
        "train_ratio": len(train_ids) / len(cohort),
        "val_ratio": len(val_ids) / len(cohort),
        "test_ratio": len(test_ids) / len(cohort),
        "data_seed": seed,
        "cohort_criteria": (
            "Patient-level split; H_ICU hospitalizations are admissions (hadm_id) with "
            ">=1 ICU stay record in icu/icustays and hospital LOS >=24h in hosp/admissions."
        )
    }
    
    with open(output_dir / "cohort_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Saved cohort files to {output_dir}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Extract Exp3 H_ICU cohort (LOS>=24h + linked ICU stay) for data parity"
    )
    parser.add_argument(
        "--mimic_dir",
        type=Path,
        required=True,
        help="Path to MIMIC-IV directory (e.g., physionet.org/files/mimiciv/3.1)"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output directory for cohort files"
    )
    parser.add_argument(
        "--data_seed",
        type=int,
        default=42,
        help="Random seed for patient splitting (default: 42)"
    )
    parser.add_argument(
        "--min_los_hours",
        type=float,
        default=24.0,
        help="Minimum length of stay in hours (default: 24.0)"
    )
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("ICU Cohort Extraction for Experiment 3")
    logger.info("=" * 60)
    logger.info(f"MIMIC directory: {args.mimic_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Data seed: {args.data_seed}")
    logger.info(f"Min LOS: {args.min_los_hours} hours")
    logger.info("")
    
    # Load data
    long_stay_patients, long_stay_hadm, subject_to_hadm = load_admissions(args.mimic_dir)
    icu_patients, icu_hadm = load_icustays(args.mimic_dir)
    
    # Extract cohort
    cohort, cohort_hadm_ids = extract_icu_cohort(
        long_stay_patients, long_stay_hadm, icu_patients, icu_hadm, args.min_los_hours
    )
    
    # Split
    train_ids, val_ids, test_ids = patient_level_split(
        list(cohort),
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2,
        seed=args.data_seed
    )
    
    # Save
    stats = save_cohort(
        cohort,
        train_ids,
        val_ids,
        test_ids,
        cohort_hadm_ids,
        subject_to_hadm,
        args.output_dir,
        args.data_seed,
    )
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("Cohort Statistics")
    logger.info("=" * 60)
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    logger.info("")
    logger.info("Done! Use these patient IDs for both MEDS and CLIF pipelines.")


if __name__ == "__main__":
    main()
