#!/usr/bin/env python3
"""
Cohort alignment script for Experiment 3.

This script extracts the common patient cohort (ICU patients with ≥24h stays)
from MIMIC-IV data and creates patient ID lists that can be applied to both
MEDS and CLIF pipelines to ensure data parity.

The cohort criteria match CLIF's natural scope:
- Patients with at least one ICU stay
- Hospital stay duration ≥24 hours
- Patient-level split: 70/10/20 (train/val/test)

Usage:
    python scripts/align_cohorts.py \
        --mimic_dir /path/to/physionet.org/files/mimiciv/3.1 \
        --output_dir /path/to/output \
        --data_seed 42

Output:
    - cohort_patient_ids.csv: All patient IDs in the cohort
    - train_patient_ids.csv: Training set patient IDs
    - val_patient_ids.csv: Validation set patient IDs
    - test_patient_ids.csv: Test set patient IDs
    - cohort_stats.json: Summary statistics

Attribution:
    This script is part of the Input Representation Benchmark.
    The MEDS extraction pipeline used in this benchmark is adapted from ETHOS-ARES:
    https://github.com/ipolharvard/ethos-ares (MIT License, Copyright © 2024 Paweł Renc)
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_admissions(mimic_dir: Path) -> pd.DataFrame:
    """Load admissions table with length of stay calculation."""
    admissions_path = mimic_dir / "hosp" / "admissions.csv.gz"
    logger.info(f"Loading admissions from {admissions_path}")
    
    df = pd.read_csv(
        admissions_path,
        usecols=["subject_id", "hadm_id", "admittime", "dischtime"],
        parse_dates=["admittime", "dischtime"]
    )
    
    # Calculate length of stay in hours
    df["los_hours"] = (df["dischtime"] - df["admittime"]).dt.total_seconds() / 3600
    
    return df


def load_icustays(mimic_dir: Path) -> pd.DataFrame:
    """Load ICU stays table."""
    icustays_path = mimic_dir / "icu" / "icustays.csv.gz"
    logger.info(f"Loading ICU stays from {icustays_path}")
    
    df = pd.read_csv(
        icustays_path,
        usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime"],
        parse_dates=["intime", "outtime"]
    )
    
    return df


def extract_icu_cohort(
    admissions: pd.DataFrame,
    icustays: pd.DataFrame,
    min_los_hours: float = 24.0
) -> Set[int]:
    """
    Extract ICU cohort: patients with ≥1 ICU stay AND hospital stay ≥24h.
    
    Args:
        admissions: Hospital admissions dataframe
        icustays: ICU stays dataframe
        min_los_hours: Minimum length of stay in hours
        
    Returns:
        Set of subject_ids meeting criteria
    """
    # Get patients with at least one ICU stay
    icu_patients = set(icustays["subject_id"].unique())
    logger.info(f"Patients with ICU stays: {len(icu_patients):,}")
    
    # Get admissions with LOS ≥24h
    long_stay_admissions = admissions[admissions["los_hours"] >= min_los_hours]
    long_stay_patients = set(long_stay_admissions["subject_id"].unique())
    logger.info(f"Patients with LOS ≥{min_los_hours}h: {len(long_stay_patients):,}")
    
    # Intersection: ICU patients with long stays
    cohort = icu_patients & long_stay_patients
    logger.info(f"ICU cohort (≥{min_los_hours}h): {len(cohort):,} patients")
    
    return cohort


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
    
    # Shuffle with fixed seed
    rng = np.random.RandomState(seed)
    shuffled = rng.permutation(patient_ids).tolist()
    
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
    pd.DataFrame({"subject_id": sorted(cohort)}).to_csv(
        output_dir / "cohort_patient_ids.csv", index=False
    )
    pd.DataFrame({"subject_id": train_ids}).to_csv(
        output_dir / "train_patient_ids.csv", index=False
    )
    pd.DataFrame({"subject_id": val_ids}).to_csv(
        output_dir / "val_patient_ids.csv", index=False
    )
    pd.DataFrame({"subject_id": test_ids}).to_csv(
        output_dir / "test_patient_ids.csv", index=False
    )
    
    # Compute and save statistics
    stats = {
        "total_patients": len(cohort),
        "train_patients": len(train_ids),
        "val_patients": len(val_ids),
        "test_patients": len(test_ids),
        "train_ratio": len(train_ids) / len(cohort),
        "val_ratio": len(val_ids) / len(cohort),
        "test_ratio": len(test_ids) / len(cohort),
        "data_seed": seed,
        "cohort_criteria": "ICU patients with hospital stay ≥24 hours"
    }
    
    with open(output_dir / "cohort_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Saved cohort files to {output_dir}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Extract ICU cohort for Experiment 3 data parity"
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
    admissions = load_admissions(args.mimic_dir)
    icustays = load_icustays(args.mimic_dir)
    
    # Extract cohort
    cohort = extract_icu_cohort(admissions, icustays, args.min_los_hours)
    
    # Split
    train_ids, val_ids, test_ids = patient_level_split(
        list(cohort),
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2,
        seed=args.data_seed
    )
    
    # Save
    stats = save_cohort(cohort, train_ids, val_ids, test_ids, args.output_dir, args.data_seed)
    
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
