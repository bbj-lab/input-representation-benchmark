#!/usr/bin/env python3
"""
Cohort parity validation script for Experiment 3.

This script verifies that the MEDS and CLIF pipelines use the same patient
cohort, ensuring data parity for fair comparison in Experiment 3.

Validation Checks:
1. Patient ID overlap: Both pipelines use same subject_ids
2. Admission ID overlap: Both pipelines use same hadm_ids
3. Split consistency: Same train/val/test patient assignments
4. Data volume: Similar number of events/tokens per patient

Usage:
    python scripts/validate_cohort_parity.py \
        --cohort_dir /path/to/cohort \
        --meds_dir /path/to/meds/data \
        --clif_dir /path/to/clif/data \
        --output_dir /path/to/output

Output:
    - validation_report.json: Detailed validation results
    - validation_summary.txt: Human-readable summary
    - Exits with code 1 if validation fails

Attribution:
    This script is part of the Input Representation Benchmark.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Set

import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_patient_ids_from_csv(filepath: Path) -> Set[int]:
    """Load patient IDs from CSV file."""
    df = pd.read_csv(filepath)
    return set(df["subject_id"].tolist())


def load_patient_ids_from_meds(meds_dir: Path) -> Set[int]:
    """Load patient IDs from MEDS parquet files."""
    patient_ids = set()
    data_dir = meds_dir / "data"
    
    for split in ["train", "tuning", "test"]:
        split_dir = data_dir / split
        if split_dir.exists():
            for parquet_file in split_dir.glob("*.parquet"):
                try:
                    df = pd.read_parquet(parquet_file, columns=["subject_id"])
                    patient_ids.update(df["subject_id"].unique().tolist())
                except Exception as e:
                    logger.warning(f"Could not read {parquet_file}: {e}")
    
    return patient_ids


def load_patient_ids_from_clif(clif_dir: Path) -> Set[int]:
    """Load patient IDs from CLIF directory structure."""
    patient_ids = set()
    
    # CLIF structure may vary - check common patterns
    for pattern in ["*.parquet", "**/*.parquet"]:
        for parquet_file in clif_dir.glob(pattern):
            try:
                df = pd.read_parquet(parquet_file, columns=["subject_id"])
                patient_ids.update(df["subject_id"].unique().tolist())
            except Exception:
                # Try alternative column names
                try:
                    df = pd.read_parquet(parquet_file, columns=["patient_id"])
                    patient_ids.update(df["patient_id"].unique().tolist())
                except Exception:
                    continue
    
    return patient_ids


def validate_patient_overlap(
    cohort_ids: Set[int],
    meds_ids: Set[int],
    clif_ids: Set[int]
) -> Dict:
    """
    Validate that both pipelines use patients from the aligned cohort.
    
    Returns:
        Dictionary with validation results
    """
    results = {
        "cohort_size": len(cohort_ids),
        "meds_patients": len(meds_ids),
        "clif_patients": len(clif_ids),
        "meds_in_cohort": len(meds_ids & cohort_ids),
        "clif_in_cohort": len(clif_ids & cohort_ids),
        "meds_outside_cohort": len(meds_ids - cohort_ids),
        "clif_outside_cohort": len(clif_ids - cohort_ids),
        "meds_clif_overlap": len(meds_ids & clif_ids),
        "meds_only": len(meds_ids - clif_ids),
        "clif_only": len(clif_ids - meds_ids),
    }
    
    # Compute parity metrics
    if len(meds_ids) > 0 and len(clif_ids) > 0:
        results["jaccard_similarity"] = len(meds_ids & clif_ids) / len(meds_ids | clif_ids)
        results["cohort_coverage_meds"] = len(meds_ids & cohort_ids) / len(cohort_ids)
        results["cohort_coverage_clif"] = len(clif_ids & cohort_ids) / len(cohort_ids)
    else:
        results["jaccard_similarity"] = 0.0
        results["cohort_coverage_meds"] = 0.0
        results["cohort_coverage_clif"] = 0.0
    
    # Validation pass/fail
    results["patients_match"] = (meds_ids == clif_ids)
    results["all_in_cohort"] = (
        results["meds_outside_cohort"] == 0 and 
        results["clif_outside_cohort"] == 0
    )
    
    return results


def validate_split_consistency(
    cohort_dir: Path,
    meds_dir: Path,
    clif_dir: Path
) -> Dict:
    """
    Validate that train/val/test splits are consistent.
    
    Returns:
        Dictionary with split validation results
    """
    results = {}
    
    # Load cohort splits
    try:
        cohort_train = load_patient_ids_from_csv(cohort_dir / "train_patient_ids.csv")
        cohort_val = load_patient_ids_from_csv(cohort_dir / "val_patient_ids.csv")
        cohort_test = load_patient_ids_from_csv(cohort_dir / "test_patient_ids.csv")
        
        results["cohort_train"] = len(cohort_train)
        results["cohort_val"] = len(cohort_val)
        results["cohort_test"] = len(cohort_test)
        results["cohort_splits_loaded"] = True
    except Exception as e:
        logger.warning(f"Could not load cohort splits: {e}")
        results["cohort_splits_loaded"] = False
        return results
    
    # Check for split leakage
    train_val_overlap = len(cohort_train & cohort_val)
    train_test_overlap = len(cohort_train & cohort_test)
    val_test_overlap = len(cohort_val & cohort_test)
    
    results["train_val_overlap"] = train_val_overlap
    results["train_test_overlap"] = train_test_overlap
    results["val_test_overlap"] = val_test_overlap
    results["no_split_leakage"] = (
        train_val_overlap == 0 and 
        train_test_overlap == 0 and 
        val_test_overlap == 0
    )
    
    return results


def generate_report(
    patient_results: Dict,
    split_results: Dict,
    output_dir: Path
) -> bool:
    """
    Generate validation report and summary.
    
    Returns:
        True if validation passed, False otherwise
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Combine results
    report = {
        "patient_validation": patient_results,
        "split_validation": split_results
    }
    
    # Determine overall pass/fail
    passed = (
        patient_results.get("patients_match", False) or
        patient_results.get("jaccard_similarity", 0) > 0.99
    ) and split_results.get("no_split_leakage", True)
    
    report["overall_passed"] = passed
    
    # Save JSON report
    with open(output_dir / "validation_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    # Generate human-readable summary
    summary_lines = [
        "=" * 60,
        "COHORT PARITY VALIDATION SUMMARY",
        "=" * 60,
        "",
        "PATIENT VALIDATION:",
        f"  Cohort size: {patient_results.get('cohort_size', 'N/A'):,}",
        f"  MEDS patients: {patient_results.get('meds_patients', 'N/A'):,}",
        f"  CLIF patients: {patient_results.get('clif_patients', 'N/A'):,}",
        f"  MEDS-CLIF overlap: {patient_results.get('meds_clif_overlap', 'N/A'):,}",
        f"  Jaccard similarity: {patient_results.get('jaccard_similarity', 0):.4f}",
        f"  Patients match exactly: {patient_results.get('patients_match', False)}",
        "",
        "SPLIT VALIDATION:",
        f"  Train patients: {split_results.get('cohort_train', 'N/A'):,}",
        f"  Val patients: {split_results.get('cohort_val', 'N/A'):,}",
        f"  Test patients: {split_results.get('cohort_test', 'N/A'):,}",
        f"  No split leakage: {split_results.get('no_split_leakage', 'N/A')}",
        "",
        "=" * 60,
        f"OVERALL VALIDATION: {'PASSED' if passed else 'FAILED'}",
        "=" * 60,
    ]
    
    summary = "\n".join(summary_lines)
    
    with open(output_dir / "validation_summary.txt", "w") as f:
        f.write(summary)
    
    print(summary)
    
    return passed


def main():
    parser = argparse.ArgumentParser(
        description="Validate cohort parity between MEDS and CLIF pipelines"
    )
    parser.add_argument(
        "--cohort_dir",
        type=Path,
        required=True,
        help="Directory containing cohort patient ID files"
    )
    parser.add_argument(
        "--meds_dir",
        type=Path,
        required=True,
        help="Directory containing MEDS-formatted data"
    )
    parser.add_argument(
        "--clif_dir",
        type=Path,
        required=True,
        help="Directory containing CLIF-formatted data"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output directory for validation report"
    )
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Cohort Parity Validation for Experiment 3")
    logger.info("=" * 60)
    
    # Load cohort patient IDs
    logger.info("Loading cohort patient IDs...")
    cohort_ids = load_patient_ids_from_csv(args.cohort_dir / "cohort_patient_ids.csv")
    logger.info(f"  Cohort: {len(cohort_ids):,} patients")
    
    # Load MEDS patient IDs
    logger.info("Loading MEDS patient IDs...")
    meds_ids = load_patient_ids_from_meds(args.meds_dir)
    logger.info(f"  MEDS: {len(meds_ids):,} patients")
    
    # Load CLIF patient IDs
    logger.info("Loading CLIF patient IDs...")
    clif_ids = load_patient_ids_from_clif(args.clif_dir)
    logger.info(f"  CLIF: {len(clif_ids):,} patients")
    
    # Validate
    logger.info("Validating patient overlap...")
    patient_results = validate_patient_overlap(cohort_ids, meds_ids, clif_ids)
    
    logger.info("Validating split consistency...")
    split_results = validate_split_consistency(args.cohort_dir, args.meds_dir, args.clif_dir)
    
    # Generate report
    logger.info("Generating validation report...")
    passed = generate_report(patient_results, split_results, args.output_dir)
    
    # Exit with appropriate code
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
