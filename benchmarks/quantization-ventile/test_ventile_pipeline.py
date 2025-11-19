#!/usr/bin/env python3
"""Complete pipeline verification test.

This script verifies that all components of the ventile quantization pipeline
are working correctly, from MEDS extraction through final analysis.

Usage:
    python test_ventile_pipeline.py
"""

import json
import sys
from pathlib import Path

import polars as pl
from loguru import logger


def test_meds_integration():
    """Test that MEDS data includes reference ranges."""
    logger.info("Testing MEDS data integration...")
    
    base_dir = Path(__file__).parent
    meds_file = base_dir / "data/meds/data/train/0.parquet"
    if not meds_file.exists():
        logger.error(f"MEDS file not found: {meds_file}")
        return False
    
    df = pl.scan_parquet(meds_file)
    schema = df.collect_schema()
    
    has_refs = 'ref_range_lower' in schema.names() and 'ref_range_upper' in schema.names()
    if not has_refs:
        logger.error("MEDS data missing ref_range columns")
        return False
    
    # Count events with ref ranges
    labs_with_refs = df.filter(
        pl.col('code').str.starts_with('LAB//') &
        pl.col('ref_range_lower').is_not_null()
    ).select(pl.len()).collect().item()
    
    logger.success(f"✓ MEDS data has ref_range columns")
    logger.success(f"✓ Shard 0 has {labs_with_refs:,} lab events with ref_ranges")
    return True


def test_tokenization_preservation():
    """Test that ethos tokenization preserves ref_range columns."""
    logger.info("Testing tokenization pipeline...")
    
    base_dir = Path(__file__).parent
    tokenized_file = base_dir / "data/tokenized/train/04_preprocessing_with_counts/0.parquet"
    if not tokenized_file.exists():
        logger.error(f"Tokenized file not found: {tokenized_file}")
        return False
    
    df = pl.scan_parquet(tokenized_file)
    schema = df.collect_schema()
    
    has_refs = 'ref_range_lower' in schema.names() and 'ref_range_upper' in schema.names()
    if not has_refs:
        logger.error("Tokenized data missing ref_range columns")
        return False
    
    logger.success(f"✓ Tokenized data preserves ref_range columns")
    return True


def test_ventile_quantization():
    """Test that ventile quantization produces correct results."""
    logger.info("Testing ventile quantization...")
    
    base_dir = Path(__file__).parent
    ventile_breaks_file = base_dir / "data/tokenized/train/ventile_breaks.json"
    if not ventile_breaks_file.exists():
        logger.error(f"Ventile breaks file not found: {ventile_breaks_file}")
        return False
    
    with open(ventile_breaks_file) as f:
        ventile_breaks = json.load(f)
    
    if len(ventile_breaks) == 0:
        logger.error("No ventile breaks generated")
        return False
    
    # Check that all codes have exactly 20 bins
    all_20_bins = all(len(breaks) + 1 == 20 for breaks in ventile_breaks.values())
    if not all_20_bins:
        incomplete = {code: len(breaks)+1 for code, breaks in ventile_breaks.items() if len(breaks)+1 != 20}
        logger.error(f"Found {len(incomplete)} codes without 20 bins: {list(incomplete.items())[:5]}")
        return False
    
    logger.success(f"✓ Generated ventile breaks for {len(ventile_breaks)} codes")
    logger.success(f"✓ All codes have exactly 20 bins (strict filtering working)")
    return True


def test_glucose_example():
    """Test the glucose example with 5-10-5 binning."""
    logger.info("Testing glucose example...")
    
    base_dir = Path(__file__).parent
    ventile_breaks_file = base_dir / "data/tokenized/train/ventile_breaks.json"
    with open(ventile_breaks_file) as f:
        ventile_breaks = json.load(f)
    
    glucose_code = 'LAB//Q//50931//MG/DL'
    if glucose_code not in ventile_breaks:
        logger.error(f"Glucose code {glucose_code} not found in ventile breaks")
        return False
    
    breaks = ventile_breaks[glucose_code]
    
    # Expected: ref_range is [70.0, 105.0]
    if 70.0 not in breaks or 105.0 not in breaks:
        logger.error(f"Reference bounds [70.0, 105.0] not found in breaks")
        return False
    
    # Count breaks in each region
    below = [b for b in breaks if b < 70.0]
    within = [b for b in breaks if 70.0 < b < 105.0]
    above = [b for b in breaks if b > 105.0]
    
    if len(below) + 1 != 5:
        logger.error(f"Expected 5 bins below 70, got {len(below)+1}")
        return False
    
    if len(within) + 1 != 10:
        logger.error(f"Expected 10 bins within [70-105], got {len(within)+1}")
        return False
    
    if len(above) + 1 != 5:
        logger.error(f"Expected 5 bins above 105, got {len(above)+1}")
        return False
    
    logger.success(f"✓ Glucose has correct 5-10-5 binning: {len(below)+1}+{len(within)+1}+{len(above)+1}=20")
    return True


def test_analysis_outputs():
    """Test that analysis outputs were generated."""
    logger.info("Testing analysis outputs...")
    
    base_dir = Path(__file__).parent
    expected_files = [
        base_dir / "analysis/ventile_breaks_summary.csv",
        base_dir / "analysis/reference_range_coverage.csv",
    ]
    
    all_exist = True
    for filepath in expected_files:
        p = Path(filepath)
        if not p.exists():
            logger.error(f"Missing analysis file: {filepath}")
            all_exist = False
        else:
            size_kb = p.stat().st_size / 1024
            logger.success(f"✓ {p.name} ({size_kb:.1f} KB)")
    
    return all_exist


def main():
    """Run all tests."""
    logger.info("="*70)
    logger.info("VENTILE QUANTIZATION PIPELINE TEST SUITE")
    logger.info("="*70)
    
    tests = [
        ("MEDS Integration", test_meds_integration),
        ("Tokenization Preservation", test_tokenization_preservation),
        ("Ventile Quantization", test_ventile_quantization),
        ("Glucose Example", test_glucose_example),
        ("Analysis Outputs", test_analysis_outputs),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n{'─'*70}")
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            logger.error(f"Test '{test_name}' raised exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Print summary
    logger.info(f"\n{'='*70}")
    logger.info("TEST SUMMARY")
    logger.info("="*70)
    
    all_passed = True
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"  {status}: {test_name}")
        if not passed:
            all_passed = False
    
    logger.info("="*70)
    
    if all_passed:
        logger.success("\n🎉 ALL TESTS PASSED! Pipeline is working correctly.")
        return 0
    else:
        logger.error("\n❌ SOME TESTS FAILED. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

