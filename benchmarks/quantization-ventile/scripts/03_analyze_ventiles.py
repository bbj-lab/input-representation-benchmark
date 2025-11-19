#!/usr/bin/env python3
"""Analyze ventile breaks and reference range usage.

This script analyzes the computed ventile breaks to understand how reference
ranges were used in the quantization process.

Usage:
    python 03_analyze_ventiles.py
"""

import json
from pathlib import Path

import polars as pl
from loguru import logger


def analyze_ventile_breaks(ventile_breaks_path: Path, output_dir: Path):
    """Analyze ventile breaks and categorize by reference range strategy."""
    
    logger.info(f"Loading ventile breaks from {ventile_breaks_path}")
    with open(ventile_breaks_path) as f:
        ventile_breaks = json.load(f)
    
    logger.info(f"Found {len(ventile_breaks)} codes with ventile breaks")
    
    # Categorize breaks by number of bins
    bin_counts = {}
    for code, breaks in ventile_breaks.items():
        n_bins = len(breaks) + 1
        if n_bins not in bin_counts:
            bin_counts[n_bins] = []
        bin_counts[n_bins].append(code)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("VENTILE BREAKS SUMMARY")
    logger.info("="*60)
    logger.info(f"Total codes with quantization: {len(ventile_breaks)}")
    logger.info("\nDistribution by number of bins:")
    for n_bins in sorted(bin_counts.keys()):
        count = len(bin_counts[n_bins])
        pct = count / len(ventile_breaks) * 100
        logger.info(f"  {n_bins:2d} bins: {count:4d} codes ({pct:5.1f}%)")
    
    # Show example codes with different bin counts
    logger.info("\nExample codes by bin count:")
    for n_bins in sorted(bin_counts.keys())[:5]:
        examples = bin_counts[n_bins][:3]
        logger.info(f"\n  {n_bins} bins:")
        for code in examples:
            breaks = ventile_breaks[code]
            logger.info(f"    {code}")
            logger.info(f"      Breaks: {breaks[:3]}...{breaks[-3:]} (showing first/last 3)")
    
    # Save detailed analysis
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create summary DataFrame
    summary_data = []
    for code, breaks in ventile_breaks.items():
        summary_data.append({
            "code": code,
            "n_bins": len(breaks) + 1,
            "n_breaks": len(breaks),
            "min_break": min(breaks) if breaks else None,
            "max_break": max(breaks) if breaks else None,
        })
    
    summary_df = pl.DataFrame(summary_data).sort("n_bins", descending=True)
    
    output_file = output_dir / "ventile_breaks_summary.csv"
    summary_df.write_csv(output_file)
    logger.info(f"\n✓ Detailed summary saved to: {output_file}")
    
    # Show top 20 most frequent lab codes (LAB//Q// prefix)
    lab_codes = [row for row in summary_data if "LAB//Q//" in row["code"]]
    if lab_codes:
        logger.info(f"\nTop 20 lab codes with ventile breaks:")
        for i, row in enumerate(lab_codes[:20], 1):
            logger.info(f"  {i:2d}. {row['code']:40s} - {row['n_bins']:2d} bins")
    
    logger.info("="*60)


def analyze_meds_reference_ranges(meds_dir: Path, output_dir: Path):
    """Analyze reference range coverage in MEDS data."""
    
    logger.info(f"\nAnalyzing reference ranges in {meds_dir}")
    
    # Scan ALL training files for comprehensive analysis
    train_files = list((meds_dir / "train").glob("*.parquet"))
    if not train_files:
        logger.warning("No training data found")
        return
    
    logger.info(f"Found {len(train_files)} training files - analyzing all files for complete statistics")
    
    dfs = []
    for fp in train_files:
        df = pl.scan_parquet(fp).filter(
            pl.col("code").str.starts_with("LAB//")
        )
        
        # Check schema
        schema = df.collect_schema()
        if "ref_range_lower" in schema.names() and "ref_range_upper" in schema.names():
            df = df.select([
                "code",
                "numeric_value",
                "ref_range_lower",
                "ref_range_upper",
            ])
            dfs.append(df)
    
    if not dfs:
        logger.warning("No lab data with reference ranges found")
        return
    
    logger.info("Computing reference range statistics across all training data...")
    combined = pl.concat(dfs).collect()
    
    # Compute detailed statistics per lab code
    summary = combined.group_by("code").agg([
        pl.len().alias("total_events"),
        pl.col("numeric_value").is_not_null().sum().alias("events_with_numeric"),
        pl.col("ref_range_lower").is_not_null().sum().alias("events_with_lower"),
        pl.col("ref_range_upper").is_not_null().sum().alias("events_with_upper"),
        (
            pl.col("ref_range_lower").is_not_null() 
            & pl.col("ref_range_upper").is_not_null()
        ).sum().alias("events_with_both"),
        (
            pl.col("ref_range_lower").is_null() 
            & pl.col("ref_range_upper").is_null()
        ).sum().alias("events_with_neither"),
        (
            pl.col("ref_range_lower").is_not_null() 
            & pl.col("ref_range_upper").is_null()
        ).sum().alias("events_with_lower_only"),
        (
            pl.col("ref_range_lower").is_null() 
            & pl.col("ref_range_upper").is_not_null()
        ).sum().alias("events_with_upper_only"),
        pl.col("ref_range_lower").drop_nulls().first().alias("ref_lower"),
        pl.col("ref_range_upper").drop_nulls().first().alias("ref_upper"),
    ]).with_columns([
        (pl.col("events_with_both") / pl.col("total_events") * 100).alias("pct_both"),
        (pl.col("events_with_lower_only") / pl.col("total_events") * 100).alias("pct_lower_only"),
        (pl.col("events_with_upper_only") / pl.col("total_events") * 100).alias("pct_upper_only"),
        (pl.col("events_with_neither") / pl.col("total_events") * 100).alias("pct_neither"),
    ]).sort("total_events", descending=True)
    
    # Print comprehensive summary
    total_codes = len(summary)
    total_events = summary["total_events"].sum()
    
    # Categorize codes by reference range availability
    codes_both = (summary["pct_both"] > 0).sum()
    codes_lower_only = (summary["pct_lower_only"] > 0).sum()
    codes_upper_only = (summary["pct_upper_only"] > 0).sum()
    codes_neither = (summary["pct_neither"] == 100).sum()
    
    # Event-level statistics
    events_both = summary["events_with_both"].sum()
    events_lower_only = summary["events_with_lower_only"].sum()
    events_upper_only = summary["events_with_upper_only"].sum()
    events_neither = summary["events_with_neither"].sum()
    
    logger.info("\n" + "-"*70)
    logger.info("REFERENCE RANGE COVERAGE ANALYSIS - MIMIC-IV v3.1")
    logger.info("-"*70)
    logger.info(f"\nDataset Summary:")
    logger.info(f"  Total lab codes: {total_codes}")
    logger.info(f"  Total lab events: {total_events:,}")
    
    logger.info(f"\nCoverage by Lab Code:")
    logger.info(f"  1. Both bounds (lower AND upper):")
    logger.info(f"     - Codes: {codes_both} ({codes_both/total_codes*100:.1f}%)")
    logger.info(f"     - Strategy: 5 bins below + 10 within + 5 above = 20 bins")
    logger.info(f"  2. Lower bound only:")
    logger.info(f"     - Codes: {codes_lower_only} ({codes_lower_only/total_codes*100:.1f}%)")
    logger.info(f"     - Strategy: 5 bins below + 15 above = 20 bins")
    logger.info(f"  3. Upper bound only:")
    logger.info(f"     - Codes: {codes_upper_only} ({codes_upper_only/total_codes*100:.1f}%)")
    logger.info(f"     - Strategy: 15 bins below + 5 above = 20 bins")
    logger.info(f"  4. No reference bounds:")
    logger.info(f"     - Codes: {codes_neither} ({codes_neither/total_codes*100:.1f}%)")
    logger.info(f"     - Strategy: Standard 20-bin ventile")
    
    logger.info(f"\nCoverage by Lab Event:")
    logger.info(f"  1. Both bounds: {events_both:,} events ({events_both/total_events*100:.1f}%)")
    logger.info(f"  2. Lower only: {events_lower_only:,} events ({events_lower_only/total_events*100:.1f}%)")
    logger.info(f"  3. Upper only: {events_upper_only:,} events ({events_upper_only/total_events*100:.1f}%)")
    logger.info(f"  4. No bounds: {events_neither:,} events ({events_neither/total_events*100:.1f}%)")
    
    # Show examples from each category
    logger.info(f"\nExample Labs by Reference Range Category:")
    
    both_examples = summary.filter(pl.col("pct_both") > 0).head(5)
    logger.info(f"\n  Both bounds (5 examples):")
    for row in both_examples.iter_rows(named=True):
        logger.info(
            f"    {row['code']:45s} [{row['ref_lower']:6.1f}, {row['ref_upper']:6.1f}] "
            f"({row['total_events']:,} events)"
        )
    
    lower_only = summary.filter(pl.col("pct_lower_only") > 0).head(3)
    if len(lower_only) > 0:
        logger.info(f"\n  Lower bound only (3 examples):")
        for row in lower_only.iter_rows(named=True):
            logger.info(
                f"    {row['code']:45s} lower={row['ref_lower']:6.1f} "
                f"({row['total_events']:,} events)"
            )
    
    upper_only = summary.filter(pl.col("pct_upper_only") > 0).head(3)
    if len(upper_only) > 0:
        logger.info(f"\n  Upper bound only (3 examples):")
        for row in upper_only.iter_rows(named=True):
            logger.info(
                f"    {row['code']:45s} upper={row['ref_upper']:6.1f} "
                f"({row['total_events']:,} events)"
            )
    
    neither = summary.filter(pl.col("pct_neither") == 100).head(3)
    if len(neither) > 0:
        logger.info(f"\n  No bounds (3 examples):")
        for row in neither.iter_rows(named=True):
            logger.info(f"    {row['code']:45s} ({row['total_events']:,} events)")
    
    # Save detailed analysis
    output_file = output_dir / "reference_range_coverage.csv"
    summary.write_csv(output_file)
    
    logger.info(f"\n" + "-"*70)
    logger.info(f"Detailed coverage analysis saved to: {output_file}")
    logger.info("-"*70)


def main():
    """Main analysis function."""
    # Get paths
    # scripts/ -> ventile-quantization/ -> benchmarks/ -> root/
    # We assume this script is run from benchmarks/ventile-quantization/
    # cwd = Path.cwd()
    
    experiment_dir = Path(__file__).parent.parent
    
    tokenized_dir = experiment_dir / "data/tokenized/train"
    meds_dir = experiment_dir / "data/meds/data"
    analysis_dir = experiment_dir / "analysis"
    
    logger.info("="*60)
    logger.info("Input Representation Benchmark - Ventile Analysis")
    logger.info("="*60)
    
    # Analyze quantile breaks (may be ventile_breaks.json or quantiles.json)
    ventile_breaks_path = tokenized_dir / "ventile_breaks.json"
    quantiles_path = tokenized_dir / "quantiles.json"
    
    if ventile_breaks_path.exists():
        logger.info("Analyzing ventile breaks (custom ventile quantization)")
        analyze_ventile_breaks(ventile_breaks_path, analysis_dir)
    elif quantiles_path.exists():
        logger.info("Analyzing quantiles (standard ethos quantization with num_quantiles=20)")
        analyze_ventile_breaks(quantiles_path, analysis_dir)
    else:
        logger.warning(f"No quantile breaks found at {ventile_breaks_path} or {quantiles_path}")
        logger.warning("Please run tokenization first")
    
    # Analyze reference ranges in MEDS data
    if meds_dir.exists():
        analyze_meds_reference_ranges(meds_dir, analysis_dir)
    else:
        logger.warning(f"MEDS data not found at {meds_dir}")
    
    logger.info("\n✓ Analysis complete!")
    logger.info(f"Analysis saved to: {analysis_dir}")


if __name__ == "__main__":
    main()

