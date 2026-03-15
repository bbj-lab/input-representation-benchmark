#!/usr/bin/env python3
"""
Compute Coefficient of Variation (CV) for pooled numeric values in mapped categories.
Compares True CLIF vs Randomized mapping vs Native.
"""
import json
from pathlib import Path
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    base_dir = Path(__file__).resolve().parents[2] / "artifacts" / "runs" / "exp3" / "arms"
    arms = ["meds_icu_native", "meds_mapped", "meds_randomized"]
    
    # We will focus on LAB and VITAL domains where numeric values are prominent
    domains = ["LAB", "VITAL"]
    
    stats_out = {}
    
    for arm in arms:
        print(f"Processing arm: {arm}")
        arm_dir = base_dir / arm
        if not arm_dir.exists():
            print(f"  Missing {arm_dir}")
            continue
            
        # load train split (where quantization happens)
        try:
            lf = pl.scan_parquet(arm_dir / "train" / "meds.parquet")
        except Exception as e:
            print(f"  Failed to load data: {e}")
            continue
            
        stats_out[arm] = {}
        for domain in domains:
            print(f"  Domain: {domain}")
            # get all codes in this domain
            df = (
                lf.filter(
                    pl.col("code").is_not_null() & 
                    pl.col("code").cast(pl.Utf8).str.starts_with(f"{domain}//") &
                    pl.col("numeric_value").is_not_null()
                )
                .select([
                    pl.col("code").cast(pl.Utf8).alias("code"),
                    pl.col("numeric_value").cast(pl.Float64)
                ])
                .collect(streaming=True)
            )
            
            # Compute grouped stats (mean, std, count) for each distinct pooled token
            grouped = df.group_by("code").agg([
                pl.col("numeric_value").mean().alias("mean"),
                pl.col("numeric_value").std().alias("std"),
                pl.count("numeric_value").alias("count")
            ])
            
            # Filter low-count bins to avoid noisy CV
            grouped = grouped.filter(pl.col("count") >= 10)
            
            # Calculate CV = std / |mean|
            # Add small epsilon to avoid div by zero, drop nulls
            grouped = grouped.with_columns(
                (pl.col("std") / (pl.col("mean").abs() + 1e-6)).alias("cv")
            ).drop_nulls("cv")
            
            # Record distribution of CVs
            cvs = grouped["cv"].to_numpy()
            if len(cvs) > 0:
                stats_out[arm][domain] = {
                    "median_cv": float(np.median(cvs)),
                    "mean_cv": float(np.mean(cvs)),
                    "25th_cv": float(np.percentile(cvs, 25)),
                    "75th_cv": float(np.percentile(cvs, 75)),
                    "num_categories_analyzed": len(cvs),
                    "total_events": int(grouped["count"].sum())
                }
                
    # Print results summary
    print("\n\n=== Coefficient of Variation (CV) Summary ===")
    print("Hypothesis: True CLIF groups semantically related codes, so within-category CV should be lower than Randomized (which pools arbitrary distributions).")
    
    for domain in domains:
        print(f"\n-- Domain: {domain} --")
        for arm in arms:
            if arm in stats_out and domain in stats_out[arm]:
                s = stats_out[arm][domain]
                print(f"  {arm:18s}: Median CV = {s['median_cv']:.3f} | Mean CV = {s['mean_cv']:.3f} (n={s['num_categories_analyzed']} categories from {s['total_events']:,} events)")

if __name__ == "__main__":
    main()
