#!/usr/bin/env python3
"""
Compute test-set positive-class prevalence for Table 8 footnote.

Reads pre-computed `tokens_timelines_outcomes.parquet` files (Exp3 ICU cohort) 
and directly computes outcomes from MEDS event parquets (Exp1-2 all-admission cohort).

Usage (from project root, in input-rep conda env):
    python utilities/scripts/compute_prevalence.py \
        --meds_events_dir data/clif/raw \
        --exp3_tokenized_dir artifacts/runs/exp3/meds_icu/deciles_none_unfused_time_rope_first_24h-tokenized

Outputs: prevalence percentages for each outcome in each cohort.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import polars as pl

# Re-use outcome computation from extract_outcomes_meds
from pipeline.scripts.extract_outcomes_meds import (
    _compute_outcomes_from_meds,
    _scan_meds_events,
    DEFAULT_IMV_ITEMIDS,
)


OUTCOME_COLS = [
    "same_admission_death",      # → Mortality
    "long_length_of_stay",       # → Long LoS (>7 days)  
    "icu_admission",             # → ICU Adm. (Exp 1-2)
    "imv_event",                 # → IMV
    "prolonged_icu_stay",        # → Prolonged ICU Stay (Exp 3)
]

DISPLAY_NAMES = {
    "same_admission_death": "Mortality",
    "long_length_of_stay": "Long LoS (>7d)",
    "icu_admission": "ICU Admission",
    "imv_event": "IMV",
    "prolonged_icu_stay": "Prolonged ICU Stay",
}


def _prevalence_from_outcomes(outcomes: pl.DataFrame, label: str) -> None:
    """Print prevalence for all outcome columns."""
    n = outcomes.height
    print(f"\n{'='*60}")
    print(f"{label}  (N = {n:,})")
    print(f"{'='*60}")
    for col in OUTCOME_COLS:
        if col not in outcomes.columns:
            print(f"  {DISPLAY_NAMES[col]:25s}  [column missing]")
            continue
        pos = outcomes[col].sum()
        if pos is None: pos = 0
        pct = pos / n * 100 if n > 0 else 0
        print(f"  {DISPLAY_NAMES[col]:25s}  {pos:>6,}/{n:>6,} = {pct:5.1f}%")
    print()


def main():
    parser = argparse.ArgumentParser(description="Compute prevalence for Table 8 footnote.")
    parser.add_argument(
        "--meds_events_dir", type=Path, default=None,
        help="MEDS events directory with train/val/test splits (for Exp1-2 all-admission cohort).",
    )
    parser.add_argument(
        "--exp3_tokenized_dir", type=Path, default=None,
        help="Exp3 tokenized directory containing test/tokens_timelines_outcomes.parquet.",
    )
    args = parser.parse_args()

    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    # ---------- Exp 1-2: All admissions >= 24h ----------
    meds_dir = args.meds_events_dir or PROJECT_ROOT / "data" / "clif" / "raw"
    test_split = meds_dir / "test"
    
    if test_split.exists():
        print("\n[Exp 1-2] Computing from MEDS events (CLIF raw format)...")
        # Load hospitalization for Mort/LoS
        hosp_path = test_split / "clif_hospitalization.parquet"
        if hosp_path.exists():
            hosp = pl.scan_parquet(hosp_path)
            
            # Compute Outcomes
            # Mortality: discharge_category is 'Dead/Expired' or similar? 
            # Check values: usually 'Expired', 'Dead'.
            # LoS: discharge - admission
            
            outcomes = hosp.select(
                pl.col("hospitalization_id"),
                same_admission_death=pl.col("discharge_category").str.contains("(?i)dead|expire|death"),
                length_of_stay=(pl.col("discharge_dttm") - pl.col("admission_dttm")).dt.total_hours(),
            ).with_columns(
                long_length_of_stay=pl.col("length_of_stay") > (7 * 24),
                icu_admission=pl.lit(None), # Missing in test set
                imv_event=pl.lit(None),     # Missing in test set
            ).collect()
            
            _prevalence_from_outcomes(outcomes, "Exp 1-2 test set (all admissions ≥ 24h)")
            print("(Note: ICU and IMV prevalence metrics unavailable for CLIF raw test set)")
            
        else:
            print(f"[Exp 1-2] SKIP: {hosp_path} not found")
    else:
        print(f"[Exp 1-2] SKIP: {test_split} not found")

    # ---------- Exp 3: ICU cohort ----------
    exp3_dir = args.exp3_tokenized_dir or (
        PROJECT_ROOT / "artifacts" / "runs" / "exp3" / "meds_icu"
        / "deciles_none_unfused_time_rope_first_24h-tokenized"
    )
    exp3_test = exp3_dir / "test" / "tokens_timelines_outcomes.parquet"
    
    if exp3_test.exists():
        print("[Exp 3] Reading pre-computed outcomes...")
        df = pl.read_parquet(exp3_test)
        _prevalence_from_outcomes(df, "Exp 3 test set (ICU cohort)")
    else:
        print(f"[Exp 3] SKIP: {exp3_test} not found")

    # ---------- Summary for paper ----------
    print("\n" + "="*60)
    print("COPY TO paper.tex Table 8 footnote:")
    print("="*60)
    print("Format: Mortality~$\\approx$\\,X\\%, Long LoS~$\\approx$\\,Y\\%, ...")
    print("(round to nearest integer)")


if __name__ == "__main__":
    main()
