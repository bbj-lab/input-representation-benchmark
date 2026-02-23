#!/usr/bin/env python3
"""
Extended outcome extraction (Tier 2 & 3) for regression and expanded classification.

Companion to `extract_outcomes_meds.py`.  That script produces the canonical
binary outcomes (mortality, LOS, ICU admission, IMV).  This script adds
continuous/ordinal targets suitable for regression-based probing and
threshold-based binary expansions.

Outputs (per hospitalization, joined onto existing outcomes parquet)
-------------------------------------------------------------------
Continuous regression targets (Tier 2):
    - peak_creatinine        : max creatinine during admission (mg/dL)
    - peak_troponin          : max troponin T or I during admission (ng/mL)
    - min_hemoglobin         : min hemoglobin during admission (g/dL)
    - peak_potassium         : max potassium during admission (mEq/L)
    - min_glucose            : min glucose during admission (mg/dL)
    - peak_bnp               : max BNP or NT-proBNP during admission (pg/mL)
    - time_to_icu_hours      : hours from hospital admission to first ICU admission
                               (null if no ICU admission)

Binary outcomes (Tier 3):
    - hyperkalemia           : peak potassium > 6.0 mEq/L
    - severe_anemia          : min hemoglobin < 7.0 g/dL
    - hypoglycemia           : min glucose < 50 mg/dL
    - vasopressor_initiation : any vasopressor infusion during admission
    - hypotension            : any MAP < 65 mmHg or SBP < 90 mmHg during admission

MIMIC-IV itemid reference
-------------------------
Labs (hosp/labevents → code "LAB//{itemid}//{uom}"):
    50912  : Creatinine (mg/dL)
    51003  : Troponin T (ng/mL)
    51002  : Troponin I (ng/mL)                 [less common]
    51222  : Hemoglobin (g/dL)       [blood gas]
    50811  : Hemoglobin (g/dL)       [CBC]
    50971  : Potassium (mEq/L)       [blood gas]
    50822  : Potassium (mEq/L)       [chemistry]
    50931  : Glucose (mg/dL)         [blood gas]
    50809  : Glucose (mg/dL)         [chemistry]
    50963  : BNP (pg/mL)
    50964  : NT-proBNP (pg/mL)

Vitals (icu/chartevents → code "VITAL//{itemid}//{uom}"):
    220052 : Arterial Blood Pressure mean (mmHg)        [MAP]
    220181 : Non Invasive Blood Pressure mean (mmHg)     [NIBP MAP]
    220050 : Arterial Blood Pressure systolic (mmHg)
    220179 : Non Invasive Blood Pressure systolic (mmHg)
    225309 : ART BP Systolic

Vasopressors (icu/inputevents → code "INFUSION_START//{itemid}"):
    221906 : Norepinephrine
    221289 : Epinephrine
    222315 : Vasopressin
    221749 : Phenylephrine
    229617 : Dopamine              [MIMIC-IV v3+]
    221662 : Dopamine              [legacy]
    221653 : Dobutamine

Usage
-----
python scripts/extract_extended_outcomes.py \\
    --meds_events_dir /path/to/meds/data \\
    --tokenized_dir /path/to/tokenized \\
    --base_outcomes_filename tokens_timelines_outcomes.parquet \\
    --output_filename tokens_timelines_extended_outcomes.parquet

The script reads the base outcomes parquet (which already has hospitalization_id +
all columns from extract_outcomes_meds.py) and LEFT JOINs the new columns.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import polars as pl

# ---------------------------------------------------------------------------
# MIMIC-IV itemid constants
# ---------------------------------------------------------------------------

CREATININE_ITEMIDS = (50912,)
TROPONIN_ITEMIDS = (51003, 51002)
HEMOGLOBIN_ITEMIDS = (51222, 50811)
POTASSIUM_ITEMIDS = (50971, 50822)
GLUCOSE_ITEMIDS = (50931, 50809)
BNP_ITEMIDS = (50963, 50964)

VASOPRESSOR_ITEMIDS = (221906, 221289, 222315, 221749, 229617, 221662, 221653)

# MAP
MAP_ITEMIDS = (220052, 220181)
# SBP
SBP_ITEMIDS = (220050, 220179, 225309)

# ---------------------------------------------------------------------------
# Helpers (duplicated from extract_outcomes_meds.py to keep this standalone)
# ---------------------------------------------------------------------------


def _existing_split_dir(base: Path, split: str) -> Path | None:
    p = base / split
    return p if p.exists() and p.is_dir() else None


def _resolve_meds_split_dir(meds_events_dir: Path, tokenized_split: str) -> tuple[str, Path | None]:
    if (p := _existing_split_dir(meds_events_dir, tokenized_split)) is not None:
        return tokenized_split, p
    aliases: dict[str, Sequence[str]] = {"val": ("tuning",), "tuning": ("val",)}
    for alt in aliases.get(tokenized_split, ()):
        if (p := _existing_split_dir(meds_events_dir, alt)) is not None:
            return alt, p
    return tokenized_split, None


def _scan_meds_events(split_dir: Path) -> pl.LazyFrame:
    meds_parquet = split_dir / "meds.parquet"
    if meds_parquet.exists():
        return pl.scan_parquet(meds_parquet)
    return pl.scan_parquet(str(split_dir / "*.parquet"))


def _extract_itemid(code_col: str = "code") -> pl.Expr:
    """Extract the integer itemid from a MEDS code like 'LAB//50912//mg/dL'."""
    return pl.col(code_col).str.split("//").list.get(1).cast(pl.Int64, strict=False)


# ---------------------------------------------------------------------------
# Lab-value extraction
# ---------------------------------------------------------------------------


def _lab_extremes(
    meds: pl.LazyFrame,
    itemids: tuple[int, ...],
    agg: str,
    col_name: str,
) -> pl.LazyFrame:
    """
    Extract per-admission peak (max) or trough (min) lab value.

    Parameters
    ----------
    meds : LazyFrame with columns (hadm_id, code, numeric_value)
    itemids : tuple of MIMIC itemids to filter on
    agg : 'max' or 'min'
    col_name : output column name
    """
    agg_expr = pl.col("numeric_value").max() if agg == "max" else pl.col("numeric_value").min()
    return (
        meds.filter(pl.col("hadm_id").is_not_null())
        .filter(pl.col("code").str.starts_with("LAB//"))
        .with_columns(_extract_itemid().alias("itemid"))
        .filter(pl.col("itemid").is_in(list(itemids)))
        .filter(pl.col("numeric_value").is_not_null())
        .group_by("hadm_id")
        .agg(agg_expr.alias(col_name))
    )


# ---------------------------------------------------------------------------
# Vital-sign extraction
# ---------------------------------------------------------------------------


def _vital_extremes(
    meds: pl.LazyFrame,
    itemids: tuple[int, ...],
    agg: str,
    col_name: str,
) -> pl.LazyFrame:
    """Extract per-admission min/max vital sign value."""
    agg_expr = pl.col("numeric_value").max() if agg == "max" else pl.col("numeric_value").min()
    return (
        meds.filter(pl.col("hadm_id").is_not_null())
        .filter(pl.col("code").str.starts_with("VITAL//"))
        .with_columns(_extract_itemid().alias("itemid"))
        .filter(pl.col("itemid").is_in(list(itemids)))
        .filter(pl.col("numeric_value").is_not_null())
        .group_by("hadm_id")
        .agg(agg_expr.alias(col_name))
    )


# ---------------------------------------------------------------------------
# Main outcome computation
# ---------------------------------------------------------------------------


def _compute_extended_outcomes(meds: pl.LazyFrame) -> pl.LazyFrame:
    """
    Compute extended outcomes keyed by `hospitalization_id` (string hadm_id).

    Returns a LazyFrame with one row per hadm_id.
    """
    # Keep only columns we need (avoid scanning wide frames)
    meds = meds.select("subject_id", "hadm_id", "time", "code", "numeric_value")

    # ---- Tier 2: Continuous regression targets ----

    peak_creatinine = _lab_extremes(meds, CREATININE_ITEMIDS, "max", "peak_creatinine")
    peak_troponin = _lab_extremes(meds, TROPONIN_ITEMIDS, "max", "peak_troponin")
    min_hemoglobin = _lab_extremes(meds, HEMOGLOBIN_ITEMIDS, "min", "min_hemoglobin")
    peak_potassium = _lab_extremes(meds, POTASSIUM_ITEMIDS, "max", "peak_potassium")
    min_glucose = _lab_extremes(meds, GLUCOSE_ITEMIDS, "min", "min_glucose")
    peak_bnp = _lab_extremes(meds, BNP_ITEMIDS, "max", "peak_bnp")

    # Time-to-ICU (hours from hospital admission to first ICU admission)
    admissions = (
        meds.filter(pl.col("hadm_id").is_not_null())
        .filter(pl.col("code").str.starts_with("HOSPITAL_ADMISSION"))
        .group_by("hadm_id")
        .agg(admission_time=pl.col("time").min())
    )
    icu_adm = (
        meds.filter(pl.col("hadm_id").is_not_null())
        .filter(pl.col("code").str.starts_with("ICU_ADMISSION"))
        .group_by("hadm_id")
        .agg(icu_admission_time=pl.col("time").min())
    )
    time_to_icu = (
        admissions.join(icu_adm, on="hadm_id", how="left", maintain_order="left")
        .with_columns(
            time_to_icu_hours=(
                (pl.col("icu_admission_time") - pl.col("admission_time"))
                .dt.total_seconds()
                / 3600.0
            )
        )
        .select("hadm_id", "time_to_icu_hours")
    )

    # ---- Tier 3: Binary expansions ----

    # Vasopressor initiation (any vasopressor infusion)
    vasopressor = (
        meds.filter(pl.col("hadm_id").is_not_null())
        .filter(pl.col("code").str.starts_with("INFUSION_START//"))
        .with_columns(_extract_itemid().alias("itemid"))
        .filter(pl.col("itemid").is_in(list(VASOPRESSOR_ITEMIDS)))
        .group_by("hadm_id")
        .agg(vasopressor_initiation=pl.lit(True))
    )

    # Hypotension: any MAP < 65 or SBP < 90
    min_map = _vital_extremes(meds, MAP_ITEMIDS, "min", "min_map")
    min_sbp = _vital_extremes(meds, SBP_ITEMIDS, "min", "min_sbp")

    # ---- Assemble ----

    # Start from all hadm_ids we know about
    all_hadm = (
        meds.filter(pl.col("hadm_id").is_not_null())
        .select("hadm_id")
        .unique()
    )

    result = all_hadm
    for df in [
        peak_creatinine,
        peak_troponin,
        min_hemoglobin,
        peak_potassium,
        min_glucose,
        peak_bnp,
        time_to_icu,
        vasopressor,
        min_map,
        min_sbp,
    ]:
        result = result.join(df, on="hadm_id", how="left", maintain_order="left")

    # Compute derived binary columns
    result = result.with_columns(
        # Tier 3: threshold-based binary outcomes
        hyperkalemia=(pl.col("peak_potassium") > 6.0).fill_null(False),
        severe_anemia=(pl.col("min_hemoglobin") < 7.0).fill_null(False),
        hypoglycemia=(pl.col("min_glucose") < 50.0).fill_null(False),
        vasopressor_initiation=pl.col("vasopressor_initiation").fill_null(False),
        hypotension=(
            (pl.col("min_map") < 65.0) | (pl.col("min_sbp") < 90.0)
        ).fill_null(False),
    )

    # Standardize key column and drop intermediate vitals
    result = (
        result
        .with_columns(pl.col("hadm_id").cast(pl.String).alias("hospitalization_id"))
        .drop("hadm_id", "min_map", "min_sbp")
        .sort("hospitalization_id")
    )

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract extended outcomes (Tier 2 regression + Tier 3 binary) from MEDS events."
    )
    parser.add_argument(
        "--meds_events_dir",
        type=Path,
        required=True,
        help="Directory containing MEDS parquet shards per split.",
    )
    parser.add_argument(
        "--tokenized_dir",
        type=Path,
        required=True,
        help="Directory containing tokenized split subdirs.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,val,test",
        help="Comma-separated split names (default: train,val,test).",
    )
    parser.add_argument(
        "--base_outcomes_filename",
        type=str,
        default="tokens_timelines_outcomes.parquet",
        help="Input parquet with base outcomes to augment (default: tokens_timelines_outcomes.parquet).",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="tokens_timelines_extended_outcomes.parquet",
        help="Output parquet filename (default: tokens_timelines_extended_outcomes.parquet).",
    )
    args = parser.parse_args(argv)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    meds_events_dir = args.meds_events_dir.expanduser().resolve()
    tokenized_dir = args.tokenized_dir.expanduser().resolve()

    for split in splits:
        tok_split_dir = tokenized_dir / split
        base_path = tok_split_dir / args.base_outcomes_filename
        if not base_path.exists():
            print(f"[extract_extended_outcomes] Skip split={split}: missing {base_path}")
            continue

        meds_split_name, meds_split_dir = _resolve_meds_split_dir(meds_events_dir, split)
        if meds_split_dir is None:
            print(
                f"[extract_extended_outcomes] Skip split={split}: no MEDS split dir "
                f"under {meds_events_dir}"
            )
            continue

        print(f"[extract_extended_outcomes] Processing split={split} (meds={meds_split_name})...")

        meds = _scan_meds_events(meds_split_dir)
        extended = _compute_extended_outcomes(meds)

        # Load base outcomes and join
        base = pl.scan_parquet(base_path).with_columns(
            pl.col("hospitalization_id").cast(pl.String)
        )

        out = base.join(
            extended,
            on="hospitalization_id",
            how="left",
            maintain_order="left",
        )

        output_path = tok_split_dir / args.output_filename
        out.sink_parquet(output_path)

        # Quick summary statistics
        collected = pl.scan_parquet(output_path)
        n_total = collected.select(pl.len()).collect().item()

        stat_cols = [
            "peak_creatinine", "peak_troponin", "min_hemoglobin",
            "peak_potassium", "min_glucose", "peak_bnp", "time_to_icu_hours",
        ]
        binary_cols = [
            "hyperkalemia", "severe_anemia", "hypoglycemia",
            "vasopressor_initiation", "hypotension",
        ]

        print(f"  Wrote {output_path} ({n_total} rows)")
        for col in stat_cols:
            try:
                stats = (
                    collected
                    .select(
                        pl.col(col).is_not_null().sum().alias("n_valid"),
                        pl.col(col).mean().alias("mean"),
                        pl.col(col).median().alias("median"),
                        pl.col(col).std().alias("std"),
                    )
                    .collect()
                    .row(0, named=True)
                )
                print(
                    f"  {col}: n={stats['n_valid']}, "
                    f"mean={stats['mean']:.2f}, median={stats['median']:.2f}, "
                    f"std={stats['std']:.2f}"
                )
            except Exception:
                print(f"  {col}: (could not compute stats)")

        for col in binary_cols:
            try:
                ct = (
                    collected
                    .select(pl.col(col).sum().alias("pos"), pl.len().alias("total"))
                    .collect()
                    .row(0, named=True)
                )
                print(f"  {col}: {ct['pos']}/{ct['total']} ({100*ct['pos']/ct['total']:.1f}%)")
            except Exception:
                print(f"  {col}: (could not compute prevalence)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
