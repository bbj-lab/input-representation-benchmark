#!/usr/bin/env python3
"""
Extended outcome extraction for regression and additional binary evaluation.

Companion to `extract_outcomes_meds.py`.  That script produces the canonical
binary outcomes (mortality, LOS, ICU admission, IMV).  This script adds
continuous/ordinal outcomes suitable for regression-based probing and
threshold-based additional binary outcomes.

Outputs (per hospitalization, joined onto existing outcomes parquet)
-------------------------------------------------------------------
Continuous regression outcomes:
    - peak_creatinine        : max creatinine after 24h (mg/dL)
    - peak_troponin          : max troponin T or I after 24h (ng/mL)
    - min_hemoglobin         : min hemoglobin after 24h (g/dL)
    - peak_potassium         : max potassium after 24h (mEq/L)
    - min_potassium          : min potassium after 24h (mEq/L)
    - min_glucose            : min glucose after 24h (mg/dL)
    - min_sodium             : min sodium after 24h (mEq/L)
    - max_sodium             : max sodium after 24h (mEq/L)
    - peak_bnp               : max BNP or NT-proBNP after 24h (pg/mL)
    - max_heart_rate         : max heart rate after 24h (bpm)
    - max_sbp               : max systolic blood pressure after 24h (mmHg)
    - max_dbp               : max diastolic blood pressure after 24h (mmHg)

Additional binary outcomes:
    - hyperkalemia           : peak potassium after 24h >= 6.5 mEq/L
    - severe_hypokalemia     : min potassium after 24h < 2.5 mEq/L
    - severe_anemia          : min hemoglobin after 24h < 7.0 g/dL
    - hypoglycemia           : min glucose after 24h < 54 mg/dL
    - profound_hyponatremia  : min sodium after 24h < 125 mEq/L
    - severe_hypernatremia   : max sodium after 24h >= 160 mEq/L
    - tachycardia_hr130      : max heart rate after 24h >= 130 bpm
    - severe_hypertension    : max SBP >= 180 mmHg or max DBP >= 120 mmHg after 24h
    - vasopressor_initiation : any vasopressor infusion after 24h
    - hypotension            : any MAP < 65 mmHg or SBP < 90 mmHg after 24h
    - crrt_initiation        : any CRRT event after 24h
    - hemodialysis_initiation: any hemodialysis event after 24h

For measurement-based binary outcomes, admissions with no qualifying post-24h
measurement are left null (unobserved), not forced negative. For binary
outcomes, parallel `*_24h` columns record whether the admission already
satisfied the endpoint within the first 24h so downstream evaluation can
exclude those admissions to prevent temporal leakage.

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
SODIUM_ITEMIDS = (50983, 50824, 52623)

VASOPRESSOR_ITEMIDS = (221906, 221289, 222315, 221749, 229617, 221662, 221653)
CRRT_CHARTEVENT_ITEMIDS = (227290,)
CRRT_PROCEDURE_ITEMIDS = (225802,)
HEMODIALYSIS_CHARTEVENT_ITEMIDS = (226499,)
HEMODIALYSIS_PROCEDURE_ITEMIDS = (225441,)

# MAP
MAP_ITEMIDS = (220052, 220181)
# SBP
SBP_ITEMIDS = (220050, 220179, 225309)
# DBP
DBP_ITEMIDS = (220180, 220051, 225310, 224643, 227242)
# HR
HEART_RATE_ITEMIDS = (220045,)

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


def _admission_times(meds: pl.LazyFrame) -> pl.LazyFrame:
    return (
        meds.filter(pl.col("hadm_id").is_not_null())
        .filter(pl.col("code").str.starts_with("HOSPITAL_ADMISSION"))
        .group_by("hadm_id")
        .agg(admission_time=pl.col("time").min())
    )


def _with_observation_window(
    meds: pl.LazyFrame,
    admissions: pl.LazyFrame,
    *,
    observation_hours: float,
    within_window: bool,
) -> pl.LazyFrame:
    obs = pl.duration(hours=float(observation_hours))
    cmp = (
        pl.col("time") <= (pl.col("admission_time") + obs)
        if within_window
        else pl.col("time") > (pl.col("admission_time") + obs)
    )
    return (
        meds.filter(pl.col("hadm_id").is_not_null())
        .join(admissions, on="hadm_id", how="inner", maintain_order="left")
        .filter(cmp)
    )


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


def _binary_event_presence(
    meds: pl.LazyFrame,
    *,
    prefix: str,
    itemids: tuple[int, ...],
    col_name: str,
) -> pl.LazyFrame:
    return (
        meds.filter(pl.col("hadm_id").is_not_null())
        .filter(pl.col("code").str.starts_with(prefix))
        .with_columns(_extract_itemid().alias("itemid"))
        .filter(pl.col("itemid").is_in(list(itemids)))
        .group_by("hadm_id")
        .agg(pl.lit(True).alias(col_name))
    )


def _binary_event_presence_multi(
    meds: pl.LazyFrame,
    *,
    specs: list[tuple[str, tuple[int, ...]]],
    col_name: str,
) -> pl.LazyFrame:
    parts: list[pl.LazyFrame] = []
    for prefix, itemids in specs:
        parts.append(
            meds.filter(pl.col("hadm_id").is_not_null())
            .filter(pl.col("code").str.starts_with(prefix))
            .with_columns(_extract_itemid().alias("itemid"))
            .filter(pl.col("itemid").is_in(list(itemids)))
            .select("hadm_id")
        )
    if not parts:
        return pl.LazyFrame({"hadm_id": [], col_name: []}, schema={"hadm_id": pl.Int64, col_name: pl.Boolean})
    return (
        pl.concat(parts, how="vertical")
        .unique()
        .group_by("hadm_id")
        .agg(pl.lit(True).alias(col_name))
    )


# ---------------------------------------------------------------------------
# Main outcome computation
# ---------------------------------------------------------------------------


def _compute_extended_outcomes(
    meds: pl.LazyFrame,
    *,
    observation_hours: float = 24.0,
) -> pl.LazyFrame:
    """
    Compute extended outcomes keyed by `hospitalization_id` (string hadm_id).

    Returns a LazyFrame with one row per hadm_id.
    """
    # Keep only columns we need (avoid scanning wide frames)
    meds = meds.select("subject_id", "hadm_id", "time", "code", "numeric_value")
    admissions = _admission_times(meds)
    meds_pre24h = _with_observation_window(
        meds, admissions, observation_hours=observation_hours, within_window=True
    )
    meds_post24h = _with_observation_window(
        meds, admissions, observation_hours=observation_hours, within_window=False
    )

    # ---- Regression outcomes ----

    peak_creatinine = _lab_extremes(
        meds_post24h, CREATININE_ITEMIDS, "max", "peak_creatinine"
    )
    peak_troponin = _lab_extremes(
        meds_post24h, TROPONIN_ITEMIDS, "max", "peak_troponin"
    )
    min_hemoglobin = _lab_extremes(
        meds_post24h, HEMOGLOBIN_ITEMIDS, "min", "min_hemoglobin"
    )
    peak_potassium = _lab_extremes(
        meds_post24h, POTASSIUM_ITEMIDS, "max", "peak_potassium"
    )
    min_potassium = _lab_extremes(
        meds_post24h, POTASSIUM_ITEMIDS, "min", "min_potassium"
    )
    min_glucose = _lab_extremes(
        meds_post24h, GLUCOSE_ITEMIDS, "min", "min_glucose"
    )
    min_sodium = _lab_extremes(
        meds_post24h, SODIUM_ITEMIDS, "min", "min_sodium"
    )
    max_sodium = _lab_extremes(
        meds_post24h, SODIUM_ITEMIDS, "max", "max_sodium"
    )
    peak_bnp = _lab_extremes(meds_post24h, BNP_ITEMIDS, "max", "peak_bnp")
    max_heart_rate = _vital_extremes(
        meds_post24h, HEART_RATE_ITEMIDS, "max", "max_heart_rate"
    )
    max_sbp = _vital_extremes(meds_post24h, SBP_ITEMIDS, "max", "max_sbp")
    max_dbp = _vital_extremes(meds_post24h, DBP_ITEMIDS, "max", "max_dbp")

    # First-24h extrema for leakage-safe exclusion flags
    peak_potassium_24h = _lab_extremes(
        meds_pre24h, POTASSIUM_ITEMIDS, "max", "peak_potassium_24h"
    )
    min_potassium_24h = _lab_extremes(
        meds_pre24h, POTASSIUM_ITEMIDS, "min", "min_potassium_24h"
    )
    min_hemoglobin_24h = _lab_extremes(
        meds_pre24h, HEMOGLOBIN_ITEMIDS, "min", "min_hemoglobin_24h"
    )
    min_glucose_24h = _lab_extremes(
        meds_pre24h, GLUCOSE_ITEMIDS, "min", "min_glucose_24h"
    )
    min_sodium_24h = _lab_extremes(
        meds_pre24h, SODIUM_ITEMIDS, "min", "min_sodium_24h"
    )
    max_sodium_24h = _lab_extremes(
        meds_pre24h, SODIUM_ITEMIDS, "max", "max_sodium_24h"
    )
    max_heart_rate_24h = _vital_extremes(
        meds_pre24h, HEART_RATE_ITEMIDS, "max", "max_heart_rate_24h"
    )
    max_sbp_24h = _vital_extremes(
        meds_pre24h, SBP_ITEMIDS, "max", "max_sbp_24h"
    )
    max_dbp_24h = _vital_extremes(
        meds_pre24h, DBP_ITEMIDS, "max", "max_dbp_24h"
    )

    # ---- Additional binary outcomes ----

    vasopressor = _binary_event_presence(
        meds_post24h,
        prefix="INFUSION_START//",
        itemids=VASOPRESSOR_ITEMIDS,
        col_name="vasopressor_initiation",
    )
    vasopressor_24h = _binary_event_presence(
        meds_pre24h,
        prefix="INFUSION_START//",
        itemids=VASOPRESSOR_ITEMIDS,
        col_name="vasopressor_initiation_24h",
    )
    crrt = _binary_event_presence_multi(
        meds_post24h,
        specs=[
            ("VITAL//", CRRT_CHARTEVENT_ITEMIDS),
            ("PROCEDURE//", CRRT_PROCEDURE_ITEMIDS),
        ],
        col_name="crrt_initiation",
    )
    crrt_24h = _binary_event_presence_multi(
        meds_pre24h,
        specs=[
            ("VITAL//", CRRT_CHARTEVENT_ITEMIDS),
            ("PROCEDURE//", CRRT_PROCEDURE_ITEMIDS),
        ],
        col_name="crrt_initiation_24h",
    )
    hemodialysis = _binary_event_presence_multi(
        meds_post24h,
        specs=[
            ("VITAL//", HEMODIALYSIS_CHARTEVENT_ITEMIDS),
            ("PROCEDURE//", HEMODIALYSIS_PROCEDURE_ITEMIDS),
        ],
        col_name="hemodialysis_initiation",
    )
    hemodialysis_24h = _binary_event_presence_multi(
        meds_pre24h,
        specs=[
            ("VITAL//", HEMODIALYSIS_CHARTEVENT_ITEMIDS),
            ("PROCEDURE//", HEMODIALYSIS_PROCEDURE_ITEMIDS),
        ],
        col_name="hemodialysis_initiation_24h",
    )

    # Hypotension: any MAP < 65 or SBP < 90 after 24h, with first-24h exclusion flag
    min_map = _vital_extremes(meds_post24h, MAP_ITEMIDS, "min", "min_map")
    min_sbp = _vital_extremes(meds_post24h, SBP_ITEMIDS, "min", "min_sbp")
    min_map_24h = _vital_extremes(meds_pre24h, MAP_ITEMIDS, "min", "min_map_24h")
    min_sbp_24h = _vital_extremes(meds_pre24h, SBP_ITEMIDS, "min", "min_sbp_24h")

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
        min_potassium,
        min_glucose,
        min_sodium,
        max_sodium,
        peak_bnp,
        max_heart_rate,
        max_sbp,
        max_dbp,
        peak_potassium_24h,
        min_potassium_24h,
        min_hemoglobin_24h,
        min_glucose_24h,
        min_sodium_24h,
        max_sodium_24h,
        max_heart_rate_24h,
        max_sbp_24h,
        max_dbp_24h,
        vasopressor,
        vasopressor_24h,
        crrt,
        crrt_24h,
        hemodialysis,
        hemodialysis_24h,
        min_map,
        min_sbp,
        min_map_24h,
        min_sbp_24h,
    ]:
        result = result.join(df, on="hadm_id", how="left", maintain_order="left")

    # Compute derived binary columns
    result = result.with_columns(
        hyperkalemia=pl.when(pl.col("peak_potassium").is_not_null())
        .then((pl.col("peak_potassium") >= 6.5).cast(pl.Float64))
        .otherwise(None),
        hyperkalemia_24h=(pl.col("peak_potassium_24h") >= 6.5).fill_null(False),
        severe_hypokalemia=pl.when(pl.col("min_potassium").is_not_null())
        .then((pl.col("min_potassium") < 2.5).cast(pl.Float64))
        .otherwise(None),
        severe_hypokalemia_24h=(pl.col("min_potassium_24h") < 2.5).fill_null(False),
        severe_anemia=pl.when(pl.col("min_hemoglobin").is_not_null())
        .then((pl.col("min_hemoglobin") < 7.0).cast(pl.Float64))
        .otherwise(None),
        severe_anemia_24h=(pl.col("min_hemoglobin_24h") < 7.0).fill_null(False),
        hypoglycemia=pl.when(pl.col("min_glucose").is_not_null())
        .then((pl.col("min_glucose") < 54.0).cast(pl.Float64))
        .otherwise(None),
        hypoglycemia_24h=(pl.col("min_glucose_24h") < 54.0).fill_null(False),
        profound_hyponatremia=pl.when(pl.col("min_sodium").is_not_null())
        .then((pl.col("min_sodium") < 125.0).cast(pl.Float64))
        .otherwise(None),
        profound_hyponatremia_24h=(pl.col("min_sodium_24h") < 125.0).fill_null(False),
        severe_hypernatremia=pl.when(pl.col("max_sodium").is_not_null())
        .then((pl.col("max_sodium") >= 160.0).cast(pl.Float64))
        .otherwise(None),
        severe_hypernatremia_24h=(pl.col("max_sodium_24h") >= 160.0).fill_null(False),
        tachycardia_hr130=pl.when(pl.col("max_heart_rate").is_not_null())
        .then((pl.col("max_heart_rate") >= 130.0).cast(pl.Float64))
        .otherwise(None),
        tachycardia_hr130_24h=(pl.col("max_heart_rate_24h") >= 130.0).fill_null(False),
        severe_hypertension=pl.when(
            pl.col("max_sbp").is_not_null() | pl.col("max_dbp").is_not_null()
        )
        .then(((pl.col("max_sbp") >= 180.0) | (pl.col("max_dbp") >= 120.0)).cast(pl.Float64))
        .otherwise(None),
        severe_hypertension_24h=((pl.col("max_sbp_24h") >= 180.0) | (pl.col("max_dbp_24h") >= 120.0)).fill_null(False),
        vasopressor_initiation=pl.col("vasopressor_initiation")
        .cast(pl.Float64)
        .fill_null(0.0),
        vasopressor_initiation_24h=pl.col("vasopressor_initiation_24h").fill_null(False),
        hypotension=pl.when(
            pl.col("min_map").is_not_null() | pl.col("min_sbp").is_not_null()
        )
        .then(((pl.col("min_map") < 65.0) | (pl.col("min_sbp") < 90.0)).cast(pl.Float64))
        .otherwise(None),
        hypotension_24h=(
            (pl.col("min_map_24h") < 65.0) | (pl.col("min_sbp_24h") < 90.0)
        ).fill_null(False),
        crrt_initiation=pl.col("crrt_initiation").cast(pl.Float64).fill_null(0.0),
        crrt_initiation_24h=pl.col("crrt_initiation_24h").fill_null(False),
        hemodialysis_initiation=pl.col("hemodialysis_initiation").cast(pl.Float64).fill_null(0.0),
        hemodialysis_initiation_24h=pl.col("hemodialysis_initiation_24h").fill_null(False),
    )

    # Standardize key column and drop intermediate vitals
    result = (
        result
        .with_columns(pl.col("hadm_id").cast(pl.String).alias("hospitalization_id"))
        .drop(
            "hadm_id",
            "peak_potassium_24h",
            "min_potassium_24h",
            "min_hemoglobin_24h",
            "min_glucose_24h",
            "min_sodium_24h",
            "max_sodium_24h",
            "max_heart_rate_24h",
            "max_sbp_24h",
            "max_dbp_24h",
            "min_map",
            "min_sbp",
            "min_map_24h",
            "min_sbp_24h",
        )
        .sort("hospitalization_id")
    )

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract extended outcomes (regression + additional binary) from MEDS events."
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
        "--observation_hours",
        type=float,
        default=24.0,
        help="Observation window in hours used to define post-24h outcomes (default: 24).",
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
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fail on missing requested splits instead of skipping them.",
    )
    args = parser.parse_args(argv)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    meds_events_dir = args.meds_events_dir.expanduser().resolve()
    tokenized_dir = args.tokenized_dir.expanduser().resolve()

    for split in splits:
        tok_split_dir = tokenized_dir / split
        base_path = tok_split_dir / args.base_outcomes_filename
        if not base_path.exists():
            msg = f"[extract_extended_outcomes] Missing split={split}: {base_path}"
            if args.strict:
                raise FileNotFoundError(msg)
            print(msg)
            continue

        meds_split_name, meds_split_dir = _resolve_meds_split_dir(meds_events_dir, split)
        if meds_split_dir is None:
            msg = (
                f"[extract_extended_outcomes] Missing MEDS split for split={split} under "
                f"{meds_events_dir}"
            )
            if args.strict:
                raise FileNotFoundError(msg)
            print(msg)
            continue

        print(f"[extract_extended_outcomes] Processing split={split} (meds={meds_split_name})...")

        meds = _scan_meds_events(meds_split_dir)
        extended = _compute_extended_outcomes(meds, observation_hours=args.observation_hours)

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
            "peak_potassium", "min_potassium", "min_glucose", "min_sodium",
            "max_sodium", "peak_bnp", "max_heart_rate", "max_sbp", "max_dbp",
        ]
        binary_cols = [
            "hyperkalemia", "severe_hypokalemia", "severe_anemia",
            "hypoglycemia", "profound_hyponatremia", "severe_hypernatremia",
            "tachycardia_hr130", "severe_hypertension", "vasopressor_initiation",
            "hypotension", "crrt_initiation", "hemodialysis_initiation",
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
