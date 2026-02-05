#!/usr/bin/env python3
"""
Split CLIF parquet tables into train/val/test directories using cohort split lists.

Why this exists
---------------
`fms-ehrs` tokenization expects a directory structure:

  data/clif/
    raw/
      train/ clif_*.parquet
      val/   clif_*.parquet
      test/  clif_*.parquet

Upstream CLIF conversion pipelines (e.g., CLIF-MIMIC) commonly emit a *single*
directory of `clif_*.parquet` tables without train/val/test splits. This script
creates split directories by filtering tables using provided patient ID lists.

Filtering rules
---------------
For each split:
1) Filter `clif_hospitalization.parquet` by the selected cohort key:
   - patient_id ∈ split_patient_ids (patient-level cohort), OR
   - hospitalization_id ∈ split_hadm_ids (Exp3 H_ICU hospitalizations within patient splits; recommended for Exp3).
     Here H_ICU is admissions (`hadm_id`) with hospital LOS>=24h AND >=1 linked ICU stay record in
     MIMIC-IV `icu/icustays.csv.gz` for the same `hadm_id` (see `scripts/align_cohorts.py`).
2) Derive the set of `hospitalization_id` for that split from the filtered
   hospitalization table.
3) For each other `clif_*.parquet` table:
   - If it has `hospitalization_id`, keep rows with `hospitalization_id` in the split set.
   - Else if it has `patient_id`, keep rows with `patient_id` in the split set.
   - Else, skip the table (not used by tokenization configs).

Inputs
------
- `--clif_in_dir`: directory containing unsplit `clif_*.parquet` tables
- `--splits_dir`: directory containing:
    - train_patient_ids.csv
    - val_patient_ids.csv
    - test_patient_ids.csv
    - train_hadm_ids.csv (optional; required if --filter_key=hospitalization_id)
    - val_hadm_ids.csv   (optional; required if --filter_key=hospitalization_id)
    - test_hadm_ids.csv  (optional; required if --filter_key=hospitalization_id)
  produced by `scripts/align_cohorts.py` (column name: `subject_id`)

Output
------
Writes filtered parquet tables into:
  <clif_out_root>/raw/{train,val,test}/clif_*.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import polars as pl


def _load_patient_ids(csv_path: Path) -> pl.LazyFrame:
    df = pl.read_csv(csv_path)
    col = "subject_id" if "subject_id" in df.columns else "patient_id"
    return df.select(pl.col(col).cast(pl.String, strict=False).alias("patient_id")).lazy()


def _load_hadm_ids(csv_path: Path) -> pl.LazyFrame:
    df = pl.read_csv(csv_path)
    col = "hadm_id" if "hadm_id" in df.columns else "hospitalization_id"
    return df.select(pl.col(col).cast(pl.String, strict=False).alias("hospitalization_id")).lazy()


def _scan_parquet(path: Path) -> pl.LazyFrame:
    return pl.scan_parquet(path)


def _has_col(lf: pl.LazyFrame, col: str) -> bool:
    return col in lf.collect_schema().names()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Split CLIF parquet tables by patient splits.")
    p.add_argument(
        "--clif_in_dir",
        type=Path,
        required=True,
        help="Directory containing unsplit clif_*.parquet tables.",
    )
    p.add_argument(
        "--splits_dir",
        type=Path,
        required=True,
        help="Directory containing train/val/test patient ID CSVs.",
    )
    p.add_argument(
        "--filter_key",
        type=str,
        choices=["patient_id", "hospitalization_id"],
        default="patient_id",
        help=(
            "How to define the cohort filter. "
            "patient_id uses <split>_patient_ids.csv; "
            "hospitalization_id uses <split>_hadm_ids.csv (recommended for Exp3 H_ICU: LOS>=24h + linked ICU stay)."
        ),
    )
    p.add_argument(
        "--clif_out_root",
        type=Path,
        default=Path("data/clif"),
        help="Output CLIF root (default: data/clif). Writes to clif_out_root/raw/{split}/.",
    )
    p.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "val", "test"],
        help="Split names to generate (default: train val test).",
    )
    p.add_argument(
        "--glob",
        type=str,
        default="clif_*.parquet",
        help="Glob for CLIF parquet tables in clif_in_dir (default: clif_*.parquet).",
    )
    args = p.parse_args(argv)

    clif_in_dir = args.clif_in_dir.expanduser().resolve()
    splits_dir = args.splits_dir.expanduser().resolve()
    clif_out_root = args.clif_out_root.expanduser().resolve()

    if not clif_in_dir.exists():
        raise FileNotFoundError(f"Missing clif_in_dir: {clif_in_dir}")
    if not splits_dir.exists():
        raise FileNotFoundError(f"Missing splits_dir: {splits_dir}")

    hosp_fp = clif_in_dir / "clif_hospitalization.parquet"
    if not hosp_fp.exists():
        raise FileNotFoundError(
            f"Missing required table clif_hospitalization.parquet in {clif_in_dir}. "
            "Ensure your CLIF conversion outputs clif_*.parquet files with CLIF-2.1 naming."
        )

    tables = sorted(clif_in_dir.glob(args.glob))
    if not tables:
        raise FileNotFoundError(f"No parquet files matching {args.glob} in {clif_in_dir}")

    hosp = _scan_parquet(hosp_fp).with_columns(
        pl.col("patient_id").cast(pl.String, strict=False),
        pl.col("hospitalization_id").cast(pl.String, strict=False),
    )

    for split in args.splits:
        if args.filter_key == "patient_id":
            split_csv = splits_dir / f"{split}_patient_ids.csv"
            if not split_csv.exists():
                raise FileNotFoundError(f"Missing split file: {split_csv}")
            patient_ids = _load_patient_ids(split_csv)
            hosp_split = hosp.join(patient_ids, on="patient_id", how="semi")
        else:
            split_csv = splits_dir / f"{split}_hadm_ids.csv"
            if not split_csv.exists():
                raise FileNotFoundError(
                    f"Missing split file: {split_csv}. "
                    "Did you run scripts/align_cohorts.py that emits *_hadm_ids.csv?"
                )
            hadm_ids = _load_hadm_ids(split_csv)
            hosp_split = hosp.join(hadm_ids, on="hospitalization_id", how="semi")
            patient_ids = hosp_split.select("patient_id").unique()

        hosp_ids = hosp_split.select("hospitalization_id").unique()

        out_dir = clif_out_root / "raw" / split
        out_dir.mkdir(parents=True, exist_ok=True)

        # Always write hospitalization table first (required by tokenizer).
        hosp_split.sink_parquet(out_dir / hosp_fp.name)

        for fp in tables:
            if fp.name == hosp_fp.name:
                continue

            lf = _scan_parquet(fp)

            # Normalize join keys if present.
            if _has_col(lf, "hospitalization_id"):
                lf = lf.with_columns(
                    pl.col("hospitalization_id").cast(pl.String, strict=False)
                ).join(hosp_ids, on="hospitalization_id", how="semi")
            elif _has_col(lf, "patient_id"):
                lf = lf.with_columns(
                    pl.col("patient_id").cast(pl.String, strict=False)
                ).join(patient_ids, on="patient_id", how="semi")
            else:
                # Skip tables that cannot be filtered by the cohort keys; they are not
                # used by our tokenization configs and would otherwise contaminate parity.
                continue

            lf.sink_parquet(out_dir / fp.name)

    print(f"Wrote split CLIF tables to: {clif_out_root / 'raw'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

