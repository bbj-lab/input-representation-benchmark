#!/usr/bin/env python3
"""
Augment CLIF lab tables with reference range bounds for clinically anchored binning.

Motivation
----------
Experiment 1 can select a clinically anchored discretization (e.g., ventiles 5-10-5)
that requires reference interval bounds (L, U). In the MEDS pipeline, these bounds
exist per event (`ref_range_lower`, `ref_range_upper`) from MIMIC-IV `hosp/labevents`.

In the CLIF 2.1 schema (as implemented by clifpy), the labs table includes a
`reference_unit` but does not include lower/upper reference range bounds. To enable
the *same* clinically anchored binning logic in both MEDS and CLIF arms (Experiment 3),
we augment CLIF `clif_labs.parquet` with:
  - ref_range_lower: float
  - ref_range_upper: float

Approach (data-derived, MIMIC-native)
------------------------------------
We derive a reference-range lookup table from MIMIC-IV v3.1:
  1) Read `hosp/labevents.csv.gz` for (itemid, valueuom, ref_range_lower, ref_range_upper)
  2) Join to `hosp/d_labitems.csv.gz` to map itemid -> loinc_code
  3) Aggregate reference range bounds by (loinc_code, unit) using robust medians
  4) Join this lookup to CLIF labs by (lab_loinc_code, reference_unit)

This provides a consistent, empirically grounded way to attach reference ranges to CLIF
lab rows without modifying the upstream CLIF conversion pipeline.

Notes / limitations
-------------------
- Some CLIF lab rows may have missing `lab_loinc_code`; these rows will remain without
  ref ranges and will fall back to population quantile binning (as in Exp1 logic).
- Reference ranges in MIMIC may vary across events; we use medians for stability.
- This script is intended to run once and cache the lookup parquet for reuse.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import polars as pl


def _norm_unit(expr: pl.Expr) -> pl.Expr:
    """Normalize unit strings for joining across MIMIC and CLIF."""
    return (
        expr.cast(pl.String, strict=False)
        .fill_null("")
        .str.strip_chars()
        .str.to_lowercase()
        .str.replace_all(r"\s+", "")
    )


def build_ref_range_lookup(
    *,
    mimic_dir: Path,
    output_path: Path,
    force: bool = False,
) -> Path:
    """
    Build (loinc_code, unit) -> (ref_range_lower, ref_range_upper) lookup table.
    """
    if output_path.exists() and not force:
        return output_path

    labevents_path = mimic_dir / "hosp" / "labevents.csv.gz"
    d_labitems_path = mimic_dir / "hosp" / "d_labitems.csv.gz"
    if not labevents_path.exists():
        raise FileNotFoundError(f"Missing {labevents_path}")
    if not d_labitems_path.exists():
        raise FileNotFoundError(f"Missing {d_labitems_path}")

    # Read minimal labitems mapping (itemid -> loinc_code)
    labitems = (
        pl.scan_csv(d_labitems_path)
        .select(
            pl.col("itemid").cast(pl.Int64),
            pl.col("loinc_code").cast(pl.String, strict=False).fill_null(""),
        )
        .filter(pl.col("loinc_code") != "")
    )

    # Read labevents ref range columns (massive; keep only required cols)
    labs = (
        pl.scan_csv(
            labevents_path,
            columns=["itemid", "valueuom", "ref_range_lower", "ref_range_upper"],
            infer_schema_length=1_000,
        )
        .select(
            pl.col("itemid").cast(pl.Int64),
            _norm_unit(pl.col("valueuom")).alias("unit_norm"),
            pl.col("ref_range_lower").cast(pl.Float64, strict=False),
            pl.col("ref_range_upper").cast(pl.Float64, strict=False),
        )
        .filter(pl.col("ref_range_lower").is_not_null() & pl.col("ref_range_upper").is_not_null())
        .filter(pl.col("unit_norm") != "")
    )

    # Aggregate at itemid+unit first (reduces data volume before joining loinc_code)
    per_item = (
        labs.group_by(["itemid", "unit_norm"])
        .agg(
            pl.median("ref_range_lower").alias("ref_range_lower"),
            pl.median("ref_range_upper").alias("ref_range_upper"),
            pl.len().alias("n_events"),
        )
    )

    # Attach loinc_code and aggregate to loinc_code+unit
    lookup = (
        per_item.join(labitems, on="itemid", how="left")
        .filter(pl.col("loinc_code").is_not_null() & (pl.col("loinc_code") != ""))
        .group_by(["loinc_code", "unit_norm"])
        .agg(
            pl.median("ref_range_lower").alias("ref_range_lower"),
            pl.median("ref_range_upper").alias("ref_range_upper"),
            pl.sum("n_events").alias("n_events"),
        )
        .sort(["loinc_code", "unit_norm"])
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    lookup.collect(streaming=True).write_parquet(output_path)
    return output_path


def augment_one_split(
    *,
    clif_split_dir: Path,
    lookup: pl.LazyFrame,
    inplace: bool,
    output_dir: Path | None,
) -> Path:
    labs_path = clif_split_dir / "clif_labs.parquet"
    if not labs_path.exists():
        raise FileNotFoundError(f"Missing {labs_path}")

    clif_labs = pl.scan_parquet(labs_path)
    clif_labs = clif_labs.with_columns(
        _norm_unit(pl.col("reference_unit")).alias("unit_norm"),
        pl.col("lab_loinc_code").cast(pl.String, strict=False).fill_null(""),
    )

    out = (
        clif_labs.join(
            lookup,
            left_on=[pl.col("lab_loinc_code"), pl.col("unit_norm")],
            right_on=[pl.col("loinc_code"), pl.col("unit_norm")],
            how="left",
        )
        .drop("unit_norm")
        .with_columns(
            # Standardize names expected by fms-ehrs tokenizer when include_ref_ranges=True
            pl.col("ref_range_lower").cast(pl.Float64, strict=False),
            pl.col("ref_range_upper").cast(pl.Float64, strict=False),
        )
    )

    if inplace:
        out.sink_parquet(labs_path)
        return labs_path

    assert output_dir is not None
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / clif_split_dir.name / "clif_labs.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.sink_parquet(out_path)
    return out_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Augment CLIF labs with reference range bounds.")
    p.add_argument(
        "--mimic_dir",
        type=Path,
        required=True,
        help="Path to MIMIC-IV v3.1 directory (physionet.org/files/mimiciv/3.1)",
    )
    p.add_argument(
        "--clif_root",
        type=Path,
        required=True,
        help="Root directory containing CLIF split dirs (expects raw/{train,val,test}/clif_labs.parquet)",
    )
    p.add_argument(
        "--clif_version_in",
        type=str,
        default="raw",
        help="Input CLIF version directory name under clif_root (default: raw)",
    )
    p.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "val", "test"],
        help="Split names to process (default: train val test).",
    )
    p.add_argument(
        "--lookup_out",
        type=Path,
        default=None,
        help="Where to write/read the cached loinc-unit ref range lookup parquet. "
        "Default: clif_root/.ref_ranges_by_loinc_unit.parquet",
    )
    p.add_argument(
        "--force_recompute_lookup",
        action="store_true",
        help="Recompute the lookup even if the cached file exists.",
    )
    p.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite clif_labs.parquet in-place (recommended if clif_root is a working directory).",
    )
    p.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="If not inplace, write augmented labs to this directory preserving split structure.",
    )
    args = p.parse_args(argv)

    clif_in_root = args.clif_root / args.clif_version_in
    if not clif_in_root.exists():
        raise FileNotFoundError(f"Missing CLIF input dir: {clif_in_root}")

    lookup_path = args.lookup_out or (args.clif_root / ".ref_ranges_by_loinc_unit.parquet")
    lookup_path = build_ref_range_lookup(
        mimic_dir=args.mimic_dir,
        output_path=lookup_path,
        force=args.force_recompute_lookup,
    )
    lookup = pl.scan_parquet(lookup_path)

    if not args.inplace and args.output_dir is None:
        raise ValueError("Either --inplace or --output_dir must be provided.")

    written: list[Path] = []
    for split in args.splits:
        split_dir = clif_in_root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Missing split dir: {split_dir}")
        written.append(
            augment_one_split(
                clif_split_dir=split_dir,
                lookup=lookup,
                inplace=args.inplace,
                output_dir=args.output_dir,
            )
        )

    print("Wrote:")
    for pth in written:
        print(f"  - {pth}")
    print(f"Lookup cache: {lookup_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

