#!/usr/bin/env python3
"""
Create an Exp3 MEDS events directory by filtering MEDS parquet splits by ICU-eligible hadm_ids.

Why this exists
---------------
Exp3 must isolate vocabulary semantics (MEDS vs CLIF). Since CLIF is ICU-scoped,
we restrict the MEDS arm to the *same ICU-eligible hospitalizations*.

Inputs
------
- --meds_in_dir: MEDS events directory containing {train,tuning,test}/ with parquet shards
- --splits_dir: directory containing train/val/test hadm lists (from scripts/align_cohorts.py):
    - train_hadm_ids.csv
    - val_hadm_ids.csv
    - test_hadm_ids.csv

Output
------
Writes an events directory at --meds_out_dir with {train,val,test}/meds.parquet
containing only rows whose hadm_id is in the corresponding split list.

Notes
-----
- We write one consolidated parquet per split (streaming) for simplicity.
- We keep *all columns* to preserve compatibility with the existing MEDS tokenizer config.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl


def _load_ids(csv_path: Path, col: str) -> pl.DataFrame:
    df = pl.read_csv(csv_path)
    if col not in df.columns:
        raise ValueError(f"Missing column {col} in {csv_path} (columns={df.columns})")
    return df.select(pl.col(col).cast(pl.Int64, strict=False).alias(col)).unique()


def _scan_split_parquets(split_dir: Path) -> pl.LazyFrame:
    fps = sorted(split_dir.glob("*.parquet"))
    if not fps:
        raise FileNotFoundError(f"No parquet shards found under {split_dir}")
    return pl.scan_parquet([str(p) for p in fps])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Filter MEDS events dir by split hadm_id lists for Exp3.")
    p.add_argument("--meds_in_dir", type=Path, required=True, help="MEDS events directory (contains train/tuning/test).")
    p.add_argument("--splits_dir", type=Path, required=True, help="Cohort splits dir (contains *_hadm_ids.csv).")
    p.add_argument("--meds_out_dir", type=Path, required=True, help="Output MEDS events directory (train/val/test).")
    p.add_argument(
        "--val_split_name",
        type=str,
        default="tuning",
        choices=["tuning", "val"],
        help="Which MEDS split is used as validation in the input dir (default: tuning).",
    )
    args = p.parse_args(argv)

    meds_in = args.meds_in_dir.expanduser().resolve()
    splits = args.splits_dir.expanduser().resolve()
    meds_out = args.meds_out_dir.expanduser().resolve()

    if not meds_in.exists():
        raise FileNotFoundError(f"Missing meds_in_dir: {meds_in}")
    if not splits.exists():
        raise FileNotFoundError(f"Missing splits_dir: {splits}")

    # map our tokenization expectations: output uses val, but MEDS extraction uses tuning
    in_map = {"train": "train", "val": args.val_split_name, "test": "test"}

    for split in ["train", "val", "test"]:
        ids_csv = splits / f"{split}_hadm_ids.csv"
        if not ids_csv.exists():
            raise FileNotFoundError(f"Missing {ids_csv}")
        keep = _load_ids(ids_csv, "hadm_id")

        src_split = meds_in / in_map[split]
        lf = _scan_split_parquets(src_split)
        if "hadm_id" not in lf.collect_schema().names():
            raise ValueError(f"Expected hadm_id column in MEDS parquet shards under {src_split}")

        out_dir = meds_out / split
        out_dir.mkdir(parents=True, exist_ok=True)
        out_fp = out_dir / "meds.parquet"

        # Semi-join by hadm_id (streaming)
        lf = lf.join(keep.lazy(), on="hadm_id", how="semi")
        lf.sink_parquet(out_fp)

    print(f"Wrote ICU-aligned MEDS events to: {meds_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

