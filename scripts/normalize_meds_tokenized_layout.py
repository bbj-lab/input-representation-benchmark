#!/usr/bin/env python3
"""
Normalize MEDS tokenized artifacts to match fms-ehrs downstream expectations.

Why
---
fms-ehrs training/evaluation utilities assume:
- split directory name: `val` (not `tuning`)
- subject id column: `hospitalization_id` (not `hadm_id`)

MEDS tokenization via `mimic-meds-ed.yaml` naturally uses `hadm_id` as the subject_id,
and some MEDS pipelines use `tuning` for validation.

This script performs a deterministic, explicit normalization step so we can keep
fms-ehrs downstream scripts unchanged.

What it does
------------
- If a `tuning/` split exists and `val/` does not, it renames `tuning/` → `val/`
  (or copies when using --output_dir).
- Renames `hadm_id` → `hospitalization_id` in:
  - tokens_timelines.parquet
  - tokens_timelines_outcomes.parquet (if present)
- Casts `hospitalization_id` to string for consistency with CLIF splits.

Safety
------
By default, writes into a new directory (--output_dir). Use --inplace only when
you are sure you want to modify the input directory.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable

import polars as pl


TOKEN_FILES = ("tokens_timelines.parquet", "tokens_timelines_outcomes.parquet")


def _iter_splits(base: Path) -> list[Path]:
    return sorted(p for p in base.iterdir() if p.is_dir() and not p.name.startswith("."))


def _normalize_split_dir(src_split: Path, dst_split: Path) -> None:
    dst_split.mkdir(parents=True, exist_ok=True)

    for fn in TOKEN_FILES:
        src = src_split / fn
        if not src.exists():
            continue

        df = pl.read_parquet(src)

        if "hospitalization_id" not in df.columns and "hadm_id" in df.columns:
            df = df.rename({"hadm_id": "hospitalization_id"})

        if "hospitalization_id" in df.columns:
            df = df.with_columns(pl.col("hospitalization_id").cast(pl.String))

        df.write_parquet(dst_split / fn)

    # Copy vocab if present (train split), preserving naming
    vocab = src_split / "vocab.gzip"
    if vocab.exists():
        shutil.copy2(vocab, dst_split / "vocab.gzip")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Normalize MEDS tokenized layout for fms-ehrs compatibility.")
    p.add_argument("--input_dir", type=Path, required=True, help="Tokenized base dir containing split subdirs.")
    p.add_argument("--output_dir", type=Path, default=None, help="Output base dir (default: <input_dir>_normalized).")
    p.add_argument(
        "--inplace",
        action="store_true",
        help="Modify input_dir in place (dangerous). If set, --output_dir is ignored.",
    )
    args = p.parse_args(argv)

    input_dir = args.input_dir.expanduser().resolve()
    if args.inplace:
        out_dir = input_dir
        tmp_dir = input_dir.parent / f".{input_dir.name}.tmp_normalize"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = (
            args.output_dir.expanduser().resolve()
            if args.output_dir is not None
            else input_dir.parent / f"{input_dir.name}_normalized"
        )
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp_dir = out_dir

    splits = _iter_splits(input_dir)

    # Handle tuning→val mapping
    split_names = {s.name for s in splits}
    rename_map: dict[str, str] = {}
    if "tuning" in split_names and "val" not in split_names:
        rename_map["tuning"] = "val"

    for s in splits:
        dst_name = rename_map.get(s.name, s.name)
        _normalize_split_dir(s, tmp_dir / dst_name)

    if args.inplace:
        # Replace contents atomically-ish: remove old split dirs then move tmp into place
        for s in _iter_splits(input_dir):
            shutil.rmtree(s)
        for s in _iter_splits(tmp_dir):
            shutil.move(str(s), str(input_dir / s.name))
        shutil.rmtree(tmp_dir)

    print(f"[normalize_meds_tokenized_layout] Wrote normalized tokenized dataset to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

