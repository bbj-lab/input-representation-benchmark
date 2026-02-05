#!/usr/bin/env python3
"""
Build Experiment 3 MEDS-only vocabulary-semantics arms (row-fixed).

This script produces the 4 *primary* Exp3 arms:
  1) meds_icu (native)
  2) meds_mapped (MEDS→CLIF categories; row-fixed)
  3) meds_randomized (null: randomized mapping within domain)
  4) meds_freqmatched (null: frequency-matched mapping within domain)

All arms are constructed from the same MEDS events directory that has already been
filtered to the Exp3 ICU-hospitalization cohort \(H_{\mathrm{ICU}}\):
hospital admissions (`hadm_id`) with (i) hospital LOS \(\ge\) 24h in
`hosp/admissions.csv.gz` and (ii) \(\ge\)1 linked ICU stay record in
`icu/icustays.csv.gz` for the same `hadm_id` (see `scripts/align_cohorts.py`).

Input format: {train,val,test}/meds.parquet. We only rewrite the `code` column for
LAB/VITAL events; timestamps and numeric_values are left unchanged.

Mapping source
--------------
We use CLIF-MIMIC mapping tables:
  - CLIF-MIMIC/data/mappings/mimic-to-clif-mappings - labs.csv
  - CLIF-MIMIC/data/mappings/mimic-to-clif-mappings - vitals.csv

The MEDS codes are expected to include an itemid-like identifier as the second
field when split on "//", e.g.:
  LAB//50931//mg/dL
  VITAL//220045//bpm

Unmapped codes are mapped to a sentinel category per domain:
  LAB//__unmapped__
  VITAL//__unmapped__
This keeps the transformation total and makes mapping coverage measurable.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import polars as pl


def _read_split(meds_dir: Path, split: str) -> pl.LazyFrame:
    fp = meds_dir / split / "meds.parquet"
    if not fp.exists():
        raise FileNotFoundError(f"Missing MEDS split parquet: {fp}")
    return pl.scan_parquet(str(fp))


def _write_split(lf: pl.LazyFrame, out_dir: Path, split: str) -> None:
    d = out_dir / split
    d.mkdir(parents=True, exist_ok=True)
    lf.sink_parquet(d / "meds.parquet")


def _load_lab_mapping(labs_csv: Path) -> dict[str, str]:
    df = pl.read_csv(labs_csv, ignore_errors=True)
    # expected cols: lab_category, decision, itemid, ...
    if "itemid" not in df.columns or "lab_category" not in df.columns:
        raise ValueError(f"Unexpected labs mapping schema in {labs_csv} (cols={df.columns})")
    out: dict[str, str] = {}
    for row in df.select(pl.col("itemid"), pl.col("lab_category")).iter_rows(named=True):
        itemid = row["itemid"]
        cat = row["lab_category"]
        if itemid is None or cat is None:
            continue
        try:
            out[str(int(itemid))] = str(cat)
        except Exception:
            continue
    return out


def _load_vital_mapping(vitals_csv: Path) -> dict[str, str]:
    df = pl.read_csv(vitals_csv, ignore_errors=True)
    # expected cols: vital_category, ..., itemid, ...
    if "itemid" not in df.columns or "vital_category" not in df.columns:
        raise ValueError(f"Unexpected vitals mapping schema in {vitals_csv} (cols={df.columns})")
    out: dict[str, str] = {}
    for row in df.select(pl.col("itemid"), pl.col("vital_category")).iter_rows(named=True):
        itemid = row["itemid"]
        cat = row["vital_category"]
        if itemid is None or cat is None:
            continue
        try:
            out[str(int(itemid))] = str(cat)
        except Exception:
            continue
    return out


def _extract_itemid(code: str) -> str | None:
    # MEDS code format: PREFIX//<id>//...
    parts = code.split("//")
    if len(parts) < 2:
        return None
    return parts[1]


def _apply_itemid_mapping(
    lf: pl.LazyFrame,
    *,
    domain_prefix: str,
    itemid_to_cat: dict[str, str],
    out_prefix: str,
    unmapped_cat: str,
) -> tuple[pl.LazyFrame, dict[str, int]]:
    # Track coverage stats on the split.
    # We compute counts in eager mode (streaming) to avoid needing to materialize full data.
    df_codes = (
        lf.filter(pl.col("code").is_not_null() & pl.col("code").cast(pl.Utf8).str.starts_with(domain_prefix))
        .select(pl.col("code").cast(pl.Utf8).alias("code"))
        .collect(streaming=True)
    )
    codes = df_codes["code"].to_list()
    n_total = len(codes)
    n_mapped = 0
    rep: dict[str, str] = {}
    for c in codes:
        itemid = _extract_itemid(str(c))
        if itemid is None:
            rep[str(c)] = f"{out_prefix}//{unmapped_cat}"
            continue
        cat = itemid_to_cat.get(str(itemid))
        if cat is None or cat.strip() == "" or cat.upper() == "NO MAPPING":
            rep[str(c)] = f"{out_prefix}//{unmapped_cat}"
            continue
        n_mapped += 1
        rep[str(c)] = f"{out_prefix}//{cat}"

    stats = {
        "n_total": n_total,
        "n_mapped": n_mapped,
        "n_unmapped": n_total - n_mapped,
    }

    out_lf = lf.with_columns(
        pl.when(pl.col("code").cast(pl.Utf8).str.starts_with(domain_prefix))
        .then(pl.col("code").cast(pl.Utf8).replace(rep, default=pl.col("code").cast(pl.Utf8)))
        .otherwise(pl.col("code").cast(pl.Utf8))
        .alias("code")
    )
    return out_lf, stats


def _domain_counts_codes(lf: pl.LazyFrame, *, prefix: str) -> Counter[str]:
    df = (
        lf.filter(pl.col("code").is_not_null() & pl.col("code").cast(pl.Utf8).str.starts_with(prefix))
        .select(pl.col("code").cast(pl.Utf8).alias("code"))
        .collect(streaming=True)
    )
    return Counter(df["code"].to_list())


def _make_random_mapping(*, src_codes: list[str], tgt_categories: list[str], rng: random.Random) -> dict[str, str]:
    if not tgt_categories:
        raise ValueError("Target category list is empty.")
    tgt = list(tgt_categories)
    rng.shuffle(tgt)
    out: dict[str, str] = {}
    for i, c in enumerate(sorted(src_codes)):
        out[c] = tgt[i % len(tgt)]
    return out


def _make_freqmatched_mapping(*, src_counts: Counter[str], tgt_counts: Counter[str]) -> dict[str, str]:
    if not tgt_counts:
        raise ValueError("Target count distribution is empty.")
    remaining = {k: int(v) for k, v in tgt_counts.items()}
    cats = sorted(remaining.keys(), key=lambda k: remaining[k], reverse=True)
    mapping: dict[str, str] = {}
    for code, cnt in sorted(src_counts.items(), key=lambda kv: kv[1], reverse=True):
        cats.sort(key=lambda k: (remaining[k], k), reverse=True)
        chosen = cats[0]
        mapping[code] = chosen
        remaining[chosen] = max(0, remaining[chosen] - int(cnt))
    return mapping


def _apply_code_to_code_mapping(lf: pl.LazyFrame, *, domain_prefix: str, mapping: dict[str, str]) -> pl.LazyFrame:
    rep = {k: v for k, v in mapping.items()}
    return lf.with_columns(
        pl.when(pl.col("code").cast(pl.Utf8).str.starts_with(domain_prefix))
        .then(pl.col("code").cast(pl.Utf8).replace(rep, default=pl.col("code").cast(pl.Utf8)))
        .otherwise(pl.col("code").cast(pl.Utf8))
        .alias("code")
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build Exp3 MEDS-only semantics arms (native + mapped + null controls).")
    p.add_argument(
        "--meds_in_dir",
        type=Path,
        required=True,
        help=(
            "MEDS events dir filtered to Exp3 ICU-hospitalization cohort H_ICU, with "
            "{train,val,test}/meds.parquet. H_ICU is admissions (hadm_id) with LOS>=24h "
            "AND >=1 linked ICU stay record in icu/icustays for the same hadm_id."
        ),
    )
    p.add_argument("--out_root", type=Path, default=Path("data/exp3/arms"), help="Output root (default: data/exp3/arms).")
    p.add_argument("--seed", type=int, default=42, help="Seed for randomized mapping control.")
    p.add_argument(
        "--clif_mimic_repo",
        type=Path,
        default=Path("../CLIF-MIMIC"),
        help="Path to CLIF-MIMIC repo (default: ../CLIF-MIMIC). Used for mapping CSVs.",
    )
    args = p.parse_args(argv)

    meds_in = args.meds_in_dir.expanduser().resolve()
    out_root = args.out_root.expanduser().resolve()
    clif_mimic_repo = args.clif_mimic_repo.expanduser().resolve()
    rng = random.Random(int(args.seed))

    labs_csv = clif_mimic_repo / "data" / "mappings" / "mimic-to-clif-mappings - labs.csv"
    vitals_csv = clif_mimic_repo / "data" / "mappings" / "mimic-to-clif-mappings - vitals.csv"
    if not labs_csv.exists() or not vitals_csv.exists():
        raise FileNotFoundError(f"Missing CLIF-MIMIC mapping CSVs under {clif_mimic_repo}/data/mappings/")

    lab_itemid_to_cat = _load_lab_mapping(labs_csv)
    vit_itemid_to_cat = _load_vital_mapping(vitals_csv)

    # 1) MEDS→CLIF mapped arm (row-fixed)
    mapped_dir = out_root / "meds_mapped"
    mapped_dir.mkdir(parents=True, exist_ok=True)
    coverage: dict[str, dict[str, int]] = {}
    for split in ["train", "val", "test"]:
        lf = _read_split(meds_in, split)
        lf, lab_stats = _apply_itemid_mapping(
            lf,
            domain_prefix="LAB",
            itemid_to_cat=lab_itemid_to_cat,
            out_prefix="LAB",
            unmapped_cat="__unmapped__",
        )
        lf, vit_stats = _apply_itemid_mapping(
            lf,
            domain_prefix="VITAL",
            itemid_to_cat=vit_itemid_to_cat,
            out_prefix="VITAL",
            unmapped_cat="__unmapped__",
        )
        coverage[split] = {"LAB": lab_stats, "VITAL": vit_stats}
        _write_split(lf, mapped_dir, split)

    (mapped_dir / "mappings").mkdir(parents=True, exist_ok=True)
    (mapped_dir / "mappings" / "meta.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "seed": int(args.seed),
                "mapping_source": "CLIF-MIMIC mapping CSVs (labs.csv, vitals.csv)",
                "coverage": coverage,
                "note": "Row-fixed mapping: only MEDS `code` is rewritten for LAB/VITAL; times/numeric_values unchanged.",
            },
            indent=2,
            sort_keys=True,
        )
    )

    # 2) Build null controls using the standardized target vocabulary induced by meds_mapped (train split).
    mapped_train = _read_split(mapped_dir, "train")
    lab_tgt_counts = _domain_counts_codes(mapped_train, prefix="LAB//")
    vit_tgt_counts = _domain_counts_codes(mapped_train, prefix="VITAL//")
    lab_tgt_categories = list(lab_tgt_counts.keys())
    vit_tgt_categories = list(vit_tgt_counts.keys())

    # MEDS source distributions (train split) for mapping controls
    meds_train = _read_split(meds_in, "train")
    lab_src_counts = _domain_counts_codes(meds_train, prefix="LAB")
    vit_src_counts = _domain_counts_codes(meds_train, prefix="VITAL")

    # Randomized mapping: map each raw code to a target category code (within domain).
    lab_rand = _make_random_mapping(src_codes=list(lab_src_counts.keys()), tgt_categories=lab_tgt_categories, rng=rng)
    vit_rand = _make_random_mapping(src_codes=list(vit_src_counts.keys()), tgt_categories=vit_tgt_categories, rng=rng)

    # Frequency-matched mapping: map raw codes to target categories so category freq profile matches meds_mapped.
    lab_freq = _make_freqmatched_mapping(src_counts=lab_src_counts, tgt_counts=lab_tgt_counts)
    vit_freq = _make_freqmatched_mapping(src_counts=vit_src_counts, tgt_counts=vit_tgt_counts)

    for arm_name, (lab_map, vit_map) in {
        "meds_randomized": (lab_rand, vit_rand),
        "meds_freqmatched": (lab_freq, vit_freq),
    }.items():
        arm_dir = out_root / arm_name
        arm_dir.mkdir(parents=True, exist_ok=True)
        (arm_dir / "mappings").mkdir(parents=True, exist_ok=True)
        (arm_dir / "mappings" / "lab_mapping.json").write_text(json.dumps(lab_map, indent=2, sort_keys=True))
        (arm_dir / "mappings" / "vital_mapping.json").write_text(json.dumps(vit_map, indent=2, sort_keys=True))
        (arm_dir / "mappings" / "meta.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "seed": int(args.seed),
                    "target_distribution": "Derived from meds_mapped train split (row-fixed standardized identifier scheme).",
                },
                indent=2,
                sort_keys=True,
            )
        )
        for split in ["train", "val", "test"]:
            lf = _read_split(meds_in, split)
            lf = _apply_code_to_code_mapping(lf, domain_prefix="LAB", mapping=lab_map)
            lf = _apply_code_to_code_mapping(lf, domain_prefix="VITAL", mapping=vit_map)
            _write_split(lf, arm_dir, split)

    print("OK")
    print(f"  Wrote meds_mapped under:      {mapped_dir}")
    print(f"  Wrote meds_randomized under:  {out_root / 'meds_randomized'}")
    print(f"  Wrote meds_freqmatched under: {out_root / 'meds_freqmatched'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

