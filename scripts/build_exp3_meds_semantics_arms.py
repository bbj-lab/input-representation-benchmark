#!/usr/bin/env python3
"""
Build Experiment 3 MEDS-only vocabulary-semantics arms (row-fixed, full-breadth).

This script produces the 4 *primary* Exp3 arms:
  1) meds_icu (native)   — unchanged raw MEDS codes
  2) meds_mapped          — MEDS→CLIF categories; row-fixed
  3) meds_randomized      — null: randomized mapping within domain
  4) meds_freqmatched     — null: frequency-matched mapping within domain

All arms are constructed from the same MEDS events directory that has already been
filtered to the Exp3 ICU-hospitalization cohort H_ICU:
hospital admissions (hadm_id) with (i) hospital LOS >= 24h in
hosp/admissions.csv.gz and (ii) >= 1 linked ICU stay record in
icu/icustays.csv.gz for the same hadm_id (see scripts/align_cohorts.py).

Input format: {train,val,test}/meds.parquet. We only rewrite the `code` column for
mapped events; timestamps and numeric_values are left unchanged.

Mapped domains (4)
------------------
  - LAB      (itemid → lab_category)
  - VITAL    (itemid → vital_category)
  - MEDICATION   (drug name → med_category)
  - INFUSION_START / INFUSION_END  (itemid → med_category)

Mapping source
--------------
We use CLIF-MIMIC mapping tables:
  - CLIF-MIMIC/data/mappings/mimic-to-clif-mappings - labs.csv
  - CLIF-MIMIC/data/mappings/mimic-to-clif-mappings - vitals.csv
  - CLIF-MIMIC/data/mappings/mimic-to-clif-mappings - med_category.csv

The MEDS codes are expected to include an identifier as the second field when
split on "//":
  LAB//50931//mg/dL               → itemid-based
  VITAL//220045//bpm              → itemid-based
  MEDICATION//Norepinephrine//Started → drug-name-based
  INFUSION_START//221906          → itemid-based

Unmapped codes are mapped to a sentinel category per domain:
  LAB//__unmapped__
  VITAL//__unmapped__
  MEDICATION//__unmapped__//<action>
  INFUSION_START//__unmapped__
This keeps the transformation total and makes mapping coverage measurable.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _read_split(meds_dir: Path, split: str) -> pl.LazyFrame:
    fp = meds_dir / split / "meds.parquet"
    if not fp.exists():
        raise FileNotFoundError(f"Missing MEDS split parquet: {fp}")
    return pl.scan_parquet(str(fp))


def _write_with_replace(lf: pl.LazyFrame, out_dir: Path, split: str, rep: dict[str, str]) -> None:
    """Apply a single unified replacement dict and write to parquet.

    Using one `replace()` call (not 5 chained ones) avoids 5× full-scan overhead.
    """
    d = out_dir / split
    d.mkdir(parents=True, exist_ok=True)
    if rep:
        lf = lf.with_columns(
            pl.col("code").cast(pl.Utf8).replace(rep, default=pl.col("code").cast(pl.Utf8)).alias("code")
        )
    lf.sink_parquet(d / "meds.parquet")


# ---------------------------------------------------------------------------
# Mapping loaders
# ---------------------------------------------------------------------------

def _load_lab_mapping(labs_csv: Path) -> dict[str, str]:
    df = pl.read_csv(labs_csv, ignore_errors=True)
    if "itemid" not in df.columns or "lab_category" not in df.columns:
        raise ValueError(f"Unexpected labs mapping schema in {labs_csv} (cols={df.columns})")
    out: dict[str, str] = {}
    for row in df.select(pl.col("itemid"), pl.col("lab_category")).iter_rows(named=True):
        itemid = row["itemid"]
        cat = row["lab_category"]
        if itemid is None or cat is None:
            continue
        try:
            out[str(int(itemid))] = str(cat).strip()
        except Exception:
            continue
    return out


def _load_vital_mapping(vitals_csv: Path) -> dict[str, str]:
    df = pl.read_csv(vitals_csv, ignore_errors=True)
    if "itemid" not in df.columns or "vital_category" not in df.columns:
        raise ValueError(f"Unexpected vitals mapping schema in {vitals_csv} (cols={df.columns})")
    out: dict[str, str] = {}
    for row in df.select(pl.col("itemid"), pl.col("vital_category")).iter_rows(named=True):
        itemid = row["itemid"]
        cat = row["vital_category"]
        if itemid is None or cat is None:
            continue
        try:
            out[str(int(itemid))] = str(cat).strip()
        except Exception:
            continue
    return out


def _load_med_mapping(med_csv: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Load med_category.csv and return two dicts:
      - itemid_to_cat: str(itemid) → med_category   (for INFUSION_START/END)
      - label_to_cat:  normalized_label → med_category (for MEDICATION)

    Only rows with decision != 'NO MAPPING' are included.
    """
    df = pl.read_csv(med_csv, ignore_errors=True)
    required = {"itemid", "label", "med_category", "decision"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {missing} in {med_csv} (cols={df.columns})")

    itemid_to_cat: dict[str, str] = {}
    label_to_cat: dict[str, str] = {}

    for row in df.select("itemid", "label", "med_category", "decision").iter_rows(named=True):
        decision = str(row["decision"]).strip().upper() if row["decision"] else ""
        if decision == "NO MAPPING":
            continue
        cat = row["med_category"]
        if cat is None or str(cat).strip() == "":
            continue
        cat = str(cat).strip()

        # itemid mapping (for infusions)
        itemid = row["itemid"]
        if itemid is not None:
            try:
                itemid_to_cat[str(int(itemid))] = cat
            except (ValueError, TypeError):
                pass

        # label mapping (for MEDICATION drug names)
        label = row["label"]
        if label is not None and str(label).strip():
            norm_label = _normalize_drug_name(str(label).strip())
            label_to_cat[norm_label] = cat

    return itemid_to_cat, label_to_cat


def _normalize_drug_name(name: str) -> str:
    """Normalize a drug name for fuzzy matching.

    Strategy: lowercase, strip parenthetical suffixes, collapse whitespace.
    E.g., "HYDROmorphone (Dilaudid)" → "hydromorphone"
          "Sodium Chloride 0.9%  Flush" → "sodium chloride 0.9% flush"
    """
    s = name.lower().strip()
    # Remove parenthetical suffixes like (Dilaudid), (Precedex), etc.
    s = re.sub(r"\s*\(.*?\)\s*", " ", s)
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ---------------------------------------------------------------------------
# Code extraction helpers
# ---------------------------------------------------------------------------

def _extract_field(code: str, index: int) -> str | None:
    """Extract the nth field from a '//' delimited MEDS code."""
    parts = code.split("//")
    if len(parts) <= index:
        return None
    return parts[index]


# ---------------------------------------------------------------------------
# Mapping application
# ---------------------------------------------------------------------------

def _apply_itemid_mapping(
    lf: pl.LazyFrame,
    *,
    domain_prefix: str,
    itemid_to_cat: dict[str, str],
    out_prefix: str,
    unmapped_cat: str,
    itemid_field_index: int = 1,
    preserve_suffix: bool = False,
) -> tuple[pl.LazyFrame, dict[str, int]]:
    """Map codes by extracting an itemid field and looking it up in itemid_to_cat.

    Memory-optimised: groups by unique codes first, then builds the replacement dict.
    If preserve_suffix is True, keeps fields after the itemid (e.g., units).
    """
    # Collect UNIQUE codes with their counts (not all rows)
    df_unique = (
        lf.filter(pl.col("code").is_not_null() & pl.col("code").cast(pl.Utf8).str.starts_with(domain_prefix + "//"))
        .select(pl.col("code").cast(pl.Utf8).alias("code"))
        .group_by("code")
        .agg(pl.len().alias("count"))
        .collect(streaming=True)
    )
    n_total = int(df_unique["count"].sum())
    n_mapped = 0
    rep: dict[str, str] = {}

    for row in df_unique.iter_rows(named=True):
        c = str(row["code"])
        cnt = int(row["count"])
        itemid = _extract_field(c, itemid_field_index)
        if itemid is None:
            rep[c] = f"{out_prefix}//{unmapped_cat}"
            continue
        cat = itemid_to_cat.get(str(itemid))
        if cat is None or cat.strip() == "" or cat.upper() == "NO MAPPING":
            rep[c] = f"{out_prefix}//{unmapped_cat}"
            continue
        n_mapped += cnt
        if preserve_suffix:
            parts = c.split("//")
            suffix = "//".join(parts[itemid_field_index + 1:]) if len(parts) > itemid_field_index + 1 else ""
            rep[c] = f"{out_prefix}//{cat}//{suffix}" if suffix else f"{out_prefix}//{cat}"
        else:
            rep[c] = f"{out_prefix}//{cat}"

    stats = {
        "n_total": n_total,
        "n_mapped": n_mapped,
        "n_unmapped": n_total - n_mapped,
    }

    out_lf = lf.with_columns(
        pl.when(pl.col("code").cast(pl.Utf8).str.starts_with(domain_prefix + "//"))
        .then(pl.col("code").cast(pl.Utf8).replace(rep, default=pl.col("code").cast(pl.Utf8)))
        .otherwise(pl.col("code").cast(pl.Utf8))
        .alias("code")
    )
    return out_lf, stats


def _apply_medication_name_mapping(
    lf: pl.LazyFrame,
    *,
    label_to_cat: dict[str, str],
    unmapped_cat: str = "__unmapped__",
) -> tuple[pl.LazyFrame, dict[str, int]]:
    """Map MEDICATION codes by matching the drug name field to CLIF labels.

    Memory-optimised: groups by unique codes first.
    MEDS format: MEDICATION//<drugname>//<action>
    Output:      MEDICATION//<med_category>//<action>
    """
    # Collect UNIQUE codes with counts
    df_unique = (
        lf.filter(
            pl.col("code").is_not_null()
            & pl.col("code").cast(pl.Utf8).str.starts_with("MEDICATION//")
        )
        .select(pl.col("code").cast(pl.Utf8).alias("code"))
        .group_by("code")
        .agg(pl.len().alias("count"))
        .collect(streaming=True)
    )
    n_total = int(df_unique["count"].sum())
    n_mapped = 0
    rep: dict[str, str] = {}

    for row in df_unique.iter_rows(named=True):
        c_str = str(row["code"])
        cnt = int(row["count"])
        parts = c_str.split("//")
        if len(parts) < 2:
            rep[c_str] = f"MEDICATION//{unmapped_cat}"
            continue
        drug_name = parts[1]
        action = parts[2] if len(parts) > 2 else ""
        norm_name = _normalize_drug_name(drug_name)

        cat = label_to_cat.get(norm_name)
        if cat is None or cat.strip() == "":
            rep[c_str] = f"MEDICATION//{unmapped_cat}//{action}" if action else f"MEDICATION//{unmapped_cat}"
            continue

        n_mapped += cnt
        rep[c_str] = f"MEDICATION//{cat}//{action}" if action else f"MEDICATION//{cat}"

    stats = {
        "n_total": n_total,
        "n_mapped": n_mapped,
        "n_unmapped": n_total - n_mapped,
    }

    out_lf = lf.with_columns(
        pl.when(pl.col("code").cast(pl.Utf8).str.starts_with("MEDICATION//"))
        .then(pl.col("code").cast(pl.Utf8).replace(rep, default=pl.col("code").cast(pl.Utf8)))
        .otherwise(pl.col("code").cast(pl.Utf8))
        .alias("code")
    )
    return out_lf, stats


# ---------------------------------------------------------------------------
# Domain frequency helpers
# ---------------------------------------------------------------------------

def _domain_counts_codes(lf: pl.LazyFrame, *, prefix: str) -> Counter[str]:
    """Get code→count mapping for a domain prefix. Memory-optimised via group-by."""
    df = (
        lf.filter(pl.col("code").is_not_null() & pl.col("code").cast(pl.Utf8).str.starts_with(prefix))
        .select(pl.col("code").cast(pl.Utf8).alias("code"))
        .group_by("code")
        .agg(pl.len().alias("count"))
        .collect(streaming=True)
    )
    return Counter({str(r["code"]): int(r["count"]) for r in df.iter_rows(named=True)})


# ---------------------------------------------------------------------------
# Null control construction
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Pooled numeric stats for frequency-matched controls
# ---------------------------------------------------------------------------

def _compute_pooled_numeric_stats(
    meds_dir: Path,
    *,
    mapping: dict[str, str],
    domain_prefix: str,
    num_bins: int = 10,
    split: str = "train",
) -> dict[str, dict]:
    lf = _read_split(meds_dir, split)
    df = (
        lf.filter(
            pl.col("code").is_not_null()
            & pl.col("code").cast(pl.Utf8).str.starts_with(domain_prefix)
            & pl.col("numeric_value").is_not_null()
        )
        .select(
            pl.col("code").cast(pl.Utf8).alias("code"),
            pl.col("numeric_value").cast(pl.Float64).alias("numeric_value"),
        )
        .collect(streaming=True)
    )

    cat_to_src: dict[str, list[str]] = {}
    for src, tgt in mapping.items():
        cat_to_src.setdefault(tgt, []).append(src)

    stats: dict[str, dict] = {}
    for tgt_cat, src_codes in sorted(cat_to_src.items()):
        vals = df.filter(pl.col("code").is_in(src_codes))["numeric_value"].to_numpy()
        vals = vals[np.isfinite(vals)]
        if len(vals) < num_bins:
            stats[tgt_cat] = {
                "n_values": int(len(vals)),
                "boundaries": [],
                "src_codes_pooled": len(src_codes),
            }
            continue
        quantiles = np.linspace(0, 1, num_bins + 1)
        boundaries = np.quantile(vals, quantiles).tolist()
        stats[tgt_cat] = {
            "n_values": int(len(vals)),
            "boundaries": boundaries,
            "src_codes_pooled": len(src_codes),
            "median": float(np.median(vals)),
            "iqr": float(np.quantile(vals, 0.75) - np.quantile(vals, 0.25)),
        }
    return stats


# ---------------------------------------------------------------------------
# Code-to-code mapping for null controls
# ---------------------------------------------------------------------------

def _apply_code_to_code_mapping(lf: pl.LazyFrame, *, domain_prefix: str, mapping: dict[str, str]) -> pl.LazyFrame:
    rep = {k: v for k, v in mapping.items()}
    return lf.with_columns(
        pl.when(pl.col("code").cast(pl.Utf8).str.starts_with(domain_prefix))
        .then(pl.col("code").cast(pl.Utf8).replace(rep, default=pl.col("code").cast(pl.Utf8)))
        .otherwise(pl.col("code").cast(pl.Utf8))
        .alias("code")
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# Domain registry: (prefix, mapping_key_type)
DOMAINS = [
    ("LAB", "itemid"),
    ("VITAL", "itemid"),
    ("MEDICATION", "label"),
    ("INFUSION_START", "itemid"),
    ("INFUSION_END", "itemid"),
]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build Exp3 MEDS-only semantics arms (native + mapped + null controls).")
    p.add_argument(
        "--meds_in_dir",
        type=Path,
        required=True,
        help=(
            "MEDS events dir filtered to Exp3 ICU-hospitalization cohort H_ICU, with "
            "{train,val,test}/meds.parquet."
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

    mappings_dir = clif_mimic_repo / "data" / "mappings"

    # Load all mapping CSVs
    labs_csv = mappings_dir / "mimic-to-clif-mappings - labs.csv"
    vitals_csv = mappings_dir / "mimic-to-clif-mappings - vitals.csv"
    med_csv = mappings_dir / "mimic-to-clif-mappings - med_category.csv"

    for f in (labs_csv, vitals_csv, med_csv):
        if not f.exists():
            raise FileNotFoundError(f"Missing CLIF-MIMIC mapping CSV: {f}")

    lab_itemid_to_cat = _load_lab_mapping(labs_csv)
    vit_itemid_to_cat = _load_vital_mapping(vitals_csv)
    med_itemid_to_cat, med_label_to_cat = _load_med_mapping(med_csv)

    print(f"Loaded mappings:")
    print(f"  LAB:       {len(lab_itemid_to_cat)} itemid→category")
    print(f"  VITAL:     {len(vit_itemid_to_cat)} itemid→category")
    print(f"  MED(id):   {len(med_itemid_to_cat)} itemid→med_category (for INFUSION)")
    print(f"  MED(name): {len(med_label_to_cat)} label→med_category (for MEDICATION)")

    # ===================================================================
    # Helper: build a unified old→new replacement dict from unique codes
    # ===================================================================
    def _build_mapped_rep(unique_codes: list[str]) -> tuple[dict[str, str], dict[str, dict[str, int]]]:
        """Build one flat {old_code: new_code} dict across all 5 domains.

        Returns (rep_dict, per_domain_stats).
        """
        rep: dict[str, str] = {}
        stats: dict[str, dict[str, int]] = {}

        for c in unique_codes:
            c_str = str(c)

            # LAB: LAB//<itemid>//... → LAB//<lab_category>//...
            if c_str.startswith("LAB//"):
                itemid = _extract_field(c_str, 1)
                if itemid is not None:
                    cat = lab_itemid_to_cat.get(str(itemid))
                    if cat and cat.strip() and cat.upper() != "NO MAPPING":
                        parts = c_str.split("//")
                        suffix = "//".join(parts[2:]) if len(parts) > 2 else ""
                        rep[c_str] = f"LAB//{cat}//{suffix}" if suffix else f"LAB//{cat}"
                    else:
                        # keep original for unmapped
                        pass

            # VITAL: VITAL//<itemid>//... → VITAL//<vital_category>//...
            elif c_str.startswith("VITAL//"):
                itemid = _extract_field(c_str, 1)
                if itemid is not None:
                    cat = vit_itemid_to_cat.get(str(itemid))
                    if cat and cat.strip() and cat.upper() != "NO MAPPING":
                        parts = c_str.split("//")
                        suffix = "//".join(parts[2:]) if len(parts) > 2 else ""
                        rep[c_str] = f"VITAL//{cat}//{suffix}" if suffix else f"VITAL//{cat}"

            # MEDICATION: MEDICATION//<drugname>//<action> → MEDICATION//<med_cat>//<action>
            elif c_str.startswith("MEDICATION//"):
                parts = c_str.split("//")
                if len(parts) >= 2:
                    drug_name = parts[1]
                    action = parts[2] if len(parts) > 2 else ""
                    norm_name = _normalize_drug_name(drug_name)
                    cat = med_label_to_cat.get(norm_name)
                    if cat and cat.strip():
                        rep[c_str] = f"MEDICATION//{cat}//{action}" if action else f"MEDICATION//{cat}"

            # INFUSION_START: INFUSION_START//<itemid> → INFUSION_START//<med_cat>
            elif c_str.startswith("INFUSION_START//"):
                itemid = _extract_field(c_str, 1)
                if itemid is not None:
                    cat = med_itemid_to_cat.get(str(itemid))
                    if cat and cat.strip() and cat.upper() != "NO MAPPING":
                        rep[c_str] = f"INFUSION_START//{cat}"

            # INFUSION_END: INFUSION_END//<itemid> → INFUSION_END//<med_cat>
            elif c_str.startswith("INFUSION_END//"):
                itemid = _extract_field(c_str, 1)
                if itemid is not None:
                    cat = med_itemid_to_cat.get(str(itemid))
                    if cat and cat.strip() and cat.upper() != "NO MAPPING":
                        rep[c_str] = f"INFUSION_END//{cat}"

        return rep, {}

    # ===================================================================
    # 1) MEDS→CLIF mapped arm (single-pass per split)
    # ===================================================================
    mapped_dir = out_root / "meds_mapped"
    mapped_dir.mkdir(parents=True, exist_ok=True)
    coverage: dict[str, dict] = {}

    for split in ["train", "val", "test"]:
        print(f"\n--- Mapping {split} ---")
        lf = _read_split(meds_in, split)

        # Read unique codes with counts — single group-by scan
        df_unique = (
            lf.select(pl.col("code").cast(pl.Utf8).alias("code"))
            .filter(pl.col("code").is_not_null())
            .group_by("code")
            .agg(pl.len().alias("count"))
            .collect(streaming=True)
        )
        unique_codes = df_unique["code"].to_list()
        code_counts = {str(r["code"]): int(r["count"]) for r in df_unique.iter_rows(named=True)}
        print(f"  Unique codes: {len(unique_codes):,}")

        # Build unified replacement dict
        rep, _ = _build_mapped_rep(unique_codes)
        print(f"  Unified replacement dict: {len(rep):,} entries")

        # Compute per-domain stats from the replacement dict + code counts
        split_coverage: dict[str, dict] = {}
        for prefix in ["LAB", "VITAL", "MEDICATION", "INFUSION_START", "INFUSION_END"]:
            domain_codes = [c for c in unique_codes if c.startswith(prefix + "//")]
            n_total = sum(code_counts.get(c, 0) for c in domain_codes)
            n_mapped = sum(code_counts.get(c, 0) for c in domain_codes if c in rep)
            split_coverage[prefix] = {"n_total": n_total, "n_mapped": n_mapped, "n_unmapped": n_total - n_mapped}
            print(f"  {prefix:20s}: {n_mapped:>8,} / {n_total:>8,} mapped")

        coverage[split] = split_coverage

        # Write in a single pass
        lf = _read_split(meds_in, split)  # re-scan to avoid stale LazyFrame
        _write_with_replace(lf, mapped_dir, split, rep)
        print(f"  Wrote {split}")

    (mapped_dir / "mappings").mkdir(parents=True, exist_ok=True)
    (mapped_dir / "mappings" / "meta.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "seed": int(args.seed),
                "mapping_source": "CLIF-MIMIC mapping CSVs (labs.csv, vitals.csv, med_category.csv)",
                "domains_mapped": ["LAB", "VITAL", "MEDICATION", "INFUSION_START", "INFUSION_END"],
                "coverage": coverage,
                "note": "Row-fixed mapping: only MEDS `code` is rewritten for mapped domains; times/numeric_values unchanged.",
            },
            indent=2,
            sort_keys=True,
        )
    )

    # ===================================================================
    # 2) Build null controls using the target vocabulary from meds_mapped
    # ===================================================================
    mapped_train = _read_split(mapped_dir, "train")

    # Collect target distributions per domain
    domain_tgt_counts: dict[str, Counter[str]] = {}
    domain_tgt_categories: dict[str, list[str]] = {}
    domain_src_counts: dict[str, Counter[str]] = {}

    meds_train = _read_split(meds_in, "train")

    for prefix, _ in DOMAINS:
        domain_tgt_counts[prefix] = _domain_counts_codes(mapped_train, prefix=prefix + "//")
        domain_tgt_categories[prefix] = list(domain_tgt_counts[prefix].keys())
        domain_src_counts[prefix] = _domain_counts_codes(meds_train, prefix=prefix + "//")
        print(f"\n{prefix}: {len(domain_src_counts[prefix])} src codes → {len(domain_tgt_categories[prefix])} tgt categories")

    # Build randomized and freq-matched mappings for all domains
    rand_mappings: dict[str, dict[str, str]] = {}
    freq_mappings: dict[str, dict[str, str]] = {}
    for prefix, _ in DOMAINS:
        if not domain_tgt_categories[prefix]:
            print(f"  WARNING: no target categories for {prefix}, skipping null controls for this domain.")
            continue
        rand_mappings[prefix] = _make_random_mapping(
            src_codes=list(domain_src_counts[prefix].keys()),
            tgt_categories=domain_tgt_categories[prefix],
            rng=rng,
        )
        freq_mappings[prefix] = _make_freqmatched_mapping(
            src_counts=domain_src_counts[prefix],
            tgt_counts=domain_tgt_counts[prefix],
        )

    for arm_name, domain_maps in {
        "meds_randomized": rand_mappings,
        "meds_freqmatched": freq_mappings,
    }.items():
        arm_dir = out_root / arm_name
        arm_dir.mkdir(parents=True, exist_ok=True)
        (arm_dir / "mappings").mkdir(parents=True, exist_ok=True)

        # Save per-domain mappings
        for prefix, mapping in domain_maps.items():
            (arm_dir / "mappings" / f"{prefix.lower()}_mapping.json").write_text(
                json.dumps(mapping, indent=2, sort_keys=True)
            )

        (arm_dir / "mappings" / "meta.json").write_text(
            json.dumps(
                {
                    "schema_version": 3,
                    "seed": int(args.seed),
                    "domains": list(domain_maps.keys()),
                    "target_distribution": "Derived from meds_mapped train split (row-fixed standardized identifier scheme).",
                    "numeric_matched": True,
                    "note": (
                        "Decile boundaries are computed on pooled numeric_value "
                        "distributions from all native codes mapped to the same "
                        "target category. This guarantees that any AUROC gain in "
                        "the true CLIF arm is from semantics, not from smoother "
                        "decile boundaries caused by merging codes."
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )

        # Merge all domain mappings into a single flat dict — single pass per split
        unified_null_rep: dict[str, str] = {}
        for prefix, mapping in domain_maps.items():
            unified_null_rep.update(mapping)
        print(f"\n{arm_name}: unified null replacement dict has {len(unified_null_rep):,} entries")

        for split in ["train", "val", "test"]:
            lf = _read_split(meds_in, split)
            _write_with_replace(lf, arm_dir, split, unified_null_rep)
            print(f"  Wrote {split}")

        # Compute pooled numeric distributions for numeric domains
        for prefix in ["LAB", "VITAL"]:
            if prefix in domain_maps:
                numeric_stats = _compute_pooled_numeric_stats(
                    meds_in, mapping=domain_maps[prefix], domain_prefix=prefix,
                    num_bins=10, split="train",
                )
                (arm_dir / "mappings" / f"{prefix.lower()}_numeric_stats.json").write_text(
                    json.dumps(numeric_stats, indent=2, sort_keys=True)
                )
                print(f"  Computed pooled numeric stats for {arm_name}/{prefix}: {len(numeric_stats)} categories")

    print("\nOK")
    print(f"  Wrote meds_mapped under:      {mapped_dir}")
    print(f"  Wrote meds_randomized under:  {out_root / 'meds_randomized'}")
    print(f"  Wrote meds_freqmatched under: {out_root / 'meds_freqmatched'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
