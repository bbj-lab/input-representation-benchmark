#!/usr/bin/env python3
"""
Build Experiment 3 semantic-control arms as MEDS-format datasets.

Goal (internal validity)
---------------------------------------
Exp3 compares MEDS-native vs CLIF-standardized vocabularies on an ICU-aligned cohort.
To validate that any observed CLIF advantage is due to *semantics* (not merely
vocabulary size or regularization), we build two explicit null-model arms:

1) Randomized mapping control:
   - Map MEDS LAB/VITAL codes to CLIF lab_category/vital_category *at random* (within domain).
   - Preserves the target vocabulary *support* (same category names) but destroys semantics.

2) Frequency-matched collapse control:
   - Map MEDS LAB/VITAL codes to CLIF categories such that the resulting category
     frequencies on the MEDS train split approximately match CLIF category frequencies.
   - Tests whether gains come from controlled-vocab frequency/regularization rather than semantics.

These are written out as MEDS-parquet splits that can be tokenized with the existing
MEDS Exp3 tokenizer config (events filtered by code prefix LAB/VITAL).

Validity note
-------------
- Internal validity: these controls are designed to falsify the claim
  “semantic standardization improves downstream performance” by holding constant
  the cohort, timestamps, and numeric values while perturbing only the mapping from
  native codes → standardized concepts.
- External validity: Exp3 is intentionally restricted to a matched-signal ICU cohort
  (vitals+labs) for parity with CLIF; this improves internal validity but reduces
  generalizability to other care settings/modalities.

Inputs
------
- --meds_in_dir: ICU-aligned MEDS events dir with {train,val,test}/meds.parquet
- --clif_raw_dir: CLIF split root containing raw/{train,val,test}/clif_labs.parquet and clif_vitals.parquet

Outputs
-------
Writes:
- <out_root>/meds_randomized/{train,val,test}/meds.parquet
- <out_root>/meds_freqmatched/{train,val,test}/meds.parquet
and mapping JSONs under each arm dir.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

try:
    import polars as pl  # type: ignore
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "polars is required for Exp3 control-arm construction. "
        "Install via this repo's dependencies (see pyproject.toml) or run in the configured environment."
    ) from e


def _read_meds_split(meds_dir: Path, split: str) -> pl.LazyFrame:
    fp = meds_dir / split / "meds.parquet"
    if not fp.exists():
        raise FileNotFoundError(f"Missing MEDS split parquet: {fp}")
    return pl.scan_parquet(str(fp))


def _read_clif_split(clif_raw_dir: Path, split: str, fname: str) -> pl.LazyFrame:
    fp = clif_raw_dir / split / fname
    if not fp.exists():
        raise FileNotFoundError(f"Missing CLIF split parquet: {fp}")
    return pl.scan_parquet(str(fp))


def _domain_counts_meds(lf: pl.LazyFrame, *, prefix: str) -> Counter[str]:
    # Count events per raw MEDS code for a domain (LAB or VITAL)
    df = (
        lf.filter(pl.col("code").is_not_null() & pl.col("code").cast(pl.Utf8).str.starts_with(prefix))
        .select(pl.col("code").cast(pl.Utf8).alias("code"))
        .collect(streaming=True)
    )
    return Counter(df["code"].to_list())


def _domain_counts_clif(lf: pl.LazyFrame, *, col: str) -> Counter[str]:
    df = lf.select(pl.col(col).cast(pl.Utf8).alias(col)).collect(streaming=True)
    vals = [v for v in df[col].to_list() if v is not None and v != ""]
    return Counter(vals)


def _make_random_mapping(
    *,
    src_codes: list[str],
    tgt_categories: list[str],
    rng: random.Random,
) -> dict[str, str]:
    # Deterministic, coverage-guaranteed mapping:
    # Shuffle target categories and assign by cycling, so all categories appear.
    if not tgt_categories:
        raise ValueError("Target category list is empty.")
    tgt = list(tgt_categories)
    rng.shuffle(tgt)
    out: dict[str, str] = {}
    for i, c in enumerate(sorted(src_codes)):
        out[c] = tgt[i % len(tgt)]
    return out


def _make_freqmatched_mapping(
    *,
    src_counts: Counter[str],
    tgt_counts: Counter[str],
) -> dict[str, str]:
    """
    Greedy frequency-matching:
    - Sort MEDS codes by descending frequency.
    - Maintain remaining target counts per category.
    - Assign each MEDS code to the category with the largest remaining budget.
    This is a simple, deterministic baseline; good enough for a null model.
    """
    if not tgt_counts:
        raise ValueError("Target count distribution is empty.")
    remaining = {k: int(v) for k, v in tgt_counts.items()}
    cats = sorted(remaining.keys(), key=lambda k: remaining[k], reverse=True)
    mapping: dict[str, str] = {}

    for code, cnt in sorted(src_counts.items(), key=lambda kv: kv[1], reverse=True):
        # pick category with max remaining budget (ties broken lexicographically)
        cats.sort(key=lambda k: (remaining[k], k), reverse=True)
        chosen = cats[0]
        mapping[code] = chosen
        remaining[chosen] = max(0, remaining[chosen] - int(cnt))
    return mapping


def _apply_mapping(
    lf: pl.LazyFrame,
    *,
    domain_prefix: str,
    mapping: dict[str, str],
    out_prefix: str,
) -> pl.LazyFrame:
    # Replace codes in the given domain with a category-based code that keeps the
    # original domain prefix (so the tokenizer filter_expr still applies).
    #
    # Example:
    #   LAB//50931//mg/dL  -> LAB//glucose_serum
    #   VITAL//220045//bpm -> VITAL//heart_rate
    rep = {k: f"{out_prefix}//{v}" for k, v in mapping.items()}
    return lf.with_columns(
        pl.when(pl.col("code").cast(pl.Utf8).str.starts_with(domain_prefix))
        .then(pl.col("code").cast(pl.Utf8).replace(rep, default=pl.col("code").cast(pl.Utf8)))
        .otherwise(pl.col("code").cast(pl.Utf8))
        .alias("code")
    )


def _write_split(lf: pl.LazyFrame, out_dir: Path, split: str) -> None:
    d = out_dir / split
    d.mkdir(parents=True, exist_ok=True)
    lf.sink_parquet(d / "meds.parquet")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build Exp3 semantic-control arms (MEDS randomized + freq-matched).")
    p.add_argument("--meds_in_dir", type=Path, required=True, help="ICU-aligned MEDS events dir (train/val/test/meds.parquet).")
    p.add_argument("--clif_raw_dir", type=Path, required=True, help="CLIF split root containing raw/{train,val,test}/clif_*.parquet.")
    p.add_argument("--out_root", type=Path, default=Path("data/exp3/arms"), help="Output root (default: data/exp3/arms).")
    p.add_argument("--seed", type=int, default=42, help="Seed for randomized mapping control.")
    args = p.parse_args(argv)

    meds_in = args.meds_in_dir.expanduser().resolve()
    clif_raw = args.clif_raw_dir.expanduser().resolve()
    out_root = args.out_root.expanduser().resolve()
    rng = random.Random(int(args.seed))

    # CLIF category distributions (train split)
    clif_labs = _read_clif_split(clif_raw, "train", "clif_labs.parquet")
    clif_vitals = _read_clif_split(clif_raw, "train", "clif_vitals.parquet")
    lab_tgt = _domain_counts_clif(clif_labs, col="lab_category")
    vit_tgt = _domain_counts_clif(clif_vitals, col="vital_category")

    # MEDS code distributions (train split)
    meds_train = _read_meds_split(meds_in, "train")
    lab_src = _domain_counts_meds(meds_train, prefix="LAB")
    vit_src = _domain_counts_meds(meds_train, prefix="VITAL")

    # Build mappings
    lab_rand = _make_random_mapping(src_codes=list(lab_src.keys()), tgt_categories=list(lab_tgt.keys()), rng=rng)
    vit_rand = _make_random_mapping(src_codes=list(vit_src.keys()), tgt_categories=list(vit_tgt.keys()), rng=rng)

    lab_freq = _make_freqmatched_mapping(src_counts=lab_src, tgt_counts=lab_tgt)
    vit_freq = _make_freqmatched_mapping(src_counts=vit_src, tgt_counts=vit_tgt)

    # Apply per-split
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
                    "internal_validity": (
                        "Null model for semantics: preserves cohort/timestamps/values while perturbing "
                        "native→standardized mapping within domain."
                    ),
                    "external_validity": (
                        "Restricted to ICU-aligned vitals+labs cohort for matched-signal parity with CLIF."
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )

        for split in ["train", "val", "test"]:
            lf = _read_meds_split(meds_in, split)
            lf = _apply_mapping(lf, domain_prefix="LAB", mapping=lab_map, out_prefix="LAB")
            lf = _apply_mapping(lf, domain_prefix="VITAL", mapping=vit_map, out_prefix="VITAL")
            _write_split(lf, arm_dir, split)

    print("OK")
    print(f"  Wrote control arms under: {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

