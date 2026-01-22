#!/usr/bin/env python3
"""
Quantify notation ambiguity in MEDS-formatted data.

Motivation (internal validity of representation experiments):
- Single-site EHR data can still exhibit substantial within-site heterogeneity in how
  clinically similar concepts are recorded (e.g., qualitative lab results and other
  text-valued fields).
- This script measures the *pervasiveness* of text-valued events in MEDS and summarizes
  common surface-form variants for a small, explicitly defined lexicon of canonical labels
  (e.g., "positive" vs. "negative").

This script is designed to be safe on large datasets:
- Uses Polars lazy scanning over Parquet shards.
- Produces exact counts for simple aggregates.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import fire
import polars as pl


@dataclass(frozen=True)
class Lexicon:
    positive: tuple[str, ...] = (
        "+",
        "pos",
        "positive",
        "detected",
        "reactive",
        "present",
        "yes",
        "y",
        "true",
    )
    negative: tuple[str, ...] = (
        "-",
        "neg",
        "negative",
        "not detected",
        "nondetected",
        "non reactive",
        "nonreactive",
        "absent",
        "none",
        "no",
        "n",
        "false",
    )
    trace: tuple[str, ...] = ("trace",)
    equivocal: tuple[str, ...] = ("equivocal", "indeterminate", "borderline")
    unknown: tuple[str, ...] = ("unk", "unknown", "na", "n/a", "not available")


def _domain_expr() -> pl.Expr:
    # MEDS code format often looks like "LAB//50931//mg/dL" or "GENDER//M".
    # We treat the first token before the first delimiter as a coarse domain.
    return pl.col("code").str.split_exact("//", 1).struct.field("field_0").alias("domain")


def _text_stripped_expr() -> pl.Expr:
    return (
        pl.col("text_value")
        .fill_null("")
        .cast(pl.Utf8)
        .str.strip_chars()
        .alias("text_stripped")
    )


def _text_norm_expr() -> pl.Expr:
    # Normalize for surface-form counting:
    # - lowercase
    # - trim + collapse whitespace
    # - remove most punctuation but keep + and - (common shorthand)
    return (
        pl.col("text_stripped")
        .str.to_lowercase()
        .str.replace_all(r"[\s\t\n\r]+", " ")
        .str.replace_all(r"[^a-z0-9\+\-\.\ ]", "")
        .str.strip_chars()
        .alias("text_norm")
    )


def _canon_expr(lex: Lexicon) -> pl.Expr:
    t = pl.col("text_norm")

    # Exact-match lexicon mapping (deliberately conservative for reproducibility).
    return (
        pl.when(t.is_in(list(lex.positive)))
        .then(pl.lit("positive"))
        .when(t.is_in(list(lex.negative)))
        .then(pl.lit("negative"))
        .when(t.is_in(list(lex.trace)))
        .then(pl.lit("trace"))
        .when(t.is_in(list(lex.equivocal)))
        .then(pl.lit("equivocal"))
        .when(t.is_in(list(lex.unknown)))
        .then(pl.lit("unknown"))
        .otherwise(pl.lit("other"))
        .alias("canon")
    )


def main(
    *,
    meds_data_dir: str = "benchmarks/mimic-meds-extraction/data/meds/data",
    split: str = "train",
    out_dir: str = "outputs/meds_ambiguity",
    top_k_forms: int = 25,
) -> None:
    meds_data_dir_p = Path(meds_data_dir).expanduser().resolve()
    split_dir = meds_data_dir_p / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split dir not found: {split_dir}")

    out_dir_p = Path(out_dir).expanduser().resolve()
    out_dir_p.mkdir(parents=True, exist_ok=True)

    lf = pl.scan_parquet(str(split_dir / "*.parquet")).with_columns(
        _domain_expr(), _text_stripped_expr()
    )
    has_text = pl.col("text_stripped").str.len_chars() > 0

    # (1) Pervasiveness: text-valued rows by domain (exact; no sampling).
    by_domain = (
        lf.group_by("domain")
        .agg(
            total=pl.len(),
            text_rows=has_text.cast(pl.Int64).sum(),
        )
        .with_columns(text_frac=(pl.col("text_rows") / pl.col("total")))
        .sort("text_frac", descending=True)
        .collect(streaming=True)
    )

    # (2) Surface-form ambiguity: counts of normalized strings and canonical labels.
    lex = Lexicon()
    lf_text = (
        lf.filter(has_text)
        .with_columns(_text_norm_expr())
        .with_columns(_canon_expr(lex))
    )

    canon_counts = (
        lf_text.group_by("canon")
        .agg(n=pl.len(), n_unique_norm=pl.col("text_norm").n_unique())
        .sort("n", descending=True)
        .collect(streaming=True)
    )

    # Top surface forms per canonical label (rank within canon).
    forms = (
        lf_text.group_by(["canon", "text_norm"])
        .agg(n=pl.len())
        .with_columns(rank=pl.col("n").rank("dense", descending=True).over("canon"))
        .filter(pl.col("rank") <= top_k_forms)
        .sort(["canon", "n"], descending=[False, True])
        .collect(streaming=True)
    )

    # Save artifacts
    summary = {
        "meds_data_dir": str(meds_data_dir_p),
        "split": split,
        "top_k_forms": top_k_forms,
        "by_domain": by_domain.to_dicts(),
        "canon_counts": canon_counts.to_dicts(),
        "top_forms": forms.to_dicts(),
        "lexicon": {
            "positive": list(lex.positive),
            "negative": list(lex.negative),
            "trace": list(lex.trace),
            "equivocal": list(lex.equivocal),
            "unknown": list(lex.unknown),
        },
    }
    (out_dir_p / f"{split}_summary.json").write_text(json.dumps(summary, indent=2))

    # Human-readable stdout
    print("=== MEDS notation ambiguity QC ===")
    print(f"split_dir: {split_dir}")
    print("")
    print("== text-valued rows by domain (text_value non-empty) ==")
    print(by_domain)
    print("")
    print("== canonical label summary (within text-valued rows) ==")
    print(canon_counts)
    print("")
    print(f"== top {top_k_forms} normalized surface forms per canon ==")
    print(forms)
    print("")
    print(f"Wrote: {(out_dir_p / f'{split}_summary.json')}")


if __name__ == "__main__":
    fire.Fire(main)

