#!/usr/bin/env python3
"""
Validate MEDS vs CLIF outcome label parity (with emphasis on IMV).

This script is intended for Experiment 3 parity checks, where MEDS and CLIF are
run on the *same ICU ≥24h cohort*.

Why this matters
----------------
The CLIF pipeline in `fms-ehrs` was designed and tested end-to-end. For MEDS,
outcome extraction requires special handling (notably IMV timing) because MEDS
tokenization can append procedures as suffix tokens at discharge time, which
breaks 24h-window logic if computed via token presence.

Inputs
------
Provide base directories for MEDS and CLIF tokenized datasets. Each directory
should contain split subdirectories (train/val/test) with
`tokens_timelines_outcomes.parquet`.

Outputs
-------
Writes a JSON report including confusion matrices and disagreement rates for:
  - same_admission_death
  - long_length_of_stay
  - icu_admission
  - imv_event
  - icu_admission_24h
  - imv_event_24h

CLIF is treated as the reference label source.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import polars as pl


DEFAULT_LABEL_COLS: tuple[str, ...] = (
    "same_admission_death",
    "long_length_of_stay",
    "icu_admission",
    "imv_event",
    "icu_admission_24h",
    "imv_event_24h",
)


def _confusion_counts(df: pl.DataFrame, *, y_true: str, y_pred: str) -> dict[str, int]:
    # Ensure boolean, fill null as False.
    t = df[y_true].fill_null(False).cast(pl.Boolean)
    p = df[y_pred].fill_null(False).cast(pl.Boolean)

    tp = int(((t) & (p)).sum())
    tn = int((~t & ~p).sum())
    fp = int((~t & p).sum())
    fn = int((t & ~p).sum())
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def _rates_from_counts(c: dict[str, int]) -> dict[str, float]:
    tp, tn, fp, fn = c["tp"], c["tn"], c["fp"], c["fn"]
    total = tp + tn + fp + fn
    pos = tp + fn
    neg = tn + fp

    def div(a: float, b: float) -> float:
        return float(a) / float(b) if b else float("nan")

    return {
        "n": float(total),
        "prevalence": div(pos, total),
        "accuracy": div(tp + tn, total),
        "sensitivity": div(tp, pos),  # recall
        "specificity": div(tn, neg),
        "ppv": div(tp, tp + fp),  # precision
        "npv": div(tn, tn + fn),
        "fpr": div(fp, neg),
        "fnr": div(fn, pos),
    }


def _load_outcomes(path: Path) -> pl.LazyFrame:
    return pl.scan_parquet(path).with_columns(pl.col("hospitalization_id").cast(pl.String))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate MEDS vs CLIF label parity (IMV-focused).")
    parser.add_argument("--meds_dir", type=Path, required=True, help="MEDS tokenized base dir with splits.")
    parser.add_argument("--clif_dir", type=Path, required=True, help="CLIF tokenized base dir with splits.")
    parser.add_argument(
        "--splits",
        type=str,
        default="train,val,test",
        help="Comma-separated split names to compare (default: train,val,test).",
    )
    parser.add_argument(
        "--outcomes_filename",
        type=str,
        default="tokens_timelines_outcomes.parquet",
        help="Outcome parquet filename inside each split (default: tokens_timelines_outcomes.parquet).",
    )
    parser.add_argument(
        "--label_cols",
        type=str,
        default=",".join(DEFAULT_LABEL_COLS),
        help="Comma-separated label columns to compare.",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        required=True,
        help="Path to write JSON report.",
    )
    args = parser.parse_args(argv)

    meds_dir = args.meds_dir.expanduser().resolve()
    clif_dir = args.clif_dir.expanduser().resolve()
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    label_cols = [c.strip() for c in args.label_cols.split(",") if c.strip()]

    report: dict[str, Any] = {
        "meds_dir": str(meds_dir),
        "clif_dir": str(clif_dir),
        "splits": splits,
        "label_cols": label_cols,
        "by_split": {},
    }

    for split in splits:
        meds_path = meds_dir / split / args.outcomes_filename
        clif_path = clif_dir / split / args.outcomes_filename

        if not meds_path.exists() or not clif_path.exists():
            report["by_split"][split] = {
                "skipped": True,
                "reason": f"missing meds={meds_path.exists()} clif={clif_path.exists()}",
            }
            continue

        meds = _load_outcomes(meds_path)
        clif = _load_outcomes(clif_path)

        # Join on hospitalization_id (inner = only comparable admissions)
        joined = (
            clif.select(["hospitalization_id"] + label_cols)
            .rename({c: f"{c}_clif" for c in label_cols if c != "hospitalization_id"})
            .join(
                meds.select(["hospitalization_id"] + label_cols).rename(
                    {c: f"{c}_meds" for c in label_cols if c != "hospitalization_id"}
                ),
                on="hospitalization_id",
                how="inner",
                validate="1:1",
            )
        )

        # Compute counts without materializing full joined table.
        base_stats = joined.select(
            n_overlap=pl.len(),
        ).collect().to_dicts()[0]

        label_stats: dict[str, Any] = {}
        for c in label_cols:
            y_true = f"{c}_clif"
            y_pred = f"{c}_meds"

            # Compute confusion matrix counts via aggregation.
            counts_row = (
                joined.select(
                    tp=((pl.col(y_true).fill_null(False) & pl.col(y_pred).fill_null(False)).sum()).alias("tp"),
                    tn=((~pl.col(y_true).fill_null(False) & ~pl.col(y_pred).fill_null(False)).sum()).alias("tn"),
                    fp=((~pl.col(y_true).fill_null(False) & pl.col(y_pred).fill_null(False)).sum()).alias("fp"),
                    fn=((pl.col(y_true).fill_null(False) & ~pl.col(y_pred).fill_null(False)).sum()).alias("fn"),
                    disagreement_rate=(pl.col(y_true).fill_null(False) != pl.col(y_pred).fill_null(False))
                    .mean()
                    .alias("disagreement_rate"),
                    prevalence_clif=pl.col(y_true).fill_null(False).mean().alias("prevalence_clif"),
                    prevalence_meds=pl.col(y_pred).fill_null(False).mean().alias("prevalence_meds"),
                )
                .collect()
                .to_dicts()[0]
            )

            counts = {k: int(counts_row[k]) for k in ("tp", "tn", "fp", "fn")}
            rates = _rates_from_counts(counts)
            label_stats[c] = {
                "counts": counts,
                "rates": rates,
                "disagreement_rate": float(counts_row["disagreement_rate"]),
                "prevalence_clif": float(counts_row["prevalence_clif"]),
                "prevalence_meds": float(counts_row["prevalence_meds"]),
            }

        report["by_split"][split] = {
            "skipped": False,
            **base_stats,
            "labels": label_stats,
        }

    output_json = args.output_json.expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2))
    print(f"[validate_imv_detection] Wrote report: {output_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

