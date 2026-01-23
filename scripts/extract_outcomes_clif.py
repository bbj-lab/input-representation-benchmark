#!/usr/bin/env python3
"""
CLIF-format outcome extraction for representation-based evaluation.

Why this exists
---------------
The reference fms-ehrs outcome extractor (`fms_ehrs/scripts/extract_outcomes.py`) infers outcomes
from token presence (e.g., `RESP_IMV`, `ADT_icu`). That approach is convenient when the tokenizer
includes those event families.

In Experiment 3 (vocabulary semantics), we intentionally tokenize only a matched-signal subset
(labs + vitals) to reduce confounding from format-specific feature availability. Therefore, outcome
labels must be computed from *raw CLIF tables* (e.g., ADT and respiratory support), not from
tokens that are absent by design.

Outputs (per hospitalization)
-----------------------------
- length_of_stay (hours)
- icu_length_of_stay (hours; summed over ICU ADT segments where possible)
- same_admission_death
- long_length_of_stay  (length_of_stay > 7 days)
- icu_admission        (ICU stay exists at any time during admission)
- imv_event            (IMV respiratory support event at any time during admission)
- prolonged_icu_stay   (icu_length_of_stay > threshold; default 48h)
- icu_admission_24h    (ICU admission within first 24h)
- imv_event_24h        (IMV event within first 24h)

Assumed CLIF layout
-------------------
`data_dir/raw/{train,val,test}/` contains at least:
- clif_hospitalization.parquet (admission/discharge + discharge_category)
- clif_adt.parquet (in_dttm/out_dttm + location_category)
- clif_respiratory_support_processed.parquet (recorded_dttm + mode_category/device_category)

`data_dir/<data_version>-tokenized/{train,val,test}/tokens_timelines.parquet` exists and contains
`hospitalization_id`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl


def _read_raw_table(raw_split_dir: Path, fname: str) -> pl.LazyFrame:
    fp = raw_split_dir / fname
    if not fp.exists():
        raise FileNotFoundError(f"Missing raw CLIF table: {fp}")
    return pl.scan_parquet(str(fp))


def _is_icu_loc(expr: pl.Expr) -> pl.Expr:
    # Robust ICU location detection for clif_adt.location_category
    return expr.cast(pl.Utf8).str.to_lowercase().str.contains("icu")


def _is_imv(expr: pl.Expr) -> pl.Expr:
    # Robust IMV detection for clif_respiratory_support_processed mode/device categories
    return expr.cast(pl.Utf8).str.to_lowercase().str.contains("imv")


def _compute_outcomes_for_split(
    *,
    raw_split_dir: Path,
    los_days: float,
    observation_hours: float,
    icu_los_hours_threshold: float,
) -> pl.LazyFrame:
    hosp = _read_raw_table(raw_split_dir, "clif_hospitalization.parquet").select(
        "hospitalization_id", "admission_dttm", "discharge_dttm", "discharge_category"
    )
    hosp = hosp.with_columns(
        hospitalization_id=pl.col("hospitalization_id").cast(pl.String),
        admission_dttm=pl.col("admission_dttm").cast(pl.Datetime(time_unit="ms")),
        discharge_dttm=pl.col("discharge_dttm").cast(pl.Datetime(time_unit="ms")),
        discharge_category=pl.col("discharge_category").cast(pl.Utf8),
    )

    # ICU ADT segments
    adt = _read_raw_table(raw_split_dir, "clif_adt.parquet").select(
        "hospitalization_id", "in_dttm", "out_dttm", "location_category"
    )
    adt = adt.with_columns(
        hospitalization_id=pl.col("hospitalization_id").cast(pl.String),
        in_dttm=pl.col("in_dttm").cast(pl.Datetime(time_unit="ms")),
        out_dttm=pl.col("out_dttm").cast(pl.Datetime(time_unit="ms")),
        location_category=pl.col("location_category").cast(pl.Utf8),
    ).filter(_is_icu_loc(pl.col("location_category")))

    # Sum ICU segment durations (hours) where both in/out are available; keep first ICU admission time.
    icu = (
        adt.filter(pl.col("in_dttm").is_not_null())
        .with_columns(
            seg_hours=(pl.col("out_dttm") - pl.col("in_dttm")).dt.total_hours(),
        )
        .with_columns(seg_hours=pl.col("seg_hours").fill_null(0.0).clip(lower_bound=0.0))
        .group_by("hospitalization_id")
        .agg(
            icu_admission_time=pl.col("in_dttm").min(),
            icu_length_of_stay=pl.col("seg_hours").sum(),
        )
    )

    # IMV event (earliest respiratory support record with IMV in mode/device)
    resp = _read_raw_table(raw_split_dir, "clif_respiratory_support_processed.parquet").select(
        "hospitalization_id", "recorded_dttm", "mode_category", "device_category"
    )
    resp = resp.with_columns(
        hospitalization_id=pl.col("hospitalization_id").cast(pl.String),
        recorded_dttm=pl.col("recorded_dttm").cast(pl.Datetime(time_unit="ms")),
        mode_category=pl.col("mode_category").cast(pl.Utf8),
        device_category=pl.col("device_category").cast(pl.Utf8),
    )
    imv = (
        resp.filter(
            pl.col("recorded_dttm").is_not_null()
            & (_is_imv(pl.col("mode_category")) | _is_imv(pl.col("device_category")))
        )
        .group_by("hospitalization_id")
        .agg(imv_time=pl.col("recorded_dttm").min())
    )

    obs = pl.duration(hours=float(observation_hours))
    los_threshold_hours = float(los_days) * 24.0
    icu_threshold_hours = float(icu_los_hours_threshold)

    discharge_is_death = (
        pl.col("discharge_category")
        .fill_null("")
        .cast(pl.Utf8)
        .str.to_lowercase()
        .str.contains("expired|died|death|dead")
    )

    base = hosp.join(icu, on="hospitalization_id", how="left").join(imv, on="hospitalization_id", how="left")

    return (
        base.with_columns(
            length_of_stay=(pl.col("discharge_dttm") - pl.col("admission_dttm")).dt.total_hours(),
            same_admission_death=discharge_is_death,
            icu_admission=pl.col("icu_admission_time").is_not_null(),
            imv_event=pl.col("imv_time").is_not_null(),
            prolonged_icu_stay=pl.col("icu_length_of_stay").fill_null(0.0) > icu_threshold_hours,
            icu_admission_24h=pl.col("icu_admission_time").is_not_null()
            & ((pl.col("icu_admission_time") - pl.col("admission_dttm")) <= obs),
            imv_event_24h=pl.col("imv_time").is_not_null()
            & ((pl.col("imv_time") - pl.col("admission_dttm")) <= obs),
        )
        .with_columns(long_length_of_stay=pl.col("length_of_stay") > los_threshold_hours)
        .with_columns(
            # normalize nulls for downstream expectations
            pl.col("icu_length_of_stay").fill_null(0.0),
            pl.col("same_admission_death").fill_null(False),
            pl.col("long_length_of_stay").fill_null(False),
            pl.col("icu_admission").fill_null(False),
            pl.col("imv_event").fill_null(False),
            pl.col("prolonged_icu_stay").fill_null(False),
            pl.col("icu_admission_24h").fill_null(False),
            pl.col("imv_event_24h").fill_null(False),
        )
        .select(
            "hospitalization_id",
            "length_of_stay",
            "icu_length_of_stay",
            "same_admission_death",
            "long_length_of_stay",
            "icu_admission",
            "imv_event",
            "prolonged_icu_stay",
            "icu_admission_24h",
            "imv_event_24h",
        )
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Compute CLIF outcomes from raw tables and join onto tokenized timelines.")
    p.add_argument("--data_dir", type=Path, required=True, help="CLIF dataset root containing raw/ and *-tokenized/.")
    p.add_argument("--ref_version", type=str, required=True, help="Full-stay tokenization version (unused for now; kept for interface parity).")
    p.add_argument("--data_version", type=str, required=True, help="24h tokenization version base (without -tokenized).")
    p.add_argument("--splits", type=str, default="train,val,test", help="Comma-separated splits (default: train,val,test).")
    p.add_argument("--los_days", type=float, default=7.0, help="Threshold (days) for long_length_of_stay (default: 7).")
    p.add_argument("--observation_hours", type=float, default=24.0, help="Observation window for *_24h flags (default: 24).")
    p.add_argument("--icu_los_hours_threshold", type=float, default=48.0, help="Threshold (hours) for prolonged_icu_stay (default: 48).")
    args = p.parse_args(argv)

    data_dir = args.data_dir.expanduser().resolve()
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    for split in splits:
        raw_split_dir = data_dir / "raw" / split
        tok_split_dir = data_dir / f"{args.data_version}-tokenized" / split
        timeline_fp = tok_split_dir / "tokens_timelines.parquet"
        if not timeline_fp.exists():
            print(f"[extract_outcomes_clif] Skip split={split}: missing {timeline_fp}")
            continue

        outcomes = _compute_outcomes_for_split(
            raw_split_dir=raw_split_dir,
            los_days=args.los_days,
            observation_hours=args.observation_hours,
            icu_los_hours_threshold=args.icu_los_hours_threshold,
        )

        tt = pl.scan_parquet(timeline_fp).with_columns(pl.col("hospitalization_id").cast(pl.String))
        out = tt.join(outcomes, on="hospitalization_id", how="left", validate="1:1", maintain_order="left")
        out_fp = tok_split_dir / "tokens_timelines_outcomes.parquet"
        out.sink_parquet(out_fp)
        print(f"[extract_outcomes_clif] Wrote {out_fp}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

