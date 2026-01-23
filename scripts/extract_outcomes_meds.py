#!/usr/bin/env python3
"""
MEDS-format outcome extraction for representation-based evaluation.

Why this exists
---------------
The fms-ehrs outcome extractor (`fms_ehrs/scripts/extract_outcomes.py`) computes
several labels via token-presence checks on tokenized timelines (full stay and a
24h-truncated view). This works for CLIF because key events (e.g., `RESP_IMV`)
are inserted as time-stamped tokens.

For MEDS tokenization with `fms-ehrs/fms_ehrs/config/mimic-meds-ed.yaml`,
procedures are aggregated into `proc_list` and appended as **suffix** tokens at
discharge time (`suffix: PROC`). This destroys event timing for procedures, and
in particular makes `imv_event_24h` computed from a 24h-truncated timeline
systematically incorrect.

This script computes the fms-ehrs-compatible outcomes directly from MEDS event
timestamps (storetime semantics via the MEDS extraction config) and joins them
onto tokenized timelines to produce `tokens_timelines_outcomes.parquet`.

Outputs (per hospitalization)
-----------------------------
- length_of_stay (hours)
- icu_length_of_stay (hours; derived from ICU admission/discharge events)
- same_admission_death
- long_length_of_stay  (length_of_stay > 7 days)
- icu_admission        (ICU admission at any time during stay)
- imv_event            (IMV procedure at any time during stay)
- prolonged_icu_stay   (icu_length_of_stay > threshold; default 48h)
- icu_admission_24h    (ICU admission within first 24h)
- imv_event_24h        (IMV procedure within first 24h)

Notes on "after 24h" prediction tasks
-------------------------------------
Following the fms-ehrs convention, downstream analyses can define "ICU admission
after 24h" as:
  icu_admission_after_24h = icu_admission & (~icu_admission_24h)
and similarly for IMV.

Input expectations
------------------
1) MEDS events
   Either:
   - A directory with split subdirectories (e.g., `train/`, `tuning/`, `val/`,
     `test/`) each containing one or more `*.parquet` shards (as produced by the
     adapted ETHOS-ARES MEDS pipeline), OR
   - A single `meds.parquet` file per split (not required).

   Required columns in the MEDS event tables:
   - subject_id (patient id)
   - hadm_id (hospitalization id; may be null for patient-level events)
   - time (datetime)
   - code (string)

2) Tokenized timelines
   A directory with split subdirectories, each containing `tokens_timelines.parquet`.
   The timeline id column may be `hadm_id` (MEDS configs) or `hospitalization_id`
   (CLIF configs). This script standardizes to `hospitalization_id` in the output.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import polars as pl


DEFAULT_IMV_ITEMIDS: tuple[int, ...] = (224385, 225792)


def _existing_split_dir(base: Path, split: str) -> Path | None:
    """Return base/split if it exists, else None."""
    p = base / split
    return p if p.exists() and p.is_dir() else None


def _resolve_meds_split_dir(meds_events_dir: Path, tokenized_split: str) -> tuple[str, Path | None]:
    """
    Map tokenized split name to MEDS split name.

    MEDS_transforms pipelines often use `tuning` where other pipelines use `val`.
    """
    # Prefer exact match first.
    if (p := _existing_split_dir(meds_events_dir, tokenized_split)) is not None:
        return tokenized_split, p

    # Common alias: val <-> tuning
    aliases: dict[str, Sequence[str]] = {
        "val": ("tuning",),
        "tuning": ("val",),
    }
    for alt in aliases.get(tokenized_split, ()):
        if (p := _existing_split_dir(meds_events_dir, alt)) is not None:
            return alt, p

    return tokenized_split, None


def _scan_meds_events(split_dir: Path) -> pl.LazyFrame:
    """
    Scan MEDS events from a split directory.

    Supports:
    - One or more parquet shards: `split_dir/*.parquet`
    - A single `meds.parquet` inside the split directory
    """
    meds_parquet = split_dir / "meds.parquet"
    if meds_parquet.exists():
        return pl.scan_parquet(meds_parquet)

    # MEDS_transforms pipeline produces shard files like `0.parquet`, `1.parquet`, ...
    return pl.scan_parquet(str(split_dir / "*.parquet"))


def _compute_outcomes_from_meds(
    meds: pl.LazyFrame,
    *,
    imv_itemids: Sequence[int],
    los_days: float,
    observation_hours: float,
    icu_los_hours_threshold: float,
) -> pl.LazyFrame:
    """Compute outcome labels keyed by `hospitalization_id` (string)."""
    meds = meds.select("subject_id", "hadm_id", "time", "code")

    # Admission/discharge (by hadm_id)
    admissions = (
        meds.filter(pl.col("hadm_id").is_not_null())
        .filter(pl.col("code").str.starts_with("HOSPITAL_ADMISSION"))
        .group_by("hadm_id")
        .agg(
            admission_time=pl.col("time").min(),
            subject_id=pl.col("subject_id").drop_nulls().first(),
        )
    )

    discharges = (
        meds.filter(pl.col("hadm_id").is_not_null())
        .filter(pl.col("code").str.starts_with("HOSPITAL_DISCHARGE"))
        .group_by("hadm_id")
        .agg(
            discharge_time=pl.col("time").min(),
            discharge_code=pl.col("code").sort_by("time").first(),
        )
        .with_columns(
            discharge_loc=pl.col("discharge_code")
            .str.split("//")
            .list.get(1)
            .cast(pl.String, strict=False)
            .str.to_lowercase()
        )
        .drop("discharge_code")
    )

    # ICU admission time (by hadm_id)
    icu_adm = (
        meds.filter(pl.col("hadm_id").is_not_null())
        .filter(pl.col("code").str.starts_with("ICU_ADMISSION"))
        .group_by("hadm_id")
        .agg(icu_admission_time=pl.col("time").min())
    )

    # ICU discharge time (by hadm_id) – use max to handle multiple segments.
    icu_dsc = (
        meds.filter(pl.col("hadm_id").is_not_null())
        .filter(pl.col("code").str.starts_with("ICU_DISCHARGE"))
        .group_by("hadm_id")
        .agg(icu_discharge_time=pl.col("time").max())
    )

    # IMV time (by hadm_id) from procedureevents itemids
    imv = (
        meds.filter(pl.col("hadm_id").is_not_null())
        .filter(pl.col("code").str.starts_with("PROCEDURE//"))
        .with_columns(
            procedure_itemid=pl.col("code")
            .str.split("//")
            .list.get(1)
            .cast(pl.Int64, strict=False)
        )
        .filter(pl.col("procedure_itemid").is_in(list(imv_itemids)))
        .group_by("hadm_id")
        .agg(imv_time=pl.col("time").min())
    )

    # Death time (by subject_id) – may be null if not deceased
    death = (
        meds.filter(pl.col("code") == "MEDS_DEATH")
        .group_by("subject_id")
        .agg(death_time=pl.col("time").min())
    )

    base = (
        admissions.join(discharges, on="hadm_id", how="left", validate="1:1", maintain_order="left")
        .join(icu_adm, on="hadm_id", how="left", validate="1:1", maintain_order="left")
        .join(icu_dsc, on="hadm_id", how="left", validate="1:1", maintain_order="left")
        .join(imv, on="hadm_id", how="left", validate="1:1", maintain_order="left")
        .join(death, on="subject_id", how="left", validate="m:1", maintain_order="left")
    )

    obs = pl.duration(hours=float(observation_hours))
    los_threshold_hours = float(los_days) * 24.0
    icu_threshold_hours = float(icu_los_hours_threshold)

    discharge_is_death = (
        pl.col("discharge_loc")
        .fill_null("")
        .str.contains("died")
        | pl.col("discharge_loc").fill_null("").str.contains("expired")
        | (pl.col("discharge_loc").fill_null("") == "dead")
        | (pl.col("discharge_loc").fill_null("") == "death")
    )

    death_between = (
        pl.col("death_time").is_not_null()
        & pl.col("admission_time").is_not_null()
        & pl.col("discharge_time").is_not_null()
        & pl.col("death_time").is_between(pl.col("admission_time"), pl.col("discharge_time"), closed="both")
    )

    outcomes = (
        base.with_columns(
            hospitalization_id=pl.col("hadm_id").cast(pl.Int64).cast(pl.String),
            length_of_stay=(pl.col("discharge_time") - pl.col("admission_time")).dt.total_hours(),
            # ICU LOS: approximate as (last ICU discharge - first ICU admission).
            # If ICU discharge is missing but ICU admission exists, fall back to hospital discharge.
            icu_length_of_stay=pl.when(pl.col("icu_admission_time").is_null())
            .then(0.0)
            .when(pl.col("icu_discharge_time").is_not_null())
            .then((pl.col("icu_discharge_time") - pl.col("icu_admission_time")).dt.total_hours())
            .when(pl.col("discharge_time").is_not_null())
            .then((pl.col("discharge_time") - pl.col("icu_admission_time")).dt.total_hours())
            .otherwise(0.0),
            same_admission_death=(discharge_is_death | death_between),
            icu_admission=pl.col("icu_admission_time").is_not_null(),
            imv_event=pl.col("imv_time").is_not_null(),
            icu_admission_24h=pl.col("icu_admission_time").is_not_null()
            & ((pl.col("icu_admission_time") - pl.col("admission_time")) <= obs),
            imv_event_24h=pl.col("imv_time").is_not_null()
            & ((pl.col("imv_time") - pl.col("admission_time")) <= obs),
        )
        .with_columns(
            prolonged_icu_stay=pl.col("icu_admission_time").is_not_null()
            & (pl.col("icu_length_of_stay") > icu_threshold_hours),
        )
        .with_columns(long_length_of_stay=pl.col("length_of_stay") > los_threshold_hours)
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
        .with_columns(
            # Ensure boolean columns are non-null (align with fms-ehrs expectations).
            pl.col("same_admission_death").fill_null(False),
            pl.col("long_length_of_stay").fill_null(False),
            pl.col("icu_admission").fill_null(False),
            pl.col("imv_event").fill_null(False),
            pl.col("prolonged_icu_stay").fill_null(False),
            pl.col("icu_admission_24h").fill_null(False),
            pl.col("imv_event_24h").fill_null(False),
        )
        .sort("hospitalization_id")
    )

    return outcomes


def _infer_timeline_id_col(tt_schema: pl.Schema) -> str:
    if "hospitalization_id" in tt_schema:
        return "hospitalization_id"
    if "hadm_id" in tt_schema:
        return "hadm_id"
    # Fall back to the first column (but this should not happen in practice).
    return tt_schema.names()[0]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute MEDS outcomes and join onto tokenized timelines.")
    parser.add_argument(
        "--meds_events_dir",
        type=Path,
        required=True,
        help="Directory containing MEDS parquet shards per split (e.g., .../data/meds/data).",
    )
    parser.add_argument(
        "--tokenized_dir",
        type=Path,
        required=True,
        help="Directory containing tokenized split subdirs with tokens_timelines.parquet.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,val,test",
        help="Comma-separated split names to process, or 'auto' to use subdirs in --tokenized_dir (default: train,val,test).",
    )
    parser.add_argument(
        "--imv_itemids",
        type=str,
        default=",".join(map(str, DEFAULT_IMV_ITEMIDS)),
        help="Comma-separated MIMIC procedureevents itemids defining IMV (default: 224385,225792).",
    )
    parser.add_argument(
        "--los_days",
        type=float,
        default=7.0,
        help="Threshold (days) for long length-of-stay (default: 7).",
    )
    parser.add_argument(
        "--observation_hours",
        type=float,
        default=24.0,
        help="Observation window (hours) for *_24h flags (default: 24).",
    )
    parser.add_argument(
        "--icu_los_hours_threshold",
        type=float,
        default=48.0,
        help="Threshold (hours) for prolonged_icu_stay (default: 48).",
    )
    parser.add_argument(
        "--timeline_filename",
        type=str,
        default="tokens_timelines.parquet",
        help="Timeline parquet filename inside each split directory (default: tokens_timelines.parquet).",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="tokens_timelines_outcomes.parquet",
        help="Output filename to write inside each tokenized split directory (default: tokens_timelines_outcomes.parquet).",
    )
    args = parser.parse_args(argv)

    if args.splits.strip().lower() == "auto":
        splits = sorted(
            p.name
            for p in args.tokenized_dir.expanduser().resolve().iterdir()
            if p.is_dir() and not p.name.startswith(".")
        )
    else:
        splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    imv_itemids = tuple(int(x) for x in args.imv_itemids.split(",") if x.strip())

    meds_events_dir = args.meds_events_dir.expanduser().resolve()
    tokenized_dir = args.tokenized_dir.expanduser().resolve()

    for split in splits:
        tok_split_dir = tokenized_dir / split
        timeline_path = tok_split_dir / args.timeline_filename
        if not timeline_path.exists():
            print(f"[extract_outcomes_meds] Skip split={split}: missing {timeline_path}")
            continue

        meds_split_name, meds_split_dir = _resolve_meds_split_dir(meds_events_dir, split)
        if meds_split_dir is None:
            print(
                f"[extract_outcomes_meds] Skip split={split}: no MEDS split dir found under {meds_events_dir} "
                f"(looked for {split!r} and common aliases)."
            )
            continue

        meds = _scan_meds_events(meds_split_dir)
        outcomes = _compute_outcomes_from_meds(
            meds,
            imv_itemids=imv_itemids,
            los_days=args.los_days,
            observation_hours=args.observation_hours,
            icu_los_hours_threshold=args.icu_los_hours_threshold,
        )

        tt = pl.scan_parquet(timeline_path)
        id_col = _infer_timeline_id_col(tt.collect_schema())

        # Standardize to `hospitalization_id` (string) in the output file.
        tt_std = (
            tt.with_columns(pl.col(id_col).cast(pl.String).alias("hospitalization_id"))
            .drop(id_col)
            if id_col != "hospitalization_id"
            else tt.with_columns(pl.col("hospitalization_id").cast(pl.String))
        )

        out = tt_std.join(
            outcomes,
            on="hospitalization_id",
            how="left",
            validate="1:1",
            maintain_order="left",
        )

        output_path = tok_split_dir / args.output_filename
        out.sink_parquet(output_path)
        print(
            f"[extract_outcomes_meds] Wrote {output_path} "
            f"(token_split={split}, meds_split={meds_split_name})"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

