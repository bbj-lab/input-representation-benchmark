#!/usr/bin/env python3
"""Materialize a CLIF-schematized MIMIC arm in the MEDS event interface.

The conversion keeps the Exp3 cohort and train/validation/test assignment from
the raw-MIMIC MEDS arm. Outcomes continue to be extracted from that raw arm, so
the two representation arms differ only in their input schema.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl


EVENT_TABLES = (
    "clif_adt",
    "clif_code_status",
    "clif_crrt_therapy",
    "clif_ecmo_mcs",
    "clif_hospital_diagnosis",
    "clif_labs",
    "clif_medication_admin_continuous",
    "clif_medication_admin_intermittent",
    "clif_patient_assessments",
    "clif_patient_procedures",
    "clif_position",
    "clif_respiratory_support",
    "clif_sofa",
    "clif_successful_extubation",
    "clif_vitals",
)
TIME_COLUMNS = (
    "recorded_dttm",
    "admin_dttm",
    "lab_result_dttm",
    "lab_collect_dttm",
    "in_dttm",
    "procedure_billed_dttm",
    "assessment_dttm",
)
NUMERIC_COLUMNS = (
    "lab_value_numeric",
    "vital_value",
    "med_dose",
    "total_score",
)
ID_COLUMNS = {
    "patient_id",
    "hospitalization_id",
    "hospitalization_joined_id",
    "hospital_id",
    "med_order_id",
    "device_id",
    "billing_provider_id",
    "performing_provider_id",
}


def _schema(path: Path) -> dict[str, pl.DataType]:
    return dict(pl.scan_parquet(path).collect_schema())


def _event_frame(
    *,
    path: Path,
    table_name: str,
    cohort: pl.LazyFrame,
) -> pl.LazyFrame | None:
    schema = _schema(path)
    if "hospitalization_id" not in schema:
        return None
    time_column = next((column for column in TIME_COLUMNS if column in schema), None)
    if time_column is None:
        return None

    text_columns = [
        column
        for column, dtype in schema.items()
        if column not in ID_COLUMNS
        and column != time_column
        and dtype in (pl.String, pl.Categorical, pl.Enum)
    ]
    numeric_column = next(
        (column for column in NUMERIC_COLUMNS if column in schema),
        None,
    )
    descriptor = (
        pl.concat_str(
            [pl.col(column).cast(pl.String).fill_null("unknown") for column in text_columns[:3]],
            separator="//",
        )
        if text_columns
        else pl.lit("event")
    )
    numeric_value = (
        pl.col(numeric_column).cast(pl.Float64, strict=False)
        if numeric_column is not None
        else pl.lit(None).cast(pl.Float64)
    )
    return (
        pl.scan_parquet(path)
        .join(cohort, on="hospitalization_id", how="inner")
        .select(
            pl.col("patient_id").cast(pl.String).alias("subject_id"),
            pl.col("hospitalization_id").cast(pl.Int64, strict=False).alias("hadm_id"),
            pl.col(time_column)
            .cast(pl.Datetime("us", "UTC"), strict=False)
            .alias("time"),
            (
                pl.lit(f"CLIF_{table_name.removeprefix('clif_').upper()}//")
                + descriptor
            ).alias("code"),
            numeric_value.alias("numeric_value"),
        )
        .filter(pl.col("hadm_id").is_not_null() & pl.col("time").is_not_null())
    )


def _admission_frames(hospitalizations: pl.LazyFrame, cohort: pl.LazyFrame) -> list[pl.LazyFrame]:
    base = hospitalizations.join(cohort, on="hospitalization_id", how="inner")
    columns = set(hospitalizations.collect_schema().names())
    admission_type = (
        pl.col("admission_type_category").cast(pl.String).fill_null("unknown")
        if "admission_type_category" in columns
        else pl.lit("unknown")
    )
    discharge_type = (
        pl.col("discharge_category").cast(pl.String).fill_null("unknown")
        if "discharge_category" in columns
        else pl.lit("unknown")
    )
    return [
        base.select(
            pl.col("patient_id").cast(pl.String).alias("subject_id"),
            pl.col("hospitalization_id").cast(pl.Int64, strict=False).alias("hadm_id"),
            pl.col("admission_dttm")
            .cast(pl.Datetime("us", "UTC"), strict=False)
            .alias("time"),
            (pl.lit("HOSPITAL_ADMISSION//") + admission_type).alias("code"),
            pl.lit(None).cast(pl.Float64).alias("numeric_value"),
        ),
        base.select(
            pl.col("patient_id").cast(pl.String).alias("subject_id"),
            pl.col("hospitalization_id").cast(pl.Int64, strict=False).alias("hadm_id"),
            pl.col("discharge_dttm")
            .cast(pl.Datetime("us", "UTC"), strict=False)
            .alias("time"),
            (pl.lit("HOSPITAL_DISCHARGE//") + discharge_type).alias("code"),
            pl.lit(None).cast(pl.Float64).alias("numeric_value"),
        ),
    ]


def convert_split(
    *,
    clif_dir: Path,
    source_meds_dir: Path,
    output_dir: Path,
    split: str,
) -> dict[str, int]:
    source_path = source_meds_dir / split / "meds.parquet"
    if not source_path.exists():
        raise FileNotFoundError(f"Missing raw-MIMIC MEDS split: {source_path}")
    hospitalizations_path = clif_dir / "clif_hospitalization.parquet"
    if not hospitalizations_path.exists():
        raise FileNotFoundError(f"Missing CLIF hospitalization table: {hospitalizations_path}")

    cohort_ids = (
        pl.scan_parquet(source_path)
        .select(pl.col("hadm_id").cast(pl.String).drop_nulls().unique().alias("hospitalization_id"))
        .collect(engine="streaming")
    )
    hospitalizations = pl.scan_parquet(hospitalizations_path)
    cohort = (
        hospitalizations.join(cohort_ids.lazy(), on="hospitalization_id", how="inner")
        .select("hospitalization_id", "patient_id")
    )
    frames = _admission_frames(hospitalizations, cohort_ids.lazy())
    tables_used = 0
    for table_name in EVENT_TABLES:
        table_path = clif_dir / f"{table_name}.parquet"
        if not table_path.exists():
            continue
        frame = _event_frame(path=table_path, table_name=table_name, cohort=cohort)
        if frame is not None:
            frames.append(frame)
            tables_used += 1

    target = output_dir / split
    target.mkdir(parents=True, exist_ok=True)
    output_path = target / "meds.parquet"
    (
        pl.concat(frames, how="vertical_relaxed")
        .filter(pl.col("hadm_id").is_not_null() & pl.col("time").is_not_null())
        .sink_parquet(output_path)
    )
    n_rows = pl.scan_parquet(output_path).select(pl.len()).collect().item()
    return {
        "cohort_admissions": cohort_ids.height,
        "event_tables": tables_used,
        "rows": int(n_rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clif-dir", type=Path, required=True)
    parser.add_argument("--source-meds-dir", type=Path, required=True)
    parser.add_argument("--output-meds-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=("train", "val", "test"))
    args = parser.parse_args()

    for split in args.splits:
        summary = convert_split(
            clif_dir=args.clif_dir,
            source_meds_dir=args.source_meds_dir,
            output_dir=args.output_meds_dir,
            split=split,
        )
        print(f"{split}: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
