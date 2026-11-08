#!/usr/bin/env python3
"""Bin benchmark hospitalizations by how many stays belong to the same patient.

The visit count is the number of distinct hospitalizations for that
subject_id inside the scope being tabulated. Bins are 1, 2, 3, 4, and 5+.
A patient with three stays contributes one patient and three hospitalizations
to the "3" row.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MEDS = (
    REPO_ROOT / "benchmarks/mimic-meds-extraction/data/meds/data/raw"
)
DEFAULT_TOKENIZED = (
    REPO_ROOT
    / "outputs/runs/tokenized/mimiciv-3.1_meds_70-10-20"
    / "deciles_none_unfused_time_tokens_first_24h-tokenized"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "outputs/runs/ml4h_2026/metrics/hospitalizations_by_patient_visits.md"
)
SPLITS = ("train", "val", "test")
BIN_ORDER = ("1", "2", "3", "4", "5+")


def unique_stay_pairs(parquet_paths: list[Path]) -> dict[int, int]:
    """Map each hospitalization id to one patient id.

    Event rows are read in batches so the full MEDS tables are not held
    in memory. The returned map contains one entry per hospitalization.
    """
    votes: dict[int, dict[int, int]] = {}
    for path in parquet_paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(
            columns=["subject_id", "hadm_id"],
            batch_size=250_000,
        ):
            table = pa.Table.from_batches([batch])
            table = table.filter(pc.is_valid(table["hadm_id"]))
            table = table.filter(pc.is_valid(table["subject_id"]))
            for subject_id, hadm_id in zip(
                table["subject_id"].to_pylist(),
                table["hadm_id"].to_pylist(),
            ):
                hadm = int(hadm_id)
                subject = int(subject_id)
                bucket = votes.get(hadm)
                if bucket is None:
                    votes[hadm] = {subject: 1}
                else:
                    bucket[subject] = bucket.get(subject, 0) + 1
    hadm_to_subject: dict[int, int] = {}
    n_ambiguous = 0
    for hadm, bucket in votes.items():
        if len(bucket) > 1:
            n_ambiguous += 1
        hadm_to_subject[hadm] = max(bucket, key=bucket.get)
    print(
        f"MEDS hospitalizations: {len(hadm_to_subject):,}; "
        f"ambiguous patient ids resolved by majority event count: {n_ambiguous:,}"
    )
    return hadm_to_subject


def timeline_ids(path: Path, id_column: str) -> set[int]:
    table = pq.read_table(path, columns=[id_column])
    values = table[id_column].drop_null().to_pylist()
    return {int(value) for value in values}


def bin_label(n_visits: int) -> str:
    if n_visits >= 5:
        return "5+"
    return str(n_visits)


def tabulate(stays: list[tuple[int, int]]) -> list[dict[str, object]]:
    """stays are (subject_id, hadm_id), already unique."""
    visits_by_subject: dict[int, int] = {}
    for subject_id, _hadm_id in stays:
        visits_by_subject[subject_id] = visits_by_subject.get(subject_id, 0) + 1

    patients = {label: 0 for label in BIN_ORDER}
    hospitalizations = {label: 0 for label in BIN_ORDER}
    for subject_id, n_visits in visits_by_subject.items():
        label = bin_label(n_visits)
        patients[label] += 1
        hospitalizations[label] += n_visits

    total_hospitalizations = sum(hospitalizations.values())
    rows = []
    for label in BIN_ORDER:
        n_hosp = hospitalizations[label]
        share = (
            n_hosp / total_hospitalizations if total_hospitalizations else 0.0
        )
        rows.append(
            {
                "n_visits": label,
                "n_patients": patients[label],
                "n_hospitalizations": n_hosp,
                "pct_hospitalizations": 100.0 * share,
            }
        )
    return rows


def restrict_to_ids(
    hadm_to_subject: dict[int, int],
    hadm_ids: set[int],
) -> list[tuple[int, int]]:
    stays = []
    missing = 0
    for hadm_id in hadm_ids:
        subject_id = hadm_to_subject.get(hadm_id)
        if subject_id is None:
            missing += 1
            continue
        stays.append((subject_id, hadm_id))
    if missing:
        raise SystemExit(
            f"{missing} timeline hospitalizations have no MEDS subject_id"
        )
    return stays


def markdown_table(scope: str, rows: list[dict[str, object]]) -> str:
    lines = [
        f"### {scope}",
        "",
        "| Visits by the same patient | Patients | Hospitalizations | Share of hospitalizations |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {n_visits} | {n_patients:,} | {n_hospitalizations:,} | {pct_hospitalizations:.1f}% |".format(
                **row
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meds-raw", type=Path, default=DEFAULT_MEDS)
    parser.add_argument("--tokenized", type=Path, default=DEFAULT_TOKENIZED)
    parser.add_argument("--id-column", default="hadm_id")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    shard_paths = sorted(args.meds_raw.glob("*/*.parquet"))
    if not shard_paths:
        raise SystemExit(f"No MEDS parquet shards under {args.meds_raw}")
    hadm_to_subject = unique_stay_pairs(shard_paths)

    sections = ["# Hospitalizations by visits from the same patient", ""]
    sections.append(
        "Visit count is the number of benchmark hospitalizations for that "
        "subject_id inside the named scope. The 5+ row is five or more visits."
    )
    sections.append("")

    pooled: list[tuple[int, int]] = []
    for split in SPLITS:
        timeline = args.tokenized / split / "tokens_timelines.parquet"
        hadm_ids = timeline_ids(timeline, args.id_column)
        stays = restrict_to_ids(hadm_to_subject, hadm_ids)
        pooled.extend(stays)
        sections.append(markdown_table(split, tabulate(stays)))

    sections.append(markdown_table("train + val + test", tabulate(pooled)))
    text = "\n".join(sections)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
    print(text)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
