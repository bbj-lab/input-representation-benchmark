#!/usr/bin/env python3
"""Validate the 24-to-48-hour metric tables in outputs/runs/ml4h_2026/metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs/runs/ml4h_2026/metrics"

EXPECTED = {
    "exp1": {"runs": 72, "binary": 17, "regression": 20, "test_rows": 85_057},
    "exp2": {"runs": 72, "binary": 17, "regression": 20, "test_rows": 85_057},
    "exp3": {"runs": 12, "binary": 16, "regression": 20, "test_rows": 16_529},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=tuple(EXPECTED),
        default=list(EXPECTED),
        help="Experiments that must be present in the metric tables.",
    )
    return parser.parse_args()


def resolve_table(output_root: Path, filename: str) -> Path:
    direct = output_root / filename
    if direct.is_file():
        return direct
    nested = output_root / "00_data" / filename
    if nested.is_file():
        return nested
    raise SystemExit(f"Missing metric table {filename} under {output_root}")


def main() -> int:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    requested = tuple(
        name for name in ("exp1", "exp2", "exp3") if name in args.experiments
    )
    manifest_path = output_root / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["summary"]["analysis"] != "24to48h":
            raise SystemExit("Unexpected analysis identifier in manifest.")
        manifest_experiments = tuple(
            manifest["summary"].get("requested_experiments", [])
        )
        if manifest_experiments and manifest_experiments != requested:
            raise SystemExit(
                "Manifest experiments "
                f"{list(manifest_experiments)} do not match requested {list(requested)}."
            )

    registry = pd.read_parquet(resolve_table(output_root, "run_registry.parquet"))
    metrics = pd.read_parquet(resolve_table(output_root, "metrics_long.parquet"))
    coverage = pd.read_parquet(
        resolve_table(output_root, "label_coverage_long.parquet")
    )
    expected_metric_rows = 0
    for experiment in requested:
        contract = EXPECTED[experiment]
        subset_registry = registry.loc[registry["experiment"] == experiment]
        if len(subset_registry) != contract["runs"]:
            raise SystemExit(
                f"{experiment}: expected {contract['runs']} registry rows, "
                f"found {len(subset_registry)}."
            )
        subset_metrics = metrics.loc[metrics["experiment"] == experiment]
        expected_rows = contract["runs"] * (
            contract["binary"] * 4 + contract["regression"] * 4
        )
        expected_metric_rows += expected_rows
        if len(subset_metrics) != expected_rows:
            raise SystemExit(
                f"{experiment}: expected {expected_rows} metrics, "
                f"found {len(subset_metrics)}."
            )
        if set(subset_metrics["test_cohort_rows"].astype(int)) != {
            contract["test_rows"]
        }:
            raise SystemExit(f"{experiment}: unexpected feature-cohort row count.")
        binary = coverage.loc[
            (coverage["experiment"] == experiment)
            & (coverage["task_type"] == "binary")
        ]
        regression = coverage.loc[
            (coverage["experiment"] == experiment)
            & (coverage["task_type"] == "regression")
        ]
        if binary["outcome"].nunique() != contract["binary"]:
            raise SystemExit(f"{experiment}: unexpected binary coverage contract.")
        if regression["outcome"].nunique() != contract["regression"]:
            raise SystemExit(f"{experiment}: unexpected regression coverage contract.")
        if (coverage["n_labeled"] > coverage["test_cohort_rows"]).any():
            raise SystemExit("A label-coverage row exceeds its feature cohort.")
        if (coverage["n_unlabeled"] < 0).any():
            raise SystemExit("A label-coverage row has a negative unlabeled count.")
    if len(metrics) != expected_metric_rows:
        raise SystemExit(
            f"Expected {expected_metric_rows} total metrics, found {len(metrics)}."
        )

    print(
        "Validated 24-to-48h metric tables: "
        f"{len(metrics):,} metric rows ({', '.join(requested)})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
