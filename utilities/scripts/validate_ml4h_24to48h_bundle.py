#!/usr/bin/env python3
"""Validate the isolated ML4H 24-to-48-hour provisional result bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


EXPECTED = {
    "exp1": {"runs": 72, "binary": 17, "regression": 20, "test_rows": 85_057},
    "exp2": {"runs": 72, "binary": 17, "regression": 20, "test_rows": 85_057},
    "exp3": {"runs": 12, "binary": 16, "regression": 20, "test_rows": 16_529},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--primary-snapshot",
        type=Path,
        default=None,
        help="Optional JSON file containing the pre-render 07_ML4H_ready hash map.",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=tuple(EXPECTED),
        default=list(EXPECTED),
        help="Experiments that must be present in this bundle.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    manifest_path = output_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["summary"]["analysis"] != "24to48h":
        raise SystemExit("Unexpected analysis identifier in provisional manifest.")
    requested = tuple(
        name for name in ("exp1", "exp2", "exp3") if name in args.experiments
    )
    expected_figures = 4 * len(requested)
    if (
        int(manifest["figure_count"]) != expected_figures
        or len(manifest["figures"]) != expected_figures
    ):
        raise SystemExit(
            f"Expected {expected_figures} provisional figure pairs, "
            f"found {manifest['figure_count']}."
        )
    manifest_experiments = tuple(manifest["summary"].get("requested_experiments", []))
    if manifest_experiments and manifest_experiments != requested:
        raise SystemExit(
            "Manifest experiments "
            f"{list(manifest_experiments)} do not match requested {list(requested)}."
        )
    for figure in manifest["figures"]:
        for key in ("pdf", "png", "source", "metadata"):
            path = output_root / figure[key]
            if not path.is_file() or path.stat().st_size == 0:
                raise SystemExit(f"Missing or empty provisional {key}: {path}")

    registry = pd.read_parquet(output_root / "00_data/run_registry.parquet")
    metrics = pd.read_parquet(output_root / "00_data/metrics_long.parquet")
    coverage = pd.read_parquet(output_root / "00_data/label_coverage_long.parquet")
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

    if args.primary_snapshot is not None:
        snapshot = json.loads(args.primary_snapshot.expanduser().read_text(encoding="utf-8"))
        for relative, expected_hash in snapshot.items():
            path = output_root.parent / "07_ML4H_ready" / relative
            if not path.is_file() or sha256(path) != expected_hash:
                raise SystemExit(f"Primary figure bundle changed: {path}")

    print(
        "Validated provisional 24-to-48h bundle: "
        f"{len(metrics):,} metric rows and {expected_figures} figure pairs "
        f"({', '.join(requested)})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
