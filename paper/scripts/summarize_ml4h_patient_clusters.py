#!/usr/bin/env python3
"""Summarize patient-cluster intervals against model-level paired t-tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = REPO_ROOT / "outputs/runs/ml4h_2026"
INTERVALS = (
    RUN_ROOT
    / "patient_cluster_bootstrap"
    / "patient_cluster_intervals.parquet"
)
MODEL_TESTS = RUN_ROOT / "model_level_tests" / "family_tests.csv"
OUTPUT_ROOT = RUN_ROOT / "patient_cluster_bootstrap"
KEYS = [
    "task_index",
    "experiment",
    "condition",
    "level",
    "task_type",
    "family",
]


def counts(frame: pd.DataFrame) -> dict[str, int]:
    """Count agreement between the two uncertainty analyses."""
    t_star = frame["star_t"]
    cluster = frame["cluster_excludes_zero"]
    return {
        "tests": int(len(frame)),
        "paired_t_significant": int(t_star.sum()),
        "cluster_excludes_zero": int(cluster.sum()),
        "both": int((t_star & cluster).sum()),
        "paired_t_only": int((t_star & ~cluster).sum()),
        "cluster_only": int((~t_star & cluster).sum()),
        "neither": int((~t_star & ~cluster).sum()),
    }


def main() -> int:
    """Write overall, table-level, and discordant summaries."""
    intervals = pd.read_parquet(INTERVALS)
    model = pd.read_csv(MODEL_TESTS)
    selected = model[KEYS + ["table", "delta", "p_t_adj", "star_t"]]
    merged = intervals.merge(
        selected,
        on=KEYS,
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != 136:
        raise ValueError(f"Expected 136 merged rows, found {len(merged)}")
    point_gap = float(np.max(np.abs(merged["point"] - merged["delta"])))
    if point_gap > 1e-12:
        raise ValueError(f"Point estimates differ by {point_gap:.3g}")

    merged["star_t"] = merged["star_t"].astype(bool)
    merged["cluster_excludes_zero"] = ~merged[
        "cluster_ci_contains_zero"
    ].astype(bool)
    overall = counts(merged)
    by_table = {
        str(table): counts(part)
        for table, part in merged.groupby("table", sort=False)
    }
    summary = {
        "overall": overall,
        "by_table": by_table,
        "max_point_gap": point_gap,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_ROOT / "agreement_summary.json"
    json_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    discordant = merged.loc[
        merged["star_t"] != merged["cluster_excludes_zero"],
        [
            *KEYS,
            "table",
            "point",
            "cluster_ci_low",
            "cluster_ci_high",
            "p_t_adj",
            "star_t",
            "cluster_excludes_zero",
        ],
    ].copy()
    discordant.to_csv(OUTPUT_ROOT / "agreement_discordant.csv", index=False)

    lines = [
        "scope\ttests\tpaired_t_significant\tcluster_excludes_zero\tboth\tpaired_t_only\tcluster_only\tneither"
    ]
    for scope, values in [("all", overall), *by_table.items()]:
        lines.append(
            "\t".join(
                [
                    scope,
                    str(values["tests"]),
                    str(values["paired_t_significant"]),
                    str(values["cluster_excludes_zero"]),
                    str(values["both"]),
                    str(values["paired_t_only"]),
                    str(values["cluster_only"]),
                    str(values["neither"]),
                ]
            )
        )
    text_path = OUTPUT_ROOT / "agreement_summary.txt"
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
