#!/usr/bin/env python3
"""Draw the clinical-boundary probe with a reference-status baseline.

Reads the existing JSON and validates its recorded counts; does not reload
checkpoints.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JSON = (
    REPO_ROOT / "outputs/runs/ml4h_2026/metrics/clinical_boundary_probe_results.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT.parent / "ML4H2026/ML4H/figures/clinical_boundary_probe.png"
)
GRANULARITY_ORDER = ("Deciles", "Ventiles", "Trentiles", "Centiles")


def reference_status_prevalence(record: dict) -> float:
    n_total = int(record.get("n_total") or 0)
    if n_total <= 0:
        raise ValueError("Probe record must contain a positive n_total")
    n_within = int(record.get("n_normal") or 0)
    n_outside = int(record.get("n_abnormal") or 0)
    if n_within + n_outside != n_total:
        raise ValueError(
            "Reference-status counts do not sum to n_total: "
            f"{n_within} + {n_outside} != {n_total}"
        )

    labels = record.get("labels")
    if labels is not None:
        observed_within = sum(int(label) == 0 for label in labels)
        observed_outside = sum(int(label) == 1 for label in labels)
        if (observed_within, observed_outside) != (n_within, n_outside):
            raise ValueError(
                "Recorded reference-status counts do not match labels: "
                f"{(n_within, n_outside)} != "
                f"{(observed_within, observed_outside)}"
            )

    accuracy = record.get("accuracy")
    n_correct = record.get("n_correct")
    if accuracy is not None and n_correct is not None:
        expected_accuracy = int(n_correct) / n_total
        if not np.isclose(float(accuracy), expected_accuracy):
            raise ValueError(
                "Recorded accuracy does not match n_correct / n_total: "
                f"{accuracy} != {n_correct} / {n_total}"
            )
    return max(n_within, n_outside) / n_total


def draw(results: dict, output: Path) -> None:
    granularities = [key for key in GRANULARITY_ORDER if key in results]
    measurements: list[str] = []
    names: list[str] = []
    first = results[granularities[0]]
    for key, record in first.items():
        if key.startswith("_"):
            continue
        measurements.append(key)
        names.append(record.get("name", key))

    n_groups = len(measurements)
    n_bars = len(granularities)
    x = np.arange(n_groups)
    width = 0.8 / n_bars
    fig, ax = plt.subplots(figsize=(14.5, 7.0))
    colors = plt.cm.Set2(np.linspace(0, 1, n_bars))
    baseline_handles = []

    for i, gran in enumerate(granularities):
        accs = []
        baselines = []
        for measurement in measurements:
            record = results[gran].get(measurement, {})
            accs.append(float(record.get("accuracy") or 0.0))
            baselines.append(reference_status_prevalence(record))
        bars = ax.bar(
            x + i * width - (n_bars - 1) * width / 2,
            accs,
            width * 0.9,
            label=gran,
            color=colors[i],
            edgecolor="gray",
            linewidth=0.5,
        )
        for bar, acc, baseline in zip(bars, accs, baselines):
            if acc > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{acc:.0%}",
                    ha="center",
                    va="bottom",
                    fontsize=12,
                )
            handle = ax.scatter(
                bar.get_x() + bar.get_width() / 2,
                baseline,
                marker="D",
                s=42,
                facecolors="white",
                edgecolors="black",
                linewidths=1.2,
                zorder=4,
            )
            if not baseline_handles:
                baseline_handles.append(handle)

    ax.set_xlabel("Lab test", fontsize=18)
    ax.set_ylabel("Leave-one-out accuracy", fontsize=18)
    ax.set_title(
        "Clinical-boundary probe",
        fontsize=19,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=15)
    ax.tick_params(axis="y", labelsize=15)
    ax.set_ylim(0, 1.08)
    handles, labels = ax.get_legend_handles_labels()
    if baseline_handles:
        handles.append(baseline_handles[0])
        labels.append("Majority-class baseline")
    ax.legend(handles, labels, fontsize=14, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    results = json.loads(args.results_json.read_text(encoding="utf-8"))
    draw(results, args.output.expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
