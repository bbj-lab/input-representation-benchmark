#!/usr/bin/env python3
"""Draw the ML4H representation-benchmark schematic (appendix Figure 1).

Only the three performed axes. Window names: hospital mortality, still
hospitalized at hour 48, still in ICU at hour 48. Task counts: 17 binary +
20 regression (Experiment 3: 16 + 20). No randomized or frequency-matched
mapping, no LOS hours, no AUPRC.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT.parent / "ML4H2026/ML4H/figures/figure-1.png"


def box(ax, x, y, w, h, title, lines, facecolor="#f4f4f4"):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=0.8,
        edgecolor="#222222",
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.02,
        y + h - 0.04,
        title,
        fontsize=9.5,
        fontweight="bold",
        va="top",
        ha="left",
        color="#111111",
    )
    ax.text(
        x + 0.02,
        y + h - 0.12,
        "\n".join(lines),
        fontsize=8.0,
        va="top",
        ha="left",
        color="#222222",
        linespacing=1.35,
    )


def draw(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.2, 4.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Representation axes and 24--48h evaluation", fontsize=12, pad=8)

    box(
        ax,
        0.02,
        0.08,
        0.30,
        0.84,
        "Experiment 1  Quantization and fusion",
        [
            "Population $K$: deciles, ventiles,",
            "trentiles, centiles",
            "Reference-range 5--10--5 and 10--10--10",
            "Unfused code + shared $Q_k$",
            "versus fused $(c,k)$ token",
            "",
            "Cohort: hospital LOS $\\geq$ 24h",
        ],
        facecolor="#eef3f8",
    )
    box(
        ax,
        0.34,
        0.08,
        0.30,
        0.84,
        "Experiment 2  Value and time",
        [
            "Numeric maps: discrete bins,",
            "soft interpolation, multiplicative",
            "xVal, affine xVal",
            "Time: inserted tokens, event order,",
            "admission-relative RoPE",
            "",
            "Unfused population deciles",
        ],
        facecolor="#eef6ef",
    )
    box(
        ax,
        0.66,
        0.52,
        0.32,
        0.40,
        "Experiment 3  Native vs CLIF",
        [
            "Same ICU admission lists and raw-MIMIC labels",
            "Native MEDS versus CLIF 2.1.0 event set",
            "Selection, collapse, and preprocessing move together",
        ],
        facecolor="#f7f1e8",
    )
    box(
        ax,
        0.66,
        0.08,
        0.32,
        0.40,
        "Frozen-probe readout",
        [
            "Train on first 24h; probe last-token state",
            "Documented outcomes in $(24,48]$",
            "Hospital mortality in hours 24--48;",
            "still hospitalized at 48; still in ICU at 48",
            "17 binary + 20 regression (Exp.\\ 3: 16 + 20)",
            "AUROC and Spearman $\\rho$ family means",
        ],
        facecolor="#f3eee8",
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")
    print(f"wrote {path.with_suffix('.pdf')}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    draw(args.output.expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
