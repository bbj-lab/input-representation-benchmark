#!/usr/bin/env python3
"""Generate Qwen3 fused-vs-unfused outcome forest plot for the MLHC appendix."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from generate_mlhc_paper_figures import (
    BINARY_OUTCOME_ORDER_EXP12,
    FOREST_BINARY_AUROC_X_LO_FLOOR,
    FOREST_FIGSIZE,
    FOREST_HANDLE_SPREAD,
    FOREST_LABEL_SIZE,
    FOREST_LAYOUT_TOP_EDGE,
    FOREST_LINE_WIDTH,
    FOREST_MARKER_SIZE,
    FOREST_OUTCOME_LABELS,
    FOREST_TICK_SIZE,
    FOREST_TITLE_SIZE,
    FOREST_YLIM_PAD,
    REGRESSION_OUTCOME_ORDER,
    _add_forest_knob_legends,
    _add_forest_outcome_banding,
    _add_forest_outcome_separators,
    _forest_color_handles,
    _forest_legend_anchor_y,
    _forest_marker_handles,
    _legend_row_count,
    _save_source,
    _style,
    _xlim_from_points,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATS_ROOT = ROOT / "outputs" / "runs" / "statistics" / "generalizability_tests"
DEFAULT_OUT_DIR = ROOT.parent / "MLHC2026" / "MLHC" / "figures"

QWEN3_ORDER = [
    ("qwen3_depth8_deciles_unfused", "8-layer scaled", "unfused"),
    ("qwen3_depth8_deciles_fused", "8-layer scaled", "fused"),
    ("qwen3_depth16_deciles_unfused", "16-layer scaled", "unfused"),
    ("qwen3_depth16_deciles_fused", "16-layer scaled", "fused"),
    ("qwen3_deciles_unfused", "0.6 billion default", "unfused"),
    ("qwen3_deciles_fused", "0.6 billion default", "fused"),
]

QWEN3_COLORS = {
    "8-layer scaled": "#59A14F",
    "16-layer scaled": "#E15759",
    "0.6 billion default": "#4E79A7",
}

METRIC_FILES = [
    DEFAULT_STATS_ROOT / "qwen3_0p6b_llama10ep" / "qwen3_primary_binary" / "qwen3_primary_binary-metrics.csv",
    DEFAULT_STATS_ROOT / "qwen3_0p6b_llama10ep" / "qwen3_additional_binary" / "qwen3_additional_binary-metrics.csv",
    DEFAULT_STATS_ROOT / "qwen3_0p6b_llama10ep" / "qwen3_length_of_stay" / "qwen3_length_of_stay-metrics.csv",
    DEFAULT_STATS_ROOT / "qwen3_0p6b_llama10ep" / "qwen3_extended_regression" / "qwen3_extended_regression-metrics.csv",
    DEFAULT_STATS_ROOT / "qwen3_scaled" / "qwen3_scaled_primary_binary" / "qwen3_scaled_primary_binary-metrics.csv",
    DEFAULT_STATS_ROOT / "qwen3_scaled" / "qwen3_scaled_additional_binary" / "qwen3_scaled_additional_binary-metrics.csv",
    DEFAULT_STATS_ROOT / "qwen3_scaled" / "qwen3_scaled_length_of_stay" / "qwen3_scaled_length_of_stay-metrics.csv",
    DEFAULT_STATS_ROOT / "qwen3_scaled" / "qwen3_scaled_extended_regression" / "qwen3_scaled_extended_regression-metrics.csv",
]


def load_qwen3_metrics() -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in METRIC_FILES]
    metrics = pd.concat(frames, ignore_index=True)
    selected = BINARY_OUTCOME_ORDER_EXP12 + REGRESSION_OUTCOME_ORDER
    return metrics[
        metrics["outcome"].isin(selected)
        & metrics["metric"].isin(["roc_auc", "spearman_rho"])
    ].copy()


def _plot_qwen3_panel(
    ax,
    metrics: pd.DataFrame,
    *,
    outcome_order: list[str],
    metric: str,
    panel_title: str,
    xlabel: str,
    x_lo_floor: float | None,
) -> None:
    handle_order = [handle for handle, _, _ in QWEN3_ORDER]
    handle_map = {handle: (variant, tokenization) for handle, variant, tokenization in QWEN3_ORDER}
    n_out = len(outcome_order)
    n_h = len(handle_order)
    base_y = np.arange(n_out, dtype=float)[::-1]
    offsets = (
        np.linspace((n_h - 1) * FOREST_HANDLE_SPREAD, -(n_h - 1) * FOREST_HANDLE_SPREAD, n_h)
        if n_h > 1
        else np.array([0.0])
    )
    pts: list[float] = []
    for o_idx, outcome in enumerate(outcome_order):
        sub = metrics[(metrics["metric"] == metric) & (metrics["outcome"] == outcome)]
        for h_idx, handle in enumerate(handle_order):
            row = sub[sub["handle"] == handle]
            if row.empty:
                continue
            r = row.iloc[0]
            pts.append(float(r["point"]))
            variant, tokenization = handle_map[handle]
            marker = "o" if tokenization == "unfused" else "s"
            alpha = 0.55 if tokenization == "unfused" else 0.95
            color = QWEN3_COLORS[variant]
            y = base_y[o_idx] + offsets[h_idx]
            ax.hlines(y, r["ci_lo"], r["ci_hi"], color=color, linewidth=FOREST_LINE_WIDTH, alpha=alpha)
            ax.scatter(
                r["point"],
                y,
                color=color,
                marker=marker,
                s=FOREST_MARKER_SIZE,
                edgecolor="#222222",
                linewidth=0.25,
                alpha=alpha,
                zorder=4,
            )
    ax.set_yticks(base_y)
    ax.set_yticklabels([FOREST_OUTCOME_LABELS[o] for o in outcome_order], fontsize=FOREST_TICK_SIZE)
    ax.set_title(panel_title, fontsize=FOREST_TITLE_SIZE, pad=4)
    ax.set_xlabel(xlabel, fontsize=FOREST_LABEL_SIZE)
    ax.tick_params(axis="x", labelsize=FOREST_TICK_SIZE, length=2)
    ax.grid(axis="x", alpha=0.20, linewidth=0.5)
    if pts:
        _xlim_from_points(ax, np.array(pts), x_lo_floor=x_lo_floor, x_hi_ceil=None)
    ax.set_ylim(base_y.min() - FOREST_YLIM_PAD, base_y.max() + FOREST_YLIM_PAD)
    _add_forest_outcome_banding(ax, base_y)
    _add_forest_outcome_separators(ax, base_y)


def build_qwen3_outcome_forests(metrics: pd.DataFrame, out_dir: Path) -> None:
    source = metrics[
        ["family_name", "outcome", "handle", "metric", "point", "ci_lo", "ci_hi"]
    ].copy()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sources").mkdir(parents=True, exist_ok=True)
    _save_source(source, out_dir / "sources" / "qwen3_outcome_forests_source.csv")

    fig, axes = plt.subplots(1, 2, figsize=FOREST_FIGSIZE, sharey=False)
    _plot_qwen3_panel(
        axes[0],
        metrics,
        outcome_order=BINARY_OUTCOME_ORDER_EXP12,
        metric="roc_auc",
        panel_title="Binary outcomes",
        xlabel="AUROC",
        x_lo_floor=FOREST_BINARY_AUROC_X_LO_FLOOR,
    )
    _plot_qwen3_panel(
        axes[1],
        metrics,
        outcome_order=REGRESSION_OUTCOME_ORDER,
        metric="spearman_rho",
        panel_title="Regression outcomes",
        xlabel=r"Spearman $\rho$",
        x_lo_floor=None,
    )
    color_handles = _forest_color_handles(list(QWEN3_COLORS.items()))
    marker_handles = _forest_marker_handles([
        ("Unfused", "o", 0.55),
        ("Fused", "s", 0.95),
    ])
    layout_top = FOREST_LAYOUT_TOP_EDGE
    fig.tight_layout(rect=[0, 0, 1, layout_top])
    legend_rows = max(
        _legend_row_count(len(color_handles), 3),
        _legend_row_count(len(marker_handles), 2),
    )
    _add_forest_knob_legends(
        fig,
        left_handles=color_handles,
        left_title="Model scale (color)",
        right_handles=marker_handles,
        right_title="Tokenization (marker)",
        top_edge=_forest_legend_anchor_y(layout_top=layout_top, legend_rows=legend_rows),
        left_ncol=3,
        right_ncol=2,
    )
    fig.savefig(out_dir / "qwen3_outcome_forests.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "qwen3_outcome_forests.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--figures_dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    _style()
    metrics = load_qwen3_metrics()
    build_qwen3_outcome_forests(metrics, args.figures_dir.expanduser().resolve())
    print(args.figures_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
