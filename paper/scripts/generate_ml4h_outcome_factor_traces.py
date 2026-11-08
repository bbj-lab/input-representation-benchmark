#!/usr/bin/env python3
"""Render outcome x factor traces for the ML4H 24-48h provisional bundle.

This alternative replaces the radar encoding. Outcome is a print-safe Paul Tol
color, the factor level is the x-axis position and the same haloed marker used
on the spider charts, and absolute performance is the y-axis. Each plotted
point is the mean of three seeds and the two architectures. Panels that share
a metric also share a y-scale so a small change cannot look large. A companion
figure shows the best-minus-worst gap so the reader can see where representation
choices actually move the metric.

It reads the already materialized provisional metrics; it does not rerun any
Stage 3 prediction scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = REPO_ROOT.parent / "Archive/ml4h-dump/08_ML4H_provisional"
TRACE_SECTION = "08_outcome_factor_traces"
EXPERIMENTS = ("exp1", "exp2", "exp3")

TASK_ORDER = ("binary", "regression")
TASK_METRIC_LABELS = {"binary": "AUROC", "regression": r"Spearman $\rho$"}
TASK_BASELINE_DELTA_LABELS = {
    "binary": r"$\Delta$ baseline AUROC",
    "regression": r"$\Delta$ baseline Spearman $\rho$",
}
TASK_DELTA_LABELS = {
    "binary": r"$\Delta$ AUROC (best $-$ worst)",
    "regression": r"$\Delta$ Spearman $\rho$ (best $-$ worst)",
}
TASK_KEY_TITLES = {
    "binary": "Binary outcomes (AUROC)",
    "regression": r"Regression outcomes (Spearman $\rho$)",
}

# Paul Tol's bright, colorblind-safe hues. Color encodes a clinical family,
# not an individual outcome: each panel therefore uses only four hues.
FAMILY_COLORS = {
    "Hospital outcomes": "#0077BB",
    "Organ support": "#CC3311",
    "Laboratory": "#009988",
    "Electrolytes": "#009988",
    "Biomarkers": "#EE7733",
    "Hematology": "#AA4499",
    "Vitals": "#332288",
}
FAMILY_ORDER = {
    "binary": ("Hospital outcomes", "Organ support", "Laboratory", "Vitals"),
    "regression": ("Electrolytes", "Biomarkers", "Hematology", "Vitals"),
}
OUTCOME_FAMILIES = {
    "same_admission_death": "Hospital outcomes",
    "long_length_of_stay": "Hospital outcomes",
    "icu_admission": "Hospital outcomes",
    "prolonged_icu_stay": "Hospital outcomes",
    "imv_event": "Organ support",
    "vasopressor_initiation": "Organ support",
    "crrt_initiation": "Organ support",
    "hemodialysis_initiation": "Organ support",
    "hyperkalemia": "Laboratory",
    "severe_hypokalemia": "Laboratory",
    "profound_hyponatremia": "Laboratory",
    "severe_hypernatremia": "Laboratory",
    "hypoglycemia": "Laboratory",
    "severe_anemia": "Laboratory",
    "tachycardia_hr130": "Vitals",
    "severe_hypertension": "Vitals",
    "hypotension": "Vitals",
    "min_potassium": "Electrolytes",
    "peak_potassium": "Electrolytes",
    "min_sodium": "Electrolytes",
    "max_sodium": "Electrolytes",
    "min_creatinine": "Biomarkers",
    "peak_creatinine": "Biomarkers",
    "min_troponin": "Biomarkers",
    "peak_troponin": "Biomarkers",
    "min_bnp": "Biomarkers",
    "peak_bnp": "Biomarkers",
    "min_glucose": "Biomarkers",
    "max_glucose": "Biomarkers",
    "min_hemoglobin": "Hematology",
    "max_hemoglobin": "Hematology",
    "min_heart_rate": "Vitals",
    "max_heart_rate": "Vitals",
    "min_sbp": "Vitals",
    "max_sbp": "Vitals",
    "min_dbp": "Vitals",
    "max_dbp": "Vitals",
}
# Six dashes, shared across families. Styles cycle if a family has more than six
# members (Biomarkers has eight after glucose). Long strokes keep patterns distinct.
FAMILY_LINESTYLES = (
    "solid",
    "dash",
    "dash_dot",
    "dash_dot_dot",
    "dash_dot_dot_dot",
    "dotted",
)
LINESTYLE_PATTERNS = {
    "solid": "solid",
    "dash": (0, (7.0, 3.0)),
    "dash_dot": (0, (6.0, 2.4, 1.2, 2.4)),
    "dash_dot_dot": (0, (6.0, 2.2, 1.2, 2.2, 1.2, 2.2)),
    "dash_dot_dot_dot": (0, (6.0, 2.0, 1.2, 2.0, 1.2, 2.0, 1.2, 2.0)),
    "dotted": (0, (1.1, 2.4)),
}
# Copy the spider-chart marker recipe: small filled glyph, white halo, thin dark rim.
MARKER_FACE_AREA = 36.0
MARKER_HALO_AREA = 78.0
MARKER_AREA_SCALE = {
    "o": 1.00,
    "s": 0.95,
    "^": 1.20,
    "D": 0.95,
    "v": 1.20,
    "P": 1.35,
    "X": 1.25,
    "h": 1.10,
}
MARKER_EDGE_COLOR = "#111111"
MARKER_EDGE_WIDTH = 0.60
TRACE_HALO_WIDTH = 3.6
TRACE_LINE_WIDTH = 2.35
TRACE_LINE_ALPHA = 0.95
GRID_COLOR = "#D4D4D4"
SPINE_COLOR = "#333333"
REFERENCE_COLOR = "#8A8A8A"
FIGURE_PNG_DPI = 600
# 20 degrees below the axis, reading to the right of each tick.
LEVEL_LABEL_ROTATION = -20.0

CELL_WIDTH_IN = 9.2
TRACE_GUTTER_WIDTH_IN = 1.15
TRACE_AXIS_HEIGHT_IN = 5.1
LABEL_ROW_EXTRA_IN = 0.72
LETTER_ROW_HEIGHT_IN = 0.40
SHAPE_LEGEND_HEIGHT_IN = 1.75
KEY_GAP_HEIGHT_IN = 0.55
OUTCOME_KEY_HEIGHT_IN = 5.8
DELTA_ROW_HEIGHT_IN = 0.38
DELTA_LABEL_PT = 16.0
POOLED_BACKBONE = "llama_qwen_mean"
SHAPE_LEGEND_FACE = "#111111"
SHAPE_LEGEND_MARKER_SIZE = 8.8


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_spider_renderer():
    return load_module(
        "ml4h_spider_renderer_for_traces",
        REPO_ROOT / "paper/scripts/generate_ml4h_spider_charts.py",
    )


def load_alternatives_renderer():
    return load_module(
        "ml4h_visual_alternatives_for_traces",
        REPO_ROOT / "paper/scripts/generate_ml4h_provisional_visual_alternatives.py",
    )


def primary_baseline_level_id(experiment: str, *, spider) -> str:
    baseline = spider.baseline_filters(experiment)
    for item in spider.experiment_series(experiment):
        if item.filters == baseline:
            return item.level_id
    raise ValueError(f"{experiment}: no primary baseline series.")


def baseline_outcome_order(
    source: pd.DataFrame,
    task_type: str,
    *,
    experiment: str,
    spider,
) -> tuple[str, ...]:
    """Highest baseline AUROC or Spearman ρ first, matching the traces' vertical order."""
    available = set(source.loc[source["task_type"] == task_type, "outcome"].astype(str))
    fallback = spider.outcome_order(task_type, available)
    level_id = primary_baseline_level_id(experiment, spider=spider)
    baseline = source.loc[
        (source["task_type"] == task_type) & (source["level_id"].astype(str) == level_id)
    ]
    if baseline.empty:
        return fallback
    ranked = (
        baseline.groupby(baseline["outcome"].astype(str), sort=False)["point"]
        .mean()
        .sort_values(ascending=False)
    )
    ordered = [outcome for outcome in ranked.index if outcome in available]
    leftover = [outcome for outcome in fallback if outcome not in set(ordered)]
    return tuple(ordered + leftover)


def average_over_architectures(source: pd.DataFrame, *, spider) -> pd.DataFrame:
    """Equal-weight mean of the Llama and Qwen seed-means (3 seeds × 2 architectures)."""
    expected = tuple(spider.BACKBONE_ORDER)
    if expected != ("llama", "qwen"):
        raise ValueError(f"Unexpected architectures: {expected}")
    present = set(source["backbone"].astype(str))
    if present != set(expected):
        raise ValueError(f"Need both architectures to pool; found {sorted(present)}")
    group_keys = [
        "experiment",
        "condition",
        "condition_title",
        "task_type",
        "metric",
        "outcome",
        "outcome_label",
        "level_id",
        "level_label",
        "color",
        "marker",
        "linestyle",
        "is_baseline",
        "component_config_ids",
    ]
    grouped = source.groupby(group_keys, as_index=False, sort=False).agg(
        point=("point", "mean"),
        lo=("point", "min"),
        hi=("point", "max"),
        n_seeds=("n_seeds", "first"),
        n_architectures=("backbone", "nunique"),
    )
    if grouped.empty:
        raise ValueError("No rows left after averaging architectures.")
    if not (grouped["n_architectures"] == 2).all():
        raise ValueError("Every pooled cell must include both architectures.")
    if not (grouped["n_seeds"] == 3).all():
        raise ValueError("Every pooled cell must keep the three-seed mean.")
    grouped["backbone"] = POOLED_BACKBONE
    return grouped


def outcome_linestyles(
    source: pd.DataFrame,
    task_type: str,
    *,
    experiment: str,
    spider,
) -> dict[str, str]:
    """Cycle four distinct dashes inside each family, highest baseline first."""
    ordered = baseline_outcome_order(
        source, task_type, experiment=experiment, spider=spider
    )
    styles: dict[str, str] = {}
    for family in FAMILY_ORDER[task_type]:
        members = [
            outcome
            for outcome in ordered
            if OUTCOME_FAMILIES.get(outcome) == family
        ]
        styles.update(
            {
                outcome: FAMILY_LINESTYLES[index % len(FAMILY_LINESTYLES)]
                for index, outcome in enumerate(members)
            }
        )
    return styles


def line_style(style: str):
    """Return the Matplotlib pattern for a compact, publication-safe style name."""
    try:
        return LINESTYLE_PATTERNS[style]
    except KeyError as error:
        raise ValueError(f"Unknown outcome line style: {style}") from error


def level_layout(experiment: str, *, spider) -> dict[str, tuple[dict[str, object], ...]]:
    """Return ordered level metadata (position, shape, label) per condition."""
    layout: dict[str, tuple[dict[str, object], ...]] = {}
    for column in spider.panel_columns(experiment):
        levels = []
        for index, item in enumerate(column.series):
            label = item.label
            if spider.series_is_baseline(experiment, item):
                label = f"{label} (baseline)"
            levels.append(
                {
                    "level_id": item.level_id,
                    "label": label,
                    "position": float(index),
                    "marker": item.marker,
                    "fillstyle": "full",
                }
            )
        layout[column.key] = tuple(levels)
    return layout


def trace_source(
    metrics: pd.DataFrame,
    experiment: str,
    *,
    spider,
    alternatives,
) -> pd.DataFrame:
    """Seed-mean, then architecture-mean, scores with colors, shapes, best/worst flags."""
    source = average_over_architectures(
        alternatives.factor_seed_interval_source(
            metrics, experiment, spider=spider
        ),
        spider=spider,
    )
    layout = level_layout(experiment, spider=spider)
    color_index: dict[tuple[str, str], int] = {}
    linestyle_index: dict[tuple[str, str], str] = {}
    for task_type in TASK_ORDER:
        outcomes = baseline_outcome_order(
            source, task_type, experiment=experiment, spider=spider
        )
        for index, outcome in enumerate(outcomes):
            color_index[(task_type, outcome)] = index
        linestyle_index.update(
            {
                (task_type, outcome): linestyle
                for outcome, linestyle in outcome_linestyles(
                    source, task_type, experiment=experiment, spider=spider
                ).items()
            }
        )
    out = source.copy()
    keys = list(zip(out["task_type"].astype(str), out["outcome"].astype(str)))
    unknown = {
        outcome
        for _, outcome in keys
        if outcome not in OUTCOME_FAMILIES
    }
    if unknown:
        raise ValueError(f"{experiment}: outcomes missing family mapping: {sorted(unknown)}")
    out["outcome_index"] = [color_index[key] for key in keys]
    out["outcome_family"] = [OUTCOME_FAMILIES[outcome] for _, outcome in keys]
    out["outcome_color"] = [
        FAMILY_COLORS[OUTCOME_FAMILIES[outcome]] for _, outcome in keys
    ]
    out["outcome_linestyle"] = [linestyle_index[key] for key in keys]
    level_keys = list(zip(out["condition"].astype(str), out["level_id"].astype(str)))
    level_lookup = {
        (condition, level["level_id"]): level
        for condition, levels in layout.items()
        for level in levels
    }
    missing = {key for key in level_keys if key not in level_lookup}
    if missing:
        raise ValueError(f"{experiment}: levels missing from layout: {sorted(missing)}")
    out["level_position"] = [level_lookup[key]["position"] for key in level_keys]
    out["level_marker"] = [level_lookup[key]["marker"] for key in level_keys]
    out["level_fillstyle"] = [level_lookup[key]["fillstyle"] for key in level_keys]
    out["level_label"] = [level_lookup[key]["label"] for key in level_keys]

    grouped = out.groupby(["condition", "task_type", "outcome"])["point"]
    panel_max = grouped.transform("max")
    panel_min = grouped.transform("min")
    varies = panel_max > panel_min
    out["level_delta"] = panel_max - panel_min
    out["is_best"] = varies & (out["point"] == panel_max)
    out["is_worst"] = varies & (out["point"] == panel_min)
    return out


def delta_summary(source: pd.DataFrame) -> pd.DataFrame:
    """Best-minus-worst gap per panel and outcome, with the winning/losing levels."""
    records: list[dict[str, object]] = []
    keys = ["experiment", "condition", "condition_title", "task_type", "outcome"]
    for key, group in source.groupby(keys, sort=False):
        best = group.loc[group["point"].idxmax()]
        worst = group.loc[group["point"].idxmin()]
        records.append(
            {
                **dict(zip(keys, key)),
                "outcome_label": best["outcome_label"],
                "outcome_family": best["outcome_family"],
                "outcome_color": best["outcome_color"],
                "outcome_linestyle": best["outcome_linestyle"],
                "delta": float(best["point"]) - float(worst["point"]),
                "best_level_id": best["level_id"],
                "best_level_label": best["level_label"],
                "best_marker": best["level_marker"],
                "best_fillstyle": best["level_fillstyle"],
                "best_point": float(best["point"]),
                "worst_level_id": worst["level_id"],
                "worst_level_label": worst["level_label"],
                "worst_marker": worst["level_marker"],
                "worst_fillstyle": worst["level_fillstyle"],
                "worst_point": float(worst["point"]),
            }
        )
    return pd.DataFrame(records)


def trace_panel_col(panel_index: int) -> int:
    return panel_index * 2


def padded_limits(values, *, lower: float | None = None) -> tuple[float, float]:
    """Shared axis range so neighboring panels stay comparable."""
    clean = np.asarray(list(values), dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return (0.0, 1.0)
    low, high = float(clean.min()), float(clean.max())
    pad = 0.06 * max(high - low, 1e-3)
    if lower is None:
        return low - pad, high + pad
    return lower, high + pad


def marker_area(marker: str, base: float) -> float:
    return base * MARKER_AREA_SCALE.get(marker, 1.0)


def draw_spider_marker(ax, x, y, *, marker: str, color: str, scale: float = 1.0, zorder: float = 3.0) -> None:
    """Same filled glyph + white halo + dark rim used on the spider charts."""
    ax.scatter(
        x,
        y,
        marker=marker,
        s=marker_area(marker, MARKER_HALO_AREA) * scale,
        facecolors="white",
        edgecolors="none",
        zorder=zorder,
    )
    ax.scatter(
        x,
        y,
        marker=marker,
        s=marker_area(marker, MARKER_FACE_AREA) * scale,
        facecolors=color,
        edgecolors=MARKER_EDGE_COLOR,
        linewidths=MARKER_EDGE_WIDTH,
        zorder=zorder + 0.02,
    )


def column_major(items: list, ncol: int) -> list:
    """Reorder so a row-major legend fills top-to-bottom in each column."""
    if ncol <= 1 or len(items) <= ncol:
        return list(items)
    nrow = int(math.ceil(len(items) / ncol))
    grid = [[None] * ncol for _ in range(nrow)]
    index = 0
    for col in range(ncol):
        for row in range(nrow):
            if index < len(items):
                grid[row][col] = items[index]
                index += 1
    ordered = []
    for row in grid:
        for item in row:
            if item is not None:
                ordered.append(item)
    return ordered


def style_axis(ax, *, spider) -> None:
    ax.tick_params(
        axis="both",
        labelsize=spider.NEURIPS_SMALL_PT,
        colors=spider.TEXT_COLOR,
        length=3.2,
        width=0.7,
        direction="out",
    )
    for label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        label.set_fontfamily("serif")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(0.8)
        ax.spines[spine].set_color(SPINE_COLOR)


def draw_panel_letter(ax, letter: str, *, spider) -> None:
    """Place the panel letter in its own row, not on the plot or architecture title."""
    ax.set_axis_off()
    ax.text(
        0.0,
        0.10,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=spider.PANEL_LETTER_PT,
        fontweight="normal",
        color=spider.TEXT_COLOR,
        fontfamily="serif",
        clip_on=False,
        zorder=10,
    )


def level_xlim(levels: tuple[dict[str, object], ...]) -> tuple[float, float]:
    positions = [float(level["position"]) for level in levels]
    return min(positions) - 0.55, max(positions) + 0.55


def wrap_level_label(label: str) -> str:
    """Keep the −20° tilt, but stop one long line from reaching the next panel."""
    text = str(label)
    if len(text) <= 24:
        return text
    for separator in (", ", " + ", " ("):
        if separator in text:
            head, tail = text.split(separator, 1)
            if separator == " (":
                return f"{head}\n({tail}"
            return f"{head}{separator.rstrip()}\n{tail}"
    return text


def draw_trace_panel(
    ax,
    rows: pd.DataFrame,
    *,
    levels: tuple[dict[str, object], ...],
    outcomes: tuple[str, ...],
    task_type: str,
    show_level_labels: bool,
    show_ylabel: bool,
    spider,
    ylim: tuple[float, float],
    marker_scale: float = 1.0,
    ylabel: str | None = None,
    reference_y: float | None = None,
    show_range: bool = False,
) -> None:
    """One condition x task axis: colored outcome traces over factor levels.

    By default the binary panels draw a chance line at 0.5 when it falls inside
    the y-limits. Pass ``reference_y`` to force a different horizontal guide
    (for example 0.0 for baseline-relative deltas) and skip the chance line.
    """
    positions = [float(level["position"]) for level in levels]
    for outcome in outcomes:
        outcome_rows = rows.loc[rows["outcome"] == outcome].set_index("level_id")
        if outcome_rows.empty:
            continue
        color = str(outcome_rows.iloc[0]["outcome_color"])
        linestyle = line_style(str(outcome_rows.iloc[0]["outcome_linestyle"]))
        xs: list[float] = []
        ys: list[float] = []
        y_lo: list[float] = []
        y_hi: list[float] = []
        for level in levels:
            level_id = str(level["level_id"])
            if level_id not in outcome_rows.index:
                continue
            record = outcome_rows.loc[level_id]
            xs.append(float(level["position"]))
            ys.append(float(record["point"]))
            if show_range:
                y_lo.append(float(record["lo"]))
                y_hi.append(float(record["hi"]))
        if not xs:
            continue
        if show_range:
            lower = np.maximum(np.asarray(ys) - np.asarray(y_lo), 0.0)
            upper = np.maximum(np.asarray(y_hi) - np.asarray(ys), 0.0)
            ax.errorbar(
                xs,
                ys,
                yerr=np.vstack([lower, upper]),
                fmt="none",
                ecolor=color,
                elinewidth=0.7,
                capsize=2.0,
                capthick=0.6,
                zorder=2.5,
            )
        ax.plot(
            xs,
            ys,
            color="white",
            linewidth=TRACE_HALO_WIDTH,
            linestyle=linestyle,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=2,
        )
        ax.plot(
            xs,
            ys,
            color=color,
            linewidth=TRACE_LINE_WIDTH,
            alpha=TRACE_LINE_ALPHA,
            linestyle=linestyle,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=2.01,
        )
        for level in levels:
            level_id = str(level["level_id"])
            if level_id not in outcome_rows.index:
                continue
            record = outcome_rows.loc[level_id]
            draw_spider_marker(
                ax,
                float(level["position"]),
                float(record["point"]),
                marker=str(level["marker"]),
                color=color,
                scale=marker_scale,
                zorder=3.0,
            )
    ax.set_xticks(positions)
    if show_level_labels:
        ax.set_xticklabels(
            [wrap_level_label(str(level["label"])) for level in levels],
            rotation=LEVEL_LABEL_ROTATION,
            rotation_mode="anchor",
            ha="left",
            va="top",
        )
        ax.tick_params(axis="x", pad=3, labelsize=spider.NEURIPS_SMALL_PT)
    else:
        ax.set_xticklabels([])
    ax.set_xlim(*level_xlim(levels))
    ax.set_ylim(*ylim)
    guide = reference_y
    if guide is None and task_type == "binary" and ylim[0] <= 0.5 <= ylim[1]:
        guide = 0.5
    if guide is not None and ylim[0] <= guide <= ylim[1]:
        ax.axhline(
            guide,
            color=REFERENCE_COLOR,
            linewidth=0.8,
            linestyle=(0, (3, 2.4)),
            zorder=1,
        )
    if show_ylabel:
        ax.set_ylabel(
            TASK_METRIC_LABELS[task_type] if ylabel is None else ylabel,
            fontsize=spider.NEURIPS_BODY_PT,
            color=spider.TEXT_COLOR,
            fontfamily="serif",
        )
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.55, zorder=0)
    ax.set_axisbelow(True)
    style_axis(ax, spider=spider)


def draw_delta_panel(
    ax,
    rows: pd.DataFrame,
    *,
    task_type: str,
    spider,
    xlim: tuple[float, float],
    marker_scale: float = 1.0,
) -> None:
    """Sorted best-minus-worst gaps for one condition x task panel."""
    ordered = rows.sort_values("delta", ascending=False).reset_index(drop=True)
    y_positions = np.arange(len(ordered), dtype=float)[::-1]
    for index, record in ordered.iterrows():
        y = y_positions[index]
        color = str(record["outcome_color"])
        linestyle = line_style(str(record["outcome_linestyle"]))
        ax.plot(
            [0.0, float(record["delta"])],
            [y, y],
            color="white",
            linewidth=TRACE_HALO_WIDTH,
            linestyle=linestyle,
            solid_capstyle="round",
            zorder=2,
        )
        ax.plot(
            [0.0, float(record["delta"])],
            [y, y],
            color=color,
            linewidth=TRACE_LINE_WIDTH,
            linestyle=linestyle,
            solid_capstyle="round",
            zorder=2.01,
        )
        draw_spider_marker(
            ax,
            0.0,
            y,
            marker=str(record["worst_marker"]),
            color=color,
            scale=marker_scale,
            zorder=4.0,
        )
        draw_spider_marker(
            ax,
            float(record["delta"]),
            y,
            marker=str(record["best_marker"]),
            color=color,
            scale=marker_scale,
            zorder=5.0,
        )
    ax.set_yticks(y_positions)
    ax.set_yticklabels([str(label) for label in ordered["outcome_label"]])
    ax.set_ylim(y_positions.min() - 0.7, y_positions.max() + 0.7)
    ax.set_xlim(*xlim)
    ax.set_xlabel(
        TASK_DELTA_LABELS[task_type],
        fontsize=spider.NEURIPS_BODY_PT,
        color=spider.TEXT_COLOR,
        fontfamily="serif",
        labelpad=8,
    )
    ax.xaxis.grid(True, color=GRID_COLOR, linewidth=0.55, zorder=0)
    ax.set_axisbelow(True)
    for index in range(len(ordered)):
        if index % 2:
            ax.axhspan(
                y_positions[index] - 0.5,
                y_positions[index] + 0.5,
                color="#F7F7F7",
                zorder=0,
            )
    style_axis(ax, spider=spider)
    ax.tick_params(axis="y", length=0, pad=3, labelsize=DELTA_LABEL_PT)
    for label in ax.get_yticklabels():
        label.set_fontsize(DELTA_LABEL_PT)
        label.set_fontfamily("serif")


def figure_width_ratios(n_cond: int) -> list[float]:
    """Multi-condition: one column per condition. One-condition: AUROC | Spearman."""
    panel_count = 2 if n_cond == 1 else n_cond
    ratios: list[float] = []
    for panel_index in range(panel_count):
        ratios.append(CELL_WIDTH_IN)
        if panel_index < panel_count - 1:
            ratios.append(TRACE_GUTTER_WIDTH_IN)
    return ratios


def plot_column(n_cond: int, column_index: int) -> int:
    if n_cond == 1:
        return 0
    return trace_panel_col(column_index)


def task_panel_col(task_index: int) -> int:
    """Grid column for a side-by-side task panel (AUROC left, Spearman right)."""
    return task_index * 2


def shape_handles(levels: tuple[dict[str, object], ...]) -> list[Line2D]:
    handles = []
    for level in levels:
        handles.append(
            Line2D(
                [0],
                [0],
                marker=str(level["marker"]),
                color=MARKER_EDGE_COLOR,
                linestyle="None",
                markerfacecolor=SHAPE_LEGEND_FACE,
                markeredgecolor=MARKER_EDGE_COLOR,
                markeredgewidth=MARKER_EDGE_WIDTH,
                linewidth=0,
                markersize=SHAPE_LEGEND_MARKER_SIZE,
                label=str(level["label"]),
            )
        )
    return handles


def place_shape_legend(ax, levels, *, title: str, spider) -> None:
    ax.set_axis_off()
    ncol = 2 if len(levels) >= 2 else 1
    legend = ax.legend(
        handles=column_major(shape_handles(levels), ncol),
        title=title,
        loc="center",
        bbox_to_anchor=(0.5, 0.5),
        ncol=ncol,
        frameon=False,
        fontsize=spider.LEGEND_PT,
        title_fontsize=spider.COLUMN_TITLE_PT,
        handlelength=2.0,
        columnspacing=1.55,
        handletextpad=0.50,
        labelspacing=0.42,
        borderaxespad=0.0,
        borderpad=0.2,
        prop={"family": "serif", "weight": "normal", "size": spider.LEGEND_PT},
    )
    legend._legend_box.align = "center"
    legend.get_title().set_fontfamily("serif")
    legend.get_title().set_ha("center")
    legend.get_title().set_color(spider.TEXT_COLOR)
    ax.add_artist(legend)


def family_ordered_outcomes(
    source: pd.DataFrame,
    task_type: str,
    *,
    experiment: str,
    spider,
) -> tuple[str, ...]:
    """Baseline-score order within the fixed clinical-family order."""
    ranked = baseline_outcome_order(
        source, task_type, experiment=experiment, spider=spider
    )
    return tuple(
        outcome
        for family in FAMILY_ORDER[task_type]
        for outcome in ranked
        if OUTCOME_FAMILIES[outcome] == family
    )


def place_outcome_key(
    ax,
    source: pd.DataFrame,
    *,
    experiment: str,
    spider,
) -> None:
    """Full-name family key with baseline-score order within each family."""
    ax.set_axis_off()
    anchors = {"binary": 0.22, "regression": 0.78}
    for task_type in TASK_ORDER:
        outcomes = family_ordered_outcomes(
            source, task_type, experiment=experiment, spider=spider
        )
        styles = (
            source.loc[
                source["task_type"] == task_type,
                ["outcome", "outcome_color", "outcome_linestyle"],
            ]
            .drop_duplicates()
            .set_index("outcome")
        )
        ax.text(
            anchors[task_type],
            1.0,
            TASK_KEY_TITLES[task_type],
            ha="center",
            va="top",
            fontsize=spider.COLUMN_TITLE_PT,
            color=spider.TEXT_COLOR,
            fontfamily="serif",
            transform=ax.transAxes,
        )
        for family_index, family in enumerate(FAMILY_ORDER[task_type]):
            members = [
                outcome for outcome in outcomes if OUTCOME_FAMILIES[outcome] == family
            ]
            handles = [
                Line2D(
                    [0],
                    [0],
                    color=str(styles.loc[outcome, "outcome_color"]),
                    linestyle=line_style(str(styles.loc[outcome, "outcome_linestyle"])),
                    linewidth=2.4,
                    marker="",
                    label=spider.FULL_OUTCOME_LABELS.get(outcome, outcome),
                )
                for outcome in members
            ]
            column = family_index % 2
            row = family_index // 2
            legend = ax.legend(
                handles=handles,
                title=family,
                loc="upper center",
                bbox_to_anchor=(
                    anchors[task_type] + (-0.115 if column == 0 else 0.115),
                    0.88 if row == 0 else 0.43,
                ),
                ncol=1,
                frameon=False,
                fontsize=spider.OUTCOME_KEY_PT,
                title_fontsize=spider.OUTCOME_KEY_PT,
                handlelength=2.6,
                handletextpad=0.45,
                labelspacing=0.32,
                borderaxespad=0.0,
                prop={
                    "family": "serif",
                    "weight": "normal",
                    "size": spider.OUTCOME_KEY_PT,
                },
            )
            legend.get_title().set_fontfamily("serif")
            legend.get_title().set_fontweight("bold")
            legend.get_title().set_color(spider.TEXT_COLOR)
            ax.add_artist(legend)


def plot_outcome_factor_traces(
    *,
    experiment: str,
    source: pd.DataFrame,
    output_root: Path,
    entries: list[dict[str, object]],
    input_hashes: dict[str, str],
    alternatives,
    spider,
) -> None:
    spider.style()
    columns = spider.panel_columns(experiment)
    layout = level_layout(experiment, spider=spider)
    n_cond = len(columns)
    side_by_side_tasks = n_cond == 1
    width_ratios = figure_width_ratios(n_cond)
    n_cols = len(width_ratios)
    width = max(sum(width_ratios), 12.5)
    height_ratios = [SHAPE_LEGEND_HEIGHT_IN, LETTER_ROW_HEIGHT_IN]
    if side_by_side_tasks:
        height_ratios.append(TRACE_AXIS_HEIGHT_IN + LABEL_ROW_EXTRA_IN)
    else:
        for task_index, _task_type in enumerate(TASK_ORDER):
            extra = LABEL_ROW_EXTRA_IN if task_index == len(TASK_ORDER) - 1 else 0.0
            height_ratios.append(TRACE_AXIS_HEIGHT_IN + extra)
    height_ratios.extend([KEY_GAP_HEIGHT_IN, OUTCOME_KEY_HEIGHT_IN])
    fig = plt.figure(figsize=(width, sum(height_ratios)))
    grid = GridSpec(
        len(height_ratios),
        n_cols,
        figure=fig,
        height_ratios=height_ratios,
        width_ratios=width_ratios,
        left=0.07,
        right=0.985,
        top=0.985,
        bottom=0.015,
        hspace=0.18,
        wspace=0.04,
    )
    y_limits = {
        task_type: padded_limits(source.loc[source["task_type"] == task_type, "point"])
        for task_type in TASK_ORDER
    }
    if side_by_side_tasks:
        column = columns[0]
        place_shape_legend(
            fig.add_subplot(grid[0, :]),
            layout[column.key],
            title=column.title,
            spider=spider,
        )
        for task_index, _task_type in enumerate(TASK_ORDER):
            draw_panel_letter(
                fig.add_subplot(grid[1, task_panel_col(task_index)]),
                spider.PANEL_LETTERS[task_index],
                spider=spider,
            )
        for task_index, task_type in enumerate(TASK_ORDER):
            outcomes = baseline_outcome_order(
                source, task_type, experiment=experiment, spider=spider
            )
            axis = fig.add_subplot(grid[2, task_panel_col(task_index)])
            rows = source.loc[
                (source["condition"] == column.key)
                & (source["task_type"] == task_type)
            ]
            draw_trace_panel(
                axis,
                rows,
                levels=layout[column.key],
                outcomes=outcomes,
                task_type=task_type,
                show_level_labels=True,
                show_ylabel=True,
                spider=spider,
                ylim=y_limits[task_type],
            )
    else:
        for column_index, column in enumerate(columns):
            col = plot_column(n_cond, column_index)
            place_shape_legend(
                fig.add_subplot(grid[0, col]),
                layout[column.key],
                title=column.title,
                spider=spider,
            )
            draw_panel_letter(
                fig.add_subplot(grid[1, col]),
                spider.PANEL_LETTERS[column_index],
                spider=spider,
            )
        for task_index, task_type in enumerate(TASK_ORDER):
            row = 2 + task_index
            outcomes = baseline_outcome_order(
                source, task_type, experiment=experiment, spider=spider
            )
            for column_index, column in enumerate(columns):
                axis = fig.add_subplot(grid[row, plot_column(n_cond, column_index)])
                rows = source.loc[
                    (source["condition"] == column.key)
                    & (source["task_type"] == task_type)
                ]
                draw_trace_panel(
                    axis,
                    rows,
                    levels=layout[column.key],
                    outcomes=outcomes,
                    task_type=task_type,
                    show_level_labels=task_index == len(TASK_ORDER) - 1,
                    show_ylabel=column_index == 0,
                    spider=spider,
                    ylim=y_limits[task_type],
                )
    fig.add_subplot(grid[-2, :]).set_axis_off()
    place_outcome_key(
        fig.add_subplot(grid[-1, :]),
        source,
        experiment=experiment,
        spider=spider,
    )
    write_figure(
        output_root=output_root,
        entries=entries,
        figure=fig,
        stem=f"{experiment}_outcome_factor_traces",
        source=source,
        description=(
            f"{spider.EXPERIMENT_DISPLAY[experiment]} 24-48h absolute outcome traces: "
            "mean of 3 seeds and 2 architectures; Paul Tol print-safe outcome color; "
            "shared y-scale per metric; factor level on the x-axis using the same "
            "marker shapes as the spider charts."
        ),
        input_hashes=input_hashes,
    )


def plot_outcome_factor_deltas(
    *,
    experiment: str,
    source: pd.DataFrame,
    summary: pd.DataFrame,
    output_root: Path,
    entries: list[dict[str, object]],
    input_hashes: dict[str, str],
    alternatives,
    spider,
) -> None:
    spider.style()
    columns = spider.panel_columns(experiment)
    layout = level_layout(experiment, spider=spider)
    n_cond = len(columns)
    side_by_side_tasks = n_cond == 1
    n_cols = 2 if side_by_side_tasks else n_cond
    task_counts = {
        task_type: len(
            baseline_outcome_order(
                source, task_type, experiment=experiment, spider=spider
            )
        )
        for task_type in TASK_ORDER
    }
    height_ratios = [SHAPE_LEGEND_HEIGHT_IN, LETTER_ROW_HEIGHT_IN]
    if side_by_side_tasks:
        height_ratios.append(
            max(3.8, max(task_counts.values()) * DELTA_ROW_HEIGHT_IN)
        )
    else:
        for task_type in TASK_ORDER:
            height_ratios.append(max(3.8, task_counts[task_type] * DELTA_ROW_HEIGHT_IN))
    x_limits = {
        task_type: padded_limits(
            summary.loc[summary["task_type"] == task_type, "delta"],
            lower=0.0,
        )
        for task_type in TASK_ORDER
    }
    width = max(CELL_WIDTH_IN * n_cols, 12.5)
    fig = plt.figure(figsize=(width, sum(height_ratios)))
    grid = GridSpec(
        len(height_ratios),
        n_cols,
        figure=fig,
        height_ratios=height_ratios,
        width_ratios=[CELL_WIDTH_IN] * n_cols,
        left=0.20,
        right=0.99,
        top=0.985,
        bottom=0.04,
        hspace=0.22,
        wspace=0.42,
    )
    if side_by_side_tasks:
        column = columns[0]
        place_shape_legend(
            fig.add_subplot(grid[0, :]),
            layout[column.key],
            title=column.title,
            spider=spider,
        )
        for task_index, task_type in enumerate(TASK_ORDER):
            draw_panel_letter(
                fig.add_subplot(grid[1, task_index]),
                spider.PANEL_LETTERS[task_index],
                spider=spider,
            )
            rows = summary.loc[
                (summary["condition"] == column.key)
                & (summary["task_type"] == task_type)
            ]
            draw_delta_panel(
                fig.add_subplot(grid[2, task_index]),
                rows,
                task_type=task_type,
                spider=spider,
                xlim=x_limits[task_type],
            )
    else:
        for column_index, column in enumerate(columns):
            place_shape_legend(
                fig.add_subplot(grid[0, column_index]),
                layout[column.key],
                title=column.title,
                spider=spider,
            )
            draw_panel_letter(
                fig.add_subplot(grid[1, column_index]),
                spider.PANEL_LETTERS[column_index],
                spider=spider,
            )
        for task_index, task_type in enumerate(TASK_ORDER):
            row = 2 + task_index
            for column_index, column in enumerate(columns):
                rows = summary.loc[
                    (summary["condition"] == column.key)
                    & (summary["task_type"] == task_type)
                ]
                draw_delta_panel(
                    fig.add_subplot(grid[row, column_index]),
                    rows,
                    task_type=task_type,
                    spider=spider,
                    xlim=x_limits[task_type],
                )
    write_figure(
        output_root=output_root,
        entries=entries,
        figure=fig,
        stem=f"{experiment}_outcome_factor_delta",
        source=summary,
        description=(
            f"{spider.EXPERIMENT_DISPLAY[experiment]} 24-48h best-minus-worst gap per "
            "outcome after averaging 3 seeds and 2 architectures; left marker is the "
            "worst level and right marker is the best level."
        ),
        input_hashes=input_hashes,
    )


def write_figure(
    *,
    output_root: Path,
    entries: list[dict[str, object]],
    figure: plt.Figure,
    stem: str,
    source: pd.DataFrame,
    description: str,
    input_hashes: dict[str, str],
) -> None:
    section_dir = output_root / TRACE_SECTION
    sources_dir = output_root / "sources"
    metadata_dir = output_root / "metadata"
    for directory in (section_dir, sources_dir, metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)
    source_path = sources_dir / f"{stem}.csv"
    metadata_path = metadata_dir / f"{stem}.json"
    pdf_path = section_dir / f"{stem}.pdf"
    png_path = section_dir / f"{stem}.png"
    source.to_csv(source_path, index=False)
    figure.savefig(pdf_path, bbox_inches="tight", pad_inches=0.10)
    figure.savefig(png_path, dpi=FIGURE_PNG_DPI, bbox_inches="tight", pad_inches=0.10)
    plt.close(figure)
    metadata_path.write_text(
        json.dumps(
            {
                "analysis": "24to48h",
                "created_at": utc_now(),
                "description": description,
                "source": str(source_path.relative_to(output_root)),
                "source_sha256": sha256(source_path),
                "input_hashes": input_hashes,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    entries.append(
        {
            "section": TRACE_SECTION,
            "stem": stem,
            "description": description,
            "pdf": str(pdf_path.relative_to(output_root)),
            "png": str(png_path.relative_to(output_root)),
            "source": str(source_path.relative_to(output_root)),
            "metadata": str(metadata_path.relative_to(output_root)),
        }
    )


def write_section_manifest(
    *,
    output_root: Path,
    entries: list[dict[str, object]],
    input_hashes: dict[str, str],
) -> None:
    section_dir = output_root / TRACE_SECTION
    section_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(entries).to_csv(section_dir / "figure_index.csv", index=False)
    (section_dir / "manifest.json").write_text(
        json.dumps(
            {
                "created_at": utc_now(),
                "generator": str(Path(__file__).resolve()),
                "analysis": "24to48h",
                "section": TRACE_SECTION,
                "figure_count": len(entries),
                "figures": entries,
                "input_hashes": input_hashes,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def generate(
    *,
    output_root: Path,
    experiments: tuple[str, ...] = EXPERIMENTS,
) -> dict[str, object]:
    if not experiments:
        raise ValueError("At least one experiment is required.")
    unknown = set(experiments) - set(EXPERIMENTS)
    if unknown:
        raise ValueError(f"Unsupported experiments: {sorted(unknown)}")
    spider = load_spider_renderer()
    alternatives = load_alternatives_renderer()
    metrics_path = output_root / "00_data/metrics_long.parquet"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing provisional metrics: {metrics_path}")
    metrics = pd.read_parquet(metrics_path)
    script_path = Path(__file__).resolve()
    input_hashes = {
        str(metrics_path): sha256(metrics_path),
        str(script_path): sha256(script_path),
        str(REPO_ROOT / "paper/scripts/generate_ml4h_spider_charts.py"): sha256(
            REPO_ROOT / "paper/scripts/generate_ml4h_spider_charts.py"
        ),
    }
    entries: list[dict[str, object]] = []
    for experiment in experiments:
        source = trace_source(
            metrics, experiment, spider=spider, alternatives=alternatives
        )
        plot_outcome_factor_traces(
            experiment=experiment,
            source=source,
            output_root=output_root,
            entries=entries,
            input_hashes=input_hashes,
            alternatives=alternatives,
            spider=spider,
        )
        plot_outcome_factor_deltas(
            experiment=experiment,
            source=source,
            summary=delta_summary(source),
            output_root=output_root,
            entries=entries,
            input_hashes=input_hashes,
            alternatives=alternatives,
            spider=spider,
        )
    write_section_manifest(
        output_root=output_root,
        entries=entries,
        input_hashes=input_hashes,
    )
    return {
        "output_root": str(output_root),
        "section": TRACE_SECTION,
        "experiments": list(experiments),
        "figure_count": len(entries),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=EXPERIMENTS,
        default=list(EXPERIMENTS),
    )
    args = parser.parse_args()
    result = generate(
        output_root=args.output_root.expanduser().resolve(),
        experiments=tuple(args.experiments),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
