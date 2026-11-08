#!/usr/bin/env python3
"""Render advisor-facing alternatives for the ML4H 24–48h provisional bundle.

This renderer updates the existing absolute outcome forests to use the
spider-chart display contract, adds factor-separated absolute forests, and
creates aggregate delta and delta-heatmap alternatives for the radar plots.
It reads the already materialized provisional metrics and spider sources; it
does not rerun Stage 3 prediction scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = REPO_ROOT.parent / "Archive/ml4h-dump/08_ML4H_provisional"
FOREST_SECTION = "02_outcome_forests"
AGGREGATE_SECTION = "05_aggregate"
HEATMAP_SECTION = "06_heatmaps"
BAR_SECTION = "07_factor_bars"
TRACE_SECTION = "08_outcome_factor_traces"
EXPERIMENTS = ("exp1", "exp2", "exp3")

BACKBONE_COLORS = {"llama": "#0072B2", "qwen": "#D55E00"}
POOLED_BACKBONE = "llama_qwen_mean"
TASK_MARKERS = {"binary": "o", "regression": "s"}
TASK_LABELS = {"binary": "Binary AUROC", "regression": r"Regression Spearman $\rho$"}
TASK_XLABELS = {"binary": "AUROC", "regression": r"Spearman $\rho$"}
TASK_FLOORS = {"binary": 0.52, "regression": None}
SHAPE_LEGEND_FACE = "#111111"
# Factor-forest dodge: fill most of the ±0.5 outcome stripe so 2- and 6-level
# series stay inside the gray/white bin without stacking on the midline.
FACTOR_FOREST_OFFSET_HALF_BAND = 0.36
BAR_OUTCOME_TICK_PT = 8.0
# Forest y-labels: shorten only collision-prone names; Min./Max. applied below.
FOREST_OUTCOME_LABEL_OVERRIDES = {
    "crrt_initiation": "Continuous renal replacement",
}
BEST_OUTLINE = "#1B9E77"
WORST_OUTLINE = "#D62728"
# Distinct outcome colors for traces; factor levels use marker shape, not these.
OUTCOME_TRACE_COLORS = (
    "#5A8FB8",
    "#8B9C42",
    "#2A6B52",
    "#C4644E",
    "#D0909A",
    "#7A2B40",
    "#9D88C2",
    "#C8A520",
    "#4A8868",
    "#DD9E48",
    "#8A558A",
    "#38708C",
    "#B07255",
    "#538A3C",
    "#C44E80",
    "#6A7C8A",
    "#D4A055",
    "#3C6878",
    "#9E5878",
    "#5CA07C",
)


def forest_outcome_label(outcome: str, *, spider) -> str:
    """Forest-axis label: CRRT short form, Min./Max. for extrema, no NP shorthand."""
    label = FOREST_OUTCOME_LABEL_OVERRIDES.get(
        outcome,
        spider.FULL_OUTCOME_LABELS.get(outcome, outcome),
    )
    if label.startswith("Minimum "):
        return f"Min. {label[len('Minimum '):]}"
    if label.startswith("Maximum "):
        return f"Max. {label[len('Maximum '):]}"
    return label



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
        "ml4h_spider_renderer",
        REPO_ROOT / "paper/scripts/generate_ml4h_spider_charts.py",
    )


def load_primary_renderer():
    return load_module(
        "ml4h_primary_renderer_for_provisional_visuals",
        REPO_ROOT / "paper/scripts/generate_ml4h_figures.py",
    )


def apply_figure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
            "font.size": 8.5,
            "axes.titleweight": "normal",
            "axes.labelweight": "normal",
            "figure.titleweight": "normal",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def selected_metrics(metrics: pd.DataFrame, experiment: str) -> pd.DataFrame:
    """Keep the absolute 24–48h performance rows used by forests."""
    selected = metrics.loc[
        (metrics["experiment"] == experiment)
        & (
            (
                (metrics["task_type"] == "binary")
                & (metrics["metric"] == "roc_auc")
            )
            | (
                (metrics["task_type"] == "regression")
                & (metrics["metric"] == "spearman_rho")
            )
        )
    ].copy()
    if selected.empty:
        raise ValueError(f"{experiment}: no absolute performance rows.")
    existing_source = (
        selected["source"].fillna("")
        if "source" in selected.columns
        else pd.Series("", index=selected.index)
    )
    selected["source"] = np.where(
        selected["config_id"].astype(str).str.startswith("meds_icu_"),
        "Native MIMIC",
        np.where(
            selected["config_id"].astype(str).str.startswith("meds_clif_"),
            "CLIF harmonized",
            existing_source,
        ),
    )
    return selected


def is_baseline_config(
    experiment: str,
    config_id: str,
    *,
    primary,
) -> bool:
    if experiment == "exp1":
        fields = primary.config_parts(config_id)
        return (
            fields["quantizer"] == "deciles"
            and fields["anchoring"] == "none"
            and fields["fusion"] == "unfused"
        )
    if experiment == "exp2":
        representation, temporal = primary._exp2_representation_and_temporal(config_id)
        return representation == "discrete" and temporal == "time_tokens"
    if experiment == "exp3":
        return config_id.startswith("meds_icu_")
    raise ValueError(f"Unsupported experiment: {experiment}")


def display_config_label(
    experiment: str,
    config_id: str,
    *,
    primary,
    spider,
) -> str:
    """Use full spider-style names for the absolute full-configuration forests."""
    if experiment == "exp1":
        fields = primary.config_parts(config_id)
        try:
            granularity = next(
                label
                for quantizer, anchoring, label, _, _ in spider.EXP1_GRANULARITY
                if quantizer == fields["quantizer"] and anchoring == fields["anchoring"]
            )
        except StopIteration as error:
            raise ValueError(f"Unknown Exp1 display configuration: {config_id}") from error
        label = f"{granularity} · {str(fields['fusion']).title()}"
    elif experiment == "exp2":
        representation, temporal = primary._exp2_representation_and_temporal(config_id)
        representation_labels = {
            "discrete": "Discrete",
            "soft": "Soft discretization",
            "xval": "xVal",
            "xval_affine": "xVal-affine",
        }
        temporal_labels = {
            "time_tokens": "Time tokens",
            "event_order": "Event stream only",
            "time_rope": "Admission-relative RoPE",
        }
        label = f"{representation_labels[representation]} · {temporal_labels[temporal]}"
    elif experiment == "exp3":
        label = "Native MIMIC" if config_id.startswith("meds_icu_") else "CLIF harmonized"
    else:
        raise ValueError(f"Unsupported experiment: {experiment}")
    if is_baseline_config(experiment, config_id, primary=primary):
        return f"{label} (baseline)"
    return label


def display_config_order(
    metrics: pd.DataFrame,
    experiment: str,
    *,
    primary,
) -> tuple[str, ...]:
    """Return the provisional-only forest order, including every Exp2 encoder."""
    available = set(metrics["config_id"].astype(str).unique())
    spec = primary.EXPERIMENT_SPECS[experiment]
    if experiment != "exp2":
        return tuple(config for config in spec.config_order if config in available)

    representation_order = {
        "discrete": 0,
        "soft": 1,
        "xval": 2,
        "xval_affine": 3,
    }
    temporal_order = {"time_tokens": 0, "event_order": 1, "time_rope": 2}
    configs = [
        config
        for config in spec.config_order
        if config in available
    ]
    return tuple(
        sorted(
            configs,
            key=lambda config: (
                representation_order[
                    primary._exp2_representation_and_temporal(config)[0]
                ],
                temporal_order[
                    primary._exp2_representation_and_temporal(config)[1]
                ],
            ),
        )
    )


def display_config_style(
    experiment: str,
    config_id: str,
    *,
    primary,
) -> tuple[str, str]:
    spec = primary.EXPERIMENT_SPECS[experiment]
    return (
        primary.experiment_config_color(spec, config_id),
        primary.experiment_config_marker(spec, config_id),
    )


def outcome_order_for_source(
    source: pd.DataFrame,
    task_type: str,
    *,
    spider,
) -> tuple[str, ...]:
    return spider.outcome_order(task_type, set(source["outcome"].astype(str)))


def outcome_labels(source: pd.DataFrame, *, spider) -> pd.DataFrame:
    out = source.copy()
    out["outcome_label"] = out["outcome"].map(
        lambda outcome: spider.FULL_OUTCOME_LABELS.get(str(outcome), str(outcome))
    )
    return out


def seed_interval_source(
    metrics: pd.DataFrame,
    experiment: str,
    *,
    primary,
    spider,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Aggregate a display-filtered configuration to seed means and ranges."""
    selected = selected_metrics(metrics, experiment)
    configs = display_config_order(selected, experiment, primary=primary)
    selected = selected.loc[selected["config_id"].isin(configs)].copy()
    if selected.empty:
        raise ValueError(f"{experiment}: no display configurations remain.")
    grouped = (
        selected.groupby(
            ["backbone", "config_id", "task_type", "outcome", "metric"],
            as_index=False,
        )
        .agg(
            point=("point", "mean"),
            lo=("point", "min"),
            hi=("point", "max"),
            n_seeds=("seed", "nunique"),
        )
    )
    if not (grouped["n_seeds"] == 3).all():
        raise ValueError(f"{experiment}: each forest point must have three seeds.")
    grouped["panel"] = np.where(
        grouped["task_type"] == "binary", "binary", "regression"
    )
    grouped["config_label"] = grouped["config_id"].map(
        lambda config: display_config_label(
            experiment, str(config), primary=primary, spider=spider
        )
    )
    grouped["is_baseline"] = grouped["config_id"].map(
        lambda config: is_baseline_config(experiment, str(config), primary=primary)
    )
    styles = grouped["config_id"].map(
        lambda config: display_config_style(experiment, str(config), primary=primary)
    )
    grouped["color"] = styles.map(lambda value: value[0])
    grouped["marker"] = styles.map(lambda value: value[1])
    order_map = {config: index for index, config in enumerate(configs)}
    grouped["config_order"] = grouped["config_id"].map(order_map)
    grouped = grouped.sort_values(
        ["config_order", "backbone", "task_type", "outcome"],
        kind="stable",
    ).reset_index(drop=True)
    return outcome_labels(grouped, spider=spider), configs


def pooled_interval_source(source: pd.DataFrame) -> pd.DataFrame:
    """Pool the already display-filtered model configurations by outcome."""
    return (
        source.groupby(
            ["task_type", "outcome", "outcome_label", "metric"],
            as_index=False,
        )
        .agg(
            point=("point", "mean"),
            lo=("lo", "min"),
            hi=("hi", "max"),
            n_configurations=("config_id", "nunique"),
            n_seed_points=("n_seeds", "sum"),
        )
    )


def draw_pooled_forest(
    axes: np.ndarray,
    source: pd.DataFrame,
    *,
    spider,
) -> None:
    for axis, task_type in zip(axes, ("binary", "regression"), strict=True):
        order = outcome_order_for_source(source, task_type, spider=spider)
        panel = source.loc[source["task_type"] == task_type].set_index("outcome").reindex(order)
        y = np.arange(len(panel), dtype=float)
        axis.hlines(y, panel["lo"], panel["hi"], color="#9ECAE1", linewidth=3)
        axis.scatter(panel["point"], y, color="#08519C", s=27, zorder=3)
        axis.set_yticks(y)
        axis.set_yticklabels(panel["outcome_label"], fontsize=7)
        axis.set_title(TASK_LABELS[task_type], fontsize=10)
        axis.set_xlabel(TASK_XLABELS[task_type], fontsize=9)
        axis.grid(axis="x", alpha=0.25)


def draw_config_forest_panel(
    ax,
    source: pd.DataFrame,
    *,
    level_ids: tuple[str, ...],
    outcomes: tuple[str, ...],
    task_type: str,
    title: str,
    spider,
    panel_letter: str | None = None,
    match_spider_legibility: bool = False,
    title_pad: float | None = None,
    letter_y: float | None = None,
    color_by_family: bool = False,
    offset_span: float | None = None,
) -> None:
    """Draw one absolute forest axis with mean points and seed-range whiskers."""
    if match_spider_legibility:
        label_pt = spider.OUTCOME_KEY_PT
        title_pt = spider.COLUMN_TITLE_PT
        xlabel_pt = spider.NEURIPS_BODY_PT
        tick_pt = spider.NEURIPS_SMALL_PT
        line_width = 1.5
        marker_size = 42
        marker_edge = 0.40
        alpha_non_baseline = 0.88
        default_offset_span = 0.035
        default_title_pad = 14
        grid_alpha = 0.28
        grid_width = 0.7
    else:
        label_pt = 6.0
        title_pt = 8.6
        xlabel_pt = 7.8
        tick_pt = 6.6
        line_width = 0.9
        marker_size = 15
        marker_edge = 0.25
        alpha_non_baseline = 0.82
        default_offset_span = 0.025
        default_title_pad = 4
        grid_alpha = 0.22
        grid_width = 0.5
    title_pad = default_title_pad if title_pad is None else title_pad
    offset_span = default_offset_span if offset_span is None else offset_span
    base_y = np.arange(len(outcomes), dtype=float)[::-1]
    offsets = np.linspace(
        (len(level_ids) - 1) * offset_span,
        -(len(level_ids) - 1) * offset_span,
        len(level_ids),
    )
    point_limits: list[float] = []
    for outcome_index, outcome in enumerate(outcomes):
        rows = source.loc[
            (source["task_type"] == task_type) & (source["outcome"] == outcome)
        ]
        for level_index, level_id in enumerate(level_ids):
            row = rows.loc[rows["level_id"] == level_id]
            if row.empty:
                continue
            record = row.iloc[0]
            y = base_y[outcome_index] + offsets[level_index]
            point_limits.extend([float(record["lo"]), float(record["hi"])])
            alpha = 1.0 if bool(record["is_baseline"]) else alpha_non_baseline
            color = (
                spider.outcome_family_color(outcome)
                if color_by_family
                else record["color"]
            )
            ax.hlines(
                y,
                float(record["lo"]),
                float(record["hi"]),
                color=color,
                linewidth=line_width,
                alpha=alpha,
                zorder=3,
            )
            ax.scatter(
                float(record["point"]),
                y,
                color=color,
                marker=record["marker"],
                s=marker_size,
                edgecolor="#222222",
                linewidth=marker_edge,
                alpha=alpha,
                zorder=4,
            )
    labels = [forest_outcome_label(outcome, spider=spider) for outcome in outcomes]
    ax.set_yticks(base_y)
    ylabel_kwargs = {"fontsize": label_pt}
    if match_spider_legibility:
        ylabel_kwargs.update(
            {
                "fontfamily": "serif",
                "color": spider.TEXT_COLOR,
            }
        )
    ax.set_yticklabels(labels, **ylabel_kwargs)
    if color_by_family:
        for tick, outcome in zip(ax.get_yticklabels(), outcomes, strict=True):
            tick.set_color(spider.outcome_family_color(outcome))
    title_kwargs = {
        "fontsize": title_pt,
        "pad": title_pad,
    }
    if match_spider_legibility:
        title_kwargs.update(
            {
                "fontweight": "normal",
                "color": spider.TEXT_COLOR,
                "fontfamily": "serif",
            }
        )
    ax.set_title(title, **title_kwargs)
    if match_spider_legibility:
        ax.set_xlabel("")
        ax.tick_params(
            axis="x",
            labelsize=tick_pt,
            length=3,
            colors=spider.TEXT_COLOR,
        )
        # Keep room between y-labels and the letter parked above the frame.
        ax.tick_params(axis="y", length=0, pad=8)
    else:
        xlabel_kwargs = {"fontsize": xlabel_pt}
        ax.set_xlabel(TASK_XLABELS[task_type], **xlabel_kwargs)
        ax.tick_params(axis="x", labelsize=tick_pt, length=2)
    ax.set_ylim(base_y.min() - 0.55, base_y.max() + 0.55)
    if point_limits:
        low, high = min(point_limits), max(point_limits)
        pad = 0.04 * max(high - low, 1e-3)
        left = low - pad
        if TASK_FLOORS[task_type] is not None:
            left = max(left, float(TASK_FLOORS[task_type]))
        ax.set_xlim(left, high + pad)
    for index in range(len(base_y)):
        if index % 2:
            ax.axhspan(
                base_y[index] - 0.5,
                base_y[index] + 0.5,
                color="#F4F4F4",
                zorder=0,
            )
    if match_spider_legibility:
        ax.grid(axis="x", alpha=grid_alpha, linewidth=grid_width, color="#C8C8C8")
    else:
        ax.grid(axis="x", alpha=grid_alpha, linewidth=grid_width)
    if panel_letter is not None:
        # Sit above the axes so the letter does not cover the first outcome label.
        default_letter_y = 1.06 if match_spider_legibility else 0.985
        letter_va = "bottom" if match_spider_legibility else "top"
        ax.text(
            0.0,
            default_letter_y if letter_y is None else letter_y,
            panel_letter,
            transform=ax.transAxes,
            ha="left",
            va=letter_va,
            fontsize=spider.PANEL_LETTER_PT,
            fontweight="normal",
            color=spider.TEXT_COLOR,
            fontfamily="serif",
            clip_on=False,
            zorder=10,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.92,
                "pad": 0.25,
            },
        )


def plot_absolute_outcome_forest(
    *,
    experiment: str,
    source: pd.DataFrame,
    output_root: Path,
    entries: list[dict[str, object]],
    input_hashes: dict[str, str],
    spider,
) -> None:
    pooled = pooled_interval_source(source)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 11.2), sharex=False)
    draw_pooled_forest(axes, pooled, spider=spider)
    fig.suptitle(
        f"{spider.EXPERIMENT_DISPLAY[experiment]} — outcome-level performance, 24–48h follow-up\n"
        "Point: mean across displayed model runs; line: observed displayed model-run range.",
        y=0.99,
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    display_scope = "All in-scope configurations are displayed."
    write_figure(
        output_root=output_root,
        entries=entries,
        figure=fig,
        section=FOREST_SECTION,
        stem=f"{experiment}_outcome_forest",
        source=pooled,
        description=(
            "Outcome-level absolute performance across displayed configurations, "
            f"architectures, and seeds; {display_scope}"
        ),
        input_hashes=input_hashes,
    )


def plot_seed_interval_forests(
    *,
    experiment: str,
    source: pd.DataFrame,
    config_order: tuple[str, ...],
    output_root: Path,
    entries: list[dict[str, object]],
    input_hashes: dict[str, str],
    primary,
    spider,
) -> None:
    # Match spider typography and canvas scale so the same pt sizes stay
    # equally legible when both PDFs are viewed at page width.
    spider.style()
    task_columns = ("binary", "regression")
    fig = plt.figure(figsize=(18.5, spider.FIGSIZE_EXP12[1]))
    # No figure title: compact legend band, then Llama / Qwen rows.
    # Extra wspace leaves room for single-line outcome labels between panels.
    grid = GridSpec(
        3,
        3,
        figure=fig,
        height_ratios=[0.15, 1.0, 1.0],
        width_ratios=[0.055, 1.0, 1.0],
        left=0.028,
        right=0.995,
        top=0.99,
        bottom=0.03,
        hspace=0.30,
        wspace=0.88,
    )
    handles = []
    for config_id in config_order:
        record = source.loc[source["config_id"] == config_id].iloc[0]
        handles.append(
            Line2D(
                [0],
                [0],
                marker=record["marker"],
                color=record["color"],
                markerfacecolor=record["color"],
                markeredgecolor="#222222",
                linewidth=0,
                markersize=9.0,
                label=record["config_label"],
            )
        )
    legend_ax = fig.add_subplot(grid[0, :])
    legend_ax.set_axis_off()
    legend_ax.legend(
        handles=handles,
        title="Configuration",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=3 if len(handles) > 6 else 2,
        fontsize=spider.LEGEND_PT,
        title_fontsize=spider.NEURIPS_SMALL_PT,
        frameon=False,
        handlelength=1.5,
        columnspacing=1.1,
        labelspacing=0.30,
        borderaxespad=0.0,
        alignment="center",
        prop={"family": "serif", "weight": "normal", "size": spider.LEGEND_PT},
    )
    for row_index, backbone in enumerate(spider.BACKBONE_ORDER):
        gutter = fig.add_subplot(grid[1 + row_index, 0])
        gutter.set_axis_off()
        gutter.text(
            0.5,
            0.5,
            spider.BACKBONE_DISPLAY[backbone],
            fontsize=spider.COLUMN_TITLE_PT,
            fontweight="normal",
            color=spider.TEXT_COLOR,
            ha="center",
            va="center",
            rotation=90,
            transform=gutter.transAxes,
            fontfamily="serif",
        )
        for col_index, task_type in enumerate(task_columns):
            axis = fig.add_subplot(grid[1 + row_index, 1 + col_index])
            outcomes = outcome_order_for_source(source, task_type, spider=spider)
            subset = source.loc[source["backbone"] == backbone]
            draw_config_forest_panel(
                axis,
                subset.rename(columns={"config_id": "level_id"}),
                level_ids=config_order,
                outcomes=outcomes,
                task_type=task_type,
                title=TASK_LABELS[task_type],
                panel_letter=spider.PANEL_LETTERS[row_index * 2 + col_index],
                match_spider_legibility=True,
                spider=spider,
            )
    display_scope = "All in-scope configurations are displayed."
    write_figure(
        output_root=output_root,
        entries=entries,
        figure=fig,
        section=FOREST_SECTION,
        stem=f"{experiment}__outcome_forests",
        source=source,
        description=(
            f"{spider.EXPERIMENT_DISPLAY[experiment]} 24–48h absolute outcome forest "
            f"with architecture rows and spider-matching panel letters/fonts; {display_scope}"
        ),
        input_hashes=input_hashes,
    )
    apply_figure_style()


def filter_series_rows(rows: pd.DataFrame, filters: dict[str, str]) -> pd.DataFrame:
    subset = rows
    for column, value in filters.items():
        if column not in subset.columns:
            raise ValueError(f"Missing factor column {column!r}.")
        subset = subset.loc[subset[column] == value]
    return subset


def factor_seed_interval_source(
    metrics: pd.DataFrame,
    experiment: str,
    *,
    spider,
) -> pd.DataFrame:
    """Build absolute factor panels using the exact spider-series membership."""
    selected = selected_metrics(metrics, experiment)
    records: list[pd.DataFrame] = []
    group_seed_keys = ["backbone", "task_type", "metric", "outcome", "seed"]
    group_keys = ["backbone", "task_type", "metric", "outcome"]
    for column in spider.panel_columns(experiment):
        for item in column.series:
            subset = filter_series_rows(selected, item.filters)
            component_ids: list[str] = []
            if item.average_over:
                for factor, allowed in item.average_over:
                    subset = subset.loc[subset[factor].isin(allowed)]
                component_ids = sorted(subset["config_id"].astype(str).unique())
                per_seed = (
                    subset.groupby(group_seed_keys, as_index=False)
                    .agg(point=("point", "mean"))
                )
            else:
                component_ids = sorted(subset["config_id"].astype(str).unique())
                per_seed = subset[group_seed_keys + ["point"]].copy()
            grouped = (
                per_seed.groupby(group_keys, as_index=False)
                .agg(
                    point=("point", "mean"),
                    lo=("point", "min"),
                    hi=("point", "max"),
                    n_seeds=("seed", "nunique"),
                )
            )
            if grouped.empty:
                raise ValueError(
                    f"{experiment}/{column.key}/{item.level_id}: no factor forest rows."
                )
            if not (grouped["n_seeds"] == 3).all():
                raise ValueError(
                    f"{experiment}/{column.key}/{item.level_id}: expected three seeds."
                )
            grouped["experiment"] = experiment
            grouped["condition"] = column.key
            grouped["condition_title"] = column.title
            grouped["level_id"] = item.level_id
            grouped["level_label"] = item.label
            grouped["color"] = item.color
            grouped["marker"] = item.marker
            grouped["linestyle"] = item.linestyle
            grouped["is_baseline"] = spider.series_is_baseline(experiment, item)
            grouped["component_config_ids"] = "|".join(component_ids)
            records.append(grouped)
    return outcome_labels(pd.concat(records, ignore_index=True), spider=spider)


def factor_per_model_source(
    metrics: pd.DataFrame,
    experiment: str,
    *,
    spider,
) -> pd.DataFrame:
    """One row per backbone × seed × outcome × displayed factor level."""
    selected = selected_metrics(metrics, experiment)
    records: list[pd.DataFrame] = []
    group_seed_keys = ["backbone", "task_type", "metric", "outcome", "seed"]
    for column in spider.panel_columns(experiment):
        for item in column.series:
            subset = filter_series_rows(selected, item.filters)
            if item.average_over:
                for factor, allowed in item.average_over:
                    subset = subset.loc[subset[factor].isin(allowed)]
                per_seed = (
                    subset.groupby(group_seed_keys, as_index=False)
                    .agg(point=("point", "mean"))
                )
            else:
                per_seed = subset[list(group_seed_keys) + ["point"]].copy()
            if per_seed.empty:
                raise ValueError(
                    f"{experiment}/{column.key}/{item.level_id}: no per-model rows."
                )
            n_models = int(per_seed.groupby(["backbone", "seed"], sort=False).ngroups)
            if n_models != 6:
                raise ValueError(
                    f"{experiment}/{column.key}/{item.level_id}: "
                    f"expected 6 models, found {n_models}."
                )
            per_seed = per_seed.copy()
            per_seed["experiment"] = experiment
            per_seed["condition"] = column.key
            per_seed["level_id"] = item.level_id
            records.append(per_seed)
    return pd.concat(records, ignore_index=True)


def average_factor_source_over_architectures(
    source: pd.DataFrame, *, spider
) -> pd.DataFrame:
    """Equal-weight mean of Llama and Qwen seed-means for one factor forest."""
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
        lo=("lo", "min"),
        hi=("hi", "max"),
        n_seeds=("n_seeds", "first"),
        n_architectures=("backbone", "nunique"),
    )
    if grouped.empty:
        raise ValueError("No rows left after averaging architectures.")
    if not (grouped["n_architectures"] == 2).all():
        raise ValueError("Every pooled factor-forest cell must include both architectures.")
    if not (grouped["n_seeds"] == 3).all():
        raise ValueError("Every pooled factor-forest cell must keep the three-seed mean.")
    grouped["backbone"] = POOLED_BACKBONE
    return grouped


def factor_forest_legend_handles(experiment: str, column, *, spider) -> list[Line2D]:
    handles: list[Line2D] = []
    for item in column.series:
        label = item.label
        if spider.series_is_baseline(experiment, item):
            label = f"{label} (baseline)"
        handles.append(
            Line2D(
                [0],
                [0],
                marker=item.marker,
                color=SHAPE_LEGEND_FACE,
                markerfacecolor=SHAPE_LEGEND_FACE,
                markeredgecolor="#222222",
                linewidth=0,
                markersize=9.0,
                label=label,
            )
        )
    return handles


def place_factor_forest_family_key(ax, *, task_type: str, spider) -> None:
    """Family color chips under one metric column, matching the family traces."""
    ax.set_axis_off()
    handles = [
        Line2D(
            [0],
            [0],
            color=spider.FAMILY_COLORS[family],
            linewidth=6.0,
            solid_capstyle="butt",
            label=family,
        )
        for family in spider.FAMILY_ORDER[task_type]
    ]
    legend = ax.legend(
        handles=handles,
        loc="center",
        ncol=2,
        frameon=False,
        fontsize=spider.NEURIPS_SMALL_PT,
        handlelength=1.4,
        handletextpad=0.45,
        columnspacing=1.4,
        labelspacing=0.28,
        borderaxespad=0.0,
        alignment="center",
        prop={"family": "serif", "weight": "normal", "size": spider.NEURIPS_SMALL_PT},
    )
    ax.add_artist(legend)


def plot_factor_forests(
    *,
    experiment: str,
    source: pd.DataFrame,
    output_root: Path,
    entries: list[dict[str, object]],
    input_hashes: dict[str, str],
    spider,
) -> None:
    """Architecture-averaged factor forests in the outcome-forest layout."""
    spider.style()
    pooled = average_factor_source_over_architectures(source, spider=spider)
    columns = spider.panel_columns(experiment)
    n_rows = len(columns)
    task_columns = ("binary", "regression")
    height_ratios: list[float] = []
    legend_rows: list[int] = []
    panel_rows: list[int] = []
    for column in columns:
        legend_rows.append(len(height_ratios))
        height_ratios.append(0.155 if len(column.series) >= 6 else 0.125)
        panel_rows.append(len(height_ratios))
        height_ratios.append(1.22)
    height_ratios.append(0.14)
    figsize = (18.5, spider.FIGSIZE_EXP12[1] * n_rows / 2 * 1.14 + 0.8)
    fig = plt.figure(figsize=figsize)
    grid = GridSpec(
        len(height_ratios),
        3,
        figure=fig,
        height_ratios=height_ratios,
        width_ratios=[0.055, 1.0, 1.0],
        left=0.028,
        right=0.995,
        top=0.99,
        bottom=0.03,
        hspace=0.10,
        wspace=0.88,
    )
    for row_index, column in enumerate(columns):
        legend_ax = fig.add_subplot(grid[legend_rows[row_index], 1:])
        legend_ax.set_axis_off()
        handles = factor_forest_legend_handles(experiment, column, spider=spider)
        n_handles = len(handles)
        if experiment == "exp2" or n_handles <= 3:
            ncol = n_handles
        else:
            ncol = 3 if n_handles >= 6 else 2
        legend_ax.legend(
            handles=handles,
            title=column.title,
            loc="upper center",
            ncol=ncol,
            fontsize=spider.LEGEND_PT,
            title_fontsize=spider.NEURIPS_SMALL_PT,
            frameon=False,
            handlelength=1.5,
            columnspacing=2.0 if experiment == "exp2" else 1.2,
            handletextpad=0.50,
            labelspacing=0.28,
            borderaxespad=0.0,
            borderpad=0.10,
            alignment="center",
            prop={"family": "serif", "weight": "normal", "size": spider.LEGEND_PT},
        )
        gutter = fig.add_subplot(grid[panel_rows[row_index], 0])
        gutter.set_axis_off()
        gutter.text(
            0.5,
            0.5,
            column.title,
            fontsize=spider.COLUMN_TITLE_PT,
            fontweight="normal",
            color=spider.TEXT_COLOR,
            ha="center",
            va="center",
            rotation=90,
            transform=gutter.transAxes,
            fontfamily="serif",
        )
        level_ids = tuple(item.level_id for item in column.series)
        n_levels = max(len(level_ids), 1)
        offset_span = FACTOR_FOREST_OFFSET_HALF_BAND / max(n_levels - 1, 1)
        for col_index, task_type in enumerate(task_columns):
            axis = fig.add_subplot(grid[panel_rows[row_index], 1 + col_index])
            outcomes = outcome_order_for_source(pooled, task_type, spider=spider)
            draw_config_forest_panel(
                axis,
                pooled.loc[pooled["condition"] == column.key],
                level_ids=level_ids,
                outcomes=outcomes,
                task_type=task_type,
                title=TASK_LABELS[task_type],
                panel_letter=spider.PANEL_LETTERS[row_index * 2 + col_index],
                match_spider_legibility=True,
                title_pad=6,
                letter_y=1.01,
                color_by_family=True,
                offset_span=offset_span,
                spider=spider,
            )
    for col_index, task_type in enumerate(task_columns):
        place_factor_forest_family_key(
            fig.add_subplot(grid[-1, 1 + col_index]),
            task_type=task_type,
            spider=spider,
        )
    display_scope = "Uses every spider-chart factor level."
    write_figure(
        output_root=output_root,
        entries=entries,
        figure=fig,
        section=FOREST_SECTION,
        stem=f"{experiment}_factor_forests",
        source=pooled,
        description=(
            f"{spider.EXPERIMENT_DISPLAY[experiment]} 24–48h factor-separated forests "
            "after averaging Llama and Qwen seed-means; condition rows, spider-matching "
            "panel letters/fonts, and outcome colors by clinical family; "
            f"{display_scope}"
        ),
        input_hashes=input_hashes,
    )
    apply_figure_style()


def factor_bar_panel_shape(experiment: str, *, spider) -> tuple[int, int]:
    """Return the data-panel grid used by the spider-matched bar figures."""
    n_conditions = len(spider.panel_columns(experiment))
    return (2, n_conditions) if n_conditions > 1 else (1, 2)


def factor_bar_outcomes(
    source: pd.DataFrame,
    *,
    spider,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Return binary, regression, and combined outcome orders for one figure."""
    binary = outcome_order_for_source(source, "binary", spider=spider)
    regression = outcome_order_for_source(source, "regression", spider=spider)
    return binary, regression, (*binary, *regression)


def factor_bar_source(
    source: pd.DataFrame,
    *,
    spider,
) -> pd.DataFrame:
    """Attach shared x-axis positions to the absolute factor source."""
    binary, regression, outcomes = factor_bar_outcomes(source, spider=spider)
    outcome_positions = {outcome: index for index, outcome in enumerate(outcomes)}
    out = outcome_labels(source, spider=spider)
    out["outcome_position"] = out["outcome"].map(outcome_positions)
    out["outcome_group"] = np.where(
        out["outcome"].isin(binary),
        "binary",
        "regression",
    )
    if out["outcome_position"].isna().any():
        raise ValueError("Factor bar source contains an unordered outcome.")
    return out


def _factor_bar_handles(
    experiment: str,
    *,
    column,
    spider,
) -> list[Patch]:
    handles: list[Patch] = []
    for item in column.series:
        label = item.label
        if spider.series_is_baseline(experiment, item):
            label = f"{label} (baseline)"
        handles.append(
            Patch(
                facecolor=item.color,
                edgecolor="#222222",
                linewidth=0.5,
                label=label,
            )
        )
    return handles


def draw_factor_bar_panel(
    ax,
    source: pd.DataFrame,
    *,
    experiment: str,
    column,
    backbone: str,
    panel_letter: str,
    show_xlabels: bool,
    spider,
) -> None:
    """Render overlaid absolute-score bars for one architecture × factor panel."""
    binary, _, outcomes = factor_bar_outcomes(source, spider=spider)
    positions = np.arange(len(outcomes), dtype=float)
    panel = source.loc[
        (source["backbone"] == backbone) & (source["condition"] == column.key)
    ]
    series = column.series
    # Configurations share the same outcome center. Successively narrower bars
    # preserve each raw score while leaving visible color bands at the sides,
    # rather than summing scores as a stacked bar would.
    bar_widths = np.linspace(0.82, 0.30, len(series))
    points: list[float] = []
    for bar_width, item in zip(bar_widths, series, strict=True):
        values = (
            panel.loc[panel["level_id"] == item.level_id]
            .set_index("outcome")
            .reindex(outcomes)["point"]
        )
        if values.isna().any():
            missing = list(values.index[values.isna()])
            raise ValueError(
                f"{experiment}/{column.key}/{backbone}/{item.level_id}: "
                f"missing absolute score(s) for {missing}."
            )
        points.extend(values.astype(float).tolist())
        ax.bar(
            positions,
            values.to_numpy(dtype=float),
            width=bar_width,
            color=item.color,
            edgecolor="#222222",
            linewidth=0.22,
            zorder=3,
        )
    lower = max(0.0, min(points) - 0.045)
    upper = min(1.05, max(points) + 0.045)
    ax.set_ylim(lower, upper)
    divider = ax.axvline(
        len(binary) - 0.5,
        color=spider.RING_COLOR,
        linewidth=2.6,
        zorder=5,
    )
    ax.text(
        (len(binary) - 1) / 2.0,
        1.015,
        "Binary AUROC",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=16,
        color=spider.TEXT_COLOR,
    )
    ax.text(
        len(binary) + (len(outcomes) - len(binary) - 1) / 2.0,
        1.015,
        r"Regression Spearman $\rho$",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=16,
        color=spider.TEXT_COLOR,
    )
    ax.set_xlim(-0.65, len(outcomes) - 0.35)
    ax.set_xticks(positions)
    if show_xlabels:
        labels = [
            spider.wrap_outcome_label(
                spider.FULL_OUTCOME_LABELS.get(outcome, outcome),
                width=18,
            )
            for outcome in outcomes
        ]
        ax.set_xticklabels(
            labels,
            rotation=45,
            rotation_mode="anchor",
            ha="right",
            va="top",
            fontsize=BAR_OUTCOME_TICK_PT,
        )
    else:
        ax.set_xticklabels([])
    ax.tick_params(axis="x", length=0, pad=4)
    ax.tick_params(axis="y", labelsize=12, length=3)
    ax.set_ylabel(
        r"Absolute score (AUROC | Spearman $\rho$)",
        fontsize=14,
    )
    ax.grid(axis="y", color="#D4D4D4", linewidth=0.65, alpha=0.7, zorder=0)
    ax.text(
        0.012,
        0.985,
        panel_letter,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=spider.PANEL_LETTER_PT,
        fontweight="normal",
        color=spider.TEXT_COLOR,
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.82,
            "pad": 0.35,
        },
        zorder=6,
    )
    ax._factor_bar_divider = divider
    ax._factor_bar_outcomes = outcomes
    ax._factor_bar_show_xlabels = show_xlabels


def plot_factor_bars(
    *,
    experiment: str,
    source: pd.DataFrame,
    output_root: Path,
    entries: list[dict[str, object]],
    input_hashes: dict[str, str],
    spider,
) -> None:
    """Render spider-layout absolute grouped bar alternatives."""
    source = factor_bar_source(source, spider=spider)
    columns = spider.panel_columns(experiment)
    n_conditions = len(columns)
    data_axes = []
    if n_conditions == 1:
        column = columns[0]
        fig = plt.figure(figsize=(32.0, 16.5))
        grid = GridSpec(
            3,
            2,
            figure=fig,
            height_ratios=[0.08, 0.20, 1.0],
            left=0.035,
            right=0.99,
            top=0.975,
            bottom=0.36,
            hspace=0.34,
            wspace=0.12,
        )
        title_ax = fig.add_subplot(grid[0, :])
        title_ax.set_axis_off()
        title_ax.text(
            0.5,
            0.50,
            column.title,
            fontsize=spider.COLUMN_TITLE_PT,
            fontweight="normal",
            color=spider.TEXT_COLOR,
            ha="center",
            va="center",
            transform=title_ax.transAxes,
        )
        legend_ax = fig.add_subplot(grid[1, :])
        legend_ax.set_axis_off()
        legend_ax.legend(
            handles=_factor_bar_handles(experiment, column=column, spider=spider),
            title=column.title,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=len(column.series),
            fontsize=spider.LEGEND_PT,
            title_fontsize=spider.NEURIPS_SMALL_PT,
            frameon=False,
            handlelength=1.4,
            columnspacing=1.2,
        )
        for column_index, backbone in enumerate(spider.BACKBONE_ORDER):
            axis = fig.add_subplot(grid[2, column_index])
            draw_factor_bar_panel(
                axis,
                source,
                experiment=experiment,
                column=column,
                backbone=backbone,
                panel_letter=spider.PANEL_LETTERS[column_index],
                show_xlabels=True,
                spider=spider,
            )
            axis.text(
                0.5,
                1.15,
                spider.BACKBONE_DISPLAY[backbone],
                transform=axis.transAxes,
                fontsize=spider.COLUMN_TITLE_PT,
                fontweight="normal",
                color=spider.TEXT_COLOR,
                ha="center",
                va="bottom",
            )
            data_axes.append(axis)
    else:
        fig = plt.figure(figsize=(32.0, 24.0))
        grid = GridSpec(
            4,
            1 + n_conditions,
            figure=fig,
            height_ratios=[0.08, 0.22, 1.0, 1.0],
            width_ratios=[0.07, *[1.0] * n_conditions],
            left=0.035,
            right=0.99,
            top=0.975,
            bottom=0.34,
            hspace=0.42,
            wspace=0.10,
        )
        title_ax = fig.add_subplot(grid[0, 1:])
        title_ax.set_axis_off()
        for column_index, column in enumerate(columns):
            title_ax.text(
                (column_index + 0.5) / n_conditions,
                0.50,
                column.title,
                fontsize=spider.COLUMN_TITLE_PT,
                fontweight="normal",
                color=spider.TEXT_COLOR,
                ha="center",
                va="center",
                transform=title_ax.transAxes,
            )
            legend_ax = fig.add_subplot(grid[1, 1 + column_index])
            legend_ax.set_axis_off()
            legend_ax.legend(
                handles=_factor_bar_handles(
                    experiment,
                    column=column,
                    spider=spider,
                ),
                title=column.title,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.0),
                ncol=2 if len(column.series) > 2 else len(column.series),
                fontsize=spider.LEGEND_PT,
                title_fontsize=spider.NEURIPS_SMALL_PT,
                frameon=False,
                handlelength=1.4,
                columnspacing=1.0,
                labelspacing=0.35,
            )
        for row_index, backbone in enumerate(spider.BACKBONE_ORDER):
            gutter = fig.add_subplot(grid[2 + row_index, 0])
            gutter.set_axis_off()
            gutter.text(
                0.5,
                0.5,
                spider.BACKBONE_DISPLAY[backbone],
                fontsize=spider.COLUMN_TITLE_PT,
                fontweight="normal",
                color=spider.TEXT_COLOR,
                ha="center",
                va="center",
                rotation=90,
                transform=gutter.transAxes,
            )
            for column_index, column in enumerate(columns):
                axis = fig.add_subplot(grid[2 + row_index, 1 + column_index])
                draw_factor_bar_panel(
                    axis,
                    source,
                    experiment=experiment,
                    column=column,
                    backbone=backbone,
                    panel_letter=spider.PANEL_LETTERS[
                        row_index * n_conditions + column_index
                    ],
                    show_xlabels=row_index == len(spider.BACKBONE_ORDER) - 1,
                    spider=spider,
                )
                data_axes.append(axis)
    fig._factor_bar_data_axes = data_axes
    write_figure(
        output_root=output_root,
        entries=entries,
        figure=fig,
        section=BAR_SECTION,
        stem=f"{experiment}_factor_bars",
        source=source,
        description=(
            "Absolute overlaid outcome bars using the same architecture × condition "
            "layout and factor membership as the spider charts; binary and regression "
            "outcomes are separated by a thick vertical divider."
        ),
        input_hashes=input_hashes,
    )


def factor_trace_panel_shape(experiment: str, *, spider) -> tuple[int, int]:
    """Architecture × condition grid used by the advisor-style traces."""
    return factor_bar_panel_shape(experiment, spider=spider)


def outcome_trace_color(outcome: str, outcomes: tuple[str, ...]) -> str:
    return OUTCOME_TRACE_COLORS[outcomes.index(outcome) % len(OUTCOME_TRACE_COLORS)]


def factor_trace_extremes(
    source: pd.DataFrame,
    *,
    outcomes: tuple[str, ...],
    level_ids: tuple[str, ...],
) -> pd.DataFrame:
    """Best/worst factor levels from seed means, within one panel only."""
    records = []
    for outcome in outcomes:
        rows = source.loc[source["outcome"] == outcome]
        if rows.empty:
            continue
        by_level = (
            rows.set_index("level_id")
            .reindex(list(level_ids))
            .dropna(subset=["point"])
        )
        if by_level.empty:
            continue
        best_id = str(by_level["point"].idxmax())
        worst_id = str(by_level["point"].idxmin())
        best = float(by_level.loc[best_id, "point"])
        worst = float(by_level.loc[worst_id, "point"])
        records.append(
            {
                "outcome": outcome,
                "best_level_id": best_id,
                "worst_level_id": worst_id,
                "best_point": best,
                "worst_point": worst,
                "spread": best - worst,
            }
        )
    return pd.DataFrame(records)


def _trace_level_labels(experiment: str, column, *, spider) -> list[str]:
    labels = []
    for item in column.series:
        label = item.label
        if spider.series_is_baseline(experiment, item):
            label = f"{label} (baseline)"
        labels.append(label)
    return labels


def _trace_shape_handles(experiment: str, column, *, spider) -> list[Line2D]:
    handles = []
    for item, label in zip(
        column.series,
        _trace_level_labels(experiment, column, spider=spider),
    ):
        handles.append(
            Line2D(
                [0],
                [0],
                marker=item.marker,
                color="#555555",
                markerfacecolor="#888888",
                markeredgecolor="#222222",
                linewidth=0,
                markersize=8.0,
                label=label,
            )
        )
    return handles


def _trace_outline_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#BBBBBB",
            markeredgecolor=BEST_OUTLINE,
            markeredgewidth=1.8,
            linewidth=0,
            markersize=8.5,
            label="Best factor level",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#BBBBBB",
            markeredgecolor=WORST_OUTLINE,
            markeredgewidth=1.8,
            linewidth=0,
            markersize=8.5,
            label="Worst factor level",
        ),
    ]


def _trace_outcome_handles(outcomes: tuple[str, ...], *, spider) -> list[Line2D]:
    handles = []
    for outcome in outcomes:
        color = outcome_trace_color(outcome, outcomes)
        handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                marker="o",
                markerfacecolor=color,
                markeredgecolor="none",
                linewidth=1.6,
                markersize=6.0,
                label=forest_outcome_label(outcome, spider=spider),
            )
        )
    return handles


def draw_factor_trace_main(
    ax,
    source: pd.DataFrame,
    *,
    experiment: str,
    column,
    outcomes: tuple[str, ...],
    extremes: pd.DataFrame,
    task_type: str,
    show_xlabels: bool,
    spider,
) -> None:
    """Absolute outcome traces vs factor levels; green/red mark within-panel extrema."""
    level_ids = tuple(item.level_id for item in column.series)
    markers = {item.level_id: item.marker for item in column.series}
    xs = np.arange(len(level_ids), dtype=float)
    extreme_map = extremes.set_index("outcome") if not extremes.empty else pd.DataFrame()
    for outcome in outcomes:
        rows = (
            source.loc[source["outcome"] == outcome]
            .set_index("level_id")
            .reindex(list(level_ids))
        )
        values = rows["point"].to_numpy(dtype=float)
        color = outcome_trace_color(outcome, outcomes)
        ax.plot(xs, values, color=color, linewidth=1.35, alpha=0.55, zorder=2)
        best_id = (
            str(extreme_map.loc[outcome, "best_level_id"])
            if outcome in extreme_map.index
            else ""
        )
        worst_id = (
            str(extreme_map.loc[outcome, "worst_level_id"])
            if outcome in extreme_map.index
            else ""
        )
        for x, level_id, value in zip(xs, level_ids, values):
            if not np.isfinite(value):
                continue
            if level_id == best_id:
                edge, width = BEST_OUTLINE, 1.8
            elif level_id == worst_id:
                edge, width = WORST_OUTLINE, 1.8
            else:
                edge, width = "#333333", 0.35
            ax.scatter(
                [x],
                [value],
                marker=markers[level_id],
                s=42,
                facecolor=color,
                edgecolor=edge,
                linewidth=width,
                alpha=0.92,
                zorder=4,
            )
    ax.set_xlim(-0.45, len(level_ids) - 0.55)
    finite = source["point"].to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size:
        low, high = float(finite.min()), float(finite.max())
        pad = 0.06 * max(high - low, 1e-3)
        left = low - pad
        if TASK_FLOORS[task_type] is not None:
            left = max(left, float(TASK_FLOORS[task_type]) - 0.01)
        ax.set_ylim(left, high + pad)
    ax.set_xticks(xs)
    if show_xlabels:
        ax.set_xticklabels(
            _trace_level_labels(experiment, column, spider=spider),
            rotation=30,
            rotation_mode="anchor",
            ha="right",
            va="top",
            fontsize=spider.NEURIPS_SMALL_PT - 4,
        )
    else:
        ax.set_xticklabels([])
    ax.tick_params(axis="x", length=3, pad=3)
    ax.tick_params(axis="y", labelsize=spider.NEURIPS_SMALL_PT - 2, length=3)
    ax.set_ylabel(TASK_XLABELS[task_type], fontsize=spider.NEURIPS_SMALL_PT)
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.7, alpha=0.75, zorder=0)
    ax.set_axisbelow(True)


def draw_factor_trace_summary(
    ax,
    *,
    outcomes: tuple[str, ...],
    extremes: pd.DataFrame,
    show_xlabels: bool,
    spider,
) -> None:
    """Full-name Δ strip: max−min across the same panel's factor levels."""
    if extremes.empty:
        ax.set_axis_off()
        return
    ranked = extremes.sort_values("spread", ascending=False).reset_index(drop=True)
    n_rows = len(ranked)
    y = np.arange(n_rows, dtype=float)[::-1]
    max_spread = max(float(ranked["spread"].max()), 1e-4)
    for row_y, record in zip(y, ranked.itertuples(index=False)):
        color = outcome_trace_color(str(record.outcome), outcomes)
        ax.hlines(row_y, 0.0, float(record.spread), color=color, linewidth=1.6, alpha=0.7)
        ax.scatter(
            [0.0],
            [row_y],
            marker="o",
            s=28,
            facecolor=color,
            edgecolor=WORST_OUTLINE,
            linewidth=1.6,
            zorder=4,
        )
        ax.scatter(
            [float(record.spread)],
            [row_y],
            marker="o",
            s=28,
            facecolor=color,
            edgecolor=BEST_OUTLINE,
            linewidth=1.6,
            zorder=4,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(
        [
            forest_outcome_label(str(outcome), spider=spider)
            for outcome in ranked["outcome"]
        ],
        fontsize=9.0,
        fontfamily="serif",
    )
    ax.set_xlim(-0.04 * max_spread, 1.08 * max_spread)
    ax.set_ylim(-0.7, n_rows - 0.3)
    ax.tick_params(axis="y", length=0, pad=3)
    if show_xlabels:
        ax.set_xlabel(r"$\Delta$ (max − min)", fontsize=spider.NEURIPS_SMALL_PT - 2)
    else:
        ax.set_xlabel("")
        ax.set_xticklabels([])
    ax.grid(axis="x", color="#D8D8D8", linewidth=0.65, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)


def plot_factor_traces(
    *,
    experiment: str,
    source: pd.DataFrame,
    output_root: Path,
    entries: list[dict[str, object]],
    input_hashes: dict[str, str],
    spider,
) -> None:
    """Advisor-style absolute traces plus within-panel Δ summaries."""
    spider.style()
    source = outcome_labels(source, spider=spider)
    columns = spider.panel_columns(experiment)
    n_conditions = len(columns)
    binary, regression, outcomes = factor_bar_outcomes(source, spider=spider)
    task_columns = (("binary", binary), ("regression", regression))
    if n_conditions == 1:
        column = columns[0]
        fig = plt.figure(figsize=(28.0, 20.0))
        grid = GridSpec(
            5,
            4,
            figure=fig,
            height_ratios=[0.10, 0.16, 0.18, 1.0, 1.0],
            width_ratios=[1.15, 0.72, 1.15, 0.72],
            left=0.07,
            right=0.995,
            top=0.98,
            bottom=0.05,
            hspace=0.22,
            wspace=0.38,
        )
        title_ax = fig.add_subplot(grid[0, :])
        title_ax.set_axis_off()
        title_ax.text(
            0.5,
            0.50,
            column.title,
            fontsize=spider.COLUMN_TITLE_PT,
            fontweight="normal",
            color=spider.TEXT_COLOR,
            ha="center",
            va="center",
            transform=title_ax.transAxes,
            fontfamily="serif",
        )
        shape_ax = fig.add_subplot(grid[1, :])
        shape_ax.set_axis_off()
        shape_ax.legend(
            handles=_trace_shape_handles(experiment, column, spider=spider)
            + _trace_outline_handles(),
            loc="center",
            bbox_to_anchor=(0.5, 0.5),
            ncol=4,
            fontsize=spider.LEGEND_PT - 2,
            frameon=False,
            alignment="center",
            prop={"family": "serif", "size": spider.LEGEND_PT - 2},
        )
        color_ax = fig.add_subplot(grid[2, :])
        color_ax.set_axis_off()
        color_ax.legend(
            handles=_trace_outcome_handles(outcomes, spider=spider),
            title="Outcomes",
            loc="center",
            bbox_to_anchor=(0.5, 0.5),
            ncol=4,
            fontsize=11,
            title_fontsize=spider.NEURIPS_SMALL_PT - 4,
            frameon=False,
            alignment="center",
            prop={"family": "serif", "size": 11},
        )
        for row_index, backbone in enumerate(spider.BACKBONE_ORDER):
            for task_index, (task_type, task_outcomes) in enumerate(task_columns):
                subset = source.loc[
                    (source["backbone"] == backbone)
                    & (source["condition"] == column.key)
                    & (source["task_type"] == task_type)
                ]
                extremes = factor_trace_extremes(
                    subset,
                    outcomes=task_outcomes,
                    level_ids=tuple(item.level_id for item in column.series),
                )
                main = fig.add_subplot(grid[3 + row_index, task_index * 2])
                draw_factor_trace_main(
                    main,
                    subset,
                    experiment=experiment,
                    column=column,
                    outcomes=task_outcomes,
                    extremes=extremes,
                    task_type=task_type,
                    show_xlabels=True,
                    spider=spider,
                )
                main.set_title(
                    f"{spider.BACKBONE_DISPLAY[backbone]} · {TASK_LABELS[task_type]}",
                    fontsize=spider.COLUMN_TITLE_PT - 4,
                    pad=8,
                    fontfamily="serif",
                )
                main.text(
                    0.0,
                    1.06,
                    spider.PANEL_LETTERS[row_index * 2 + task_index],
                    transform=main.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=spider.PANEL_LETTER_PT,
                    color=spider.TEXT_COLOR,
                    fontfamily="serif",
                    clip_on=False,
                )
                summary = fig.add_subplot(grid[3 + row_index, task_index * 2 + 1])
                draw_factor_trace_summary(
                    summary,
                    outcomes=task_outcomes,
                    extremes=extremes,
                    show_xlabels=True,
                    spider=spider,
                )
    else:
        fig = plt.figure(figsize=(34.0, 24.0))
        grid = GridSpec(
            5,
            1 + 2 * n_conditions,
            figure=fig,
            height_ratios=[0.08, 0.16, 0.22, 1.0, 1.0],
            width_ratios=[0.045, *([1.05, 0.78] * n_conditions)],
            left=0.03,
            right=0.995,
            top=0.985,
            bottom=0.045,
            hspace=0.26,
            wspace=0.42,
        )
        title_ax = fig.add_subplot(grid[0, 1:])
        title_ax.set_axis_off()
        for column_index, column in enumerate(columns):
            title_ax.text(
                (column_index + 0.5) / n_conditions,
                0.50,
                column.title,
                fontsize=spider.COLUMN_TITLE_PT,
                fontweight="normal",
                color=spider.TEXT_COLOR,
                ha="center",
                va="center",
                transform=title_ax.transAxes,
                fontfamily="serif",
            )
            shape_ax = fig.add_subplot(
                grid[1, 1 + column_index * 2 : 3 + column_index * 2]
            )
            shape_ax.set_axis_off()
            shape_ax.legend(
                handles=_trace_shape_handles(experiment, column, spider=spider)
                + _trace_outline_handles(),
                loc="center",
                bbox_to_anchor=(0.5, 0.5),
                ncol=3 if len(column.series) > 2 else 2,
                fontsize=11,
                frameon=False,
                alignment="center",
                prop={"family": "serif", "size": 11},
            )
        color_ax = fig.add_subplot(grid[2, 1:])
        color_ax.set_axis_off()
        color_ax.legend(
            handles=_trace_outcome_handles(outcomes, spider=spider),
            title="Outcomes",
            loc="center",
            bbox_to_anchor=(0.5, 0.5),
            ncol=5,
            fontsize=10.5,
            title_fontsize=12,
            frameon=False,
            alignment="center",
            prop={"family": "serif", "size": 10.5},
        )
        for row_index, backbone in enumerate(spider.BACKBONE_ORDER):
            gutter = fig.add_subplot(grid[3 + row_index, 0])
            gutter.set_axis_off()
            gutter.text(
                0.5,
                0.5,
                spider.BACKBONE_DISPLAY[backbone],
                fontsize=spider.COLUMN_TITLE_PT,
                fontweight="normal",
                color=spider.TEXT_COLOR,
                ha="center",
                va="center",
                rotation=90,
                transform=gutter.transAxes,
                fontfamily="serif",
            )
            for column_index, column in enumerate(columns):
                subset = source.loc[
                    (source["backbone"] == backbone)
                    & (source["condition"] == column.key)
                ]
                inner = grid[3 + row_index, 1 + column_index * 2].subgridspec(
                    2, 1, hspace=0.28
                )
                delta_inner = grid[3 + row_index, 2 + column_index * 2].subgridspec(
                    2, 1, hspace=0.28
                )
                letter = spider.PANEL_LETTERS[row_index * n_conditions + column_index]
                for task_index, (task_type, task_outcomes) in enumerate(task_columns):
                    task_subset = subset.loc[subset["task_type"] == task_type]
                    extremes = factor_trace_extremes(
                        task_subset,
                        outcomes=task_outcomes,
                        level_ids=tuple(item.level_id for item in column.series),
                    )
                    main = fig.add_subplot(inner[task_index, 0])
                    draw_factor_trace_main(
                        main,
                        task_subset,
                        experiment=experiment,
                        column=column,
                        outcomes=task_outcomes,
                        extremes=extremes,
                        task_type=task_type,
                        show_xlabels=task_index == 1,
                        spider=spider,
                    )
                    main.set_title(
                        TASK_LABELS[task_type],
                        fontsize=14,
                        pad=4,
                        fontfamily="serif",
                    )
                    if task_index == 0:
                        main.text(
                            0.0,
                            1.12,
                            letter,
                            transform=main.transAxes,
                            ha="left",
                            va="bottom",
                            fontsize=spider.PANEL_LETTER_PT,
                            color=spider.TEXT_COLOR,
                            fontfamily="serif",
                            clip_on=False,
                        )
                    summary = fig.add_subplot(delta_inner[task_index, 0])
                    draw_factor_trace_summary(
                        summary,
                        outcomes=task_outcomes,
                        extremes=extremes,
                        show_xlabels=task_index == 1,
                        spider=spider,
                    )
    display_scope = "Uses every spider-chart factor level."
    write_figure(
        output_root=output_root,
        entries=entries,
        figure=fig,
        section=TRACE_SECTION,
        stem=f"{experiment}_outcome_factor_traces",
        source=source,
        description=(
            "Absolute outcome traces by factor level with green/red best/worst "
            "outlines and a max−min summary strip; seed means only; "
            f"{display_scope}"
        ),
        input_hashes=input_hashes,
    )
    apply_figure_style()


def aggregate_delta_source(source: pd.DataFrame, *, spider) -> pd.DataFrame:
    """Summarize spider-source raw deltas without mixing metric families."""
    grouping = [
        "experiment",
        "condition",
        "level_id",
        "level_label",
        "is_baseline",
        "backbone",
        "task_type",
        "metric",
    ]
    aggregate = (
        source.groupby(grouping, as_index=False)
        .agg(
            mean_delta=("delta", "mean"),
            n_outcomes=("outcome", "nunique"),
            n_wins=("delta", lambda values: int((values > 0.0).sum())),
        )
    )
    aggregate["win_rate"] = aggregate["n_wins"] / aggregate["n_outcomes"]
    aggregate.loc[aggregate["is_baseline"], "win_rate"] = np.nan
    title_map = {
        column.key: column.title
        for column in spider.panel_columns(str(aggregate["experiment"].iloc[0]))
    }
    aggregate["condition_title"] = aggregate["condition"].map(title_map)
    return aggregate


def aggregate_xlim(values: pd.Series) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    max_abs = max(float(np.abs(finite).max()) if not finite.empty else 0.0, 0.005)
    return (-1.18 * max_abs, 1.18 * max_abs)


def plot_aggregate(
    *,
    experiment: str,
    source: pd.DataFrame,
    output_root: Path,
    entries: list[dict[str, object]],
    input_hashes: dict[str, str],
    spider,
) -> None:
    aggregate = aggregate_delta_source(source, spider=spider)
    columns = spider.panel_columns(experiment)
    fig, axes = plt.subplots(
        2,
        len(columns),
        figsize=(6.1 * len(columns), 8.4),
        squeeze=False,
    )
    handles = [
        Line2D(
            [0],
            [0],
            marker=TASK_MARKERS[task_type],
            color=BACKBONE_COLORS[backbone],
            markerfacecolor=BACKBONE_COLORS[backbone],
            markeredgecolor="#222222",
            linewidth=0,
            markersize=5,
            label=f"{spider.BACKBONE_DISPLAY[backbone]} · {TASK_LABELS[task_type]}",
        )
        for backbone in spider.BACKBONE_ORDER
        for task_type in ("binary", "regression")
    ]
    for column_index, column in enumerate(columns):
        level_ids = tuple(item.level_id for item in column.series)
        labels = [
            f"{item.label}{' (baseline)' if spider.series_is_baseline(experiment, item) else ''}"
            for item in column.series
        ]
        y = np.arange(len(level_ids), dtype=float)[::-1]
        condition = aggregate.loc[aggregate["condition"] == column.key]
        for row_index, (measure, xlabel) in enumerate(
            (
                ("mean_delta", "Mean Δ vs outcome-specific baseline"),
                ("win_rate", "Share of outcomes with Δ > 0"),
            )
        ):
            axis = axes[row_index, column_index]
            for backbone_index, backbone in enumerate(spider.BACKBONE_ORDER):
                for task_index, task_type in enumerate(("binary", "regression")):
                    values = (
                        condition.loc[
                            (condition["backbone"] == backbone)
                            & (condition["task_type"] == task_type)
                        ]
                        .set_index("level_id")
                        .reindex(level_ids)
                    )
                    offset = (backbone_index * 2 + task_index - 1.5) * 0.07
                    valid = values[measure].notna()
                    axis.scatter(
                        values.loc[valid, measure],
                        y[valid.to_numpy()] + offset,
                        color=BACKBONE_COLORS[backbone],
                        marker=TASK_MARKERS[task_type],
                        s=32,
                        edgecolor="#222222",
                        linewidth=0.30,
                        zorder=3,
                    )
            axis.set_yticks(y)
            axis.set_yticklabels(labels, fontsize=8)
            axis.set_xlabel(xlabel, fontsize=8.5)
            axis.grid(axis="x", alpha=0.24)
            if measure == "mean_delta":
                axis.axvline(0.0, color="#555555", linewidth=0.8, zorder=1)
                axis.set_xlim(*aggregate_xlim(condition[measure]))
                axis.set_title(column.title, fontsize=10)
            else:
                axis.axvline(0.5, color="#555555", linewidth=0.8, zorder=1)
                axis.set_xlim(0.0, 1.0)
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=2,
        fontsize=7.1,
        frameon=False,
    )
    fig.suptitle(
        f"{spider.EXPERIMENT_DISPLAY[experiment]} — aggregate outcome effects, 24–48h follow-up",
        y=0.998,
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.89])
    write_figure(
        output_root=output_root,
        entries=entries,
        figure=fig,
        section=AGGREGATE_SECTION,
        stem=f"{experiment}_aggregate",
        source=aggregate,
        description=(
            "Mean raw baseline delta and outcome win rate by factor level, "
            "architecture, and task family."
        ),
        input_hashes=input_hashes,
    )


def plot_delta_heatmap(
    *,
    experiment: str,
    source: pd.DataFrame,
    output_root: Path,
    entries: list[dict[str, object]],
    input_hashes: dict[str, str],
    spider,
) -> None:
    columns = spider.panel_columns(experiment)
    max_abs = max(float(source["delta"].abs().max()), 0.005)
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)
    fig, axes = plt.subplots(
        2,
        len(columns),
        figsize=(6.3 * len(columns), 16.5),
        squeeze=False,
    )
    image = None
    for row_index, backbone in enumerate(spider.BACKBONE_ORDER):
        for column_index, column in enumerate(columns):
            axis = axes[row_index, column_index]
            panel_source = source.loc[
                (source["backbone"] == backbone) & (source["condition"] == column.key)
            ]
            binary = outcome_order_for_source(panel_source, "binary", spider=spider)
            regression = outcome_order_for_source(panel_source, "regression", spider=spider)
            outcomes = (*binary, *regression)
            levels = tuple(item.level_id for item in column.series)
            labels = [
                f"{item.label}{' (baseline)' if spider.series_is_baseline(experiment, item) else ''}"
                for item in column.series
            ]
            matrix = (
                panel_source.pivot(index="outcome", columns="level_id", values="delta")
                .reindex(index=outcomes, columns=levels)
            )
            image = axis.imshow(
                matrix.to_numpy(dtype=float),
                aspect="auto",
                cmap="RdBu_r",
                norm=norm,
                interpolation="nearest",
            )
            axis.set_xticks(np.arange(len(levels)))
            axis.set_xticklabels(
                [spider.wrap_outcome_label(label, width=18) for label in labels],
                fontsize=7,
                rotation=0,
            )
            axis.set_yticks(np.arange(len(outcomes)))
            if column_index == 0:
                axis.set_yticklabels(
                    [spider.FULL_OUTCOME_LABELS.get(outcome, outcome) for outcome in outcomes],
                    fontsize=6.4,
                )
            else:
                axis.set_yticklabels([])
            axis.tick_params(length=0)
            axis.axhline(len(binary) - 0.5, color="#202020", linewidth=0.8)
            axis.set_title(
                f"{spider.BACKBONE_DISPLAY[backbone]} · {column.title}",
                fontsize=9,
                pad=8,
            )
            if column_index == 0:
                axis.text(
                    0.02,
                    0.98,
                    "Binary AUROC",
                    transform=axis.transAxes,
                    ha="left",
                    va="top",
                    fontsize=6.6,
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.82,
                        "pad": 0.8,
                    },
                )
                axis.text(
                    0.02,
                    1.0 - len(binary) / max(len(outcomes), 1) - 0.02,
                    r"Regression Spearman $\rho$",
                    transform=axis.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=6.6,
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.82,
                        "pad": 0.8,
                    },
                )
    assert image is not None
    fig.subplots_adjust(
        left=0.20 if len(columns) > 1 else 0.17,
        right=0.88,
        bottom=0.10,
        top=0.94,
        wspace=0.04,
        hspace=0.10,
    )
    colorbar = fig.colorbar(image, cax=fig.add_axes((0.905, 0.26, 0.018, 0.46)))
    colorbar.set_label("Δ vs outcome-specific baseline", fontsize=9)
    colorbar.ax.tick_params(labelsize=7)
    fig.suptitle(
        f"{spider.EXPERIMENT_DISPLAY[experiment]} — outcome-level baseline Δ heatmaps, "
        "24–48h follow-up",
        y=0.997,
        fontsize=12,
    )
    heatmap_source = outcome_labels(source, spider=spider)
    display_scope = "Uses every spider-chart factor level."
    write_figure(
        output_root=output_root,
        entries=entries,
        figure=fig,
        section=HEATMAP_SECTION,
        stem=f"{experiment}_delta_heatmap",
        source=heatmap_source,
        description=(
            "Raw outcome-specific baseline deltas as zero-centered factor heatmaps; "
            f"{display_scope}"
        ),
        input_hashes=input_hashes,
    )


def write_figure(
    *,
    output_root: Path,
    entries: list[dict[str, object]],
    figure: plt.Figure,
    section: str,
    stem: str,
    source: pd.DataFrame,
    description: str,
    input_hashes: dict[str, str],
) -> None:
    section_dir = output_root / section
    sources_dir = output_root / "sources"
    metadata_dir = output_root / "metadata"
    section_dir.mkdir(parents=True, exist_ok=True)
    sources_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    source_path = sources_dir / f"{stem}.csv"
    metadata_path = metadata_dir / f"{stem}.json"
    pdf_path = section_dir / f"{stem}.pdf"
    png_path = section_dir / f"{stem}.png"
    source.to_csv(source_path, index=False)
    source_hash = sha256(source_path)
    figure.savefig(pdf_path, bbox_inches="tight", pad_inches=0.08)
    figure.savefig(png_path, dpi=220, bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)
    metadata_path.write_text(
        json.dumps(
            {
                "analysis": "24to48h",
                "created_at": utc_now(),
                "description": description,
                "source": str(source_path.relative_to(output_root)),
                "source_sha256": source_hash,
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
            "section": section,
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
    section: str,
    entries: list[dict[str, object]],
    input_hashes: dict[str, str],
) -> None:
    section_dir = output_root / section
    section_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(entries).to_csv(section_dir / "figure_index.csv", index=False)
    (section_dir / "manifest.json").write_text(
        json.dumps(
            {
                "created_at": utc_now(),
                "generator": str(Path(__file__).resolve()),
                "analysis": "24to48h",
                "section": section,
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


def merge_section_entries(
    *,
    output_root: Path,
    section: str,
    new_entries: list[dict[str, object]],
    input_hashes: dict[str, str],
) -> list[dict[str, object]]:
    """Replace matching stems in an existing section index; keep other figures."""
    index_path = output_root / section / "figure_index.csv"
    existing: list[dict[str, object]] = []
    if index_path.exists():
        existing = pd.read_csv(index_path).to_dict(orient="records")
    by_stem = {str(row["stem"]): row for row in existing}
    order = [str(row["stem"]) for row in existing]
    for entry in new_entries:
        stem = str(entry["stem"])
        by_stem[stem] = entry
        if stem not in order:
            order.append(stem)
    merged = [by_stem[stem] for stem in order]
    write_section_manifest(
        output_root=output_root,
        section=section,
        entries=merged,
        input_hashes=input_hashes,
    )
    return merged


def generate(
    *,
    output_root: Path,
    experiments: tuple[str, ...] = EXPERIMENTS,
    traces_only: bool = False,
    factor_forests_only: bool = False,
) -> dict[str, object]:
    if not experiments:
        raise ValueError("At least one experiment is required.")
    if traces_only and factor_forests_only:
        raise ValueError("Choose one of traces_only or factor_forests_only.")
    unknown = set(experiments) - set(EXPERIMENTS)
    if unknown:
        raise ValueError(f"Unsupported experiments: {sorted(unknown)}")
    apply_figure_style()
    spider = load_spider_renderer()
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
    forest_entries: list[dict[str, object]] = []
    aggregate_entries: list[dict[str, object]] = []
    heatmap_entries: list[dict[str, object]] = []
    bar_entries: list[dict[str, object]] = []
    trace_entries: list[dict[str, object]] = []
    if factor_forests_only:
        for experiment in experiments:
            plot_factor_forests(
                experiment=experiment,
                source=factor_seed_interval_source(
                    metrics, experiment, spider=spider
                ),
                output_root=output_root,
                entries=forest_entries,
                input_hashes=input_hashes,
                spider=spider,
            )
        merged = merge_section_entries(
            output_root=output_root,
            section=FOREST_SECTION,
            new_entries=forest_entries,
            input_hashes=input_hashes,
        )
        return {
            "section": FOREST_SECTION,
            "factor_forest_count": len(forest_entries),
            "figure_count": len(merged),
            "output_root": str(output_root),
        }
    primary = load_primary_renderer()
    for experiment in experiments:
        factor_source = factor_seed_interval_source(
            metrics,
            experiment,
            spider=spider,
        )
        if not traces_only:
            absolute_source, config_order = seed_interval_source(
                metrics,
                experiment,
                primary=primary,
                spider=spider,
            )
            plot_absolute_outcome_forest(
                experiment=experiment,
                source=absolute_source,
                output_root=output_root,
                entries=forest_entries,
                input_hashes=input_hashes,
                spider=spider,
            )
            plot_seed_interval_forests(
                experiment=experiment,
                source=absolute_source,
                config_order=config_order,
                output_root=output_root,
                entries=forest_entries,
                input_hashes=input_hashes,
                primary=primary,
                spider=spider,
            )
            plot_factor_forests(
                experiment=experiment,
                source=factor_source,
                output_root=output_root,
                entries=forest_entries,
                input_hashes=input_hashes,
                spider=spider,
            )
            plot_factor_bars(
                experiment=experiment,
                source=factor_source,
                output_root=output_root,
                entries=bar_entries,
                input_hashes=input_hashes,
                spider=spider,
            )
        plot_factor_traces(
            experiment=experiment,
            source=factor_source,
            output_root=output_root,
            entries=trace_entries,
            input_hashes=input_hashes,
            spider=spider,
        )
        if not traces_only:
            spider_source_path = (
                output_root / "sources" / f"{experiment}_factor_spiders.csv"
            )
            if not spider_source_path.exists():
                raise FileNotFoundError(f"Missing spider source: {spider_source_path}")
            input_hashes[str(spider_source_path)] = sha256(spider_source_path)
            spider_source = pd.read_csv(spider_source_path)
            plot_aggregate(
                experiment=experiment,
                source=spider_source,
                output_root=output_root,
                entries=aggregate_entries,
                input_hashes=input_hashes,
                spider=spider,
            )
            plot_delta_heatmap(
                experiment=experiment,
                source=spider_source,
                output_root=output_root,
                entries=heatmap_entries,
                input_hashes=input_hashes,
                spider=spider,
            )
    if not traces_only:
        write_section_manifest(
            output_root=output_root,
            section=FOREST_SECTION,
            entries=forest_entries,
            input_hashes=input_hashes,
        )
        write_section_manifest(
            output_root=output_root,
            section=AGGREGATE_SECTION,
            entries=aggregate_entries,
            input_hashes=input_hashes,
        )
        write_section_manifest(
            output_root=output_root,
            section=HEATMAP_SECTION,
            entries=heatmap_entries,
            input_hashes=input_hashes,
        )
        write_section_manifest(
            output_root=output_root,
            section=BAR_SECTION,
            entries=bar_entries,
            input_hashes=input_hashes,
        )
    write_section_manifest(
        output_root=output_root,
        section=TRACE_SECTION,
        entries=trace_entries,
        input_hashes=input_hashes,
    )
    return {
        "experiments": list(experiments),
        "absolute_forest_count": len(forest_entries),
        "aggregate_count": len(aggregate_entries),
        "heatmap_count": len(heatmap_entries),
        "factor_bar_count": len(bar_entries),
        "factor_trace_count": len(trace_entries),
        "output_root": str(output_root),
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
    parser.add_argument(
        "--traces-only",
        action="store_true",
        help="Write only 08_outcome_factor_traces; leave other sections untouched.",
    )
    parser.add_argument(
        "--factor-forests-only",
        action="store_true",
        help="Rewrite only exp*_factor_forests; leave other forest figures untouched.",
    )
    args = parser.parse_args()
    result = generate(
        output_root=args.output_root.expanduser().resolve(),
        experiments=tuple(args.experiments),
        traces_only=args.traces_only,
        factor_forests_only=args.factor_forests_only,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
