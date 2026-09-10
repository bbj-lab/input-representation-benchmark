#!/usr/bin/env python3
"""Build PDF-only ML4H figure previews without changing the manuscript folder."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METRICS = REPO_ROOT / "outputs/runs/ml4h_2026/metrics/metrics_long.parquet"
DEFAULT_PCA_SOURCE = (
    REPO_ROOT
    / "outputs/runs/ml4h_2026/metrics/pca_centile_geometry_grid_source.csv"
)
DEFAULT_BOUNDARY_RESULTS = (
    REPO_ROOT
    / "outputs/runs/ml4h_2026/metrics/clinical_boundary_probe_results.json"
)
DEFAULT_TOKEN_ROOT = (
    REPO_ROOT
    / "outputs/runs/tokenized/mimiciv-3.1_meds_70-10-20"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT.parent / "ML4H-DUMP/submission-preview"

PAGE_WIDTH_IN = 7.0
TEXT_COLOR = "#202020"
GRID_COLOR = "#D8D8D8"
RANGE_COLOR = "#555555"
OKABE_ITO = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
)
MARKERS = ("o", "s", "^", "D", "P", "v")
TASK_ORDER = ("binary", "regression")
TASK_METRIC = {"binary": "roc_auc", "regression": "spearman_rho"}
TASK_TITLE = {
    "binary": "Binary outcomes",
    "regression": "Regression outcomes",
}
TASK_AXIS = {
    "binary": "Δ AUROC",
    "regression": "Δ Spearman ρ",
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
    "min_glucose": "Biomarkers",
    "max_glucose": "Biomarkers",
    "min_bnp": "Biomarkers",
    "peak_bnp": "Biomarkers",
    "min_hemoglobin": "Hematology",
    "max_hemoglobin": "Hematology",
    "min_heart_rate": "Vitals",
    "max_heart_rate": "Vitals",
    "min_sbp": "Vitals",
    "max_sbp": "Vitals",
    "min_dbp": "Vitals",
    "max_dbp": "Vitals",
}


@dataclass(frozen=True)
class Level:
    label: str
    filters: dict[str, str]
    color: str
    marker: str


@dataclass(frozen=True)
class Condition:
    key: str
    title: str
    baseline_label: str
    baseline_filters: dict[str, str]
    levels: tuple[Level, ...]
    average_factor: str | None = None
    average_levels: tuple[str, ...] = ()


EXPERIMENT_CONDITIONS = {
    "exp1": (
        Condition(
            key="granularity",
            title="Quantization granularity",
            baseline_label="Deciles · population",
            baseline_filters={
                "quantizer": "deciles",
                "anchoring": "none",
                "fusion": "unfused",
            },
            levels=(
                Level(
                    "Ventiles · population",
                    {
                        "quantizer": "ventiles",
                        "anchoring": "none",
                        "fusion": "unfused",
                    },
                    OKABE_ITO[0],
                    MARKERS[0],
                ),
                Level(
                    "Ventiles · reference",
                    {
                        "quantizer": "ventiles",
                        "anchoring": "5-10-5",
                        "fusion": "unfused",
                    },
                    OKABE_ITO[1],
                    MARKERS[1],
                ),
                Level(
                    "Trentiles · population",
                    {
                        "quantizer": "trentiles",
                        "anchoring": "none",
                        "fusion": "unfused",
                    },
                    OKABE_ITO[2],
                    MARKERS[2],
                ),
                Level(
                    "Trentiles · reference",
                    {
                        "quantizer": "trentiles",
                        "anchoring": "10-10-10",
                        "fusion": "unfused",
                    },
                    OKABE_ITO[3],
                    MARKERS[3],
                ),
                Level(
                    "Centiles · population",
                    {
                        "quantizer": "centiles",
                        "anchoring": "none",
                        "fusion": "unfused",
                    },
                    OKABE_ITO[4],
                    MARKERS[4],
                ),
            ),
        ),
        Condition(
            key="fusion",
            title="Code-value fusion",
            baseline_label="Unfused population deciles",
            baseline_filters={
                "quantizer": "deciles",
                "anchoring": "none",
                "fusion": "unfused",
            },
            levels=(
                Level(
                    "Fused",
                    {
                        "quantizer": "deciles",
                        "anchoring": "none",
                        "fusion": "fused",
                    },
                    OKABE_ITO[0],
                    MARKERS[0],
                ),
            ),
        ),
    ),
    "exp2": (
        Condition(
            key="numeric",
            title="Numeric representation",
            baseline_label="Discrete bins",
            baseline_filters={
                "representation": "discrete",
                "temporal": "time_tokens",
            },
            levels=(
                Level(
                    "Soft discretization",
                    {"representation": "soft", "temporal": "time_tokens"},
                    OKABE_ITO[0],
                    MARKERS[0],
                ),
                Level(
                    "Multiplicative xVal",
                    {"representation": "xval", "temporal": "time_tokens"},
                    OKABE_ITO[1],
                    MARKERS[1],
                ),
                Level(
                    "Affine xVal",
                    {"representation": "xval_affine", "temporal": "time_tokens"},
                    OKABE_ITO[2],
                    MARKERS[2],
                ),
            ),
        ),
        Condition(
            key="temporal",
            title="Temporal representation",
            baseline_label="Time tokens",
            baseline_filters={"temporal": "time_tokens"},
            levels=(
                Level(
                    "Event order",
                    {"temporal": "event_order"},
                    OKABE_ITO[0],
                    MARKERS[0],
                ),
                Level(
                    "Admission-relative RoPE",
                    {"temporal": "time_rope"},
                    OKABE_ITO[1],
                    MARKERS[1],
                ),
            ),
            average_factor="representation",
            average_levels=("discrete", "soft", "xval", "xval_affine"),
        ),
    ),
    "exp3": (
        Condition(
            key="schema",
            title="Data schema",
            baseline_label="Native MIMIC",
            baseline_filters={"schema": "Native MIMIC"},
            levels=(
                Level(
                    "CLIF harmonized",
                    {"schema": "CLIF harmonized"},
                    OKABE_ITO[0],
                    MARKERS[0],
                ),
            ),
        ),
    ),
}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Nimbus Sans"],
            "font.size": 8.0,
            "axes.titlesize": 8.8,
            "axes.labelsize": 8.2,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.8,
            "legend.fontsize": 7.4,
            "axes.linewidth": 0.65,
            "lines.linewidth": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "text.color": TEXT_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
        }
    )


def filter_rows(frame: pd.DataFrame, filters: dict[str, str]) -> pd.DataFrame:
    out = frame
    for column, value in filters.items():
        out = out.loc[out[column].astype(str) == value]
    return out


def prepare_run_family_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    selected = metrics.loc[
        metrics["metric"].eq(metrics["task_type"].map(TASK_METRIC))
    ].copy()
    selected["family"] = selected["outcome"].map(OUTCOME_FAMILIES)
    if selected["family"].isna().any():
        missing = sorted(selected.loc[selected["family"].isna(), "outcome"].unique())
        raise ValueError(f"Outcomes missing family assignments: {missing}")
    selected["schema"] = np.where(
        selected["config_id"].astype(str).str.startswith("meds_icu_"),
        "Native MIMIC",
        np.where(
            selected["config_id"].astype(str).str.startswith("meds_clif_"),
            "CLIF harmonized",
            "",
        ),
    )
    keys = [
        "experiment",
        "config_id",
        "backbone",
        "seed",
        "task_type",
        "family",
        "quantizer",
        "anchoring",
        "fusion",
        "representation",
        "temporal",
        "schema",
    ]
    return (
        selected.groupby(keys, as_index=False, dropna=False)["point"]
        .mean()
        .reset_index(drop=True)
    )


def model_level_table(
    run_family: pd.DataFrame,
    *,
    experiment: str,
    filters: dict[str, str],
    average_factor: str | None,
    average_levels: tuple[str, ...],
) -> pd.DataFrame:
    subset = filter_rows(
        run_family.loc[run_family["experiment"] == experiment],
        filters,
    )
    model_keys = ["backbone", "seed", "task_type", "family"]
    if average_factor is not None:
        subset = subset.loc[subset[average_factor].isin(average_levels)]
        observed = (
            subset.groupby(model_keys)[average_factor].nunique().rename("n_levels")
        )
        if not observed.eq(len(average_levels)).all():
            raise ValueError(
                f"{experiment}: incomplete {average_factor} average for {filters}"
            )
        subset = subset.groupby(model_keys, as_index=False)["point"].mean()
    elif subset.duplicated(model_keys).any():
        configs = sorted(subset["config_id"].astype(str).unique())
        raise ValueError(f"{experiment}: filters select multiple configs: {configs}")
    return subset[model_keys + ["point"]]


def build_effect_source(metrics: pd.DataFrame, experiment: str) -> pd.DataFrame:
    run_family = prepare_run_family_metrics(metrics)
    records: list[dict[str, object]] = []
    model_keys = ["backbone", "seed", "task_type", "family"]
    for condition_index, condition in enumerate(EXPERIMENT_CONDITIONS[experiment]):
        baseline = model_level_table(
            run_family,
            experiment=experiment,
            filters=condition.baseline_filters,
            average_factor=condition.average_factor,
            average_levels=condition.average_levels,
        ).rename(columns={"point": "baseline"})
        for level_index, level in enumerate(condition.levels):
            values = model_level_table(
                run_family,
                experiment=experiment,
                filters=level.filters,
                average_factor=condition.average_factor,
                average_levels=condition.average_levels,
            ).rename(columns={"point": "level_point"})
            paired = values.merge(baseline, on=model_keys, validate="one_to_one")
            paired["delta"] = paired["level_point"] - paired["baseline"]
            for (task_type, family), group in paired.groupby(
                ["task_type", "family"], sort=False
            ):
                if len(group) != 6:
                    raise ValueError(
                        f"{experiment}/{condition.key}/{level.label}/"
                        f"{task_type}/{family}: expected 6 models, found {len(group)}"
                    )
                records.append(
                    {
                        "experiment": experiment,
                        "condition": condition.key,
                        "condition_title": condition.title,
                        "baseline_label": condition.baseline_label,
                        "condition_order": condition_index,
                        "level": level.label,
                        "level_order": level_index,
                        "color": level.color,
                        "marker": level.marker,
                        "task_type": task_type,
                        "family": family,
                        "point": float(group["delta"].mean()),
                        "low": float(group["delta"].min()),
                        "high": float(group["delta"].max()),
                        "n_models": int(len(group)),
                    }
                )
    source = pd.DataFrame.from_records(records)
    expected = sum(
        len(condition.levels) * 8 for condition in EXPERIMENT_CONDITIONS[experiment]
    )
    if len(source) != expected:
        raise ValueError(f"{experiment}: expected {expected} effects, found {len(source)}")
    return source


def _symmetric_limit(source: pd.DataFrame, task_type: str) -> float:
    values = source.loc[source["task_type"] == task_type, ["low", "high"]].to_numpy()
    maximum = float(np.nanmax(np.abs(values)))
    return max(0.005, maximum * 1.16)


def _legend_handles(condition: Condition) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker=level.marker,
            color=level.color,
            linestyle="none",
            markerfacecolor=level.color,
            markeredgecolor="white",
            markeredgewidth=0.45,
            markersize=5.2,
            label=level.label,
        )
        for level in condition.levels
    ]


def draw_effect_axis(
    ax,
    source: pd.DataFrame,
    *,
    condition: Condition,
    task_type: str,
    x_limit: float,
    panel_letter: str,
) -> None:
    families = FAMILY_ORDER[task_type]
    levels = condition.levels
    base_y = np.arange(len(families), dtype=float)
    offsets = (
        np.linspace(-0.25, 0.25, len(levels)) if len(levels) > 1 else np.array([0.0])
    )
    for offset, level in zip(offsets, levels, strict=True):
        rows = (
            source.loc[
                (source["condition"] == condition.key)
                & (source["task_type"] == task_type)
                & (source["level"] == level.label)
            ]
            .set_index("family")
            .reindex(families)
        )
        values = rows["point"].to_numpy(dtype=float)
        low = rows["low"].to_numpy(dtype=float)
        high = rows["high"].to_numpy(dtype=float)
        ax.errorbar(
            values,
            base_y + offset,
            xerr=np.vstack([values - low, high - values]),
            fmt=level.marker,
            color=level.color,
            ecolor=level.color,
            elinewidth=0.8,
            capsize=2.0,
            capthick=0.8,
            markersize=4.7,
            markeredgecolor="white",
            markeredgewidth=0.45,
            zorder=3,
        )
    ax.axvline(0.0, color="#777777", linewidth=0.75, linestyle=(0, (2.2, 2.2)))
    ax.set_xlim(-x_limit, x_limit)
    ax.set_yticks(base_y)
    ax.set_yticklabels(families)
    ax.invert_yaxis()
    ax.set_xlabel(TASK_AXIS[task_type])
    ax.set_title(TASK_TITLE[task_type], loc="left", pad=4)
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.55, alpha=0.85)
    ax.grid(axis="y", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.text(
        -0.13,
        1.035,
        panel_letter,
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def render_effect_figure(
    source: pd.DataFrame,
    *,
    experiment: str,
    output_path: Path,
) -> None:
    conditions = EXPERIMENT_CONDITIONS[experiment]
    n_rows = len(conditions)
    figure_height = 3.25 if n_rows == 1 else 5.6
    bottom_margin = 0.19 if n_rows == 1 else 0.105
    figure = plt.figure(figsize=(PAGE_WIDTH_IN, figure_height))
    height_ratios: list[float] = []
    for _ in conditions:
        height_ratios.extend([0.24, 1.0])
    grid = GridSpec(
        n_rows * 2,
        2,
        figure=figure,
        height_ratios=height_ratios,
        left=0.165,
        right=0.98,
        top=0.975,
        bottom=bottom_margin,
        hspace=0.40,
        wspace=0.48,
    )
    x_limits = {
        task_type: _symmetric_limit(source, task_type) for task_type in TASK_ORDER
    }
    panel_index = 0
    for row_index, condition in enumerate(conditions):
        legend_ax = figure.add_subplot(grid[row_index * 2, :])
        legend_ax.set_axis_off()
        legend_ax.text(
            0.0,
            0.92,
            condition.title,
            transform=legend_ax.transAxes,
            fontsize=8.8,
            fontweight="bold",
            ha="left",
            va="top",
        )
        legend_ax.text(
            0.0,
            0.26,
            f"Baseline: {condition.baseline_label}",
            transform=legend_ax.transAxes,
            fontsize=7.3,
            color="#555555",
            ha="left",
            va="center",
        )
        legend_ax.legend(
            handles=_legend_handles(condition),
            loc="center right",
            bbox_to_anchor=(1.0, 0.45),
            frameon=False,
            ncol=min(3, len(condition.levels)),
            handletextpad=0.35,
            columnspacing=0.9,
            borderaxespad=0.0,
        )
        for column_index, task_type in enumerate(TASK_ORDER):
            axis = figure.add_subplot(grid[row_index * 2 + 1, column_index])
            draw_effect_axis(
                axis,
                source,
                condition=condition,
                task_type=task_type,
                x_limit=x_limits[task_type],
                panel_letter=chr(ord("A") + panel_index),
            )
            panel_index += 1
    figure.text(
        0.5,
        0.014,
        "Point = mean across six trained models; whisker = full six-model "
        "min–max range (Llama/Qwen × seeds 42–44).",
        ha="center",
        va="bottom",
        fontsize=6.9,
        color="#4A4A4A",
    )
    figure.savefig(output_path)
    plt.close(figure)


def _selected_bins(bin_ids: np.ndarray) -> set[int]:
    targets = np.array([0, 10, 25, 50, 75, 90, 99])
    return {int(bin_ids[np.argmin(np.abs(bin_ids - target))]) for target in targets}


def render_pca(source_path: Path, output_path: Path) -> None:
    source = pd.read_csv(source_path)
    order = (
        "Shared centile vocabulary",
        "Potassium",
        "Hemoglobin",
        "Creatinine",
        "Glucose",
        "Sodium",
        "Heart rate",
    )
    figure = plt.figure(figsize=(PAGE_WIDTH_IN, 4.25))
    grid = GridSpec(
        2,
        4,
        figure=figure,
        width_ratios=(1.15, 1.0, 1.0, 1.0),
        left=0.065,
        right=0.985,
        top=0.955,
        bottom=0.19,
        hspace=0.36,
        wspace=0.34,
    )
    axes = [figure.add_subplot(grid[:, 0])]
    axes.extend(
        [
            figure.add_subplot(grid[0, 1]),
            figure.add_subplot(grid[0, 2]),
            figure.add_subplot(grid[0, 3]),
            figure.add_subplot(grid[1, 1]),
            figure.add_subplot(grid[1, 2]),
            figure.add_subplot(grid[1, 3]),
        ]
    )
    cmap = plt.get_cmap("RdYlGn_r")
    norm = Normalize(vmin=0, vmax=99)
    offsets = ((4, 3), (4, -8), (-13, 3), (4, 3), (4, -8), (-13, 3), (4, 3))
    for axis, measurement in zip(axes, order, strict=True):
        rows = source.loc[source["measurement"] == measurement].sort_values("bin_id")
        if rows.empty:
            raise ValueError(f"Missing PCA panel: {measurement}")
        bins = rows["bin_id"].to_numpy(dtype=int)
        x = rows["pc1"].to_numpy(dtype=float)
        y = rows["pc2"].to_numpy(dtype=float)
        axis.plot(x, y, color="#8A8A8A", linewidth=0.55, alpha=0.65, zorder=1)
        axis.scatter(
            x,
            y,
            c=bins,
            cmap=cmap,
            norm=norm,
            s=15 if measurement == order[0] else 12,
            edgecolors="white",
            linewidths=0.35,
            zorder=2,
        )
        chosen = _selected_bins(bins)
        for annotation_index, bin_id in enumerate(sorted(chosen)):
            match = int(np.flatnonzero(bins == bin_id)[0])
            axis.annotate(
                f"Q{bin_id}",
                (x[match], y[match]),
                textcoords="offset points",
                xytext=offsets[annotation_index % len(offsets)],
                fontsize=6.0,
                color=TEXT_COLOR,
                zorder=3,
            )
        pc1 = float(rows["pc1_var_explained"].iloc[0])
        pc2 = float(rows["pc2_var_explained"].iloc[0])
        title = (
            "Shared centile tokens"
            if measurement == order[0]
            else measurement
        )
        axis.set_title(title, fontweight="bold", pad=3)
        axis.set_xlabel(f"PC1 ({100 * pc1:.1f}%)", labelpad=1.5)
        axis.set_ylabel(f"PC2 ({100 * pc2:.1f}%)", labelpad=1.5)
        axis.tick_params(length=2.5, pad=1.5)
        axis.grid(color=GRID_COLOR, linewidth=0.45, alpha=0.7)
        if measurement != order[0]:
            axis.text(
                0.97,
                0.04,
                f"{len(rows)} bins",
                transform=axis.transAxes,
                ha="right",
                va="bottom",
                fontsize=6.3,
                color="#444444",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75},
            )
    color_axis = figure.add_axes([0.18, 0.075, 0.66, 0.025])
    for bin_id in range(100):
        color_axis.axvspan(
            bin_id,
            bin_id + 1,
            ymin=0.0,
            ymax=1.0,
            facecolor=cmap(norm(bin_id)),
            edgecolor="none",
        )
    color_axis.set_xlim(0, 100)
    color_axis.set_ylim(0, 1)
    color_axis.set_yticks([])
    color_axis.set_xticks([0, 20, 40, 60, 80, 100])
    color_axis.set_xlabel("Centile index", labelpad=2)
    color_axis.tick_params(axis="x", labelsize=7.0, length=2.5)
    figure.savefig(output_path)
    plt.close(figure)


def render_boundary_probe(results_path: Path, output_path: Path) -> None:
    results = json.loads(results_path.read_text(encoding="utf-8"))
    granularities = ("Deciles", "Ventiles", "Trentiles", "Centiles")
    measurement_order = (
        "Glucose (BG)",
        "Potassium (BG)",
        "Creatinine",
        "Glucose",
        "Potassium",
        "Troponin T",
        "Hemoglobin",
    )
    by_name: dict[tuple[str, str], float] = {}
    for granularity, entries in results.items():
        for record in entries.values():
            by_name[(granularity, str(record["name"]))] = float(record["accuracy"])
    figure, axis = plt.subplots(figsize=(PAGE_WIDTH_IN, 3.65))
    x_base = np.arange(len(measurement_order), dtype=float)
    width = 0.19
    for index, granularity in enumerate(granularities):
        values = np.array(
            [by_name[(granularity, measurement)] for measurement in measurement_order]
        )
        positions = x_base + (index - 1.5) * width
        bars = axis.bar(
            positions,
            values,
            width=width * 0.88,
            color=OKABE_ITO[index],
            edgecolor="white",
            linewidth=0.45,
            label=granularity,
            zorder=3,
        )
        for bar, value in zip(bars, values, strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.016,
                f"{value:.0%}",
                ha="center",
                va="bottom",
                fontsize=6.2,
                color=TEXT_COLOR,
            )
    axis.axhline(
        0.5,
        color="#777777",
        linewidth=0.75,
        linestyle=(0, (2.2, 2.2)),
        label="Chance",
        zorder=2,
    )
    axis.set_xticks(x_base)
    axis.set_xticklabels(measurement_order, rotation=24, ha="right")
    axis.set_ylim(0.0, 1.08)
    axis.set_ylabel("Leave-one-out accuracy")
    axis.grid(axis="y", color=GRID_COLOR, linewidth=0.55)
    axis.grid(axis="x", visible=False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=5,
        frameon=False,
        columnspacing=1.0,
        handletextpad=0.35,
    )
    figure.subplots_adjust(left=0.09, right=0.985, top=0.80, bottom=0.25)
    figure.savefig(output_path)
    plt.close(figure)


def read_token_lengths(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    chunks: list[np.ndarray] = []
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(columns=["tokens"], batch_size=8192):
        chunks.append(np.asarray(batch.column(0).value_lengths(), dtype=np.int32))
    return np.concatenate(chunks)


def ecdf_points(lengths: np.ndarray, n_points: int = 1800) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.linspace(0.0, 1.0, n_points)
    values = np.quantile(lengths, probabilities, method="nearest")
    unique, indices = np.unique(values, return_index=True)
    return unique, probabilities[indices]


def render_length_comparison(
    datasets: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    labels: tuple[str, str],
    colors: tuple[str, str],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH_IN, 3.25))
    for axis, (horizon, arrays) in zip(
        axes,
        (("Full timeline", datasets["full"]), ("First 24 hours", datasets["first24"])),
        strict=True,
    ):
        maximum = 0
        for label, color, lengths in zip(labels, colors, arrays, strict=True):
            x, y = ecdf_points(lengths)
            axis.step(x, y, where="post", color=color, linewidth=1.2, label=label)
            maximum = max(maximum, int(lengths.max()))
        for threshold, linewidth, linestyle in (
            (1024, 0.65, (0, (1.4, 2.0))),
            (2048, 0.80, (0, (3.0, 2.0))),
            (4096, 1.00, "solid"),
        ):
            axis.axvline(
                threshold,
                color="#666666",
                linewidth=linewidth,
                linestyle=linestyle,
                alpha=0.85,
                zorder=1,
            )
        axis.set_xscale("log")
        axis.set_xlim(32, maximum * 1.08)
        token_ticks = [
            value
            for value in (64, 256, 1024, 2048, 4096, 16384, 65536, 262144)
            if 32 <= value <= maximum * 1.08
        ]
        axis.set_xticks(token_ticks)
        axis.set_xticklabels([f"{value:,}" for value in token_ticks], rotation=30, ha="right")
        axis.tick_params(axis="x", which="minor", labelbottom=False)
        axis.set_ylim(0.0, 1.005)
        axis.set_title(horizon, loc="left")
        axis.set_xlabel("Tokens per admission (log scale)")
        axis.grid(color=GRID_COLOR, linewidth=0.5, alpha=0.8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        annotation = "\n".join(
            f"{label}: {(lengths > 4096).mean():.3%} > 4096"
            for label, lengths in zip(labels, arrays, strict=True)
        )
        axis.text(
            0.03,
            0.07,
            annotation,
            transform=axis.transAxes,
            fontsize=6.8,
            ha="left",
            va="bottom",
            bbox={
                "facecolor": "white",
                "edgecolor": "#CCCCCC",
                "linewidth": 0.5,
                "alpha": 0.9,
                "pad": 2.0,
            },
        )
    axes[0].set_ylabel("Cumulative fraction of admissions")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.935),
        frameon=False,
        ncol=2,
    )
    figure.subplots_adjust(left=0.10, right=0.985, top=0.78, bottom=0.25, wspace=0.30)
    figure.savefig(output_path)
    plt.close(figure)


def load_length_datasets(token_root: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    def load(version: str) -> np.ndarray:
        return read_token_lengths(
            token_root / f"{version}-tokenized/train/tokens_timelines.parquet"
        )

    time_full = load("deciles_none_unfused_time_tokens")
    time_first = load("deciles_none_unfused_time_tokens_first_24h")
    return {
        "fusion_full": (
            time_full,
            load("deciles_none_fused_time_tokens"),
        ),
        "fusion_first24": (
            time_first,
            load("deciles_none_fused_time_tokens_first_24h"),
        ),
        "temporal_full": (
            time_full,
            load("deciles_none_unfused_time_rope"),
        ),
        "temporal_first24": (
            time_first,
            load("deciles_none_unfused_time_rope_first_24h"),
        ),
    }


def preflight_pdf(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Missing or empty PDF: {path}")
    fonts = subprocess.run(
        ["pdffonts", str(path)],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    if "Type 3" in fonts:
        raise ValueError(f"Type 3 font found in {path}")
    font_lines = [line for line in fonts.splitlines()[2:] if line.strip()]
    if not font_lines or any(" yes " not in f" {line} " for line in font_lines):
        raise ValueError(f"Unembedded font found in {path}\n{fonts}")
    font_names = [line.split()[0] for line in font_lines]
    if any(
        "Helvetica" not in name and "NimbusSans" not in name
        for name in font_names
    ):
        raise ValueError(f"Non-Helvetica font found in {path}: {font_names}")
    images = subprocess.run(
        ["pdfimages", "-list", str(path)],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    raster_lines = [
        line for line in images.splitlines() if line.lstrip()[:1].isdigit()
    ]
    if raster_lines:
        raise ValueError(f"Raster image embedded in vector preview {path}\n{images}")
    subprocess.run(["pdfinfo", str(path)], check=True, capture_output=True)


def write_readme(output_dir: Path, names: tuple[str, ...]) -> None:
    lines = [
        "# ML4H submission figure previews",
        "",
        "Approval copies only. Nothing in `ML4H2026/ML4H` is changed by this renderer.",
        "",
        "All figures are vector PDFs at final two-column width. Fonts are embedded,",
        "Type 3 fonts are rejected, and embedded raster images are rejected. The",
        "cluster embeds Nimbus Sans, the metric-compatible open Helvetica face.",
        "",
        "## Result figures",
        "",
        "- `exp1_effects.pdf`: family-level effects relative to each Exp1 baseline.",
        "- `exp2_effects.pdf`: numeric and temporal effects, including affine xVal.",
        "- `exp3_effects.pdf`: CLIF minus native MIMIC.",
        "- Points average six trained models; horizontal bars span their minimum and maximum.",
        "",
        "## Appendix figures",
        "",
        "- `pca_centile_geometry.pdf`: vector PCA with the prior centile palette.",
        "- `clinical_boundary_probe.pdf`: compact vector grouped-bar plot.",
        "- `length_fusion.pdf`: fused versus unfused token-length cumulative distributions.",
        "- `length_temporal.pdf`: time-token versus no-time-token cumulative distributions.",
        "",
        "Outcome membership remains in the manuscript appendix rather than inside the figures.",
        "",
        "## Files",
        "",
        *(f"- `{name}`" for name in names),
        "",
    ]
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--pca-source", type=Path, default=DEFAULT_PCA_SOURCE)
    parser.add_argument("--boundary-results", type=Path, default=DEFAULT_BOUNDARY_RESULTS)
    parser.add_argument("--token-root", type=Path, default=DEFAULT_TOKEN_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    apply_style()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.iterdir():
        if path.is_file():
            path.unlink()

    metrics = pd.read_parquet(args.metrics.expanduser().resolve())
    output_names: list[str] = []
    for experiment in ("exp1", "exp2", "exp3"):
        name = f"{experiment}_effects.pdf"
        render_effect_figure(
            build_effect_source(metrics, experiment),
            experiment=experiment,
            output_path=output_dir / name,
        )
        output_names.append(name)

    render_pca(args.pca_source.expanduser().resolve(), output_dir / "pca_centile_geometry.pdf")
    output_names.append("pca_centile_geometry.pdf")
    render_boundary_probe(
        args.boundary_results.expanduser().resolve(),
        output_dir / "clinical_boundary_probe.pdf",
    )
    output_names.append("clinical_boundary_probe.pdf")

    lengths = load_length_datasets(args.token_root.expanduser().resolve())
    render_length_comparison(
        {
            "full": lengths["fusion_full"],
            "first24": lengths["fusion_first24"],
        },
        labels=("Unfused", "Fused"),
        colors=(OKABE_ITO[0], OKABE_ITO[1]),
        output_path=output_dir / "length_fusion.pdf",
    )
    output_names.append("length_fusion.pdf")
    render_length_comparison(
        {
            "full": lengths["temporal_full"],
            "first24": lengths["temporal_first24"],
        },
        labels=("Time tokens", "No time tokens"),
        colors=(OKABE_ITO[0], OKABE_ITO[2]),
        output_path=output_dir / "length_temporal.pdf",
    )
    output_names.append("length_temporal.pdf")

    for name in output_names:
        preflight_pdf(output_dir / name)
    extras = sorted(
        path.name
        for path in output_dir.iterdir()
        if path.is_file() and path.suffix.lower() not in {".pdf", ".md"}
    )
    if extras:
        raise ValueError(f"Non-PDF preview artifacts found: {extras}")
    write_readme(output_dir, tuple(output_names))
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "pdf_count": len(output_names),
                "pdfs": output_names,
                "preflight": {
                    "embedded_fonts": True,
                    "type3_fonts": False,
                    "embedded_rasters": False,
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
