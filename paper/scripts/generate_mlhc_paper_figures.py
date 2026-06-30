#!/usr/bin/env python3
"""Generate MLHC manuscript figures from the aligned statistics files.

Outputs are written directly into the MLHC manuscript figures directory plus a
small `sources/` subdirectory containing the exact plotting tables used for each
figure.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.patheffects as pe
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MaxNLocator

from pipeline.scripts.diagnostics.diag_embedding_geometry import (
    extract_discrete_bin_embeddings,
    load_embedding_weight,
    load_vocab,
    pca_2d,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METRICS = ROOT / "outputs" / "runs" / "statistics" / "paper_stats_combined" / "all_family_metrics.csv"
DEFAULT_PAIRWISE = ROOT / "outputs" / "runs" / "statistics" / "paper_stats_combined" / "all_family_pairwise_baseline.csv"
DEFAULT_FIG_DIR = (
    ROOT.parent
    / "MLHC2026"
    / "MLHC"
    / "figures"
)
DEFAULT_TOKEN_ROOT = (
    ROOT
    / "outputs"
    / "runs"
    / "tokenized"
    / "mimiciv-3.1_meds_70-10-20"
)

EXP1_ORDER = [
    ("deciles_unfused", "Deciles (population)", "unfused"),
    ("deciles_fused", "Deciles (population)", "fused"),
    ("ventiles_unfused", "Ventiles (population)", "unfused"),
    ("ventiles_fused", "Ventiles (population)", "fused"),
    ("ventiles_5_10_5_unfused", "Ventiles (reference-range)", "unfused"),
    ("ventiles_5_10_5_fused", "Ventiles (reference-range)", "fused"),
    ("trentiles_unfused", "Trentiles (population)", "unfused"),
    ("trentiles_fused", "Trentiles (population)", "fused"),
    ("trentiles_10_10_10_unfused", "Trentiles (reference-range)", "unfused"),
    ("trentiles_10_10_10_fused", "Trentiles (reference-range)", "fused"),
    ("centiles_unfused", "Centiles (population)", "unfused"),
    ("centiles_fused", "Centiles (population)", "fused"),
]

EXP1_COLORS = {
    "Deciles (population)": "#4E79A7",
    "Ventiles (population)": "#76B7B2",
    "Ventiles (reference-range)": "#59A14F",
    "Trentiles (population)": "#E15759",
    "Trentiles (reference-range)": "#B07AA1",
    "Centiles (population)": "#F28E2B",
}

OUTCOME_LABELS = {
    "same_admission_death": "Mortality",
    "long_length_of_stay": "Long LoS",
    "icu_admission": "ICU admission",
    "prolonged_icu_stay": "Prolonged ICU stay",
    "imv_event": "IMV",
    "peak_creatinine": "Creatinine",
    "min_hemoglobin": "Hemoglobin",
    "peak_potassium": "Potassium",
    "min_glucose": "Glucose",
    "peak_troponin": "Troponin",
    "peak_bnp": "BNP",
    "hyperkalemia": "Hyperkalemia",
    "severe_anemia": "Sev. anemia",
    "hypoglycemia": "Hypoglycemia",
    "vasopressor_initiation": "Vasopressor",
    "hypotension": "Hypotension",
}

APPENDIX_OUTCOME_LABELS = {
    "same_admission_death": "Hospital mortality",
    "long_length_of_stay": "Long length of stay",
    "icu_admission": "ICU admission",
    "prolonged_icu_stay": "ICU stay >48h",
    "imv_event": "Invasive mechanical ventilation",
    "hyperkalemia": "Hyperkalemia",
    "severe_hypokalemia": "Severe hypokalemia",
    "severe_anemia": "Severe anemia",
    "hypoglycemia": "Hypoglycemia",
    "profound_hyponatremia": "Profound hyponatremia",
    "severe_hypernatremia": "Severe hypernatremia",
    "tachycardia_hr130": "Tachycardia",
    "severe_hypertension": "Severe hypertension",
    "vasopressor_initiation": "Vasopressor initiation",
    "hypotension": "Hypotension",
    "crrt_initiation": "Continuous renal replacement therapy",
    "hemodialysis_initiation": "Hemodialysis initiation",
    "length_of_stay": "Length of stay (hours)",
    "peak_creatinine": "Peak creatinine",
    "min_hemoglobin": "Minimum hemoglobin",
    "peak_potassium": "Peak potassium",
    "min_potassium": "Minimum potassium",
    "min_glucose": "Minimum glucose",
    "min_sodium": "Minimum sodium",
    "max_sodium": "Maximum sodium",
    "peak_troponin": "Peak troponin",
    "peak_bnp": "Peak natriuretic peptide",
    "max_heart_rate": "Maximum heart rate",
    "max_sbp": "Maximum systolic blood pressure",
    "max_dbp": "Maximum diastolic blood pressure",
}

FOREST_OUTCOME_LABELS = APPENDIX_OUTCOME_LABELS | {
    "imv_event": "Invasive mechanical\nventilation",
    "crrt_initiation": "Continuous renal\nreplacement therapy",
    "length_of_stay": "Length of stay\n(hours)",
    "peak_bnp": "Peak natriuretic\npeptide",
    "max_heart_rate": "Maximum heart\nrate",
    "max_sbp": "Maximum systolic\nblood pressure",
    "max_dbp": "Maximum diastolic\nblood pressure",
}

FOREST_FIGSIZE = (7.3, 7.1)
FOREST_TITLE_SIZE = 8.5
FOREST_LABEL_SIZE = 7.5
FOREST_TICK_SIZE = 6.8
FOREST_LEGEND_SIZE = 6.3
FOREST_MARKER_SIZE = 14
FOREST_LINE_WIDTH = 0.8
FOREST_HANDLE_SPREAD = 0.020
FOREST_YLIM_PAD = 0.28
FOREST_LEGEND_COLSPACING = 0.45
FOREST_LEGEND_LABELSPACING = 0.25
FOREST_OUTCOME_SEPARATOR_COLOR = "#D4D4D4"
FOREST_OUTCOME_SEPARATOR_WIDTH = 0.55
FOREST_OUTCOME_SEPARATOR_ALPHA = 0.65
FOREST_OUTCOME_BAND_COLOR = "#F4F4F4"
FOREST_OUTCOME_BAND_ALPHA = 0.9
FOREST_BINARY_AUROC_X_LO_FLOOR = 0.52
FOREST_LEGEND_TOP_EDGE = 0.925
FOREST_LEGEND_BOX_HEIGHT = 0.075
FOREST_LAYOUT_TOP_EDGE = 0.955
FOREST_EXP1_LAYOUT_TOP_EDGE = 0.935

SUMMARY_BAR_TITLE_SIZE = 9.2
SUMMARY_BAR_LABEL_SIZE = 8.4
SUMMARY_BAR_TICK_SIZE = 7.5
SUMMARY_BAR_VALUE_SIZE = 6.8
SUMMARY_BAR_WIDTH = 0.58
SUMMARY_BAR_ALPHA = 0.94
SUMMARY_BAR_EDGE_COLOR = "#FFFFFF"
SUMMARY_BAR_ERROR_COLOR = "#2F343A"
SUMMARY_BAR_GRID_COLOR = "#E5E7EB"
SUMMARY_BAR_SPINE_COLOR = "#B8BDC7"
SUMMARY_BAR_BASELINE_COLOR = "#9AA1AD"
SUMMARY_BAR_VALUE_COLOR = "#4B5563"

PCA_GRID_FIGSIZE = (12.0, 6.8)
PCA_GRID_TITLE_SIZE = 11.0
PCA_GRID_AXIS_LABEL_SIZE = 9.0
PCA_GRID_TICK_SIZE = 8.0
PCA_GRID_ANNOT_SIZE = 8.0
PCA_GRID_PANEL_NOTE_SIZE = 8.5
PCA_GRID_CBAR_LABEL_SIZE = 9.0
PCA_GRID_CBAR_TICK_SIZE = 8.0
PCA_GRID_SHARED_MARKER_SIZE = 24
PCA_GRID_FUSED_MARKER_SIZE = 20

TREND_COLORS = {
    "Binary outcomes": "#4C78A8",
    "Regression outcomes": "#F28E2B",
}

PCA_MEASUREMENTS = [
    ("Potassium", "LAB_lab//50971//meq/l"),
    ("Hemoglobin", "LAB_lab//51222//g/dl"),
    ("Creatinine", "LAB_lab//50912//mg/dl"),
    ("Glucose", "LAB_lab//50931//mg/dl"),
    ("Sodium", "LAB_lab//50983//meq/l"),
    ("Heart rate", "VTL_vital//220045//bpm"),
]

EXP1_HANDLE_SHORT_LABELS = {
    "deciles_unfused": "Deciles (population), unfused",
    "deciles_fused": "Deciles (population), fused",
    "ventiles_unfused": "Ventiles (population), unfused",
    "ventiles_fused": "Ventiles (population), fused",
    "ventiles_5_10_5_unfused": "Ventiles (clinical), unfused",
    "ventiles_5_10_5_fused": "Ventiles (clinical), fused",
    "trentiles_unfused": "Trentiles (population), unfused",
    "trentiles_fused": "Trentiles (population), fused",
    "trentiles_10_10_10_unfused": "Trentiles (clinical), unfused",
    "trentiles_10_10_10_fused": "Trentiles (clinical), fused",
    "centiles_unfused": "Centiles (population), unfused",
    "centiles_fused": "Centiles (population), fused",
}

EXP1_COLOR_LEGEND_LABELS = {
    label: label.replace(" (reference-range)", "\n(reference-range)")
    if "(reference-range)" in label
    else label
    for label in EXP1_COLORS
}

EXP1_TREND_COMPARISONS = [
    ("Dec F-U", "deciles_unfused", "deciles_fused"),
    ("Vent F-U", "ventiles_unfused", "ventiles_fused"),
    ("VentC F-U", "ventiles_5_10_5_unfused", "ventiles_5_10_5_fused"),
    ("Trent F-U", "trentiles_unfused", "trentiles_fused"),
    ("TrentC F-U", "trentiles_10_10_10_unfused", "trentiles_10_10_10_fused"),
    ("Cent F-U", "centiles_unfused", "centiles_fused"),
]

BINARY_OUTCOME_ORDER_EXP12 = [
    "same_admission_death",
    "long_length_of_stay",
    "icu_admission",
    "imv_event",
    "hyperkalemia",
    "severe_hypokalemia",
    "severe_anemia",
    "hypoglycemia",
    "profound_hyponatremia",
    "severe_hypernatremia",
    "tachycardia_hr130",
    "severe_hypertension",
    "vasopressor_initiation",
    "hypotension",
    "crrt_initiation",
    "hemodialysis_initiation",
]

BINARY_OUTCOME_ORDER_EXP3 = [
    "same_admission_death",
    "long_length_of_stay",
    "prolonged_icu_stay",
    "imv_event",
    "hyperkalemia",
    "severe_hypokalemia",
    "severe_anemia",
    "hypoglycemia",
    "profound_hyponatremia",
    "severe_hypernatremia",
    "tachycardia_hr130",
    "severe_hypertension",
    "vasopressor_initiation",
    "hypotension",
    "crrt_initiation",
    "hemodialysis_initiation",
]

REGRESSION_OUTCOME_ORDER = [
    "length_of_stay",
    "peak_creatinine",
    "min_hemoglobin",
    "peak_potassium",
    "min_potassium",
    "min_glucose",
    "min_sodium",
    "max_sodium",
    "peak_troponin",
    "peak_bnp",
    "max_heart_rate",
    "max_sbp",
    "max_dbp",
]

EXP2_HANDLE_ORDER = [
    "discrete_none",
    "soft_none",
    "xval_none",
    "xval_affine_none",
    "discrete_tt",
    "soft_tt",
    "xval_tt",
    "xval_affine_tt",
    "discrete_rope",
    "soft_rope",
    "xval_rope",
    "xval_affine_rope",
]

EXP2_HANDLE_SHORT_LABELS = {
    "discrete_none": "Discrete + event order only",
    "soft_none": "Soft discretization + event order only",
    "xval_none": "xVal (code-normalized) + event order only",
    "xval_affine_none": "xVal-affine (code-normalized + affine shift) + event order only",
    "discrete_tt": "Discrete + time tokens",
    "soft_tt": "Soft discretization + time tokens",
    "xval_tt": "xVal (code-normalized) + time tokens",
    "xval_affine_tt": "xVal-affine (code-normalized + affine shift) + time tokens",
    "discrete_rope": "Discrete + admission-relative RoPE",
    "soft_rope": "Soft discretization + admission-relative RoPE",
    "xval_rope": "xVal (code-normalized) + admission-relative RoPE",
    "xval_affine_rope": "xVal-affine (code-normalized + affine shift) + admission-relative RoPE",
}

EXP2_HANDLE_COLORS = {
    "discrete_none": "#4E79A7",
    "soft_none": "#76B7B2",
    "xval_none": "#F28E2B",
    "xval_affine_none": "#B07AA1",
    "discrete_tt": "#4E79A7",
    "soft_tt": "#76B7B2",
    "xval_tt": "#F28E2B",
    "xval_affine_tt": "#B07AA1",
    "discrete_rope": "#4E79A7",
    "soft_rope": "#76B7B2",
    "xval_rope": "#F28E2B",
    "xval_affine_rope": "#B07AA1",
}

EXP3_HANDLE_ORDER = ["meds", "mapped", "randomized", "freqmatched"]

EXP3_HANDLE_SHORT_LABELS = {
    "meds": "Native MIMIC codes",
    "mapped": "CLIF-mapped",
    "randomized": "Randomized mapped codes",
    "freqmatched": "Frequency-matched mapped codes",
}

EXP3_HANDLE_COLORS = {
    "meds": "#4E79A7",
    "mapped": "#59A14F",
    "randomized": "#E15759",
    "freqmatched": "#F28E2B",
}

EXP1_FUSION_SUMMARY_COLORS = {
    "Unfused": "#A0CBE8",
    "Fused": "#4E79A7",
}

EXP1_GRANULARITY_SUMMARY_COLORS = {
    "Deciles": "#4E79A7",
    "Ventiles": "#76B7B2",
    "Trentiles": "#E15759",
    "Centiles": "#F28E2B",
}

EXP1_ANCHOR_SUMMARY_COLORS = {
    "Population bins": "#4E79A7",
    "Clinical bins": "#59A14F",
}

EXP2_VALUE_SUMMARY_COLORS = {
    "Discrete": "#4E79A7",
    "Soft": "#76B7B2",
    "xVal (code-normalized)": "#F28E2B",
    "xVal-affine (code-normalized + affine shift)": "#B07AA1",
}

EXP2_TEMPORAL_SUMMARY_COLORS = {
    "Event order only": "#59A14F",
    "Time tokens": "#A0CBE8",
    "Admission-relative RoPE": "#4E79A7",
}

EXP2_EFFECT_COMPARISONS = [
    ("Discrete: time tokens - event order", "discrete_none", "discrete_tt", "#2A9D8F", "o"),
    ("Discrete: adm.-rel. RoPE - event order", "discrete_none", "discrete_rope", "#2A9D8F", "s"),
    ("Soft - Discrete (event order)", "discrete_none", "soft_none", "#4E79A7", "^"),
    ("Soft - Discrete (adm.-rel. RoPE)", "discrete_rope", "soft_rope", "#4E79A7", "s"),
    ("xVal-CN - Discrete (event order)", "discrete_none", "xval_none", "#6C4E9B", "^"),
    ("xVal-CN - Discrete (adm.-rel. RoPE)", "discrete_rope", "xval_rope", "#6C4E9B", "s"),
    ("xVal-Aff - xVal-CN (event order)", "xval_none", "xval_affine_none", "#E76F51", "^"),
    ("xVal-Aff - xVal-CN (adm.-rel. RoPE)", "xval_rope", "xval_affine_rope", "#E76F51", "s"),
]

EXP3_EFFECT_COMPARISONS = [
    ("CLIF-mapped - Native MIMIC codes", "meds", "mapped", "#2A9D8F", "o"),
    ("Randomized mapped codes - Native MIMIC codes", "meds", "randomized", "#E76F51", "s"),
    ("Frequency-matched mapped codes - Native MIMIC codes", "meds", "freqmatched", "#6C4E9B", "D"),
]


def _extract_fused_sparse(
    *,
    model_dir: Path,
    vocab_path: Path,
    code: str,
    num_bins: int,
) -> tuple[list[int], np.ndarray]:
    embed_weight = load_embedding_weight(model_dir)
    vocab = load_vocab(vocab_path)
    lookup = vocab["lookup"]
    present_bins: list[int] = []
    token_ids: list[int] = []
    for i in range(num_bins):
        token_id = lookup.get(f"{code}_Q{i}")
        if token_id is None:
            continue
        present_bins.append(i)
        token_ids.append(token_id)
    if not token_ids:
        raise ValueError(f"No fused tokens found for {code} in {vocab_path}")
    return present_bins, embed_weight[token_ids].float().numpy()


def _style():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "axes.edgecolor": "#333333",
            "axes.linewidth": 1.0,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )


def _save_source(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _latest_checkpoint(run_dir: Path) -> Path:
    checkpoints = sorted(
        run_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found under {run_dir}")
    return checkpoints[-1]


def _selected_bin_labels(bins: list[int]) -> set[int]:
    if len(bins) <= 12:
        return set(bins)
    anchors = [0, 10, 25, 50, 75, 90, 99]
    selected = set()
    for anchor in anchors:
        nearest = min(bins, key=lambda b: abs(b - anchor))
        selected.add(nearest)
    return selected


def _extract_pairwise_effect(
    pairwise: pd.DataFrame,
    *,
    families: list[str],
    metric: str,
    outcome: str,
    handle0: str,
    handle1: str,
) -> tuple[float, float, float, float]:
    sub = pairwise[
        pairwise["family_name"].isin(families)
        & (pairwise["metric"] == metric)
        & (pairwise["outcome"] == outcome)
    ]
    direct = sub[(sub["handle0"] == handle0) & (sub["handle1"] == handle1)]
    if not direct.empty:
        row = direct.iloc[0]
        return (
            float(row["delta_better"]),
            float(row["delta_ci_lo_better"]),
            float(row["delta_ci_hi_better"]),
            float(row["p_adj"]),
        )
    reverse = sub[(sub["handle0"] == handle1) & (sub["handle1"] == handle0)]
    if not reverse.empty:
        row = reverse.iloc[0]
        return (
            float(-row["delta_better"]),
            float(-row["delta_ci_hi_better"]),
            float(-row["delta_ci_lo_better"]),
            float(row["p_adj"]),
        )
    return (float("nan"), float("nan"), float("nan"), float("nan"))


def _count_favorable_outcomes(
    pairwise: pd.DataFrame,
    *,
    families: list[str],
    metric: str,
    handle0: str,
    handle1: str,
) -> int:
    outcomes = (
        pairwise[
        pairwise["family_name"].isin(families)
        & (pairwise["metric"] == metric)
    ]["outcome"].drop_duplicates().tolist()
    )
    count = 0
    for outcome in outcomes:
        delta, _, _, p_adj = _extract_pairwise_effect(
            pairwise,
            families=families,
            metric=metric,
            outcome=outcome,
            handle0=handle0,
            handle1=handle1,
        )
        if np.isfinite(p_adj) and np.isfinite(delta) and p_adj < 0.05 and delta > 0:
            count += 1
    return count


def _xlim_from_points(
    ax,
    points: np.ndarray,
    *,
    x_lo_floor: float | None = None,
    x_hi_ceil: float | None = None,
) -> None:
    finite = points[np.isfinite(points)]
    if finite.size == 0:
        return
    p_lo, p_hi = float(np.min(finite)), float(np.max(finite))
    span = max(p_hi - p_lo, 0.04)
    pad = max(0.012, 0.09 * span)
    lo = p_lo - pad
    hi = p_hi + pad
    if x_lo_floor is not None:
        lo = max(x_lo_floor, lo)
    if x_hi_ceil is not None:
        hi = min(x_hi_ceil, hi)
    ax.set_xlim(lo, hi)


def _collect_exp1_source(metrics: pd.DataFrame) -> pd.DataFrame:
    selected = BINARY_OUTCOME_ORDER_EXP12 + REGRESSION_OUTCOME_ORDER
    return metrics[
        metrics["family_name"].isin(
            [
                "exp1_primary_binary",
                "exp1_additional_binary",
                "exp1_length_of_stay",
                "exp1_extended_regression",
            ]
        )
        & metrics["outcome"].isin(selected)
        & metrics["metric"].isin(["roc_auc", "spearman_rho"])
    ][["family_name", "outcome", "handle", "metric", "point", "ci_lo", "ci_hi"]].copy()


def _forest_outcome_row_bounds(base_y: np.ndarray, idx: int) -> tuple[float, float]:
    if len(base_y) == 1:
        return base_y[0] - 0.35, base_y[0] + 0.35
    if idx == 0:
        y_top = base_y[0] + (base_y[0] - base_y[1]) / 2.0
    else:
        y_top = (base_y[idx - 1] + base_y[idx]) / 2.0
    if idx == len(base_y) - 1:
        y_bottom = base_y[-1] - (base_y[-2] - base_y[-1]) / 2.0
    else:
        y_bottom = (base_y[idx] + base_y[idx + 1]) / 2.0
    return y_bottom, y_top


def _add_forest_outcome_banding(ax, base_y: np.ndarray) -> None:
    x0, x1 = ax.get_xlim()
    for idx in range(len(base_y)):
        if idx % 2 == 0:
            continue
        y_bottom, y_top = _forest_outcome_row_bounds(base_y, idx)
        ax.axhspan(
            y_bottom,
            y_top,
            xmin=0,
            xmax=1,
            facecolor=FOREST_OUTCOME_BAND_COLOR,
            alpha=FOREST_OUTCOME_BAND_ALPHA,
            edgecolor="none",
            zorder=0.5,
            clip_on=True,
        )
    ax.set_xlim(x0, x1)


def _add_forest_outcome_separators(ax, base_y: np.ndarray) -> None:
    if len(base_y) < 2:
        return
    x0, x1 = ax.get_xlim()
    for idx in range(len(base_y) - 1):
        y_sep = (base_y[idx] + base_y[idx + 1]) / 2.0
        ax.hlines(
            y_sep,
            x0,
            x1,
            colors=FOREST_OUTCOME_SEPARATOR_COLOR,
            linewidth=FOREST_OUTCOME_SEPARATOR_WIDTH,
            alpha=FOREST_OUTCOME_SEPARATOR_ALPHA,
            zorder=1,
        )


def _plot_exp1_granularity_panel(
    ax,
    metrics: pd.DataFrame,
    *,
    outcome_order: list[str],
    families: list[str],
    metric: str,
    handle_order: list[str],
    handle_map: dict[str, tuple[str, str]],
    ylab_map: dict[str, str],
    xlabel: str,
    panel_title: str,
    x_lo_floor: float | None,
) -> None:
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
        sub = metrics[
            metrics["family_name"].isin(families)
            & (metrics["metric"] == metric)
            & (metrics["outcome"] == outcome)
            & metrics["handle"].isin(handle_order)
        ]
        for h_idx, handle in enumerate(handle_order):
            row = sub[sub["handle"] == handle]
            if row.empty:
                continue
            r = row.iloc[0]
            pts.append(float(r["point"]))
            granularity, tokenization = handle_map[handle]
            marker = "o" if tokenization == "unfused" else "s"
            alpha = 0.55 if tokenization == "unfused" else 0.95
            color = EXP1_COLORS[granularity]
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
    ax.set_yticklabels([ylab_map[o] for o in outcome_order], fontsize=FOREST_TICK_SIZE)
    ax.set_title(panel_title, fontsize=FOREST_TITLE_SIZE, pad=4)
    ax.set_xlabel(xlabel, fontsize=FOREST_LABEL_SIZE)
    ax.tick_params(axis="x", labelsize=FOREST_TICK_SIZE, length=2)
    ax.grid(axis="x", alpha=0.20, linewidth=0.5)
    if pts:
        _xlim_from_points(ax, np.array(pts), x_lo_floor=x_lo_floor, x_hi_ceil=None)
    ax.set_ylim(base_y.min() - FOREST_YLIM_PAD, base_y.max() + FOREST_YLIM_PAD)
    _add_forest_outcome_banding(ax, base_y)
    _add_forest_outcome_separators(ax, base_y)


def _collect_exp2_source(metrics: pd.DataFrame) -> pd.DataFrame:
    binary = metrics[
        metrics["family_name"].isin(["exp2_primary_binary", "exp2_additional_binary"])
        & (metrics["metric"] == "roc_auc")
        & metrics["outcome"].isin(BINARY_OUTCOME_ORDER_EXP12)
        & metrics["handle"].isin(EXP2_HANDLE_ORDER)
    ][["family_name", "outcome", "handle", "point", "ci_lo", "ci_hi"]].assign(panel="binary")
    regression = metrics[
        metrics["family_name"].isin(["exp2_length_of_stay", "exp2_extended_regression"])
        & (metrics["metric"] == "spearman_rho")
        & metrics["outcome"].isin(REGRESSION_OUTCOME_ORDER)
        & metrics["handle"].isin(EXP2_HANDLE_ORDER)
    ][["family_name", "outcome", "handle", "point", "ci_lo", "ci_hi"]].assign(panel="regression")
    return pd.concat([binary, regression], ignore_index=True)


def _plot_handle_metric_panel(
    ax,
    source: pd.DataFrame,
    *,
    outcome_order: list[str],
    handle_order: list[str],
    handle_colors: dict[str, str],
    marker_for_handle,
    panel_title: str,
    xlabel: str,
    x_lo_floor: float | None = None,
) -> None:
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
        sub = source[source["outcome"] == outcome]
        for h_idx, handle in enumerate(handle_order):
            row = sub[sub["handle"] == handle]
            if row.empty:
                continue
            r = row.iloc[0]
            pts.append(float(r["point"]))
            y = base_y[o_idx] + offsets[h_idx]
            color = handle_colors[handle]
            marker = marker_for_handle(handle)
            ax.hlines(y, r["ci_lo"], r["ci_hi"], color=color, linewidth=FOREST_LINE_WIDTH, alpha=0.92)
            ax.scatter(
                r["point"],
                y,
                color=color,
                marker=marker,
                s=FOREST_MARKER_SIZE,
                edgecolor="#222222",
                linewidth=0.25,
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


def _collect_exp3_source(metrics: pd.DataFrame) -> pd.DataFrame:
    binary = metrics[
        metrics["family_name"].isin(["exp3_primary_binary", "exp3_additional_binary"])
        & (metrics["metric"] == "roc_auc")
        & metrics["outcome"].isin(BINARY_OUTCOME_ORDER_EXP3)
        & metrics["handle"].isin(EXP3_HANDLE_ORDER)
    ][["family_name", "outcome", "handle", "point", "ci_lo", "ci_hi"]].assign(panel="binary")
    regression = metrics[
        metrics["family_name"].isin(["exp3_length_of_stay", "exp3_extended_regression"])
        & (metrics["metric"] == "spearman_rho")
        & metrics["outcome"].isin(REGRESSION_OUTCOME_ORDER)
        & metrics["handle"].isin(EXP3_HANDLE_ORDER)
    ][["family_name", "outcome", "handle", "point", "ci_lo", "ci_hi"]].assign(panel="regression")
    return pd.concat([binary, regression], ignore_index=True)


def _baseline_delta_source(
    metrics: pd.DataFrame,
    pairwise: pd.DataFrame,
    *,
    families: list[str],
    metric: str,
    outcome_order: list[str],
    handle_order: list[str],
    baseline_handle: str,
    panel: str,
) -> pd.DataFrame:
    source = metrics[
        metrics["family_name"].isin(families)
        & (metrics["metric"] == metric)
        & metrics["outcome"].isin(outcome_order)
        & metrics["handle"].isin(handle_order)
    ][["family_name", "outcome", "handle", "point", "ci_lo", "ci_hi"]].copy()
    baseline = (
        source[source["handle"] == baseline_handle][["outcome", "point"]]
        .rename(columns={"point": "baseline_point"})
        .drop_duplicates(subset=["outcome"])
    )
    source = source.merge(baseline, on="outcome", how="left", validate="many_to_one")
    source["delta"] = source["point"] - source["baseline_point"]

    pw = pairwise[
        pairwise["family_name"].isin(families)
        & (pairwise["metric"] == metric)
        & pairwise["outcome"].isin(outcome_order)
        & ((pairwise["handle0"] == baseline_handle) | (pairwise["handle1"] == baseline_handle))
    ][["outcome", "handle0", "handle1", "p_adj"]].copy()
    if not pw.empty:
        pw["handle"] = np.where(pw["handle0"] == baseline_handle, pw["handle1"], pw["handle0"])
        pw = pw[["outcome", "handle", "p_adj"]].drop_duplicates(subset=["outcome", "handle"])
    else:
        pw = pd.DataFrame(columns=["outcome", "handle", "p_adj"])

    source = source.merge(pw, on=["outcome", "handle"], how="left")
    source["significant"] = source["p_adj"].fillna(np.inf) < 0.05
    source.loc[source["handle"] == baseline_handle, "significant"] = False
    source["panel"] = panel
    source["baseline_handle"] = baseline_handle
    return source


def _compute_shared_heatmap_vmax(metrics: pd.DataFrame, pairwise: pd.DataFrame) -> dict[str, float]:
    del metrics, pairwise
    return {"binary": 0.15, "regression": 0.20}


def _plot_delta_heatmap_panel(
    ax,
    source: pd.DataFrame,
    *,
    outcome_order: list[str],
    handle_order: list[str],
    handle_labels: dict[str, str],
    panel_title: str,
    colorbar_label: str,
    group_breaks: list[int] | None = None,
    fixed_vmax: float | None = None,
    cbar_ticks: list[float] | None = None,
) -> None:
    matrix_full = np.full((len(outcome_order), len(handle_order)), np.nan, dtype=float)
    significant_full = np.zeros((len(outcome_order), len(handle_order)), dtype=bool)

    for row_idx, outcome in enumerate(outcome_order):
        sub = source[source["outcome"] == outcome]
        for col_idx, handle in enumerate(handle_order):
            row = sub[sub["handle"] == handle]
            if row.empty:
                continue
            rec = row.iloc[0]
            matrix_full[row_idx, col_idx] = float(rec["delta"])
            significant_full[row_idx, col_idx] = bool(rec["significant"])

    keep_rows = np.isfinite(matrix_full).any(axis=1)
    if not np.any(keep_rows):
        keep_rows = np.ones(len(outcome_order), dtype=bool)

    matrix = matrix_full[keep_rows, :]
    significant = significant_full[keep_rows, :]
    outcome_labels = [APPENDIX_OUTCOME_LABELS[outcome_order[i]] for i, keep in enumerate(keep_rows) if keep]

    finite = matrix[np.isfinite(matrix)]
    if fixed_vmax is None:
        vmax = max(float(np.nanmax(np.abs(finite))) if finite.size else 0.0, 0.02)
    else:
        vmax = max(float(fixed_vmax), 0.02)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("#F5F7FA")
    im = ax.imshow(
        matrix,
        aspect="auto",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        zorder=2,
    )

    ax.set_yticks(np.arange(len(outcome_labels)))
    ax.set_yticklabels(outcome_labels, fontsize=10)
    ax.set_xticks(np.arange(len(handle_order)))
    labels = [_wrap_bar_tick_label(handle_labels[h]) for h in handle_order]
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_title(panel_title, pad=8)
    ax.tick_params(
        axis="x",
        bottom=True,
        top=False,
        labelbottom=True,
        labeltop=False,
        length=0,
    )
    ax.tick_params(axis="y", length=0)
    ax.set_facecolor("#F7F8FA")

    ax.set_xticks(np.arange(-0.5, len(handle_order), 1.0), minor=True)
    ax.set_yticks(np.arange(-0.5, len(outcome_labels), 1.0), minor=True)
    ax.grid(which="minor", color="#FFFFFF", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    for break_idx in group_breaks or []:
        ax.axvline(break_idx + 0.5, color="#1F2937", linewidth=1.15, alpha=0.95, zorder=6)

    baseline_handle = source["baseline_handle"].iloc[0] if "baseline_handle" in source.columns and not source.empty else None
    if baseline_handle in handle_order:
        baseline_idx = handle_order.index(baseline_handle)
        ax.add_patch(
            Rectangle(
                (baseline_idx - 0.5, -0.5),
                1.0,
                len(outcome_labels),
                facecolor="none",
                edgecolor="#111827",
                linewidth=1.05,
                linestyle=(0, (2.2, 2.2)),
                zorder=7,
            )
        )

    sig_rows, sig_cols = np.where(significant & np.isfinite(matrix))
    if sig_rows.size:
        for row_idx, col_idx in zip(sig_rows.tolist(), sig_cols.tolist()):
            ax.text(
                col_idx,
                row_idx,
                r"$\ast$",
                ha="center",
                va="center",
                fontsize=9.5,
                color="#111827",
                zorder=8,
                path_effects=[pe.withStroke(linewidth=1.15, foreground="white", alpha=0.95)],
            )
    ax.set_xlim(-0.5, len(handle_order) - 0.5)
    ax.set_ylim(len(outcome_labels) - 0.5, -0.5)
    for side in ["top", "right", "bottom", "left"]:
        ax.spines[side].set_visible(False)

    cbar = plt.colorbar(
        im,
        ax=ax,
        orientation="horizontal",
        fraction=0.048,
        pad=0.08,
    )
    if cbar_ticks is not None:
        cbar.set_ticks(cbar_ticks)
    cbar.set_label(colorbar_label, fontsize=11, labelpad=3)
    cbar.ax.tick_params(labelsize=9, length=2)
    cbar.outline.set_linewidth(0.55)


def _mean_interval(source: pd.DataFrame, *, group_cols: list[str]) -> pd.DataFrame:
    return (
        source.groupby(group_cols, as_index=False)
        .agg(
            point=("point", "mean"),
            ci_lo=("ci_lo", "mean"),
            ci_hi=("ci_hi", "mean"),
            n_rows=("point", "size"),
        )
    )


def _exp1_axis_summary_source(metrics: pd.DataFrame) -> pd.DataFrame:
    source = _collect_exp1_source(metrics).copy()
    source["panel"] = np.where(source["metric"] == "roc_auc", "binary", "regression")
    handle_meta = pd.DataFrame(EXP1_ORDER, columns=["handle", "granularity_label", "fusion_raw"])
    handle_meta["granularity"] = handle_meta["granularity_label"].str.extract(
        r"^(Deciles|Ventiles|Trentiles|Centiles)"
    )
    handle_meta["anchoring"] = np.where(
        handle_meta["granularity_label"].str.contains("reference-range", regex=False),
        "Clinical bins",
        "Population bins",
    )
    handle_meta["fusion"] = handle_meta["fusion_raw"].map(
        {"unfused": "Unfused", "fused": "Fused"}
    )
    source = source.merge(
        handle_meta[["handle", "granularity", "anchoring", "fusion"]],
        on="handle",
        how="left",
    )
    anchor_source = source[source["granularity"].isin(["Ventiles", "Trentiles"])].copy()
    frames = [
        _mean_interval(
            source.assign(axis="Fusion", level=source["fusion"]),
            group_cols=["panel", "axis", "level"],
        ),
        _mean_interval(
            source.assign(axis="Granularity", level=source["granularity"]),
            group_cols=["panel", "axis", "level"],
        ),
        _mean_interval(
            anchor_source.assign(axis="Anchoring", level=anchor_source["anchoring"]),
            group_cols=["panel", "axis", "level"],
        ),
    ]
    return pd.concat(frames, ignore_index=True)


def _exp2_axis_summary_source(metrics: pd.DataFrame) -> pd.DataFrame:
    source = _collect_exp2_source(metrics).copy()
    value_family = {
        "discrete_none": "Discrete",
        "discrete_tt": "Discrete",
        "soft_none": "Soft",
        "soft_tt": "Soft",
        "xval_none": "xVal (code-normalized)",
        "xval_tt": "xVal (code-normalized)",
        "xval_affine_none": "xVal-affine (code-normalized + affine shift)",
        "xval_affine_tt": "xVal-affine (code-normalized + affine shift)",
        "discrete_rope": "Discrete",
        "soft_rope": "Soft",
        "xval_rope": "xVal (code-normalized)",
        "xval_affine_rope": "xVal-affine (code-normalized + affine shift)",
    }
    temporal_family = {
        "discrete_none": "Event order only",
        "soft_none": "Event order only",
        "xval_none": "Event order only",
        "xval_affine_none": "Event order only",
        "discrete_tt": "Time tokens",
        "soft_tt": "Time tokens",
        "xval_tt": "Time tokens",
        "xval_affine_tt": "Time tokens",
        "discrete_rope": "Admission-relative RoPE",
        "soft_rope": "Admission-relative RoPE",
        "xval_rope": "Admission-relative RoPE",
        "xval_affine_rope": "Admission-relative RoPE",
    }
    frames = [
        _mean_interval(
            source.assign(axis="Value encoder", level=source["handle"].map(value_family)),
            group_cols=["panel", "axis", "level"],
        ),
        _mean_interval(
            source.assign(
                axis="Temporal encoding",
                level=source["handle"].map(temporal_family),
            ),
            group_cols=["panel", "axis", "level"],
        ),
    ]
    return pd.concat(frames, ignore_index=True)


def _exp3_axis_summary_source(metrics: pd.DataFrame) -> pd.DataFrame:
    source = _collect_exp3_source(metrics).copy()
    level_labels = {
        "meds": "Native MIMIC codes",
        "mapped": "CLIF-mapped",
        "randomized": "Randomized mapped codes",
        "freqmatched": "Frequency-matched mapped codes",
    }
    return _mean_interval(
        source.assign(
            axis="Vocabulary arm",
            level=source["handle"].map(level_labels),
        ),
        group_cols=["panel", "axis", "level"],
    )


def _handle_average_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    exp1 = _collect_exp1_source(metrics).copy()
    exp1["panel"] = np.where(exp1["metric"] == "roc_auc", "binary", "regression")
    exp1_meta = pd.DataFrame(EXP1_ORDER, columns=["handle", "granularity_label", "fusion_raw"])
    exp1_meta["label"] = exp1_meta["handle"].map(EXP1_HANDLE_SHORT_LABELS)
    exp1_meta["color"] = exp1_meta["granularity_label"].map(EXP1_COLORS)
    exp1_frame = _mean_interval(
        exp1.merge(exp1_meta[["handle", "label", "color"]], on="handle", how="left"),
        group_cols=["panel", "handle", "label", "color"],
    )
    exp1_frame["experiment"] = "Exp1"
    exp1_frame["order"] = exp1_frame["handle"].map(
        {handle: idx for idx, (handle, _, _) in enumerate(EXP1_ORDER)}
    )
    frames.append(exp1_frame)

    exp2 = _collect_exp2_source(metrics).copy()
    exp2_frame = _mean_interval(
        exp2.assign(
            label=exp2["handle"].map(EXP2_HANDLE_SHORT_LABELS),
            color=exp2["handle"].map(EXP2_HANDLE_COLORS),
        ),
        group_cols=["panel", "handle", "label", "color"],
    )
    exp2_frame["experiment"] = "Exp2"
    exp2_frame["order"] = exp2_frame["handle"].map(
        {handle: idx for idx, handle in enumerate(EXP2_HANDLE_ORDER)}
    )
    frames.append(exp2_frame)

    exp3 = _collect_exp3_source(metrics).copy()
    exp3_frame = _mean_interval(
        exp3.assign(
            label=exp3["handle"].map(EXP3_HANDLE_SHORT_LABELS),
            color=exp3["handle"].map(EXP3_HANDLE_COLORS),
        ),
        group_cols=["panel", "handle", "label", "color"],
    )
    exp3_frame["experiment"] = "Exp3"
    exp3_frame["order"] = exp3_frame["handle"].map(
        {handle: idx for idx, handle in enumerate(EXP3_HANDLE_ORDER)}
    )
    frames.append(exp3_frame)

    return pd.concat(frames, ignore_index=True)


def _plot_summary_bar_panel(
    ax,
    source: pd.DataFrame,
    *,
    level_order: list[str],
    color_map: dict[str, str],
    title: str,
    ylabel: str,
    y_floor: float | None,
    display_labels: dict[str, str] | None = None,
) -> None:
    sub = source.copy()
    sub["level"] = pd.Categorical(sub["level"], categories=level_order, ordered=True)
    sub = sub.sort_values("level").reset_index(drop=True)
    if sub.empty:
        ax.set_axis_off()
        return
    x = np.arange(len(sub), dtype=float)
    points = sub["point"].to_numpy(dtype=float)
    ci_lo = sub["ci_lo"].to_numpy(dtype=float)
    ci_hi = sub["ci_hi"].to_numpy(dtype=float)
    err = np.vstack([points - ci_lo, ci_hi - points])
    labels = [
        display_labels.get(str(level), str(level)) if display_labels else str(level)
        for level in sub["level"]
    ]
    lower = float(np.nanmin(ci_lo)) - 0.02
    floor = lower if y_floor is None else min(float(y_floor), lower)
    upper = float(np.nanmax(ci_hi))
    pad = max(0.018, 0.12 * max(upper - floor, 0.08))
    y_top = upper + pad
    bar_heights = points - floor

    ax.set_facecolor("#FFFFFF")
    ax.bar(
        x,
        bar_heights,
        bottom=floor,
        width=SUMMARY_BAR_WIDTH,
        color=[color_map[str(level)] for level in sub["level"]],
        edgecolor=SUMMARY_BAR_EDGE_COLOR,
        linewidth=0.7,
        alpha=SUMMARY_BAR_ALPHA,
        zorder=3,
    )
    ax.errorbar(
        x,
        points,
        yerr=err,
        fmt="none",
        ecolor=SUMMARY_BAR_ERROR_COLOR,
        elinewidth=0.9,
        capsize=2.2,
        capthick=0.9,
        zorder=4,
    )
    ax.set_ylim(floor, y_top)
    ax.set_xlim(-0.55, len(sub) - 0.45)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            label if "\n" in label else "\n".join(textwrap.wrap(label, width=12, break_long_words=False))
            for label in labels
        ],
        rotation=0,
        ha="center",
        fontsize=SUMMARY_BAR_TICK_SIZE,
        linespacing=1.05,
    )
    ax.set_title(title, fontsize=SUMMARY_BAR_TITLE_SIZE, fontweight="semibold", pad=7)
    ax.set_ylabel(ylabel, fontsize=SUMMARY_BAR_LABEL_SIZE)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=SUMMARY_BAR_TICK_SIZE, length=0)
    ax.tick_params(axis="x", length=0, pad=4)
    if not ylabel:
        ax.tick_params(axis="y", labelleft=False)
        ax.spines["left"].set_visible(False)
    ax.xaxis.grid(False)
    ax.grid(axis="y", color=SUMMARY_BAR_GRID_COLOR, linestyle="-", linewidth=0.6, zorder=0)
    ax.axhline(floor, color=SUMMARY_BAR_BASELINE_COLOR, linewidth=0.7, zorder=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ["left", "bottom"]:
        if ax.spines[side].get_visible():
            ax.spines[side].set_color(SUMMARY_BAR_SPINE_COLOR)
            ax.spines[side].set_linewidth(0.7)
    for xi, yi, hi in zip(x, points, ci_hi):
        ax.annotate(
            f"{yi:.3f}",
            xy=(xi, hi),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=SUMMARY_BAR_VALUE_SIZE,
            color=SUMMARY_BAR_VALUE_COLOR,
            clip_on=False,
        )


def _exp2_marker(handle: str) -> str:
    if handle.endswith("_none"):
        return "^"
    if handle.endswith("_tt"):
        return "o"
    return "s"


EXP2_VALUE_COLOR_ITEMS = [
    ("Discrete", "#4E79A7"),
    ("Soft discretization", "#76B7B2"),
    ("xVal", "#F28E2B"),
    ("xVal-affine", "#B07AA1"),
]

EXP2_TEMPORAL_MARKER_ITEMS = [
    ("Event order only", "^", 0.95),
    ("Time tokens", "o", 0.95),
    ("Admission-relative\nRoPE", "s", 0.95),
]


EXP3_HANDLE_MARKERS = {
    "meds": "o",
    "mapped": "s",
    "randomized": "D",
    "freqmatched": "^",
}

EXP3_FOREST_LABELS = {
    "meds": "Native source codes",
    "mapped": "CLIF-mapped",
    "randomized": "Randomized mapped codes",
    "freqmatched": "Frequency-matched mapped codes",
}


def _exp3_marker(handle: str) -> str:
    return EXP3_HANDLE_MARKERS[handle]


def _wrap_bar_tick_label(label: str) -> str:
    replacements = {
        "Disc event order": "Disc\nevent\norder",
        "Soft event order": "Soft\nevent\norder",
        "xVal-CN event order": "xVal-CN\nevent\norder",
        "xVal-Aff event order": "xVal-Aff\nevent\norder",
        "Disc adm.-rel. RoPE": "Disc\nadm.-rel.\nRoPE",
        "Soft adm.-rel. RoPE": "Soft\nadm.-rel.\nRoPE",
        "xVal-CN adm.-rel. RoPE": "xVal-CN\nadm.-rel.\nRoPE",
        "xVal-Aff adm.-rel. RoPE": "xVal-Aff\nadm.-rel.\nRoPE",
        "Disc time tokens": "Disc\ntime\ntokens",
        "Soft time tokens": "Soft\ntime\ntokens",
        "xVal-CN time tokens": "xVal-CN\ntime\ntokens",
        "xVal-Aff time tokens": "xVal-Aff\ntime\ntokens",
        "VentC U": "VentC\nU",
        "VentC F": "VentC\nF",
        "TrentC U": "TrentC\nU",
        "TrentC F": "TrentC\nF",
        "CLIF-mapped": "CLIF-\nmapped",
        "Native MIMIC codes": "Native\nMIMIC\ncodes",
        "Randomized mapped codes": "Randomized\nmapped\ncodes",
        "Frequency-matched mapped codes": "Frequency-\nmatched\nmapped codes",
    }
    return replacements.get(label, label)


def _wrap_legend_label(label: str, width: int = 24) -> str:
    if "\n" in label:
        return label
    return "\n".join(textwrap.wrap(label, width=width, break_long_words=False))


def _forest_color_handles(items: list[tuple[str, str]]) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            color=color,
            markerfacecolor=color,
            markeredgecolor="#222222",
            linewidth=0,
            markersize=4.5,
            label=_wrap_legend_label(label, width=26),
        )
        for label, color in items
    ]


def _forest_marker_handles(items: list[tuple[str, str, float]]) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker=marker,
            color="#444444",
            markerfacecolor="#444444",
            markeredgecolor="#222222",
            linewidth=0,
            markersize=4.5,
            label=label,
            alpha=alpha,
        )
        for label, marker, alpha in items
    ]


def _legend_handles_for_row_major_display(handles: list[Line2D], ncol: int) -> list[Line2D]:
    if ncol <= 1:
        return handles
    nrows = (len(handles) + ncol - 1) // ncol
    ordered = []
    for col in range(ncol):
        for row in range(nrows):
            idx = row * ncol + col
            if idx < len(handles):
                ordered.append(handles[idx])
    return ordered


def _add_forest_knob_legends(
    fig: plt.Figure,
    *,
    left_handles: list[Line2D],
    left_title: str,
    right_handles: list[Line2D],
    right_title: str,
    top_edge: float,
    left_ncol: int,
    right_ncol: int,
    left_width: float = 0.62,
) -> None:
    right_x = 0.02 + left_width + 0.04
    right_width = max(0.1, 0.98 - right_x)
    fig.legend(
        handles=_legend_handles_for_row_major_display(left_handles, left_ncol),
        title=left_title,
        loc="upper left",
        bbox_to_anchor=(0.02, top_edge, left_width, FOREST_LEGEND_BOX_HEIGHT),
        ncol=left_ncol,
        mode="expand",
        frameon=False,
        fontsize=FOREST_LEGEND_SIZE,
        title_fontsize=FOREST_LEGEND_SIZE,
        columnspacing=FOREST_LEGEND_COLSPACING,
        labelspacing=FOREST_LEGEND_LABELSPACING,
        handletextpad=0.25,
        borderpad=0.0,
    )
    fig.legend(
        handles=_legend_handles_for_row_major_display(right_handles, right_ncol),
        title=right_title,
        loc="upper left",
        bbox_to_anchor=(right_x, top_edge, right_width, FOREST_LEGEND_BOX_HEIGHT),
        ncol=right_ncol,
        mode="expand",
        frameon=False,
        fontsize=FOREST_LEGEND_SIZE,
        title_fontsize=FOREST_LEGEND_SIZE,
        columnspacing=FOREST_LEGEND_COLSPACING,
        labelspacing=FOREST_LEGEND_LABELSPACING,
        handletextpad=0.25,
        borderpad=0.0,
    )


def _render_exp1_granularity_figure(metrics: pd.DataFrame, figsize: tuple[float, float]) -> plt.Figure:
    handle_order = [handle for handle, _, _ in EXP1_ORDER]
    handle_map = {handle: (granularity, tokenization) for handle, granularity, tokenization in EXP1_ORDER}
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=False)
    _plot_exp1_granularity_panel(
        axes[0],
        metrics,
        outcome_order=BINARY_OUTCOME_ORDER_EXP12,
        families=["exp1_primary_binary", "exp1_additional_binary"],
        metric="roc_auc",
        handle_order=handle_order,
        handle_map=handle_map,
        ylab_map=FOREST_OUTCOME_LABELS,
        xlabel="AUROC",
        panel_title="Binary outcomes",
        x_lo_floor=FOREST_BINARY_AUROC_X_LO_FLOOR,
    )
    _plot_exp1_granularity_panel(
        axes[1],
        metrics,
        outcome_order=REGRESSION_OUTCOME_ORDER,
        families=["exp1_length_of_stay", "exp1_extended_regression"],
        metric="spearman_rho",
        handle_order=handle_order,
        handle_map=handle_map,
        ylab_map=FOREST_OUTCOME_LABELS,
        xlabel=r"Spearman $\rho$",
        panel_title="Regression outcomes",
        x_lo_floor=None,
    )
    color_handles = _forest_color_handles([
        (EXP1_COLOR_LEGEND_LABELS[label], color) for label, color in EXP1_COLORS.items()
    ])
    marker_handles = _forest_marker_handles([
        ("Unfused", "o", 0.55),
        ("Fused", "s", 0.95),
    ])
    top_edge = FOREST_LEGEND_TOP_EDGE
    fig.tight_layout(rect=[0, 0, 1, FOREST_EXP1_LAYOUT_TOP_EDGE])
    _add_forest_knob_legends(
        fig,
        left_handles=color_handles,
        left_title="Granularity (color)",
        right_handles=marker_handles,
        right_title="Tokenization (marker)",
        top_edge=top_edge,
        left_ncol=3,
        right_ncol=2,
        left_width=0.68,
    )
    return fig


def build_exp1_figure(
    metrics: pd.DataFrame,
    pairwise: pd.DataFrame,
    out_dir: Path,
    *,
    shared_vmax: dict[str, float],
) -> None:
    handle_order = [handle for handle, _, _ in EXP1_ORDER]
    labels = EXP1_HANDLE_SHORT_LABELS.copy()
    labels["deciles_unfused"] = "Dec U\nbase"
    binary = _baseline_delta_source(
        metrics,
        pairwise,
        families=["exp1_primary_binary", "exp1_additional_binary"],
        metric="roc_auc",
        outcome_order=BINARY_OUTCOME_ORDER_EXP12,
        handle_order=handle_order,
        baseline_handle="deciles_unfused",
        panel="binary",
    )
    regression = _baseline_delta_source(
        metrics,
        pairwise,
        families=["exp1_length_of_stay", "exp1_extended_regression"],
        metric="spearman_rho",
        outcome_order=REGRESSION_OUTCOME_ORDER,
        handle_order=handle_order,
        baseline_handle="deciles_unfused",
        panel="regression",
    )
    source = pd.concat([binary, regression], ignore_index=True)
    _save_source(source, out_dir / "sources" / "exp1_granularities_source.csv")

    fig, axes = plt.subplots(1, 2, figsize=(16.2, 10.0), sharey=False)
    _plot_delta_heatmap_panel(
        axes[0],
        binary,
        outcome_order=BINARY_OUTCOME_ORDER_EXP12,
        handle_order=handle_order,
        handle_labels=labels,
        panel_title="Binary outcomes",
        colorbar_label=r"$\Delta$ AUROC vs. deciles_unfused",
        group_breaks=[1, 3, 5, 7, 9],
        fixed_vmax=shared_vmax["binary"],
        cbar_ticks=[-0.15, -0.10, -0.05, 0.00, 0.05, 0.10, 0.15],
    )
    _plot_delta_heatmap_panel(
        axes[1],
        regression,
        outcome_order=REGRESSION_OUTCOME_ORDER,
        handle_order=handle_order,
        handle_labels=labels,
        panel_title="Regression outcomes",
        colorbar_label=r"$\Delta$ Spearman $\rho$ vs. deciles_unfused",
        group_breaks=[1, 3, 5, 7, 9],
        fixed_vmax=shared_vmax["regression"],
        cbar_ticks=[-0.20, -0.10, 0.00, 0.10, 0.20],
    )
    fig.tight_layout()
    fig.savefig(out_dir / "exp1_granularities.pdf", bbox_inches="tight")
    plt.close(fig)


def build_exp2_regression_figure(
    metrics: pd.DataFrame,
    pairwise: pd.DataFrame,
    out_dir: Path,
    *,
    shared_vmax: dict[str, float],
) -> None:
    labels = EXP2_HANDLE_SHORT_LABELS.copy()
    labels["discrete_none"] = "Disc\nevent\norder\nbase"
    binary = _baseline_delta_source(
        metrics,
        pairwise,
        families=["exp2_primary_binary", "exp2_additional_binary"],
        metric="roc_auc",
        outcome_order=BINARY_OUTCOME_ORDER_EXP12,
        handle_order=EXP2_HANDLE_ORDER,
        baseline_handle="discrete_none",
        panel="binary",
    )
    regression = _baseline_delta_source(
        metrics,
        pairwise,
        families=["exp2_length_of_stay", "exp2_extended_regression"],
        metric="spearman_rho",
        outcome_order=REGRESSION_OUTCOME_ORDER,
        handle_order=EXP2_HANDLE_ORDER,
        baseline_handle="discrete_none",
        panel="regression",
    )
    source = pd.concat([binary, regression], ignore_index=True)
    _save_source(source, out_dir / "sources" / "exp2_encoding_mechanics_source.csv")

    fig, axes = plt.subplots(1, 2, figsize=(18.2, 9.8), sharey=False)
    _plot_delta_heatmap_panel(
        axes[0],
        binary,
        outcome_order=BINARY_OUTCOME_ORDER_EXP12,
        handle_order=EXP2_HANDLE_ORDER,
        handle_labels=labels,
        panel_title="Binary outcomes",
        colorbar_label=r"$\Delta$ AUROC vs. Discrete + event order only",
        group_breaks=[3, 7],
        fixed_vmax=shared_vmax["binary"],
        cbar_ticks=[-0.15, -0.10, -0.05, 0.00, 0.05, 0.10, 0.15],
    )
    _plot_delta_heatmap_panel(
        axes[1],
        regression,
        outcome_order=REGRESSION_OUTCOME_ORDER,
        handle_order=EXP2_HANDLE_ORDER,
        handle_labels=labels,
        panel_title="Regression outcomes",
        colorbar_label=r"$\Delta$ Spearman $\rho$ vs. Discrete + event order only",
        group_breaks=[3, 7],
        fixed_vmax=shared_vmax["regression"],
        cbar_ticks=[-0.20, -0.10, 0.00, 0.10, 0.20],
    )
    fig.tight_layout()
    fig.savefig(out_dir / "exp2_encoding_mechanics.pdf", bbox_inches="tight")
    plt.close(fig)


def build_exp3_binary_figure(
    metrics: pd.DataFrame,
    pairwise: pd.DataFrame,
    out_dir: Path,
    *,
    shared_vmax: dict[str, float],
) -> None:
    labels = EXP3_HANDLE_SHORT_LABELS.copy()
    labels["meds"] = "Native\nMIMIC\ncodes\nbase"
    binary = _baseline_delta_source(
        metrics,
        pairwise,
        families=["exp3_primary_binary", "exp3_additional_binary"],
        metric="roc_auc",
        outcome_order=BINARY_OUTCOME_ORDER_EXP3,
        handle_order=EXP3_HANDLE_ORDER,
        baseline_handle="meds",
        panel="binary",
    )
    regression = _baseline_delta_source(
        metrics,
        pairwise,
        families=["exp3_length_of_stay", "exp3_extended_regression"],
        metric="spearman_rho",
        outcome_order=REGRESSION_OUTCOME_ORDER,
        handle_order=EXP3_HANDLE_ORDER,
        baseline_handle="meds",
        panel="regression",
    )
    source = pd.concat([binary, regression], ignore_index=True)
    _save_source(source, out_dir / "sources" / "exp3_binary_expansion_source.csv")

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 9.8), sharey=False)
    _plot_delta_heatmap_panel(
        axes[0],
        binary,
        outcome_order=BINARY_OUTCOME_ORDER_EXP3,
        handle_order=EXP3_HANDLE_ORDER,
        handle_labels=labels,
        panel_title="Binary outcomes",
        colorbar_label=r"$\Delta$ AUROC vs. Native MIMIC codes",
        fixed_vmax=shared_vmax["binary"],
        cbar_ticks=[-0.15, -0.10, -0.05, 0.00, 0.05, 0.10, 0.15],
    )
    _plot_delta_heatmap_panel(
        axes[1],
        regression,
        outcome_order=REGRESSION_OUTCOME_ORDER,
        handle_order=EXP3_HANDLE_ORDER,
        handle_labels=labels,
        panel_title="Regression outcomes",
        colorbar_label=r"$\Delta$ Spearman $\rho$ vs. Native MIMIC codes",
        fixed_vmax=shared_vmax["regression"],
        cbar_ticks=[-0.20, -0.10, 0.00, 0.10, 0.20],
    )
    fig.tight_layout()
    fig.savefig(out_dir / "exp3_binary_expansion.pdf", bbox_inches="tight")
    plt.close(fig)


def build_centile_pca_grid(out_dir: Path) -> None:
    model_root = ROOT / "outputs" / "runs" / "models"
    token_root = ROOT / "outputs" / "runs" / "tokenized" / "mimiciv-3.1_meds_70-10-20"
    num_bins = 100

    shared_model = _latest_checkpoint(
        model_root / "exp1_meds_centiles_none_fusedFalse_discrete_time_tokens-s42" / "run-0"
    )
    shared_vocab = token_root / "centiles_none_unfused_time_tokens_first_24h-tokenized" / "train" / "vocab.gzip"
    fused_model = _latest_checkpoint(
        model_root / "exp1_meds_centiles_none_fusedTrue_discrete_time_tokens-s42" / "run-0"
    )
    fused_vocab = token_root / "centiles_none_fused_time_tokens_first_24h-tokenized" / "train" / "vocab.gzip"

    shared_emb = extract_discrete_bin_embeddings(
        shared_model,
        load_vocab(shared_vocab),
        num_bins=num_bins,
    )["shared_Q_tokens"]
    shared_bins = list(range(num_bins))
    shared_coords, shared_evr = pca_2d(shared_emb)

    source_rows: list[dict[str, object]] = []
    fig = plt.figure(figsize=PCA_GRID_FIGSIZE)
    gs = fig.add_gridspec(
        2,
        4,
        height_ratios=[1.0, 1.0],
        width_ratios=[1.12, 1.0, 1.0, 1.0],
        left=0.07,
        right=0.99,
        top=0.93,
        bottom=0.22,
        wspace=0.28,
        hspace=0.28,
    )
    shared_ax = fig.add_subplot(gs[0:2, 0])
    fused_axes = [
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[0, 3]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[1, 2]),
        fig.add_subplot(gs[1, 3]),
    ]
    cmap = plt.cm.RdYlGn_r

    shared_colors = [cmap(i / max(num_bins - 1, 1)) for i in shared_bins]
    shared_ax.scatter(
        shared_coords[:, 0],
        shared_coords[:, 1],
        c=shared_colors,
        s=PCA_GRID_SHARED_MARKER_SIZE,
        zorder=4,
        edgecolors="#222222",
        linewidths=0.3,
    )
    shared_ax.plot(shared_coords[:, 0], shared_coords[:, 1], "k--", alpha=0.25, linewidth=0.7)
    for idx, bin_id in enumerate(shared_bins):
        source_rows.append(
            {
                "measurement": "Shared centile vocabulary",
                "measurement_code": "shared_Q_tokens",
                "panel": "shared_centiles",
                "bin_id": bin_id,
                "pc1": shared_coords[idx, 0],
                "pc2": shared_coords[idx, 1],
                "pc1_var_explained": float(shared_evr[0]),
                "pc2_var_explained": float(shared_evr[1]),
                "model_dir": str(shared_model),
                "vocab_path": str(shared_vocab),
            }
        )
    for idx, bin_id in enumerate(shared_bins):
        if bin_id not in _selected_bin_labels(shared_bins):
            continue
        shared_ax.annotate(
            f"Q{bin_id}",
            (shared_coords[idx, 0], shared_coords[idx, 1]),
            textcoords="offset points",
            xytext=(4, 2),
            fontsize=PCA_GRID_ANNOT_SIZE,
        )
    shared_ax.set_title(
        "Shared centile tokens\n(one manifold for all numeric variables)",
        fontsize=PCA_GRID_TITLE_SIZE,
        fontweight="bold",
    )
    shared_ax.set_xlabel(f"PC1 ({100 * shared_evr[0]:.1f}% var)", fontsize=PCA_GRID_AXIS_LABEL_SIZE)
    shared_ax.set_ylabel(f"PC2 ({100 * shared_evr[1]:.1f}% var)", fontsize=PCA_GRID_AXIS_LABEL_SIZE)
    shared_ax.tick_params(axis="both", labelsize=PCA_GRID_TICK_SIZE)
    shared_ax.grid(alpha=0.22)

    for ax, (measurement_name, measurement_code) in zip(fused_axes, PCA_MEASUREMENTS):
        fused_bins, fused_emb = _extract_fused_sparse(
            model_dir=fused_model,
            vocab_path=fused_vocab,
            code=measurement_code,
            num_bins=num_bins,
        )
        fused_coords, fused_evr = pca_2d(fused_emb)
        colors = [cmap(i / max(num_bins - 1, 1)) for i in fused_bins]
        ax.scatter(
            fused_coords[:, 0],
            fused_coords[:, 1],
            c=colors,
            s=PCA_GRID_FUSED_MARKER_SIZE,
            zorder=4,
            edgecolors="#222222",
            linewidths=0.3,
        )
        ax.plot(fused_coords[:, 0], fused_coords[:, 1], "k--", alpha=0.25, linewidth=0.6)
        for idx, bin_id in enumerate(fused_bins):
            source_rows.append(
                {
                    "measurement": measurement_name,
                    "measurement_code": measurement_code,
                    "panel": "fused_centiles",
                    "bin_id": bin_id,
                    "pc1": fused_coords[idx, 0],
                    "pc2": fused_coords[idx, 1],
                    "pc1_var_explained": float(fused_evr[0]),
                    "pc2_var_explained": float(fused_evr[1]),
                    "model_dir": str(fused_model),
                    "vocab_path": str(fused_vocab),
                }
            )
        for idx, bin_id in enumerate(fused_bins):
            if bin_id not in _selected_bin_labels(fused_bins):
                continue
            ax.annotate(
                f"Q{bin_id}",
                (fused_coords[idx, 0], fused_coords[idx, 1]),
                textcoords="offset points",
                xytext=(3, 2),
                fontsize=PCA_GRID_ANNOT_SIZE,
            )
        ax.set_title(measurement_name, fontsize=PCA_GRID_TITLE_SIZE, fontweight="bold")
        ax.set_xlabel(f"PC1 ({100 * fused_evr[0]:.1f}% var)", fontsize=PCA_GRID_AXIS_LABEL_SIZE)
        ax.set_ylabel(f"PC2 ({100 * fused_evr[1]:.1f}% var)", fontsize=PCA_GRID_AXIS_LABEL_SIZE)
        ax.tick_params(axis="both", labelsize=PCA_GRID_TICK_SIZE)
        ax.grid(alpha=0.22)
        ax.text(
            0.97,
            0.05,
            f"{len(fused_bins)} realized bins",
            transform=ax.transAxes,
            fontsize=PCA_GRID_PANEL_NOTE_SIZE,
            fontweight="semibold",
            va="bottom",
            ha="right",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.8},
        )

    mappable = plt.cm.ScalarMappable(norm=Normalize(vmin=0, vmax=num_bins - 1), cmap=cmap)
    cbar_ax = fig.add_axes([0.12, 0.07, 0.76, 0.032])
    cbar = fig.colorbar(mappable, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Centile index (quantile bin id)", fontsize=PCA_GRID_CBAR_LABEL_SIZE)
    cbar.ax.tick_params(labelsize=PCA_GRID_CBAR_TICK_SIZE)
    _save_source(pd.DataFrame(source_rows), out_dir / "sources" / "pca_centile_geometry_grid_source.csv")
    fig.savefig(out_dir / "pca_centile_geometry_grid.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "pca_centile_geometry_grid.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_exp1_outcome_forests(metrics: pd.DataFrame, out_dir: Path) -> None:
    _save_source(
        _collect_exp1_source(metrics),
        out_dir / "sources" / "exp1_outcome_forests_source.csv",
    )
    fig = _render_exp1_granularity_figure(metrics, FOREST_FIGSIZE)
    fig.savefig(out_dir / "exp1_outcome_forests.pdf", bbox_inches="tight")
    plt.close(fig)


def build_exp2_outcome_forests(metrics: pd.DataFrame, out_dir: Path) -> None:
    source = _collect_exp2_source(metrics)
    _save_source(
        source,
        out_dir / "sources" / "exp2_outcome_forests_source.csv",
    )
    fig, axes = plt.subplots(1, 2, figsize=FOREST_FIGSIZE, sharey=False)
    bin_src = source[source["panel"] == "binary"]
    reg_src = source[source["panel"] == "regression"]
    _plot_handle_metric_panel(
        axes[0],
        bin_src,
        outcome_order=BINARY_OUTCOME_ORDER_EXP12,
        handle_order=EXP2_HANDLE_ORDER,
        handle_colors=EXP2_HANDLE_COLORS,
        marker_for_handle=_exp2_marker,
        panel_title="Binary outcomes",
        xlabel="AUROC",
        x_lo_floor=FOREST_BINARY_AUROC_X_LO_FLOOR,
    )
    _plot_handle_metric_panel(
        axes[1],
        reg_src,
        outcome_order=REGRESSION_OUTCOME_ORDER,
        handle_order=EXP2_HANDLE_ORDER,
        handle_colors=EXP2_HANDLE_COLORS,
        marker_for_handle=_exp2_marker,
        panel_title="Regression outcomes",
        xlabel=r"Spearman $\rho$",
    )
    marker_handles = _forest_marker_handles(EXP2_TEMPORAL_MARKER_ITEMS)
    color_handles = _forest_color_handles(EXP2_VALUE_COLOR_ITEMS)
    top_edge = FOREST_LEGEND_TOP_EDGE
    fig.tight_layout(rect=[0, 0, 1, FOREST_LAYOUT_TOP_EDGE])
    _add_forest_knob_legends(
        fig,
        left_handles=marker_handles,
        left_title="Temporal encoding (marker)",
        right_handles=color_handles,
        right_title="Value encoder (color)",
        top_edge=top_edge,
        left_ncol=3,
        right_ncol=2,
        left_width=0.50,
    )
    fig.savefig(out_dir / "exp2_outcome_forests.pdf", bbox_inches="tight")
    plt.close(fig)


def build_exp3_appendix_figure(metrics: pd.DataFrame, out_dir: Path) -> None:
    source = _collect_exp3_source(metrics)
    _save_source(
        source,
        out_dir / "sources" / "appendix_exp3_outcome_forests_source.csv",
    )
    fig, axes = plt.subplots(1, 2, figsize=FOREST_FIGSIZE, sharey=False)
    bin_src = source[source["panel"] == "binary"]
    reg_src = source[source["panel"] == "regression"]
    _plot_handle_metric_panel(
        axes[0],
        bin_src,
        outcome_order=BINARY_OUTCOME_ORDER_EXP3,
        handle_order=EXP3_HANDLE_ORDER,
        handle_colors=EXP3_HANDLE_COLORS,
        marker_for_handle=_exp3_marker,
        panel_title="Binary outcomes",
        xlabel="AUROC",
        x_lo_floor=FOREST_BINARY_AUROC_X_LO_FLOOR,
    )
    _plot_handle_metric_panel(
        axes[1],
        reg_src,
        outcome_order=REGRESSION_OUTCOME_ORDER,
        handle_order=EXP3_HANDLE_ORDER,
        handle_colors=EXP3_HANDLE_COLORS,
        marker_for_handle=_exp3_marker,
        panel_title="Regression outcomes",
        xlabel=r"Spearman $\rho$",
    )
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=_exp3_marker(h),
            color=EXP3_HANDLE_COLORS[h],
            markerfacecolor=EXP3_HANDLE_COLORS[h],
            markeredgecolor="#222222",
            linewidth=0,
            markersize=4.2,
            label=EXP3_FOREST_LABELS[h],
        )
        for h in EXP3_HANDLE_ORDER
    ]
    top_edge = FOREST_LEGEND_TOP_EDGE
    fig.tight_layout(rect=[0, 0, 1, FOREST_LAYOUT_TOP_EDGE])
    fig.legend(
        handles=legend_handles,
        title="Vocabulary arm",
        loc="upper left",
        bbox_to_anchor=(0.02, top_edge, 0.96, FOREST_LEGEND_BOX_HEIGHT),
        ncol=4,
        mode="expand",
        frameon=False,
        fontsize=FOREST_LEGEND_SIZE,
        title_fontsize=FOREST_LEGEND_SIZE,
        columnspacing=FOREST_LEGEND_COLSPACING,
        labelspacing=FOREST_LEGEND_LABELSPACING,
        handletextpad=0.25,
        borderpad=0.0,
    )
    fig.savefig(out_dir / "appendix_exp3_outcome_forests.pdf", bbox_inches="tight")
    plt.close(fig)


def build_exp1_appendix_summary_bars(metrics: pd.DataFrame, out_dir: Path) -> None:
    source = _exp1_axis_summary_source(metrics)
    _save_source(source, out_dir / "sources" / "appendix_exp1_axis_summary_bars_source.csv")
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 7.4))
    axis_specs = [
        ("Fusion", ["Unfused", "Fused"], EXP1_FUSION_SUMMARY_COLORS, None),
        ("Granularity", ["Deciles", "Ventiles", "Trentiles", "Centiles"], EXP1_GRANULARITY_SUMMARY_COLORS, None),
        (
            "Anchoring",
            ["Population bins", "Clinical bins"],
            EXP1_ANCHOR_SUMMARY_COLORS,
            {
                "Population bins": "Population\nbins",
                "Clinical bins": "Clinical\nbins",
            },
        ),
    ]
    panel_specs = [
        ("binary", "Mean AUROC across outcomes", 0.58),
        ("regression", r"Mean Spearman $\rho$ across outcomes", 0.0),
    ]
    for row_idx, (panel, ylabel, floor) in enumerate(panel_specs):
        for col_idx, (axis, order, colors, display) in enumerate(axis_specs):
            _plot_summary_bar_panel(
                axes[row_idx, col_idx],
                source[(source["panel"] == panel) & (source["axis"] == axis)],
                level_order=order,
                color_map=colors,
                title=axis if row_idx == 0 else "",
                ylabel=ylabel if col_idx == 0 else "",
                y_floor=floor,
                display_labels=display,
            )
    fig.tight_layout(pad=1.15, h_pad=1.75, w_pad=1.35)
    fig.savefig(out_dir / "appendix_exp1_axis_summary_bars.pdf", bbox_inches="tight")
    plt.close(fig)


def build_exp2_appendix_summary_bars(metrics: pd.DataFrame, out_dir: Path) -> None:
    source = _exp2_axis_summary_source(metrics)
    _save_source(source, out_dir / "sources" / "appendix_exp2_axis_summary_bars_source.csv")
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 7.4))
    axis_specs = [
        (
            "Value encoder",
            ["Discrete", "Soft", "xVal (code-normalized)", "xVal-affine (code-normalized + affine shift)"],
            EXP2_VALUE_SUMMARY_COLORS,
            {
                "xVal (code-normalized)": "xVal\n(code-norm.)",
                "xVal-affine (code-normalized + affine shift)": "xVal-affine",
            },
        ),
        (
            "Temporal encoding",
            ["Event order only", "Time tokens", "Admission-relative RoPE"],
            EXP2_TEMPORAL_SUMMARY_COLORS,
            {
                "Event order only": "Event\norder\nonly",
                "Time tokens": "Time\ntokens",
                "Admission-relative RoPE": "Admission-\nrelative\nRoPE",
            },
        ),
    ]
    panel_specs = [
        ("binary", "Mean AUROC across outcomes", 0.58),
        ("regression", r"Mean Spearman $\rho$ across outcomes", 0.0),
    ]
    for row_idx, (panel, ylabel, floor) in enumerate(panel_specs):
        for col_idx, (axis, order, colors, display) in enumerate(axis_specs):
            _plot_summary_bar_panel(
                axes[row_idx, col_idx],
                source[(source["panel"] == panel) & (source["axis"] == axis)],
                level_order=order,
                color_map=colors,
                title=axis if row_idx == 0 else "",
                ylabel=ylabel if col_idx == 0 else "",
                y_floor=floor,
                display_labels=display,
            )
    fig.tight_layout(pad=1.15, h_pad=1.75, w_pad=1.45)
    fig.savefig(out_dir / "appendix_exp2_axis_summary_bars.pdf", bbox_inches="tight")
    plt.close(fig)


def build_exp3_appendix_summary_bars(metrics: pd.DataFrame, out_dir: Path) -> None:
    source = _exp3_axis_summary_source(metrics)
    _save_source(source, out_dir / "sources" / "appendix_exp3_axis_summary_bars_source.csv")
    fig, axes = plt.subplots(2, 1, figsize=(5.4, 7.7))
    panel_specs = [
        ("binary", "Mean AUROC across outcomes", 0.58),
        ("regression", r"Mean Spearman $\rho$ across outcomes", 0.0),
    ]
    display = {
        "CLIF-mapped": "CLIF-mapped",
        "Native MIMIC codes": "Native MIMIC",
        "Randomized mapped codes": "Randomized",
        "Frequency-matched mapped codes": "Freq.-matched",
    }
    for ax, (panel, ylabel, floor) in zip(axes, panel_specs):
        _plot_summary_bar_panel(
            ax,
            source[source["panel"] == panel],
            level_order=[
                "Native MIMIC codes",
                "CLIF-mapped",
                "Randomized mapped codes",
                "Frequency-matched mapped codes",
            ],
            color_map={
                "Native MIMIC codes": EXP3_HANDLE_COLORS["meds"],
                "CLIF-mapped": EXP3_HANDLE_COLORS["mapped"],
                "Randomized mapped codes": EXP3_HANDLE_COLORS["randomized"],
                "Frequency-matched mapped codes": EXP3_HANDLE_COLORS["freqmatched"],
            },
            title="Vocabulary arm" if ax is axes[0] else "",
            ylabel=ylabel,
            y_floor=floor,
            display_labels=display,
        )
    fig.tight_layout(pad=1.15, h_pad=1.65)
    fig.savefig(out_dir / "appendix_exp3_axis_summary_bars.pdf", bbox_inches="tight")
    plt.close(fig)


def build_statistical_trend_figure(metrics: pd.DataFrame, out_dir: Path) -> None:
    source = _handle_average_summary(metrics)
    _save_source(source, out_dir / "sources" / "statistical_trend_summary_source.csv")

    fig, axes = plt.subplots(2, 3, figsize=(20.4, 8.4), sharey="row")
    experiments = ["Exp1", "Exp2", "Exp3"]
    panel_specs = [
        ("binary", "Mean AUROC across outcomes", 0.58),
        ("regression", r"Mean Spearman $\rho$ across outcomes", 0.0),
    ]
    for col_idx, experiment in enumerate(experiments):
        for row_idx, (panel, ylabel, floor_pref) in enumerate(panel_specs):
            ax = axes[row_idx, col_idx]
            sub = source[
                (source["experiment"] == experiment) & (source["panel"] == panel)
            ].sort_values("order")
            x = np.arange(len(sub), dtype=float)
            points = sub["point"].to_numpy(dtype=float)
            ci_lo = sub["ci_lo"].to_numpy(dtype=float)
            ci_hi = sub["ci_hi"].to_numpy(dtype=float)
            err = np.vstack([points - ci_lo, ci_hi - points])
            ax.bar(
                x,
                points,
                width=0.72,
                color=sub["color"].tolist(),
                edgecolor="#333333",
                linewidth=0.9,
                zorder=3,
            )
            ax.errorbar(
                x,
                points,
                yerr=err,
                fmt="none",
                ecolor="#222222",
                elinewidth=1.1,
                capsize=3,
                zorder=4,
            )
            ax.set_xticks(x)
            ax.set_xticklabels(
                [_wrap_bar_tick_label(label) for label in sub["label"].tolist()],
                rotation=0,
                ha="center",
                fontsize=10,
            )
            ax.set_title(experiment if row_idx == 0 else "")
            ax.set_ylabel(ylabel if col_idx == 0 else "")
            ax.grid(axis="y", alpha=0.25, zorder=0)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if panel == "binary":
                ax.axhline(0.5, color="#666666", linewidth=0.8, alpha=0.45, linestyle="--")
            lower = float(np.nanmin(ci_lo)) - 0.02
            floor = min(float(floor_pref), lower)
            upper = float(np.nanmax(ci_hi))
            pad = max(0.02, 0.10 * max(upper - floor, 0.08))
            ax.set_ylim(floor, upper + pad)
    fig.tight_layout()
    fig.savefig(out_dir / "statistical_trend_summary.pdf", bbox_inches="tight")
    plt.close(fig)


def build_legacy_statistical_count_summary(pairwise: pd.DataFrame, out_dir: Path) -> None:
    exp1_specs = EXP1_TREND_COMPARISONS
    exp2_specs = [
        ("Discrete event order -> Discrete time tokens", "discrete_none", "discrete_tt"),
        ("Discrete event order -> Discrete admission-relative RoPE", "discrete_none", "discrete_rope"),
        ("Discrete event order -> Soft event order", "discrete_none", "soft_none"),
        ("Discrete admission-relative RoPE -> Soft admission-relative RoPE", "discrete_rope", "soft_rope"),
        ("Discrete event order -> xVal-CN event order", "discrete_none", "xval_none"),
        ("Discrete admission-relative RoPE -> xVal-CN admission-relative RoPE", "discrete_rope", "xval_rope"),
        ("xVal-CN event order -> xVal-Aff event order", "xval_none", "xval_affine_none"),
        ("xVal-CN admission-relative RoPE -> xVal-Aff admission-relative RoPE", "xval_rope", "xval_affine_rope"),
    ]
    exp3_specs = [
        ("Native MIMIC codes -> CLIF-mapped", "meds", "mapped"),
        ("Native MIMIC codes -> Randomized mapped codes", "meds", "randomized"),
        ("Native MIMIC codes -> Frequency-matched mapped codes", "meds", "freqmatched"),
    ]

    rows: list[dict[str, object]] = []
    for label, handle0, handle1 in exp1_specs:
        rows.append(
            {
                "experiment": "Exp1",
                "comparison": label,
                "Binary outcomes": _count_favorable_outcomes(
                    pairwise,
                    families=["exp1_primary_binary", "exp1_additional_binary"],
                    metric="roc_auc",
                    handle0=handle0,
                    handle1=handle1,
                ),
                "Regression outcomes": _count_favorable_outcomes(
                    pairwise,
                    families=["exp1_length_of_stay", "exp1_extended_regression"],
                    metric="spearman_rho",
                    handle0=handle0,
                    handle1=handle1,
                ),
            }
        )
    for label, handle0, handle1 in exp2_specs:
        rows.append(
            {
                "experiment": "Exp2",
                "comparison": label,
                "Binary outcomes": _count_favorable_outcomes(
                    pairwise,
                    families=["exp2_primary_binary", "exp2_additional_binary"],
                    metric="roc_auc",
                    handle0=handle0,
                    handle1=handle1,
                ),
                "Regression outcomes": _count_favorable_outcomes(
                    pairwise,
                    families=["exp2_length_of_stay", "exp2_extended_regression"],
                    metric="spearman_rho",
                    handle0=handle0,
                    handle1=handle1,
                ),
            }
        )
    for label, handle0, handle1 in exp3_specs:
        rows.append(
            {
                "experiment": "Exp3",
                "comparison": label,
                "Binary outcomes": _count_favorable_outcomes(
                    pairwise,
                    families=["exp3_primary_binary", "exp3_additional_binary"],
                    metric="roc_auc",
                    handle0=handle0,
                    handle1=handle1,
                ),
                "Regression outcomes": _count_favorable_outcomes(
                    pairwise,
                    families=["exp3_length_of_stay", "exp3_extended_regression"],
                    metric="spearman_rho",
                    handle0=handle0,
                    handle1=handle1,
                ),
            }
        )

    source = pd.DataFrame(rows)
    _save_source(source, out_dir / "sources" / "statistical_trend_summary_source.csv")

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.9), sharey=True)
    width = 0.34
    for ax, experiment in zip(axes, ["Exp1", "Exp2", "Exp3"]):
        sub = source[source["experiment"] == experiment].copy().reset_index(drop=True)
        x = np.arange(len(sub))
        binary_vals = sub["Binary outcomes"].to_numpy()
        reg_vals = sub["Regression outcomes"].to_numpy()
        ax.bar(
            x - width / 2,
            binary_vals,
            width=width,
            color=TREND_COLORS["Binary outcomes"],
            edgecolor="#333333",
            label="Binary outcomes",
        )
        ax.bar(
            x + width / 2,
            reg_vals,
            width=width,
            color=TREND_COLORS["Regression outcomes"],
            edgecolor="#333333",
            label="Regression outcomes",
        )
        for xpos, val in zip(x - width / 2, binary_vals):
            ax.text(xpos, val + 0.12, str(int(val)), ha="center", va="bottom", fontsize=8)
        for xpos, val in zip(x + width / 2, reg_vals):
            ax.text(xpos, val + 0.12, str(int(val)), ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["comparison"], rotation=20, ha="right")
        ax.set_title(f"{experiment}: BH-supported favorable outcomes")
        ax.set_ylabel("Outcome count")
        ax.set_ylim(0, max(7, float(source[["Binary outcomes", "Regression outcomes"]].to_numpy().max()) + 1.2))
        ax.grid(axis="y", alpha=0.25)

    axes[-1].legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "statistical_trend_summary.pdf", bbox_inches="tight")
    plt.close(fig)


def build_maintext_selection_summary(metrics: pd.DataFrame, pairwise: pd.DataFrame, out_dir: Path) -> None:
    rows: list[dict[str, object]] = []

    exp1_specs = [
        ("Exp1", "binary", ["exp1_primary_binary", "exp1_additional_binary"], "roc_auc", BINARY_OUTCOME_ORDER_EXP12),
        ("Exp1", "regression", ["exp1_length_of_stay", "exp1_extended_regression"], "spearman_rho", REGRESSION_OUTCOME_ORDER),
    ]
    for experiment, panel, families, metric, outcome_order in exp1_specs:
        sub = metrics[
            metrics["family_name"].isin(families)
            & (metrics["metric"] == metric)
            & metrics["outcome"].isin(outcome_order)
        ].copy()
        pivot = sub.pivot(index="outcome", columns="handle", values="point").reindex(outcome_order)
        for outcome in outcome_order:
            row = pivot.loc[outcome]
            rows.append(
                {
                    "experiment": experiment,
                    "panel": panel,
                    "outcome": outcome,
                    "selection_signal": "range",
                    "score": float(row.max() - row.min()),
                    "best_handle": row.idxmax(),
                    "best_value": float(row.max()),
                    "sig_positive_count": np.nan,
                }
            )

    def _append_pairwise_summary(experiment: str, panel: str, families: list[str], metric: str, outcome_order: list[str], comparisons: list[tuple[str, str, str]]):
        for outcome in outcome_order:
            deltas = []
            sig_positive = 0
            for label, handle0, handle1 in comparisons:
                delta, _, _, p_adj = _extract_pairwise_effect(
                    pairwise,
                    families=families,
                    metric=metric,
                    outcome=outcome,
                    handle0=handle0,
                    handle1=handle1,
                )
                if not np.isfinite(delta):
                    continue
                deltas.append((label, delta))
                if np.isfinite(p_adj) and p_adj < 0.05 and delta > 0:
                    sig_positive += 1
            if deltas:
                best_label, best_delta = max(deltas, key=lambda item: item[1])
                worst_label, worst_delta = min(deltas, key=lambda item: item[1])
                spread = float(best_delta - worst_delta)
            else:
                best_label, best_delta = "", float("nan")
                worst_label, worst_delta = "", float("nan")
                spread = float("nan")
            rows.append(
                {
                    "experiment": experiment,
                    "panel": panel,
                    "outcome": outcome,
                    "selection_signal": "best_delta",
                    "score": float(best_delta),
                    "best_handle": best_label,
                    "best_value": float(best_delta),
                    "worst_handle": worst_label,
                    "worst_value": float(worst_delta),
                    "spread": spread,
                    "sig_positive_count": sig_positive,
                }
            )

    _append_pairwise_summary(
        "Exp2",
        "binary",
        ["exp2_primary_binary", "exp2_additional_binary"],
        "roc_auc",
        BINARY_OUTCOME_ORDER_EXP12,
        [(label, h0, h1) for label, h0, h1, _, _ in EXP2_EFFECT_COMPARISONS],
    )
    _append_pairwise_summary(
        "Exp2",
        "regression",
        ["exp2_length_of_stay", "exp2_extended_regression"],
        "spearman_rho",
        REGRESSION_OUTCOME_ORDER,
        [(label, h0, h1) for label, h0, h1, _, _ in EXP2_EFFECT_COMPARISONS],
    )
    _append_pairwise_summary(
        "Exp3",
        "binary",
        ["exp3_primary_binary", "exp3_additional_binary"],
        "roc_auc",
        BINARY_OUTCOME_ORDER_EXP3,
        [(label, h0, h1) for label, h0, h1, _, _ in EXP3_EFFECT_COMPARISONS],
    )
    _append_pairwise_summary(
        "Exp3",
        "regression",
        ["exp3_length_of_stay", "exp3_extended_regression"],
        "spearman_rho",
        REGRESSION_OUTCOME_ORDER,
        [(label, h0, h1) for label, h0, h1, _, _ in EXP3_EFFECT_COMPARISONS],
    )

    _save_source(pd.DataFrame(rows), out_dir / "sources" / "maintext_selection_summary.csv")


def build_length_histograms(out_dir: Path) -> None:
    """Build non-metric token-length histograms from tokenized parquet outputs."""
    script = ROOT / "utilities" / "qc" / "plot_length_histograms.py"
    base = DEFAULT_TOKEN_ROOT

    common_args = [
        sys.executable,
        str(script),
        "--base",
        str(base),
        "--split",
        "train",
    ]

    subprocess.run(
        common_args
        + [
            "--versions",
            "deciles_none_unfused_time_tokens",
            "deciles_none_fused_time_tokens",
            "deciles_none_unfused_time_tokens_first_24h",
            "deciles_none_fused_time_tokens_first_24h",
            "--labels",
            "unfused - full timeline",
            "fused - full timeline",
            "unfused - first 24h",
            "fused - first 24h",
            "--out_pdf",
            str(out_dir / "length_histograms_train.pdf"),
        ],
        check=True,
    )

    subprocess.run(
        common_args
        + [
            "--versions",
            "deciles_none_unfused_time_tokens",
            "deciles_none_unfused_time_rope",
            "deciles_none_unfused_time_tokens_first_24h",
            "deciles_none_unfused_time_rope_first_24h",
            "--labels",
            "time tokens — full timeline",
            "no time tokens (event order / admission-relative RoPE) — full timeline",
            "time tokens — first 24h",
            "no time tokens (event order / admission-relative RoPE) — first 24h",
            "--out_pdf",
            str(out_dir / "length_histograms_temporal_train.pdf"),
        ],
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_csv", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--pairwise_csv", type=Path, default=DEFAULT_PAIRWISE)
    parser.add_argument("--figures_dir", type=Path, default=DEFAULT_FIG_DIR)
    args = parser.parse_args()

    _style()
    metrics = pd.read_csv(args.metrics_csv)
    pairwise = pd.read_csv(args.pairwise_csv)
    figures_dir = args.figures_dir.expanduser().resolve()
    figures_dir.mkdir(parents=True, exist_ok=True)
    (figures_dir / "sources").mkdir(parents=True, exist_ok=True)
    shared_vmax = _compute_shared_heatmap_vmax(metrics, pairwise)

    build_exp1_figure(metrics, pairwise, figures_dir, shared_vmax=shared_vmax)
    build_exp2_regression_figure(metrics, pairwise, figures_dir, shared_vmax=shared_vmax)
    build_exp3_binary_figure(metrics, pairwise, figures_dir, shared_vmax=shared_vmax)
    build_centile_pca_grid(figures_dir)
    build_exp1_outcome_forests(metrics, figures_dir)
    build_exp2_outcome_forests(metrics, figures_dir)
    build_exp3_appendix_figure(metrics, figures_dir)
    build_exp1_appendix_summary_bars(metrics, figures_dir)
    build_exp2_appendix_summary_bars(metrics, figures_dir)
    build_exp3_appendix_summary_bars(metrics, figures_dir)
    build_statistical_trend_figure(metrics, figures_dir)
    build_maintext_selection_summary(metrics, pairwise, figures_dir)
    build_length_histograms(figures_dir)
    print(figures_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
