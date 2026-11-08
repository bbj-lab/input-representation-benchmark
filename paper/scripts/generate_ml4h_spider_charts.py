#!/usr/bin/env python3
"""Render baseline-delta ML4H spider figures and summary tables.

Each experiment uses the 24–48h horizon only. Experiments 1 and 2 are four
panels (architecture × separated condition); experiment 3 is two panels
(architecture only). Values are seed means as differences from the
experiment-specific baseline. Binary outcomes occupy the top semicircle and
regression outcomes the bottom.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DUMP_ROOT = REPO_ROOT.parent / "Archive/ml4h-dump"
DEFAULT_OUTPUT_ROOT = DEFAULT_DUMP_ROOT / "08_ML4H_provisional"
DEFAULT_PRIMARY_METRICS = DEFAULT_DUMP_ROOT / "00_data/metrics_long.parquet"
DEFAULT_PROVISIONAL_METRICS = DEFAULT_OUTPUT_ROOT / "00_data/metrics_long.parquet"
SPIDER_SECTION = "04_spider_charts"

# Okabe–Ito / Wong qualitative palette (CVD-safe).
OKABE_ITO = (
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#000000",  # black
    "#F0E442",  # yellow
)
SERIES_COLORS = OKABE_ITO
SERIES_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "h")
GRID_COLOR = "#5A5A5A"
RING_COLOR = "#2A2A2A"
TEXT_COLOR = "#1A1A1A"
TICK_COLOR = "#1A1A1A"

# Delta values map onto a fixed visual outer ring (DATA_RADIUS_MAX). Each panel
# column can use its own tick set; values outside that set are clipped/outlier-marked.
DELTA_TICKS = (-0.10, -0.05, 0.00, 0.05, 0.10)
DELTA_MIN = DELTA_TICKS[0]
DELTA_MAX = DELTA_TICKS[-1]
DATA_RADIUS_MAX = 1.12
# Spoke indices sit in a thin outer band; names live in the outcome key below.
# Leave a narrow annulus outside the data rim for outlier callouts.
DISPLAY_RADIUS_MAX = 1.38
LABEL_RADIUS_BASE = DATA_RADIUS_MAX + 0.085
LABEL_RADIUS_STEP = 0.050
LABEL_RADIUS_TIERS = 5
LABEL_RADIUS_CAP = DISPLAY_RADIUS_MAX - 0.02
LABEL_ANGLE_NUDGE_RADIANS = math.radians(3.5)
LABEL_NEIGHBOR_WINDOW = 2
DELTA_CAPTION_GAP = 0.16
HALF_GAP_RADIANS = math.radians(16.0)
# Extra radial length for dashed leaders when a spoke has outlier callouts.
OUTLIER_SPOKE_EXTRA = 0.22
PANEL_LETTERS = ("A", "B", "C", "D")
OUTCOME_LABEL_FONTSIZE = 18.0
OUTCOME_LABEL_WRAP = 12
OUTCOME_KEY_PT = 18.0
# Coreopsis reference uses ~22–42 pt on a ~14" canvas. Match that font-to-width
# ratio so spoke/legend text stays easily legible when the PDF is viewed.
NEURIPS_BODY_PT = 24.0
NEURIPS_SMALL_PT = 20.0
PANEL_LETTER_PT = 40.0
COLUMN_TITLE_PT = 26.0
LEGEND_PT = 20.0
TICK_LABEL_PT = 16.0
DELTA_CAPTION_PT = 16.0
OUTLIER_LABEL_PT = 15.0
SUMMARY_TABLE_TITLE_PT = 21.0
SUMMARY_TABLE_SUBTITLE_PT = 12.5
SUMMARY_TABLE_BODY_PT = 14.0
SUMMARY_TABLE_CONFIG_WIDTH = 0.56
SUMMARY_TABLE_METRIC_WIDTH = 0.11
SUMMARY_TABLE_AXES_BOUNDS = (0.025, 0.055, 0.95, 0.72)

# Physical figure sizes (inches): near page width so large pt sizes are not
# crushed by on-screen / paper downscaling of an oversized canvas.
FIGSIZE_EXP12 = (16.0, 20.5)
FIGSIZE_EXP3 = (15.5, 13.5)

# Per-column radial grids (requested overrides; others keep the default).
GRANULARITY_DELTA_TICKS = (-0.05, -0.025, 0.00, 0.025, 0.05)
FUSION_DELTA_TICKS = (-0.05, 0.00, 0.05, 0.10, 0.15)
VALUE_ENCODING_DELTA_TICKS = (-0.15, -0.10, -0.05, 0.00, 0.05)
TEMPORAL_DELTA_TICKS = (-0.05, -0.025, 0.00, 0.025, 0.05)
SOURCE_DELTA_TICKS = DELTA_TICKS

BACKBONE_ORDER = ("llama", "qwen")
BACKBONE_DISPLAY = {
    "llama": "Llama-3.2-1B-style",
    "qwen": "Qwen3-0.6B-style",
}
# Rest-of-stay spiders are dropped; every figure uses 24–48h only.
PRIMARY_HORIZON = "24to48h"
HORIZON_DISPLAY = {
    "restofstay": "Rest-of-stay\n(post-24h to discharge)",
    "24to48h": "24–48h follow-up",
}
TASK_ORDER = ("binary", "regression")
METRIC_BY_TASK = {"binary": "roc_auc", "regression": "spearman_rho"}
METRIC_LABEL = {"binary": "AUROC", "regression": r"Spearman $\rho$"}

# Paul Tol bright hues: color encodes clinical family on spoke numbers and the key.
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

# Full outcome names for the on-figure numbered key (no abbreviations).
FULL_OUTCOME_LABELS = {
    "same_admission_death": "Hospital mortality in hours 24--48",
    "long_length_of_stay": "Still hospitalized at hour 48",
    "icu_admission": "First ICU admission in hours 24--48",
    "prolonged_icu_stay": "Still in ICU at hour 48",
    "imv_event": "Invasive mechanical ventilation",
    "vasopressor_initiation": "Vasopressor initiation",
    "crrt_initiation": "Continuous renal replacement",
    "hemodialysis_initiation": "Hemodialysis initiation",
    "hyperkalemia": "Hyperkalemia",
    "severe_hypokalemia": "Severe hypokalemia",
    "profound_hyponatremia": "Profound hyponatremia",
    "severe_hypernatremia": "Severe hypernatremia",
    "hypoglycemia": "Hypoglycemia",
    "severe_anemia": "Severe anemia",
    "tachycardia_hr130": "Tachycardia",
    "severe_hypertension": "Severe hypertension",
    "hypotension": "Hypotension",
    "length_of_stay": "Length of stay (hours)",
    "icu_length_of_stay": "ICU length of stay (hours)",
    "min_potassium": "Minimum potassium",
    "peak_potassium": "Maximum potassium",
    "min_sodium": "Minimum sodium",
    "max_sodium": "Maximum sodium",
    "min_creatinine": "Minimum creatinine",
    "peak_creatinine": "Maximum creatinine",
    "min_troponin": "Minimum troponin",
    "peak_troponin": "Maximum troponin",
    "min_glucose": "Minimum glucose",
    "max_glucose": "Maximum glucose",
    "min_bnp": "Minimum natriuretic peptide",
    "peak_bnp": "Maximum natriuretic peptide",
    "min_hemoglobin": "Minimum hemoglobin",
    "max_hemoglobin": "Maximum hemoglobin",
    "min_heart_rate": "Minimum heart rate",
    "max_heart_rate": "Maximum heart rate",
    "min_sbp": "Minimum systolic BP",
    "max_sbp": "Maximum systolic BP",
    "min_dbp": "Minimum diastolic BP",
    "max_dbp": "Maximum diastolic BP",
}

# Spoke order follows clinical families so related outcomes form contiguous arcs.
BINARY_ORDER = (
    "same_admission_death",
    "long_length_of_stay",
    "icu_admission",
    "prolonged_icu_stay",
    "imv_event",
    "vasopressor_initiation",
    "crrt_initiation",
    "hemodialysis_initiation",
    "hyperkalemia",
    "severe_hypokalemia",
    "profound_hyponatremia",
    "severe_hypernatremia",
    "hypoglycemia",
    "severe_anemia",
    "tachycardia_hr130",
    "severe_hypertension",
    "hypotension",
)
REGRESSION_ORDER = (
    "min_potassium",
    "peak_potassium",
    "min_sodium",
    "max_sodium",
    "min_creatinine",
    "peak_creatinine",
    "min_troponin",
    "peak_troponin",
    "min_glucose",
    "max_glucose",
    "min_bnp",
    "peak_bnp",
    "min_hemoglobin",
    "max_hemoglobin",
    "min_heart_rate",
    "max_heart_rate",
    "min_sbp",
    "max_sbp",
    "min_dbp",
    "max_dbp",
    # Retained for older metric tables; filtered out when absent.
    "length_of_stay",
    "icu_length_of_stay",
)

EXPERIMENT_DISPLAY = {
    "exp1": "Experiment 1: numeric tokenization",
    "exp2": "Experiment 2: representation mechanics",
    "exp3": "Experiment 3: semantic harmonization",
}
BASELINE_FILTERS = {
    "exp1": {"quantizer": "deciles", "anchoring": "none", "fusion": "unfused"},
    "exp2": {"representation": "discrete", "temporal": "time_tokens"},
    "exp3": {"source": "Native MIMIC"},
}
BASELINE_DISPLAY = {
    "exp1": "Deciles · population · unfused",
    "exp2": "Discrete · time tokens",
    "exp3": "Native MIMIC",
}

# Color / marker identify the series within each condition panel.
# Legend labels are full phrases (no abbreviations).
EXP1_GRANULARITY = (
    ("deciles", "none", "Deciles · population", OKABE_ITO[0], "o"),
    ("ventiles", "none", "Ventiles · population", OKABE_ITO[1], "s"),
    ("ventiles", "5-10-5", "Ventiles · 5-10-5 reference", OKABE_ITO[2], "^"),
    ("trentiles", "none", "Trentiles · population", OKABE_ITO[3], "D"),
    ("trentiles", "10-10-10", "Trentiles · 10-10-10 reference", OKABE_ITO[4], "v"),
    ("centiles", "none", "Centiles · population", OKABE_ITO[5], "P"),
)
EXP1_FUSION = (
    ("unfused", "Unfused", OKABE_ITO[0], "o", "solid"),
    ("fused", "Fused", OKABE_ITO[4], "s", "dashed"),
)
EXP2_REPRESENTATION = (
    ("discrete", "Discrete", OKABE_ITO[0]),
    ("soft", "Soft discretization", OKABE_ITO[1]),
    ("xval", "xVal", OKABE_ITO[2]),
    ("xval_affine", "xVal-affine", OKABE_ITO[3]),
)
# Temporal spiders average over every numeric encoder in Experiment 2.
EXP2_TEMPORAL_AVG_REPRESENTATIONS = ("discrete", "soft", "xval", "xval_affine")
EXP2_TEMPORAL = (
    ("time_tokens", "Time tokens", "dashed", "s"),
    ("event_order", "Event stream only", "solid", "o"),
    ("time_rope", "Admission-relative RoPE", "dotted", "^"),
)


@dataclass(frozen=True)
class SeriesSpec:
    """One configuration rendered as a polygon."""

    level_id: str
    label: str
    color: str
    marker: str
    linestyle: str
    filters: dict[str, str]
    # Optional factor to average over before plotting (e.g. temporal panels).
    average_over: tuple[tuple[str, tuple[str, ...]], ...] = ()


@dataclass(frozen=True)
class DeltaGrid:
    """Signed-delta radial ticks for one panel column."""

    ticks: tuple[float, ...]

    @property
    def min(self) -> float:
        return float(self.ticks[0])

    @property
    def max(self) -> float:
        return float(self.ticks[-1])

    def to_radius(self, delta: float | np.ndarray) -> float | np.ndarray:
        """Map deltas onto the shared visual outer-ring radius."""
        span = self.max - self.min
        clipped = np.clip(delta, self.min, self.max)
        return (clipped - self.min) / span * DATA_RADIUS_MAX

    def tick_radii(self) -> tuple[float, ...]:
        return tuple(round(float(self.to_radius(delta)), 10) for delta in self.ticks)

    def is_outlier(self, delta: float) -> bool:
        return bool(delta < self.min or delta > self.max)


@dataclass(frozen=True)
class PanelColumn:
    """One figure column: a separated experimental condition."""

    key: str
    title: str
    series: tuple[SeriesSpec, ...]
    delta_grid: DeltaGrid


DEFAULT_DELTA_GRID = DeltaGrid(DELTA_TICKS)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def style() -> None:
    """NeurIPS-aligned type: Times face at 10 pt body scale."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
            "font.size": NEURIPS_BODY_PT,
            "font.weight": "normal",
            "axes.titlesize": NEURIPS_BODY_PT + 1.0,
            "axes.titleweight": "normal",
            "axes.labelsize": NEURIPS_BODY_PT,
            "axes.labelweight": "normal",
            "figure.titleweight": "normal",
            "xtick.labelsize": NEURIPS_SMALL_PT,
            "ytick.labelsize": NEURIPS_SMALL_PT,
            "legend.fontsize": LEGEND_PT,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "text.color": TEXT_COLOR,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def add_source_column(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive Exp3's source factor from config identifiers when needed."""
    out = frame.copy()
    if "source" not in out.columns:
        out["source"] = np.where(
            out["config_id"].astype(str).str.startswith("meds_icu_"),
            "Native MIMIC",
            np.where(
                out["config_id"].astype(str).str.startswith("meds_clif_"),
                "CLIF harmonized",
                "",
            ),
        )
    return out


def load_metrics(
    *,
    primary_path: Path,
    provisional_path: Path,
) -> tuple[pd.DataFrame, dict[str, str]]:
    primary = add_source_column(pd.read_parquet(primary_path))
    provisional = add_source_column(pd.read_parquet(provisional_path))
    primary["horizon"] = "restofstay"
    provisional["horizon"] = "24to48h"
    combined = pd.concat([primary, provisional], ignore_index=True)
    hashes = {
        str(primary_path): sha256(primary_path),
        str(provisional_path): sha256(provisional_path),
    }
    return combined, hashes


def outcome_order(task_type: str, available: set[str]) -> tuple[str, ...]:
    candidates = BINARY_ORDER if task_type == "binary" else REGRESSION_ORDER
    return tuple(outcome for outcome in candidates if outcome in available)


def exp1_granularity_series() -> tuple[SeriesSpec, ...]:
    """Quantization granularity arms with fusion held at the Exp1 baseline."""
    return tuple(
        SeriesSpec(
            level_id=f"{quantizer}|{anchoring}|unfused",
            label=granularity_label,
            color=color,
            marker=marker,
            linestyle="solid",
            filters={
                "quantizer": quantizer,
                "anchoring": anchoring,
                "fusion": "unfused",
            },
        )
        for quantizer, anchoring, granularity_label, color, marker in EXP1_GRANULARITY
    )


def exp1_fusion_series() -> tuple[SeriesSpec, ...]:
    """Code-value fusion arms with granularity held at the Exp1 baseline."""
    return tuple(
        SeriesSpec(
            level_id=f"deciles|none|{fusion}",
            label=label,
            color=color,
            marker=marker,
            linestyle=linestyle,
            filters={
                "quantizer": "deciles",
                "anchoring": "none",
                "fusion": fusion,
            },
        )
        for fusion, label, color, marker, linestyle in EXP1_FUSION
    )


def exp2_value_encoding_series() -> tuple[SeriesSpec, ...]:
    """Numeric representations with temporal held at the Exp2 baseline."""
    return tuple(
        SeriesSpec(
            level_id=f"{representation}|time_tokens",
            label=label,
            color=color,
            marker=SERIES_MARKERS[index],
            linestyle="solid",
            filters={"representation": representation, "temporal": "time_tokens"},
        )
        for index, (representation, label, color) in enumerate(EXP2_REPRESENTATION)
    )


def exp2_temporal_series() -> tuple[SeriesSpec, ...]:
    """Temporal arms averaged over all four Experiment-2 numeric encoders."""
    return tuple(
        SeriesSpec(
            level_id=f"avg({'+'.join(EXP2_TEMPORAL_AVG_REPRESENTATIONS)})|{temporal}",
            label=label,
            color=OKABE_ITO[index],
            marker=marker,
            linestyle=linestyle,
            filters={"temporal": temporal},
            average_over=(("representation", EXP2_TEMPORAL_AVG_REPRESENTATIONS),),
        )
        for index, (temporal, label, linestyle, marker) in enumerate(EXP2_TEMPORAL)
    )


def exp3_source_series() -> tuple[SeriesSpec, ...]:
    return (
        SeriesSpec(
            level_id="Native MIMIC",
            label="Native MIMIC",
            color=OKABE_ITO[3],
            marker="o",
            linestyle="solid",
            filters={"source": "Native MIMIC"},
        ),
        SeriesSpec(
            level_id="CLIF harmonized",
            label="CLIF harmonized",
            color=OKABE_ITO[4],
            marker="s",
            linestyle="solid",
            filters={"source": "CLIF harmonized"},
        ),
    )


def panel_columns(experiment: str) -> tuple[PanelColumn, ...]:
    """Return the separated conditions that define each figure's columns."""
    if experiment == "exp1":
        return (
            PanelColumn(
                "granularity",
                "Quantization granularity",
                exp1_granularity_series(),
                DeltaGrid(GRANULARITY_DELTA_TICKS),
            ),
            PanelColumn(
                "fusion",
                "Code-value fusion",
                exp1_fusion_series(),
                DeltaGrid(FUSION_DELTA_TICKS),
            ),
        )
    if experiment == "exp2":
        return (
            PanelColumn(
                "value_encoding",
                "Numeric representation",
                exp2_value_encoding_series(),
                DeltaGrid(VALUE_ENCODING_DELTA_TICKS),
            ),
            PanelColumn(
                "temporal",
                "Temporal representation",
                exp2_temporal_series(),
                DeltaGrid(TEMPORAL_DELTA_TICKS),
            ),
        )
    if experiment == "exp3":
        return (
            PanelColumn(
                "source",
                "Data schema",
                exp3_source_series(),
                DeltaGrid(SOURCE_DELTA_TICKS),
            ),
        )
    raise ValueError(f"Unsupported experiment: {experiment}")


def experiment_series(experiment: str) -> tuple[SeriesSpec, ...]:
    """Unique series across all columns (for summaries / inventory checks)."""
    seen: dict[str, SeriesSpec] = {}
    for column in panel_columns(experiment):
        for item in column.series:
            seen.setdefault(item.level_id, item)
    return tuple(seen.values())


def full_cross_series(experiment: str) -> tuple[SeriesSpec, ...]:
    """Compatibility alias used by older tests and callers."""
    return experiment_series(experiment)


def baseline_filters(experiment: str) -> dict[str, str]:
    """Return the primary reference configuration for an experiment."""
    try:
        return BASELINE_FILTERS[experiment].copy()
    except KeyError as error:
        raise ValueError(f"Unsupported experiment: {experiment}") from error


def series_is_baseline(experiment: str, item: SeriesSpec) -> bool:
    """True when this series is the baseline arm for its condition column."""
    baseline = baseline_filters(experiment)
    if item.filters == baseline:
        return True
    # Exp2 temporal column: Time tokens is the encoding-averaged baseline arm.
    return bool(
        item.average_over and item.filters.get("temporal") == baseline.get("temporal")
    )


def filter_rows(
    frame: pd.DataFrame,
    *,
    experiment: str,
    backbone: str,
    task_type: str,
    horizon: str = PRIMARY_HORIZON,
) -> pd.DataFrame:
    metric = METRIC_BY_TASK[task_type]
    return frame.loc[
        (frame["experiment"] == experiment)
        & (frame["backbone"] == backbone)
        & (frame["task_type"] == task_type)
        & (frame["horizon"] == horizon)
        & (frame["metric"] == metric)
    ].copy()


def _apply_filters(frame: pd.DataFrame, filters: dict[str, str]) -> pd.DataFrame:
    subset = frame
    for column, value in filters.items():
        if column not in subset:
            raise ValueError(f"Missing configuration column: {column}")
        subset = subset.loc[subset[column] == value]
    return subset


def seed_mean_by_outcome(subset: pd.DataFrame) -> tuple[dict[str, float], str]:
    """Mean seeds for exactly one configuration and return its config id."""
    config_ids = tuple(sorted(subset["config_id"].dropna().astype(str).unique()))
    if not config_ids:
        return {}, ""
    if len(config_ids) != 1:
        raise ValueError(f"Expected one configuration, found: {config_ids}")
    means = subset.groupby("outcome", as_index=False)["point"].mean()
    return (
        dict(zip(means["outcome"], means["point"], strict=True)),
        config_ids[0],
    )


def average_over_factor_deltas(
    rows: pd.DataFrame,
    *,
    item: SeriesSpec,
    outcomes: tuple[str, ...],
) -> tuple[dict[str, float], str]:
    """Average within-factor deltas, excluding levels not listed in average_over.

    For Exp2 temporal panels this is mean over encodings R of
    point(R, temporal) − point(R, time_tokens).
    """
    if item.average_over != (("representation", EXP2_TEMPORAL_AVG_REPRESENTATIONS),):
        raise ValueError(f"Unsupported average_over for {item.level_id}")
    temporal = item.filters["temporal"]
    per_outcome: dict[str, list[float]] = {outcome: [] for outcome in outcomes}
    config_ids: list[str] = []
    for representation in EXP2_TEMPORAL_AVG_REPRESENTATIONS:
        at_temporal, config_id = seed_mean_by_outcome(
            _apply_filters(
                rows,
                {"representation": representation, "temporal": temporal},
            )
        )
        at_tokens, _ = seed_mean_by_outcome(
            _apply_filters(
                rows,
                {"representation": representation, "temporal": "time_tokens"},
            )
        )
        if config_id:
            config_ids.append(config_id)
        for outcome in outcomes:
            if outcome not in at_temporal or outcome not in at_tokens:
                continue
            per_outcome[outcome].append(
                float(at_temporal[outcome]) - float(at_tokens[outcome])
            )
    deltas = {
        outcome: float(np.mean(values))
        for outcome, values in per_outcome.items()
        if values
    }
    return deltas, f"avg({'+'.join(config_ids)})" if config_ids else ""


def full_cross_delta_panel(
    rows: pd.DataFrame,
    *,
    experiment: str,
    series: tuple[SeriesSpec, ...],
    outcomes: tuple[str, ...],
    delta_grid: DeltaGrid = DEFAULT_DELTA_GRID,
) -> pd.DataFrame:
    """Seed-mean every config then calculate outcome-specific baseline deltas."""
    baseline_values, baseline_config_id = seed_mean_by_outcome(
        _apply_filters(rows, baseline_filters(experiment))
    )
    if not baseline_values:
        return pd.DataFrame()

    records: list[dict[str, object]] = []
    for item in series:
        if item.average_over:
            deltas, config_id = average_over_factor_deltas(
                rows, item=item, outcomes=outcomes
            )
            for outcome, delta in deltas.items():
                records.append(
                    {
                        "level_id": item.level_id,
                        "level_label": item.label,
                        "color": item.color,
                        "marker": item.marker,
                        "linestyle": item.linestyle,
                        "config_id": config_id,
                        "baseline_config_id": baseline_config_id,
                        "is_baseline": item.filters.get("temporal") == "time_tokens",
                        "outcome": outcome,
                        "point": float("nan"),
                        "baseline_point": float("nan"),
                        "delta": float(delta),
                        "clipped_delta": float(
                            np.clip(delta, delta_grid.min, delta_grid.max)
                        ),
                        "is_outlier": delta_grid.is_outlier(float(delta)),
                        "delta_min": delta_grid.min,
                        "delta_max": delta_grid.max,
                    }
                )
            continue

        points, config_id = seed_mean_by_outcome(_apply_filters(rows, item.filters))
        for outcome in outcomes:
            if outcome not in points or outcome not in baseline_values:
                continue
            point = float(points[outcome])
            baseline_point = float(baseline_values[outcome])
            delta = point - baseline_point
            records.append(
                {
                    "level_id": item.level_id,
                    "level_label": item.label,
                    "color": item.color,
                    "marker": item.marker,
                    "linestyle": item.linestyle,
                    "config_id": config_id,
                    "baseline_config_id": baseline_config_id,
                    "is_baseline": item.filters == baseline_filters(experiment),
                    "outcome": outcome,
                    "point": point,
                    "baseline_point": baseline_point,
                    "delta": delta,
                    "clipped_delta": float(
                        np.clip(delta, delta_grid.min, delta_grid.max)
                    ),
                    "is_outlier": delta_grid.is_outlier(delta),
                    "delta_min": delta_grid.min,
                    "delta_max": delta_grid.max,
                }
            )
    return pd.DataFrame.from_records(records)


def delta_to_radius(
    delta: float | np.ndarray,
    delta_grid: DeltaGrid = DEFAULT_DELTA_GRID,
) -> float | np.ndarray:
    """Map a signed delta onto the shared visual outer-ring radius."""
    return delta_grid.to_radius(delta)


def delta_tick_radii(delta_grid: DeltaGrid = DEFAULT_DELTA_GRID) -> tuple[float, ...]:
    return delta_grid.tick_radii()


def format_delta(value: float, *, precision: int | None = None) -> str:
    """Use a true minus sign and an explicit sign for positive deltas."""
    if precision is None:
        # Keep 0.025-style ticks readable without forcing +0.050 everywhere.
        precision = 3 if abs(round(value * 1000.0)) % 50 == 25 else 2
    rendered = f"{value:+.{precision}f}"
    return rendered.replace("-", "−")


def format_grid_ticks(ticks: tuple[float, ...]) -> str:
    return ", ".join("0" if abs(tick) < 1e-12 else format_delta(tick) for tick in ticks)


def full_outcome_label(outcome: str, fallback: str) -> str:
    """Return the display outcome name for spoke labeling."""
    if outcome in FULL_OUTCOME_LABELS:
        return FULL_OUTCOME_LABELS[outcome]
    if fallback and fallback != outcome and " " in fallback:
        return fallback
    return fallback


def wrap_outcome_label(text: str, *, width: int = OUTCOME_LABEL_WRAP) -> str:
    """Soft-wrap long labels so adjacent spokes do not collide."""
    words = text.split()
    if not words:
        return text
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        trial = " ".join([*current, word])
        if current and len(trial) > width:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return "\n".join(lines)


def estimate_label_footprint(label: str, *, fontsize: float = OUTCOME_LABEL_FONTSIZE) -> tuple[float, float]:
    """Approximate radial height and tangential half-width in data units."""
    lines = label.split("\n") or [label]
    scale = fontsize / 10.0
    # Footprints sized for local-neighbor collision checks (not global packing).
    radial_extent = 0.055 * max(1, len(lines)) * scale
    tangential_half = 0.015 * max(len(line) for line in lines) * scale
    return radial_extent, max(0.045, tangential_half)


def angular_separation(a: float, b: float) -> float:
    """Smallest absolute angle between two polar angles."""
    return abs(((a - b + math.pi) % (2.0 * math.pi)) - math.pi)


def _labels_collide(
    angle: float,
    radius: float,
    radial_extent: float,
    tangential_half: float,
    placed: list[tuple[float, float, float, float]],
) -> bool:
    """True when a candidate label overlaps any already-placed neighbor."""
    for prior_angle, prior_radius, prior_radial, prior_tangential in placed:
        separation = angular_separation(angle, prior_angle)
        # Only nearby spokes can visually collide; distant labels share radii.
        if separation > math.radians(28.0):
            continue
        mean_radius = 0.5 * (radius + prior_radius)
        need_arc = 1.35 * (tangential_half + prior_tangential)
        need_radial = 1.25 * (radial_extent + prior_radial)
        if separation * mean_radius < need_arc and abs(radius - prior_radius) < need_radial:
            return True
    return False


def assign_nonoverlapping_label_radii(
    angles: np.ndarray,
    labels: tuple[str, ...],
) -> list[float]:
    """Return radii only; prefer assign_nonoverlapping_label_placements()."""
    return [radius for radius, _ in assign_nonoverlapping_label_placements(angles, labels)]


def assign_nonoverlapping_label_placements(
    angles: np.ndarray,
    labels: tuple[str, ...],
) -> list[tuple[float, float]]:
    """Place labels with strict even/odd radial parity plus local collision push."""
    order = list(np.argsort(angles))
    footprints = [estimate_label_footprint(label) for label in labels]
    nudge = LABEL_ANGLE_NUDGE_RADIANS
    n = len(labels)
    placements: list[tuple[float, float]] = [(LABEL_RADIUS_BASE, 0.0)] * n

    # Even/odd spokes never share a radius family, so adjacent labels cannot stack.
    for rank, index in enumerate(order):
        parity = rank % 2
        placements[index] = (
            LABEL_RADIUS_BASE + parity * LABEL_RADIUS_STEP,
            (0.0, nudge)[parity],
        )

    for _ in range(6):
        moved = False
        placed_preview = [
            (
                float(angles[i]) + placements[i][1],
                placements[i][0],
                footprints[i][0],
                footprints[i][1],
            )
            for i in range(n)
        ]
        for rank, index in enumerate(order):
            parity = rank % 2
            spoke_angle = float(angles[index])
            radius, angle_offset = placements[index]
            radial_extent, tangential_half = footprints[index]
            neighbor_ranks = [
                (rank + offset) % n
                for offset in range(-LABEL_NEIGHBOR_WINDOW, LABEL_NEIGHBOR_WINDOW + 1)
                if offset != 0
            ]
            local_placed = [placed_preview[order[nr]] for nr in neighbor_ranks]
            chosen_angle = spoke_angle + angle_offset
            if not _labels_collide(
                chosen_angle, radius, radial_extent, tangential_half, local_placed
            ):
                continue
            # Stay on the same parity ladder: base + parity*step + 2*k*step.
            ladder = [
                LABEL_RADIUS_BASE + parity * LABEL_RADIUS_STEP + 2 * k * LABEL_RADIUS_STEP
                for k in range(LABEL_RADIUS_TIERS)
            ]
            ladder = [min(value, LABEL_RADIUS_CAP) for value in ladder if value <= LABEL_RADIUS_CAP + 1e-9]
            resolved = False
            for trial_offset in (angle_offset, 0.0, nudge, -nudge, 1.4 * nudge, -1.4 * nudge):
                trial_angle = spoke_angle + trial_offset
                for trial_radius in ladder:
                    if trial_radius + 1e-9 < radius:
                        continue
                    if not _labels_collide(
                        trial_angle,
                        trial_radius,
                        radial_extent,
                        tangential_half,
                        local_placed,
                    ):
                        placements[index] = (trial_radius, trial_offset)
                        placed_preview[index] = (
                            trial_angle,
                            trial_radius,
                            radial_extent,
                            tangential_half,
                        )
                        moved = True
                        resolved = True
                        break
                if resolved:
                    break
            if not resolved and ladder:
                placements[index] = (ladder[-1], angle_offset)
                placed_preview[index] = (
                    spoke_angle + angle_offset,
                    ladder[-1],
                    radial_extent,
                    tangential_half,
                )
                moved = True
        if not moved:
            break
    return placements


def label_radius(index: int) -> float:
    """Fallback staggered radius used by older callers/tests."""
    return LABEL_RADIUS_BASE + (index % LABEL_RADIUS_TIERS) * LABEL_RADIUS_STEP


def half_circle_angles(
    binary_outcomes: tuple[str, ...],
    regression_outcomes: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Place binary spokes in the visual upper half and regression below."""
    def evenly_spaced(start: float, stop: float, count: int) -> np.ndarray:
        if count == 0:
            return np.asarray([], dtype=float)
        if count == 1:
            return np.asarray([(start + stop) / 2.0], dtype=float)
        return np.linspace(start, stop, count)

    binary_angles = evenly_spaced(
        -math.pi / 2.0 + HALF_GAP_RADIANS,
        math.pi / 2.0 - HALF_GAP_RADIANS,
        len(binary_outcomes),
    )
    regression_angles = evenly_spaced(
        math.pi / 2.0 + HALF_GAP_RADIANS,
        3.0 * math.pi / 2.0 - HALF_GAP_RADIANS,
        len(regression_outcomes),
    )
    return binary_angles, regression_angles, np.concatenate(
        [binary_angles, regression_angles]
    )


def _outward_text_alignment(angle: float) -> tuple[str, str]:
    """Return horizontal / vertical alignment that grows away from the ring."""
    visual_x = math.sin(angle)
    visual_y = math.cos(angle)
    if visual_x > 0.22:
        horizontal = "left"
    elif visual_x < -0.22:
        horizontal = "right"
    else:
        horizontal = "center"
    if visual_y > 0.22:
        vertical = "bottom"
    elif visual_y < -0.22:
        vertical = "top"
    else:
        vertical = "center"
    return horizontal, vertical


def _delta_values(
    panel: pd.DataFrame,
    *,
    level_id: str,
    outcomes: tuple[str, ...],
) -> np.ndarray:
    values: list[float] = []
    for outcome in outcomes:
        match = panel.loc[
            (panel["level_id"] == level_id) & (panel["outcome"] == outcome),
            "delta",
        ]
        values.append(float(match.iloc[0]) if not match.empty else np.nan)
    return np.asarray(values, dtype=float)


def _draw_delta_grid(
    ax,
    *,
    angles: np.ndarray,
    delta_grid: DeltaGrid,
) -> None:
    """Draw rings and spokes manually so no axis ink reaches the label rim."""
    ring_theta = np.linspace(0.0, 2.0 * math.pi, 361)
    for delta, radius in zip(delta_grid.ticks, delta_grid.tick_radii(), strict=True):
        if radius == 0.0:
            ax.scatter([0.0], [0.0], s=6, color=GRID_COLOR, zorder=1)
            continue
        is_baseline = math.isclose(delta, 0.0)
        is_outer = math.isclose(delta, delta_grid.max)
        ax.plot(
            ring_theta,
            np.full_like(ring_theta, radius),
            color=RING_COLOR if (is_baseline or is_outer) else GRID_COLOR,
            linewidth=1.2 if is_baseline else (1.3 if is_outer else 0.85),
            zorder=1,
            solid_capstyle="round",
        )
    for angle in angles:
        ax.plot(
            [angle, angle],
            [0.0, DATA_RADIUS_MAX],
            color=GRID_COLOR,
            linewidth=0.72,
            zorder=1,
            solid_capstyle="butt",
        )

    # A visibly thicker diameter separates binary (top) from regression (bottom).
    data_radius_axes = 0.5 * DATA_RADIUS_MAX / DISPLAY_RADIUS_MAX
    separator = Line2D(
        [0.5 - data_radius_axes, 0.5 + data_radius_axes],
        [0.5, 0.5],
        color=RING_COLOR,
        linewidth=1.8,
        transform=ax.transAxes,
        zorder=1.8,
        solid_capstyle="butt",
    )
    separator.set_clip_on(False)
    ax.add_line(separator)


def outcome_family(outcome: str) -> str | None:
    if outcome in FAMILY_COLORS:
        return outcome
    return OUTCOME_FAMILIES.get(outcome)


def outcome_family_color(outcome: str) -> str:
    family = outcome_family(outcome)
    if family is None:
        return TICK_COLOR
    return FAMILY_COLORS[family]


def _draw_spoke_labels(
    ax,
    *,
    angles: np.ndarray,
    outcomes: tuple[str, ...],
    outcome_labels: dict[str, str],
    outlier_spokes: set[int] | None = None,
) -> list[float]:
    """Place large spoke index numbers; names live in the figure outcome key."""
    del outcome_labels
    outlier_spokes = outlier_spokes or set()
    index_labels = tuple(str(index + 1) for index in range(len(outcomes)))
    placements = assign_nonoverlapping_label_placements(angles, index_labels)
    radii: list[float] = []
    for index, (spoke_angle, label, (radius, angle_offset)) in enumerate(
        zip(angles, index_labels, placements, strict=True)
    ):
        if index in outlier_spokes:
            # Longer dashed leader clears space for outside-rim outlier values.
            radius = min(
                max(radius + OUTLIER_SPOKE_EXTRA, DATA_RADIUS_MAX + 0.20),
                LABEL_RADIUS_CAP,
            )
        radii.append(radius)
        label_angle = float(spoke_angle) + float(angle_offset)
        family_color = outcome_family_color(outcomes[index])
        ax.plot(
            [spoke_angle, label_angle],
            [DATA_RADIUS_MAX, max(DATA_RADIUS_MAX + 0.01, radius - 0.02)],
            color=GRID_COLOR,
            linestyle="--",
            linewidth=1.05 if index in outlier_spokes else 0.85,
            zorder=2,
            solid_capstyle="round",
            clip_on=False,
        )
        horizontal, vertical = _outward_text_alignment(label_angle)
        ax.text(
            label_angle,
            radius,
            label,
            fontsize=OUTCOME_LABEL_FONTSIZE,
            fontweight="normal",
            color=family_color,
            ha=horizontal,
            va=vertical,
            clip_on=False,
            zorder=6,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.96,
                "pad": 0.15,
            },
        )
    return radii


def outcome_key_entries(
    binary_outcomes: tuple[str, ...],
    regression_outcomes: tuple[str, ...],
    outcome_labels: dict[str, str],
    *,
    wrap_width: int = 28,
) -> tuple[list[str], list[str]]:
    """Build numbered key lines for the upper (binary) and lower (regression) halves."""

    def _entry(index: int, outcome: str) -> str:
        label = full_outcome_label(outcome, outcome_labels.get(outcome, outcome))
        wrapped = wrap_outcome_label(label, width=wrap_width).split("\n")
        first = f"{index}  {wrapped[0]}"
        rest = [f"    {line}" for line in wrapped[1:]]
        return "\n".join([first, *rest])

    binary_lines = [
        _entry(index + 1, outcome) for index, outcome in enumerate(binary_outcomes)
    ]
    offset = len(binary_outcomes)
    regression_lines = [
        _entry(offset + index + 1, outcome)
        for index, outcome in enumerate(regression_outcomes)
    ]
    return binary_lines, regression_lines


def draw_outcome_key(
    ax,
    *,
    binary_outcomes: tuple[str, ...],
    regression_outcomes: tuple[str, ...],
    outcome_labels: dict[str, str],
) -> None:
    """Numbered outcome key grouped by clinical family (2×2 under each task)."""
    ax.set_axis_off()
    spec = ax.get_subplotspec()
    if spec is None:
        raise RuntimeError("Outcome key axes must come from a GridSpec cell.")

    task_blocks = (
        (
            "binary",
            "Binary AUROC (upper half rim)",
            binary_outcomes,
            0,
            0,
        ),
        (
            "regression",
            r"Regression Spearman $\rho$ (lower half rim)",
            regression_outcomes,
            len(binary_outcomes),
            2,
        ),
    )
    # Size family rows by the densest cell in that row (Biomarkers needs the most).
    # Labels stay on one line, so budget is header + one line per member.
    row_line_budget = [1, 1]
    for task_type, _title, outcomes, _offset, _col0 in task_blocks:
        for family_index, family in enumerate(FAMILY_ORDER[task_type]):
            members = [
                outcome for outcome in outcomes if OUTCOME_FAMILIES.get(outcome) == family
            ]
            lines = 1 + len(members)
            row_line_budget[family_index // 2] = max(
                row_line_budget[family_index // 2], lines
            )
    key_grid = GridSpecFromSubplotSpec(
        4,
        4,
        subplot_spec=spec,
        height_ratios=[0.16, 0.11, max(2.8, 0.28 * row_line_budget[0]), max(2.2, 0.28 * row_line_budget[1])],
        hspace=0.20,
        wspace=0.22,
    )
    heading = ax.figure.add_subplot(key_grid[0, :])
    heading.set_axis_off()
    heading.text(
        0.0,
        0.40,
        "Outcome key",
        fontsize=COLUMN_TITLE_PT,
        fontweight="normal",
        color=TEXT_COLOR,
        ha="left",
        va="center",
        transform=heading.transAxes,
        fontfamily="serif",
    )

    for task_type, title, outcomes, index_offset, col0 in task_blocks:
        title_ax = ax.figure.add_subplot(key_grid[1, col0 : col0 + 2])
        title_ax.set_axis_off()
        title_ax.text(
            0.0,
            0.35,
            title,
            fontsize=NEURIPS_SMALL_PT,
            fontweight="normal",
            color=TEXT_COLOR,
            ha="left",
            va="center",
            transform=title_ax.transAxes,
            fontfamily="serif",
        )
        index_of = {
            outcome: index_offset + index + 1 for index, outcome in enumerate(outcomes)
        }
        for family_index, family in enumerate(FAMILY_ORDER[task_type]):
            row = 2 + family_index // 2
            col = col0 + family_index % 2
            family_ax = ax.figure.add_subplot(key_grid[row, col])
            family_ax.set_axis_off()
            members = [
                outcome for outcome in outcomes if OUTCOME_FAMILIES.get(outcome) == family
            ]
            color = FAMILY_COLORS[family]
            entries: list[tuple[int, str]] = []
            for outcome in members:
                label = full_outcome_label(outcome, outcome_labels.get(outcome, outcome))
                # Keep every outcome name on one line (no wrap to a second/third line).
                entries.append((index_of[outcome], " ".join(str(label).split())))
            # Shared pitch with the densest sibling in this key row so adjacent
            # columns stay aligned and wrapped lines do not collide.
            row_budget = row_line_budget[family_index // 2]
            top = 0.98
            bottom = 0.03
            step = (top - bottom) / max(row_budget, 1)
            key_pt = OUTCOME_KEY_PT if row_budget <= 9 else max(15.0, OUTCOME_KEY_PT - 2.0)
            y = top
            family_ax.text(
                0.0,
                y,
                family,
                fontsize=key_pt,
                fontweight="bold",
                color=color,
                ha="left",
                va="top",
                transform=family_ax.transAxes,
                fontfamily="serif",
                clip_on=False,
            )
            y -= step
            for global_index, label in entries:
                number = str(global_index)
                number_width = 0.08 if global_index < 10 else 0.12
                family_ax.text(
                    0.0,
                    y,
                    number,
                    fontsize=key_pt,
                    fontweight="normal",
                    color=color,
                    ha="left",
                    va="top",
                    transform=family_ax.transAxes,
                    fontfamily="serif",
                    clip_on=False,
                )
                family_ax.text(
                    number_width,
                    y,
                    label,
                    fontsize=key_pt,
                    fontweight="normal",
                    color=TEXT_COLOR,
                    ha="left",
                    va="top",
                    transform=family_ax.transAxes,
                    fontfamily="serif",
                    clip_on=False,
                )
                y -= step


def _draw_delta_tick_labels(ax, *, delta_grid: DeltaGrid) -> None:
    """Label each grid ring above its intersection with the horizontal radius."""
    # East ray is the horizontal radius; place anchors slightly north so the
    # glyphs sit above the intersection rather than on the radius stroke.
    tick_angle = math.pi / 2.0 - 0.26
    for delta, radius in zip(delta_grid.ticks, delta_grid.tick_radii(), strict=True):
        if radius <= 1e-12:
            displayed_radius = 0.055
        else:
            displayed_radius = radius + 0.012
        ax.text(
            tick_angle,
            displayed_radius,
            "0" if abs(delta) < 1e-12 else format_delta(delta),
            fontsize=TICK_LABEL_PT,
            fontweight="normal",
            color=TICK_COLOR,
            ha="center",
            va="bottom",
            zorder=7,
            bbox={
                "facecolor": (1.0, 1.0, 1.0, 0.35),
                "edgecolor": "none",
                "pad": 0.08,
            },
        )


def _draw_outlier_annotations(
    ax,
    *,
    angles: np.ndarray,
    outcomes: tuple[str, ...],
    prepared: list[tuple[SeriesSpec, np.ndarray, float]],
    delta_grid: DeltaGrid,
    label_radii: list[float] | None = None,
) -> None:
    """Mark and report every value outside the panel's delta grid range."""
    del outcomes
    by_outcome: dict[int, list[tuple[SeriesSpec, float]]] = {}
    for item, values, _ in prepared:
        for index, value in enumerate(values):
            if np.isfinite(value) and delta_grid.is_outlier(float(value)):
                by_outcome.setdefault(index, []).append((item, float(value)))

    for outcome_index, records in by_outcome.items():
        # Stable order keeps color-value annotations reproducible across runs.
        records.sort(key=lambda record: (record[0].label, record[1]))
        angle = float(angles[outcome_index])
        horizontal, vertical = _outward_text_alignment(angle)
        for annotation_index, (item, delta) in enumerate(records):
            clipped_radius = float(delta_grid.to_radius(delta))
            ax.scatter(
                [angle],
                [clipped_radius],
                marker="*",
                s=42,
                facecolors="white",
                edgecolors=item.color,
                linewidths=0.85,
                zorder=6.5,
            )
            # Outside the data rim, under the (possibly extended) spoke number.
            spoke_label_radius = (
                label_radii[outcome_index] if label_radii else LABEL_RADIUS_BASE
            )
            annotation_radius = min(
                DATA_RADIUS_MAX + 0.038 + 0.040 * annotation_index,
                max(DATA_RADIUS_MAX + 0.028, spoke_label_radius - 0.08),
            )
            ax.text(
                angle,
                annotation_radius,
                f"*{format_delta(delta, precision=3)}",
                fontsize=OUTLIER_LABEL_PT,
                fontweight="normal",
                color=item.color,
                ha=horizontal,
                va=vertical,
                clip_on=False,
                zorder=8,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.55,
                    "pad": 0.10,
                },
            )


def draw_delta_spider(
    ax,
    panel: pd.DataFrame,
    *,
    binary_outcomes: tuple[str, ...],
    regression_outcomes: tuple[str, ...],
    outcome_labels: dict[str, str],
    series: tuple[SeriesSpec, ...],
    panel_letter: str,
    delta_grid: DeltaGrid = DEFAULT_DELTA_GRID,
) -> None:
    """Draw one architecture × condition baseline-delta spider."""
    outcomes = (*binary_outcomes, *regression_outcomes)
    if not outcomes:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "No outcomes", ha="center", va="center")
        return

    _, _, angles = half_circle_angles(binary_outcomes, regression_outcomes)
    ax.set_theta_offset(math.pi / 2.0)
    ax.set_theta_direction(-1)
    ax.set_ylim(0.0, DISPLAY_RADIUS_MAX)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.spines["polar"].set_visible(False)
    _draw_delta_grid(ax, angles=angles, delta_grid=delta_grid)

    prepared: list[tuple[SeriesSpec, np.ndarray, float]] = []
    outlier_spokes: set[int] = set()
    for item in series:
        values = _delta_values(panel, level_id=item.level_id, outcomes=outcomes)
        if np.isnan(values).all():
            continue
        prepared.append((item, values, float(np.nanmean(np.abs(values)))))
        for index, value in enumerate(values):
            if np.isfinite(value) and delta_grid.is_outlier(float(value)):
                outlier_spokes.add(index)

    label_radii = _draw_spoke_labels(
        ax,
        angles=angles,
        outcomes=outcomes,
        outcome_labels=outcome_labels,
        outlier_spokes=outlier_spokes,
    )
    _draw_delta_tick_labels(ax, delta_grid=delta_grid)
    ax.text(
        0.02,
        0.98,
        panel_letter,
        transform=ax.transAxes,
        fontsize=PANEL_LETTER_PT,
        fontweight="normal",
        color=TEXT_COLOR,
        ha="left",
        va="top",
        clip_on=False,
        zorder=10,
    )

    # Draw smaller deviations first so distinctive polygons remain visible.
    prepared.sort(key=lambda item: item[2])

    for z_index, (item, deltas, _) in enumerate(prepared):
        radii = delta_grid.to_radius(deltas)
        closed_angles = np.concatenate([angles, angles[:1]])
        closed_radii = np.concatenate([radii, radii[:1]])
        z_base = 3.0 + 0.02 * z_index
        ax.plot(
            closed_angles,
            closed_radii,
            color="white",
            linestyle="solid",
            linewidth=3.4,
            zorder=z_base,
            solid_capstyle="round",
            solid_joinstyle="round",
        )
        ax.plot(
            closed_angles,
            closed_radii,
            color=item.color,
            linestyle=item.linestyle,
            linewidth=2.15,
            zorder=z_base + 0.01,
            solid_capstyle="round",
            solid_joinstyle="round",
        )
        valid = np.isfinite(radii)
        ax.scatter(
            angles[valid],
            radii[valid],
            s=30,
            marker=item.marker,
            facecolors=item.color,
            edgecolors="#111111",
            linewidths=0.55,
            zorder=z_base + 0.02,
        )

    _draw_outlier_annotations(
        ax,
        angles=angles,
        outcomes=outcomes,
        prepared=prepared,
        delta_grid=delta_grid,
        label_radii=label_radii,
    )


def build_panel_source(
    metrics: pd.DataFrame,
    *,
    experiment: str,
    backbone: str,
    series: tuple[SeriesSpec, ...],
    condition: str,
    delta_grid: DeltaGrid = DEFAULT_DELTA_GRID,
    horizon: str = PRIMARY_HORIZON,
) -> tuple[pd.DataFrame, tuple[str, ...], tuple[str, ...]]:
    """Build the combined binary + regression source rows for one spider."""
    sources: list[pd.DataFrame] = []
    outcomes_by_task: dict[str, tuple[str, ...]] = {}
    for task_type in TASK_ORDER:
        rows = filter_rows(
            metrics,
            experiment=experiment,
            backbone=backbone,
            task_type=task_type,
            horizon=horizon,
        )
        outcomes = outcome_order(task_type, set(rows["outcome"].unique()))
        outcomes_by_task[task_type] = outcomes
        part = full_cross_delta_panel(
            rows,
            experiment=experiment,
            series=series,
            outcomes=outcomes,
            delta_grid=delta_grid,
        )
        if part.empty:
            continue
        part["experiment"] = experiment
        part["backbone"] = backbone
        part["horizon"] = horizon
        part["condition"] = condition
        part["task_type"] = task_type
        part["metric"] = METRIC_BY_TASK[task_type]
        sources.append(part)
    source = (
        pd.concat(sources, ignore_index=True) if sources else pd.DataFrame()
    )
    return (
        source,
        outcomes_by_task.get("binary", ()),
        outcomes_by_task.get("regression", ()),
    )


def _legend_handles(
    series: tuple[SeriesSpec, ...],
    *,
    experiment: str,
) -> list[Line2D]:
    handles: list[Line2D] = []
    for item in series:
        label = item.label
        if series_is_baseline(experiment, item):
            label = f"{label} (baseline)"
        handles.append(
            Line2D(
                [0],
                [0],
                color=item.color,
                linestyle=item.linestyle,
                marker=item.marker,
                markersize=5.5,
                markerfacecolor=item.color,
                markeredgecolor="#111111",
                markeredgewidth=0.45,
                linewidth=1.8,
                label=label,
            )
        )
    return handles


def experiment_figure_title(experiment: str) -> str:
    return EXPERIMENT_DISPLAY[experiment]


def experiment_figure_description(experiment: str) -> str:
    """Full prose caption / description previously drawn above the spiders."""
    columns = panel_columns(experiment)
    grid_note = "; ".join(
        f"{column.title}: {format_grid_ticks(column.delta_grid.ticks)}"
        for column in columns
    )
    layout = (
        "two architecture panels side by side"
        if len(columns) == 1
        else "architecture rows × condition columns"
    )
    series_note = "; ".join(
        f"{column.title} — "
        + ", ".join(
            (
                f"{item.label} (baseline)"
                if series_is_baseline(experiment, item)
                else item.label
            )
            for item in column.series
        )
        for column in columns
    )
    return (
        f"{EXPERIMENT_DISPLAY[experiment]}. "
        f"24–48h follow-up; Δ relative to baseline ({BASELINE_DISPLAY[experiment]}). "
        f"Layout: {layout}. "
        f"Each circle: upper half rim = binary AUROC; "
        f"lower half rim = regression Spearman ρ. "
        f"Spoke numbers are colored and ordered by clinical family; the on-figure "
        f"outcome key groups the same families. "
        f"Grids — {grid_note}. "
        f"* = clipped value; adjacent number is the actual Δ. "
        f"Series: {series_note}."
    )


def _place_column_legend(
    ax,
    *,
    series: tuple[SeriesSpec, ...],
    experiment: str,
    x_center: float = 0.5,
    y: float = 0.98,
) -> None:
    """Draw a column legend in the Exp1 style: 2-wide packed block under the title."""
    n_series = len(series)
    # Exp1 uses a 2-column block for multi-series legends (and side-by-side for 2).
    ncol = 2 if n_series >= 2 else 1
    legend = ax.legend(
        handles=_legend_handles(series, experiment=experiment),
        loc="upper center",
        bbox_to_anchor=(x_center, y),
        ncol=ncol,
        frameon=False,
        fontsize=LEGEND_PT,
        handlelength=1.6,
        columnspacing=0.85,
        handletextpad=0.30,
        borderaxespad=0.0,
        labelspacing=0.28,
        borderpad=0.2,
        prop={"family": "serif", "weight": "normal", "size": LEGEND_PT},
    )
    ax.add_artist(legend)


def build_experiment_figure(
    metrics: pd.DataFrame,
    *,
    experiment: str,
) -> tuple[plt.Figure, pd.DataFrame]:
    """Render architecture × condition spiders at the 24–48h horizon.

    Titles and overall descriptions are written to FIGURE_METADATA.md rather
    than drawn on the figure. On-figure text keeps column titles, legends,
    architecture labels, and panel letters.
    """
    columns = panel_columns(experiment)
    n_cols = len(columns)
    horizontal = n_cols == 1
    label_lookup = (
        metrics.drop_duplicates("outcome")
        .set_index("outcome")["outcome_label"]
        .astype(str)
        .to_dict()
    )

    source_frames: list[pd.DataFrame] = []
    key_binary: tuple[str, ...] = ()
    key_regression: tuple[str, ...] = ()
    if horizontal:
        column = columns[0]
        fig = plt.figure(figsize=FIGSIZE_EXP3)
        # Match Exp1 title / legend / panel / key proportions.
        grid = GridSpec(
            4,
            len(BACKBONE_ORDER),
            figure=fig,
            height_ratios=[0.07, 0.28, 1.0, 1.85],
            width_ratios=[1.0] * len(BACKBONE_ORDER),
            left=0.02,
            right=0.99,
            top=0.975,
            bottom=0.015,
            hspace=0.14,
            wspace=0.18,
        )
        title_ax = fig.add_subplot(grid[0, :])
        title_ax.set_axis_off()
        title_ax.text(
            0.5,
            0.40,
            column.title,
            fontsize=COLUMN_TITLE_PT,
            fontweight="normal",
            color=TEXT_COLOR,
            ha="center",
            va="center",
            transform=title_ax.transAxes,
        )
        legend_ax = fig.add_subplot(grid[1, :])
        legend_ax.set_axis_off()
        _place_column_legend(
            legend_ax,
            series=column.series,
            experiment=experiment,
            x_center=0.5,
            y=0.98,
        )
        for column_index, backbone in enumerate(BACKBONE_ORDER):
            ax = fig.add_subplot(grid[2, column_index], projection="polar")
            panel, binary_outcomes, regression_outcomes = build_panel_source(
                metrics,
                experiment=experiment,
                backbone=backbone,
                series=column.series,
                condition=column.key,
                delta_grid=column.delta_grid,
            )
            key_binary, key_regression = binary_outcomes, regression_outcomes
            draw_delta_spider(
                ax,
                panel,
                binary_outcomes=binary_outcomes,
                regression_outcomes=regression_outcomes,
                outcome_labels=label_lookup,
                series=column.series,
                panel_letter=PANEL_LETTERS[column_index],
                delta_grid=column.delta_grid,
            )
            ax.text(
                0.5,
                1.04,
                BACKBONE_DISPLAY[backbone],
                transform=ax.transAxes,
                fontsize=COLUMN_TITLE_PT,
                fontweight="normal",
                color=TEXT_COLOR,
                ha="center",
                va="bottom",
                clip_on=False,
            )
            source_frames.append(panel)
        key_ax = fig.add_subplot(grid[3, :])
        draw_outcome_key(
            key_ax,
            binary_outcomes=key_binary,
            regression_outcomes=key_regression,
            outcome_labels=label_lookup,
        )
    else:
        fig = plt.figure(figsize=FIGSIZE_EXP12)
        width_ratios = [0.08, *[1.0] * n_cols]
        # Shared Exp1-style header: titles, then per-column 2-wide legend blocks.
        grid = GridSpec(
            5,
            1 + n_cols,
            figure=fig,
            height_ratios=[0.07, 0.28, 1.0, 1.0, 1.85],
            width_ratios=width_ratios,
            left=0.02,
            right=0.99,
            top=0.975,
            bottom=0.015,
            hspace=0.14,
            wspace=0.18,
        )
        title_ax = fig.add_subplot(grid[0, 1:])
        title_ax.set_axis_off()
        for column_index, column in enumerate(columns):
            x_center = (column_index + 0.5) / n_cols
            title_ax.text(
                x_center,
                0.40,
                column.title,
                fontsize=COLUMN_TITLE_PT,
                fontweight="normal",
                color=TEXT_COLOR,
                ha="center",
                va="center",
                transform=title_ax.transAxes,
            )
            legend_ax = fig.add_subplot(grid[1, 1 + column_index])
            legend_ax.set_axis_off()
            _place_column_legend(
                legend_ax,
                series=column.series,
                experiment=experiment,
                x_center=0.5,
                y=0.98,
            )

        for row_index, backbone in enumerate(BACKBONE_ORDER):
            gutter = fig.add_subplot(grid[2 + row_index, 0])
            gutter.set_axis_off()
            gutter.text(
                0.5,
                0.5,
                BACKBONE_DISPLAY[backbone],
                fontsize=COLUMN_TITLE_PT,
                fontweight="normal",
                color=TEXT_COLOR,
                ha="center",
                va="center",
                rotation=90,
                transform=gutter.transAxes,
            )
            for column_index, column in enumerate(columns):
                ax = fig.add_subplot(
                    grid[2 + row_index, 1 + column_index], projection="polar"
                )
                panel, binary_outcomes, regression_outcomes = build_panel_source(
                    metrics,
                    experiment=experiment,
                    backbone=backbone,
                    series=column.series,
                    condition=column.key,
                    delta_grid=column.delta_grid,
                )
                key_binary, key_regression = binary_outcomes, regression_outcomes
                panel_letter = PANEL_LETTERS[row_index * n_cols + column_index]
                draw_delta_spider(
                    ax,
                    panel,
                    binary_outcomes=binary_outcomes,
                    regression_outcomes=regression_outcomes,
                    outcome_labels=label_lookup,
                    series=column.series,
                    panel_letter=panel_letter,
                    delta_grid=column.delta_grid,
                )
                source_frames.append(panel)
        # Span the full figure width (including the architecture gutter) so the
        # outcome key can use the left-side space under the row labels.
        key_ax = fig.add_subplot(grid[4, :])
        draw_outcome_key(
            key_ax,
            binary_outcomes=key_binary,
            regression_outcomes=key_regression,
            outcome_labels=label_lookup,
        )

    source = (
        pd.concat(source_frames, ignore_index=True)
        if source_frames
        else pd.DataFrame()
    )
    return fig, source


def write_figure_metadata_markdown(
    *,
    section_dir: Path,
    experiments: tuple[str, ...],
) -> Path:
    """Write titles and overall descriptions next to the rendered figures."""
    lines = [
        "# Spider chart figure metadata",
        "",
        "Titles and overall descriptions live here so the PDF/PNG panels stay",
        "uncluttered for paper layout. On-figure text keeps column titles,",
        "full legend labels, architecture names, panel letters, spoke index",
        "numbers, a family-grouped numbered outcome key, and delta-grid annotations.",
        "",
        "Typography matches the coreopsis figure style (~22–40 pt on a page-width",
        "canvas). Long outcome names are moved to a numbered key under the panels",
        "so spoke labels stay large and non-overlapping. Spoke order and colors",
        "follow the same clinical families used in the outcome×factor traces.",
        "",
    ]
    for experiment in experiments:
        columns = panel_columns(experiment)
        lines.extend(
            [
                f"## {experiment_figure_title(experiment)}",
                "",
                f"- **File stems:** `{experiment}_factor_spiders`, "
                f"`{experiment}_factor_summary_table`",
                f"- **Baseline:** {BASELINE_DISPLAY[experiment]}",
                f"- **Horizon:** 24–48h follow-up only",
                f"- **Columns:** "
                + ", ".join(column.title for column in columns),
                "",
                "**Description.** "
                + experiment_figure_description(experiment),
                "",
                "**Legend entries (no abbreviations).**",
                "",
            ]
        )
        for column in columns:
            lines.append(f"- **{column.title}:**")
            for item in column.series:
                label = item.label
                if series_is_baseline(experiment, item):
                    label = f"{label} (baseline)"
                lines.append(f"  - {label}")
        lines.append("")
    path = section_dir / "FIGURE_METADATA.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _summary_column(backbone: str, task_type: str) -> str:
    return f"{backbone}_{task_type}"


def _summary_header(backbone: str, task_type: str) -> str:
    short_backbone = "Llama" if backbone == "llama" else "Qwen"
    short_task = "B" if task_type == "binary" else "R"
    return f"{short_backbone}\n{short_task}"


def summary_mean_table(source: pd.DataFrame, experiment: str) -> pd.DataFrame:
    """Mean baseline deltas by condition × config × architecture × task."""
    columns = panel_columns(experiment)
    records: list[dict[str, object]] = []
    for column in columns:
        for item in column.series:
            row: dict[str, object] = {
                "condition": column.title,
                "configuration": item.label,
                "is_baseline": series_is_baseline(experiment, item),
            }
            for backbone in BACKBONE_ORDER:
                for task_type in TASK_ORDER:
                    key = _summary_column(backbone, task_type)
                    match = source.loc[
                        (source["level_id"] == item.level_id)
                        & (source["condition"] == column.key)
                        & (source["backbone"] == backbone)
                        & (source["task_type"] == task_type),
                        "delta",
                    ]
                    row[key] = float(match.mean()) if not match.empty else np.nan
            records.append(row)
    return pd.DataFrame.from_records(records)


def _format_table_delta(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    return format_delta(float(value), precision=3)


def wrap_summary_label(text: str, *, width: int = 44) -> str:
    """Wrap long configuration labels so they remain inside their table cells."""
    return wrap_outcome_label(text, width=width)


def build_summary_table_figure(
    source: pd.DataFrame,
    *,
    experiment: str,
) -> plt.Figure:
    """Render one invisible-border summary table per experiment."""
    table_df = summary_mean_table(source, experiment)
    table_columns = [
        _summary_column(backbone, task_type)
        for backbone in BACKBONE_ORDER
        for task_type in TASK_ORDER
    ]
    table_headers = [
        _summary_header(backbone, task_type)
        for backbone in BACKBONE_ORDER
        for task_type in TASK_ORDER
    ]
    # A dedicated title band and a fixed-size table axes prevent the title,
    # subtitle, two-line headers, and body rows from sharing vertical space.
    fig_height = 2.15 + 0.58 * (len(table_df) + 1)
    fig = plt.figure(figsize=(15.5, fig_height))
    title = fig.suptitle(
        f"{EXPERIMENT_DISPLAY[experiment]} — 24–48h panel means",
        fontsize=SUMMARY_TABLE_TITLE_PT,
        fontweight="normal",
        color=TEXT_COLOR,
        y=0.975,
    )
    subtitle = fig.text(
        0.025,
        0.895,
        (
            f"Mean Δ from baseline ({BASELINE_DISPLAY[experiment]}) across outcomes. "
            "B = binary AUROC; R = regression Spearman ρ."
        ),
        fontsize=SUMMARY_TABLE_SUBTITLE_PT,
        fontweight="normal",
        color=TICK_COLOR,
        ha="left",
        va="center",
    )
    ax = fig.add_axes(SUMMARY_TABLE_AXES_BOUNDS)
    ax.set_axis_off()
    cell_text = []
    for _, row in table_df.iterrows():
        label = f"{row['condition']}: {row['configuration']}"
        if row["is_baseline"]:
            label = f"{label} (baseline)"
        values = [_format_table_delta(row[column]) for column in table_columns]
        cell_text.append([wrap_summary_label(label), *values])
    table = ax.table(
        cellText=cell_text,
        colLabels=["Configuration", *table_headers],
        bbox=(0.0, 0.0, 1.0, 1.0),
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(SUMMARY_TABLE_BODY_PT)
    for (row_index, column_index), cell in table.get_celld().items():
        cell.set_linewidth(0.0)
        cell.set_edgecolor("none")
        cell.set_facecolor("white")
        text = cell.get_text()
        text.set_color(TEXT_COLOR)
        text.set_fontweight("normal")
        text.set_fontfamily("serif")
        if row_index == 0:
            text.set_fontsize(SUMMARY_TABLE_BODY_PT)
        if column_index == 0:
            text.set_ha("left")
            cell.PAD = 0.035
        if row_index > 0 and row_index % 2 == 0:
            cell.set_facecolor("#F3F3F3")
    for column_index in range(len(table_columns) + 1):
        for row_index in range(len(cell_text) + 1):
            cell = table[row_index, column_index]
            cell.set_width(
                SUMMARY_TABLE_CONFIG_WIDTH
                if column_index == 0
                else SUMMARY_TABLE_METRIC_WIDTH
            )
    # Keep references on the figure for a renderer-based no-overlap test.
    fig._summary_table = table
    fig._summary_title = title
    fig._summary_subtitle = subtitle
    return fig


def write_experiment_figure(
    *,
    output_root: Path,
    experiment: str,
    figure: plt.Figure,
    source: pd.DataFrame,
    input_hashes: dict[str, str],
    table_figure: plt.Figure,
) -> dict[str, object]:
    section_dir = output_root / SPIDER_SECTION
    sources_dir = output_root / "sources"
    metadata_dir = output_root / "metadata"
    section_dir.mkdir(parents=True, exist_ok=True)
    sources_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{experiment}_factor_spiders"
    table_stem = f"{experiment}_factor_summary_table"
    pdf_path = section_dir / f"{stem}.pdf"
    png_path = section_dir / f"{stem}.png"
    source_path = sources_dir / f"{stem}.csv"
    metadata_path = metadata_dir / f"{stem}.json"
    table_pdf_path = section_dir / f"{table_stem}.pdf"
    table_png_path = section_dir / f"{table_stem}.png"
    table_source_path = sources_dir / f"{table_stem}.csv"

    source.to_csv(source_path, index=False)
    summary = summary_mean_table(source, experiment)
    summary.to_csv(table_source_path, index=False)
    figure.savefig(pdf_path, bbox_inches="tight", pad_inches=0.30)
    figure.savefig(png_path, dpi=220, bbox_inches="tight", pad_inches=0.30)
    plt.close(figure)
    table_figure.savefig(table_pdf_path, bbox_inches="tight")
    table_figure.savefig(table_png_path, dpi=220, bbox_inches="tight")
    plt.close(table_figure)

    metadata = {
        "created_at": utc_now(),
        "experiment": experiment,
        "stem": stem,
        "description": (
            f"{EXPERIMENT_DISPLAY[experiment]} baseline-delta spiders "
            "(architecture × condition panels at 24–48h; "
            "binary top, regression bottom)."
        ),
        "series_palette": {
            "name": "Okabe–Ito",
            "reference": "Okabe & Ito colorblind-safe palette; Nature CVD figure guidance",
            "colors": list(OKABE_ITO),
        },
        "baseline": {
            "filters": baseline_filters(experiment),
            "display": BASELINE_DISPLAY[experiment],
        },
        "delta_axis": {
            "by_condition": {
                column.key: {
                    "title": column.title,
                    "ticks": list(column.delta_grid.ticks),
                    "min": column.delta_grid.min,
                    "max": column.delta_grid.max,
                }
                for column in panel_columns(experiment)
            },
            "visual_outer_radius": DATA_RADIUS_MAX,
            "outliers": "clip to column grid; mark with * and actual delta annotation",
        },
        "grid": {
            "grid": GRID_COLOR,
            "ring": RING_COLOR,
            "tick": TICK_COLOR,
            "spoke_extent": "inside outer data ring only",
        },
        "layout": (
            "horizontal_architectures"
            if len(panel_columns(experiment)) == 1
            else "architecture_rows_by_condition_columns"
        ),
        "font": {
            "family": "serif",
            "stack": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
            "body_pt": NEURIPS_BODY_PT,
            "small_pt": NEURIPS_SMALL_PT,
            "weight": "regular",
            "reference": "NeurIPS formatting: 10 pt body type, Times preferred",
        },
        "input_hashes": input_hashes,
        "source_sha256": sha256(source_path),
        "outputs": {
            "pdf": str(pdf_path.relative_to(output_root)),
            "png": str(png_path.relative_to(output_root)),
            "source": str(source_path.relative_to(output_root)),
            "summary_table_pdf": str(table_pdf_path.relative_to(output_root)),
            "summary_table_png": str(table_png_path.relative_to(output_root)),
            "summary_table_source": str(table_source_path.relative_to(output_root)),
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "experiment": experiment,
        "stem": stem,
        "pdf": str(pdf_path.relative_to(output_root)),
        "png": str(png_path.relative_to(output_root)),
        "source": str(source_path.relative_to(output_root)),
        "summary_table_pdf": str(table_pdf_path.relative_to(output_root)),
        "summary_table_png": str(table_png_path.relative_to(output_root)),
        "summary_table_source": str(table_source_path.relative_to(output_root)),
        "metadata": str(metadata_path.relative_to(output_root)),
        "rows": int(len(source)),
    }


def expected_figure_stems() -> tuple[str, ...]:
    return tuple(
        f"{experiment}_factor_spiders" for experiment in ("exp1", "exp2", "exp3")
    )


def expected_table_stems() -> tuple[str, ...]:
    return tuple(
        f"{experiment}_factor_summary_table"
        for experiment in ("exp1", "exp2", "exp3")
    )


def generate(
    *,
    primary_metrics: Path,
    provisional_metrics: Path,
    output_root: Path,
    experiments: tuple[str, ...] = ("exp1", "exp2", "exp3"),
) -> dict[str, object]:
    metrics, input_hashes = load_metrics(
        primary_path=primary_metrics,
        provisional_path=provisional_metrics,
    )
    input_hashes[str(Path(__file__).resolve())] = sha256(Path(__file__).resolve())
    style()

    # Delete only prior rendered artifacts. Provenance sources live separately.
    section_dir = output_root / SPIDER_SECTION
    section_dir.mkdir(parents=True, exist_ok=True)
    for path in section_dir.glob("*"):
        if path.is_file():
            path.unlink()

    index: list[dict[str, object]] = []
    for experiment in experiments:
        figure, source = build_experiment_figure(metrics, experiment=experiment)
        table_figure = build_summary_table_figure(source, experiment=experiment)
        index.append(
            write_experiment_figure(
                output_root=output_root,
                experiment=experiment,
                figure=figure,
                source=source,
                input_hashes=input_hashes,
                table_figure=table_figure,
            )
        )

    stems = [row["stem"] for row in index]
    if stems != list(expected_figure_stems()[: len(experiments)]):
        raise ValueError(f"Unexpected spider stems: {stems}")

    metadata_md = write_figure_metadata_markdown(
        section_dir=section_dir,
        experiments=tuple(experiments),
    )

    index_path = section_dir / "figure_index.csv"
    pd.DataFrame(index).to_csv(index_path, index=False)
    manifest = {
        "created_at": utc_now(),
        "analysis": "condition_split_baseline_delta_spiders",
        "figure_count": len(index),
        "figures": index,
        "figure_metadata_markdown": str(metadata_md.relative_to(output_root)),
        "input_hashes": input_hashes,
        "contract": {
            "series": "condition_panel_seed_means",
            "baseline": "outcome_specific_per_experiment",
            "horizon": PRIMARY_HORIZON,
            "delta_grids": {
                experiment: {
                    column.key: list(column.delta_grid.ticks)
                    for column in panel_columns(experiment)
                }
                for experiment in ("exp1", "exp2", "exp3")
            },
            "packing": "architecture_by_condition_24to48h_exp3_horizontal",
            "task_halves": "binary_top_regression_bottom",
            "spoke_extent": "inside_outer_delta_ring_only",
            "outliers": "asterisk_and_actual_delta_annotation",
            "summary_tables": "invisible_border_condition_panel_means",
            "titles_and_descriptions": "FIGURE_METADATA.md",
            "legend_labels": "full_phrases_no_abbreviations",
            "font": {
                "family": "Times New Roman / Nimbus Roman",
                "body_pt": NEURIPS_BODY_PT,
                "small_pt": NEURIPS_SMALL_PT,
                "reference": "NeurIPS formatting: 10 pt body type, Times preferred",
            },
        },
    }
    (section_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "figure_count": len(index),
        "stems": stems,
        "table_stems": list(expected_table_stems()[: len(experiments)]),
        "figure_metadata_markdown": str(metadata_md.relative_to(output_root)),
        "output_root": str(output_root / SPIDER_SECTION),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-metrics", type=Path, default=DEFAULT_PRIMARY_METRICS)
    parser.add_argument(
        "--provisional-metrics",
        type=Path,
        default=DEFAULT_PROVISIONAL_METRICS,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=("exp1", "exp2", "exp3"),
        default=["exp1", "exp2", "exp3"],
    )
    args = parser.parse_args()
    summary = generate(
        primary_metrics=args.primary_metrics.expanduser().resolve(),
        provisional_metrics=args.provisional_metrics.expanduser().resolve(),
        output_root=args.output_root.expanduser().resolve(),
        experiments=tuple(args.experiments),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
