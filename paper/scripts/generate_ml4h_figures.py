#!/usr/bin/env python3
"""Build the ML4H figure bundle from completed run artifacts.

All outputs are deliberately isolated from ML4H2026.  The collector keeps one
row per model/outcome/metric so seed and architecture variation never disappear
inside a pooled paper-style estimate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import pickle
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from scipy import stats
from sklearn import metrics as skl_metrics


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_RUN_ROOT = REPO_ROOT / "outputs/runs/ml4h_2026"
DEFAULT_OUTPUT_ROOT = REPO_ROOT.parent / "Archive/ml4h-dump"
DEFAULT_TOKENIZED_ROOT = (
    REPO_ROOT / "outputs/runs/tokenized/mimiciv-3.1_meds_70-10-20"
)
sys.path.insert(0, str(REPO_ROOT.parent / "fms-ehrs"))
from fms_ehrs.framework.artifacts import model_artifact_stem

CONFIG_ORDER = (
    "meds_deciles_none_fusedFalse_discrete_time_tokens",
    "meds_deciles_none_fusedTrue_discrete_time_tokens",
    "meds_ventiles_none_fusedFalse_discrete_time_tokens",
    "meds_ventiles_none_fusedTrue_discrete_time_tokens",
    "meds_ventiles_5-10-5_fusedFalse_discrete_time_tokens",
    "meds_ventiles_5-10-5_fusedTrue_discrete_time_tokens",
    "meds_trentiles_none_fusedFalse_discrete_time_tokens",
    "meds_trentiles_none_fusedTrue_discrete_time_tokens",
    "meds_trentiles_10-10-10_fusedFalse_discrete_time_tokens",
    "meds_trentiles_10-10-10_fusedTrue_discrete_time_tokens",
    "meds_centiles_none_fusedFalse_discrete_time_tokens",
    "meds_centiles_none_fusedTrue_discrete_time_tokens",
)
BACKBONE_ORDER = ("llama", "qwen")
SEED_ORDER = (42, 43, 44)
PRIMARY_BINARY = (
    "same_admission_death",
    "long_length_of_stay",
    "icu_admission",
    "imv_event",
)
PRIMARY_REGRESSION = (
    "length_of_stay",
    "peak_creatinine",
    "min_hemoglobin",
    "peak_potassium",
)

CONFIG_COLORS = {
    "meds_deciles_none_fusedFalse_discrete_time_tokens": "#4E79A7",
    "meds_deciles_none_fusedTrue_discrete_time_tokens": "#4E79A7",
    "meds_ventiles_none_fusedFalse_discrete_time_tokens": "#76B7B2",
    "meds_ventiles_none_fusedTrue_discrete_time_tokens": "#76B7B2",
    "meds_ventiles_5-10-5_fusedFalse_discrete_time_tokens": "#59A14F",
    "meds_ventiles_5-10-5_fusedTrue_discrete_time_tokens": "#59A14F",
    "meds_trentiles_none_fusedFalse_discrete_time_tokens": "#E15759",
    "meds_trentiles_none_fusedTrue_discrete_time_tokens": "#E15759",
    "meds_trentiles_10-10-10_fusedFalse_discrete_time_tokens": "#B07AA1",
    "meds_trentiles_10-10-10_fusedTrue_discrete_time_tokens": "#B07AA1",
    "meds_centiles_none_fusedFalse_discrete_time_tokens": "#F28E2B",
    "meds_centiles_none_fusedTrue_discrete_time_tokens": "#F28E2B",
}
BACKBONE_COLORS = {"llama": "#3B6FB6", "qwen": "#C85A5A"}

# MLHC manuscript figure grammar: color encodes the granularity/anchoring regime
# and marker encodes fusion. Reused so ML4H-ready panels stay visually
# comparable with the existing manuscript figures.
MLHC_REGIME_COLORS = {
    "Deciles (population)": "#4E79A7",
    "Ventiles (population)": "#76B7B2",
    "Ventiles (reference-range)": "#59A14F",
    "Trentiles (population)": "#E15759",
    "Trentiles (reference-range)": "#B07AA1",
    "Centiles (population)": "#F28E2B",
}
MLHC_REGIME_BY_CONFIG = {
    "meds_deciles_none_fusedFalse_discrete_time_tokens": "Deciles (population)",
    "meds_deciles_none_fusedTrue_discrete_time_tokens": "Deciles (population)",
    "meds_ventiles_none_fusedFalse_discrete_time_tokens": "Ventiles (population)",
    "meds_ventiles_none_fusedTrue_discrete_time_tokens": "Ventiles (population)",
    "meds_ventiles_5-10-5_fusedFalse_discrete_time_tokens": "Ventiles (reference-range)",
    "meds_ventiles_5-10-5_fusedTrue_discrete_time_tokens": "Ventiles (reference-range)",
    "meds_trentiles_none_fusedFalse_discrete_time_tokens": "Trentiles (population)",
    "meds_trentiles_none_fusedTrue_discrete_time_tokens": "Trentiles (population)",
    "meds_trentiles_10-10-10_fusedFalse_discrete_time_tokens": "Trentiles (reference-range)",
    "meds_trentiles_10-10-10_fusedTrue_discrete_time_tokens": "Trentiles (reference-range)",
    "meds_centiles_none_fusedFalse_discrete_time_tokens": "Centiles (population)",
    "meds_centiles_none_fusedTrue_discrete_time_tokens": "Centiles (population)",
}
BACKBONE_DISPLAY = {"llama": "Llama-3.2-1B-style", "qwen": "Qwen3-0.6B-style"}

# Outcomes are grouped so each measurement's minimum and maximum sit together.
ML4H_BINARY_ORDER = (
    "same_admission_death",
    "long_length_of_stay",
    "icu_admission",
    "prolonged_icu_stay",
    "imv_event",
    "hyperkalemia",
    "severe_hypokalemia",
    "profound_hyponatremia",
    "severe_hypernatremia",
    "hypoglycemia",
    "severe_anemia",
    "tachycardia_hr130",
    "severe_hypertension",
    "hypotension",
    "vasopressor_initiation",
    "crrt_initiation",
    "hemodialysis_initiation",
)
ML4H_REGRESSION_ORDER = (
    "length_of_stay",
    "icu_length_of_stay",
    "min_creatinine",
    "peak_creatinine",
    "min_troponin",
    "peak_troponin",
    "min_hemoglobin",
    "max_hemoglobin",
    "min_potassium",
    "peak_potassium",
    "min_glucose",
    "max_glucose",
    "min_sodium",
    "max_sodium",
    "min_bnp",
    "peak_bnp",
    "min_heart_rate",
    "max_heart_rate",
    "min_sbp",
    "max_sbp",
    "min_dbp",
    "max_dbp",
)
ML4H_READY_SECTION = "07_ML4H_ready"
FOREST_HANDLE_SPREAD = 0.027
FOREST_BAND_COLOR = "#F4F4F4"
FOREST_SEPARATOR_COLOR = "#D4D4D4"


@dataclass(frozen=True)
class ExperimentSpec:
    """Manifest and display contract for one independently interpreted study."""

    experiment: str
    display_name: str
    stage3_dir: str
    expected_runs: int
    test_rows: int
    config_order: tuple[str, ...]


EXP2_CONFIG_ORDER = (
    "meds_deciles_none_fusedFalse_discrete_event_order",
    "meds_deciles_none_fusedFalse_discrete_time_tokens",
    "meds_deciles_none_fusedFalse_discrete_time_rope",
    "meds_deciles_none_fusedFalse_soft_event_order",
    "meds_deciles_none_fusedFalse_soft_time_tokens",
    "meds_deciles_none_fusedFalse_soft_time_rope",
    "meds_deciles_none_fusedFalse_xval_event_order",
    "meds_deciles_none_fusedFalse_xval_time_tokens",
    "meds_deciles_none_fusedFalse_xval_time_rope",
    "meds_deciles_none_fusedFalse_xval_affine_event_order",
    "meds_deciles_none_fusedFalse_xval_affine_time_tokens",
    "meds_deciles_none_fusedFalse_xval_affine_time_rope",
)
EXP3_CONFIG_ORDER = (
    "meds_icu_deciles_none_fusedFalse_discrete_time_tokens",
    "meds_clif_deciles_none_fusedFalse_discrete_time_tokens",
)
EXPERIMENT_SPECS = {
    "exp1": ExperimentSpec(
        experiment="exp1",
        display_name="Experiment 1: numeric tokenization",
        stage3_dir="stage3_padded_repair",
        expected_runs=72,
        test_rows=85_057,
        config_order=CONFIG_ORDER,
    ),
    "exp2": ExperimentSpec(
        experiment="exp2",
        display_name="Experiment 2: representation mechanics",
        stage3_dir="stage3_exp2",
        expected_runs=72,
        test_rows=85_057,
        config_order=EXP2_CONFIG_ORDER,
    ),
    "exp3": ExperimentSpec(
        experiment="exp3",
        display_name="Experiment 3: semantic harmonization",
        stage3_dir="stage3_exp3",
        expected_runs=12,
        test_rows=16_529,
        config_order=EXP3_CONFIG_ORDER,
    ),
}
REPRESENTATION_COLORS = {
    "discrete": "#4E79A7",
    "soft": "#76B7B2",
    "xval": "#E15759",
    "xval_affine": "#B07AA1",
}
TEMPORAL_MARKERS = {"event_order": "o", "time_tokens": "s", "time_rope": "^"}
EXP3_SOURCE_COLORS = {"meds_icu": "#4E79A7", "meds_clif": "#E15759"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def config_parts(config_id: str) -> dict[str, object]:
    parts = config_id.split("_")
    if len(parts) < 5:
        raise ValueError(f"Unexpected config id: {config_id}")
    quantizer = parts[1]
    fused_token = next(part for part in parts if part.startswith("fused"))
    fused = fused_token == "fusedTrue"
    anchor_parts = parts[2 : parts.index(fused_token)]
    anchoring = "_".join(anchor_parts)
    return {
        "quantizer": quantizer,
        "anchoring": anchoring,
        "fusion": "fused" if fused else "unfused",
        "fused": fused,
        "representation": "discrete",
        "temporal": "time_tokens",
    }


def config_label(config_id: str) -> str:
    parts = config_parts(config_id)
    quantizer = str(parts["quantizer"]).replace("iles", "ile").title()
    anchor = str(parts["anchoring"])
    anchor_label = "population" if anchor == "none" else "clinical"
    fusion = "F" if bool(parts["fused"]) else "U"
    return f"{quantizer} {anchor_label} ({fusion})"


def _exp2_representation_and_temporal(config_id: str) -> tuple[str, str]:
    for representation in sorted(REPRESENTATION_COLORS, key=len, reverse=True):
        marker = f"_{representation}_"
        if marker in config_id:
            temporal = config_id.split(marker, maxsplit=1)[1]
            if temporal in TEMPORAL_MARKERS:
                return representation, temporal
    raise ValueError(f"Could not parse Exp2 configuration: {config_id}")


def experiment_config_parts(
    spec: ExperimentSpec, config_id: str, task: dict | None = None
) -> dict[str, object]:
    """Return figure grouping fields without assuming every study shares Exp1 IDs."""
    if spec.experiment == "exp1":
        return config_parts(config_id)
    if spec.experiment == "exp2":
        representation, temporal = _exp2_representation_and_temporal(config_id)
        fields = config_parts(config_id)
        fields.update(
            {
                "representation": representation,
                "temporal": temporal,
                "fusion": "unfused",
                "fused": False,
            }
        )
        return fields
    if spec.experiment == "exp3":
        data_format = str(task.get("data_format")) if task else ""
        if not data_format:
            data_format = "meds_icu" if config_id.startswith("meds_icu_") else "meds_clif"
        return {
            "quantizer": "deciles",
            "anchoring": "none",
            "fusion": "unfused",
            "fused": False,
            "representation": "discrete",
            "temporal": "time_tokens",
            "data_format": data_format,
            "source": "Native MIMIC" if data_format == "meds_icu" else "CLIF harmonized",
        }
    raise ValueError(f"Unsupported experiment: {spec.experiment}")


def experiment_config_label(spec: ExperimentSpec, config_id: str) -> str:
    if spec.experiment == "exp1":
        return config_label(config_id)
    if spec.experiment == "exp2":
        representation, temporal = _exp2_representation_and_temporal(config_id)
        representation_label = {
            "discrete": "Discrete",
            "soft": "Soft",
            "xval": "xVal",
            "xval_affine": "xVal-affine",
        }[representation]
        temporal_label = {
            "event_order": "Event order",
            "time_tokens": "Time tokens",
            "time_rope": "Time RoPE",
        }[temporal]
        return f"{representation_label} · {temporal_label}"
    if spec.experiment == "exp3":
        return (
            "Native MIMIC"
            if config_id.startswith("meds_icu_")
            else "CLIF harmonized"
        )
    raise ValueError(f"Unsupported experiment: {spec.experiment}")


def experiment_config_color(spec: ExperimentSpec, config_id: str) -> str:
    if spec.experiment == "exp1":
        return CONFIG_COLORS[config_id]
    if spec.experiment == "exp2":
        representation, _ = _exp2_representation_and_temporal(config_id)
        return REPRESENTATION_COLORS[representation]
    if spec.experiment == "exp3":
        key = "meds_icu" if config_id.startswith("meds_icu_") else "meds_clif"
        return EXP3_SOURCE_COLORS[key]
    raise ValueError(f"Unsupported experiment: {spec.experiment}")


def experiment_config_marker(spec: ExperimentSpec, config_id: str) -> str:
    if spec.experiment == "exp1":
        return "s" if "fusedTrue" in config_id else "o"
    if spec.experiment == "exp2":
        _, temporal = _exp2_representation_and_temporal(config_id)
        return TEMPORAL_MARKERS[temporal]
    if spec.experiment == "exp3":
        return "o" if config_id.startswith("meds_icu_") else "s"
    raise ValueError(f"Unsupported experiment: {spec.experiment}")


def outcome_label(outcome: str) -> str:
    labels = {
        "same_admission_death": "Hospital mortality in hours 24--48",
        "long_length_of_stay": "Still hospitalized at hour 48",
        "icu_admission": "First ICU admission in hours 24--48",
        "prolonged_icu_stay": "Still in ICU at hour 48",
        "imv_event": "Invasive mechanical ventilation",
        "hyperkalemia": "Hyperkalemia",
        "severe_hypokalemia": "Severe hypokalemia",
        "profound_hyponatremia": "Profound hyponatremia",
        "severe_hypernatremia": "Severe hypernatremia",
        "hypoglycemia": "Hypoglycemia",
        "severe_anemia": "Severe anemia",
        "tachycardia_hr130": "Tachycardia",
        "severe_hypertension": "Severe hypertension",
        "hypotension": "Hypotension",
        "vasopressor_initiation": "Vasopressor initiation",
        "crrt_initiation": "Continuous renal replacement",
        "hemodialysis_initiation": "Hemodialysis initiation",
        "length_of_stay": "Length of stay (hours)",
        "icu_length_of_stay": "ICU length of stay (hours)",
        "min_creatinine": "Minimum creatinine",
        "peak_creatinine": "Peak creatinine",
        "min_troponin": "Minimum troponin",
        "peak_troponin": "Peak troponin",
        "min_hemoglobin": "Minimum hemoglobin",
        "max_hemoglobin": "Maximum hemoglobin",
        "min_potassium": "Minimum potassium",
        "peak_potassium": "Peak potassium",
        "min_glucose": "Minimum glucose",
        "max_glucose": "Maximum glucose",
        "min_sodium": "Minimum sodium",
        "max_sodium": "Maximum sodium",
        "min_bnp": "Minimum NT-proBNP",
        "peak_bnp": "Maximum NT-proBNP",
        "min_heart_rate": "Minimum heart rate",
        "max_heart_rate": "Maximum heart rate",
        "min_sbp": "Minimum systolic blood pressure",
        "max_sbp": "Maximum systolic blood pressure",
        "min_dbp": "Minimum diastolic blood pressure",
        "max_dbp": "Maximum diastolic blood pressure",
    }
    return labels.get(outcome, outcome.replace("_", " ").title())


def artifact_stem(model_loc: str) -> str:
    return model_artifact_stem(pathlib.Path(model_loc))


def load_json(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def ece(y_true: np.ndarray, y_score: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    bucket = np.clip(np.digitize(y_score, edges[1:-1]), 0, bins - 1)
    value = 0.0
    for index in range(bins):
        mask = bucket == index
        if mask.any():
            value += mask.mean() * abs(y_true[mask].mean() - y_score[mask].mean())
    return float(value)


def binary_metric_rows(
    y_true: np.ndarray, y_score: np.ndarray
) -> list[tuple[str, float]]:
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_score)
    y_true, y_score = y_true[mask], y_score[mask]
    rows: list[tuple[str, float]] = []
    if len(np.unique(y_true)) == 2:
        rows.extend(
            [
                ("roc_auc", float(skl_metrics.roc_auc_score(y_true, y_score))),
                (
                    "average_precision",
                    float(skl_metrics.average_precision_score(y_true, y_score)),
                ),
            ]
        )
    else:
        rows.extend([("roc_auc", float("nan")), ("average_precision", float("nan"))])
    rows.extend(
        [
            ("brier", float(skl_metrics.brier_score_loss(y_true, y_score))),
            ("ece_15", ece(y_true, y_score)),
        ]
    )
    return rows


def regression_metric_rows(
    y_true: np.ndarray, y_pred: np.ndarray
) -> list[tuple[str, float]]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    spearman = stats.spearmanr(y_true, y_pred).statistic
    return [
        ("spearman_rho", float(spearman)),
        ("r2", float(skl_metrics.r2_score(y_true, y_pred))),
        ("mae", float(skl_metrics.mean_absolute_error(y_true, y_pred))),
        ("rmse", float(math.sqrt(skl_metrics.mean_squared_error(y_true, y_pred)))),
    ]


def run_directory(model_loc: str) -> pathlib.Path:
    resolved = pathlib.Path(model_loc).resolve()
    return resolved.parent


def read_training_record(
    model_loc: str, backbone: str, seed: int, config_id: str
) -> tuple[dict[str, object], pd.DataFrame]:
    root = run_directory(model_loc)
    record_path = root / "run_record.json"
    curve_path = root / "loss_perplexity_curve.csv"
    record = load_json(record_path)
    selection = record.get("checkpoint_selection", {})
    arguments = record.get("training", {}).get("arguments", {})
    registry_values = {
        "run_directory": str(root),
        "run_record": str(record_path),
        "loss_curve": str(curve_path),
        "best_eval_loss": selection.get("best_metric"),
        "best_checkpoint": selection.get("best_model_checkpoint"),
        "global_step": arguments.get("global_step"),
        "max_steps": arguments.get("max_steps"),
    }
    curve = pd.read_csv(curve_path)
    curve["backbone"] = backbone
    curve["seed"] = seed
    curve["config_id"] = config_id
    curve["model_loc"] = model_loc
    max_step = float(curve["step"].max())
    curve["progress_fraction"] = curve["step"] / max_step if max_step else 0.0
    return registry_values, curve


def load_prediction_metrics(
    *,
    task: dict,
    kind: str,
    spec: ExperimentSpec,
    tokenized_root: pathlib.Path,
    preds_tag: str,
    expected_outcomes: tuple[str, ...],
    artifact_hashes: dict[str, str],
) -> list[dict[str, object]]:
    model_loc = str(task["model_loc"])
    stem = artifact_stem(model_loc)
    prefix = "logistic_regression" if kind == "binary" else "reg_ridge_regression"
    data_dir = pathlib.Path(task.get("data_dir", tokenized_root))
    artifact = (
        data_dir
        / f"{task['data_version']}-tokenized"
        / "test"
        / f"{prefix}-preds-{preds_tag}-{stem}.pkl"
    )
    if not artifact.is_file():
        raise FileNotFoundError(f"Missing {kind} prediction artifact: {artifact}")
    with artifact.open("rb") as handle:
        payload = pickle.load(handle)
    meta = payload["metadata"]
    observed_outcomes = tuple(meta["outcomes"])
    if set(observed_outcomes) != set(expected_outcomes):
        raise ValueError(
            f"Unexpected {kind} outcome contract in {artifact}: "
            f"expected {len(expected_outcomes)}, observed {len(observed_outcomes)}"
        )
    if set(payload["predictions"]) != set(expected_outcomes):
        raise ValueError(
            f"Incomplete {kind} artifact {artifact}: "
            f"{len(payload['predictions'])}/{len(expected_outcomes)} outcomes"
        )
    if set(meta["completed_outcomes"]) != set(expected_outcomes):
        raise ValueError(
            f"Incomplete saved outcomes in {artifact}: "
            f"{len(meta['completed_outcomes'])}/{len(expected_outcomes)}"
        )
    if meta.get("preds_tag") != preds_tag:
        raise ValueError(
            f"Unexpected prediction tag in {artifact}: "
            f"{meta.get('preds_tag')!r} != {preds_tag!r}"
        )

    config = experiment_config_parts(spec, str(task["config_id"]), task)
    artifact_key = str(artifact)
    source_sha256 = artifact_hashes.setdefault(artifact_key, sha256(artifact))
    rows: list[dict[str, object]] = []
    for outcome in expected_outcomes:
        raw_y_true = np.asarray(payload["labels"][outcome])
        y_pred = np.asarray(payload["predictions"][outcome])
        qualifier = np.asarray(payload["qualifiers"][outcome], dtype=bool)
        if len(raw_y_true) != spec.test_rows:
            raise ValueError(
                f"Unexpected test-row count in {artifact} for {outcome}: "
                f"{len(raw_y_true)} != {spec.test_rows}"
            )
        y_true = raw_y_true
        if len(y_true) != len(y_pred):
            if len(qualifier) != len(y_true) or qualifier.sum() != len(y_pred):
                raise ValueError(
                    f"Cannot align labels and predictions for {outcome} in {artifact}"
                )
            y_true = y_true[qualifier]
        metric_rows = (
            binary_metric_rows(y_true, y_pred)
            if kind == "binary"
            else regression_metric_rows(y_true, y_pred)
        )
        n = int(np.isfinite(np.asarray(y_true, dtype=float)).sum())
        for metric, point in metric_rows:
            rows.append(
                {
                    "experiment": spec.experiment,
                    "config_id": task["config_id"],
                    "config_label": experiment_config_label(
                        spec, str(task["config_id"])
                    ),
                    "backbone": task["backbone"],
                    "seed": int(task["seed"]),
                    "task_type": kind,
                    "outcome": outcome,
                    "outcome_label": outcome_label(outcome),
                    "metric": metric,
                    "point": point,
                    "n_test": n,
                    "test_cohort_rows": spec.test_rows,
                    "quantizer": config["quantizer"],
                    "anchoring": config["anchoring"],
                    "fusion": config["fusion"],
                    "representation": config["representation"],
                    "temporal": config["temporal"],
                    "effective_logreg_C": 0.01 if kind == "binary" else np.nan,
                    "effective_ridge_alpha": 100.0 if kind == "regression" else np.nan,
                    "source_artifact": str(artifact),
                    "source_sha256": source_sha256,
                }
            )
    return rows


def stage3_task_list(
    jobfile: pathlib.Path,
    model_lookup: dict[str, dict],
    *,
    expected_runs: int,
    expected_preds_tag: str,
) -> list[dict]:
    tasks: list[dict] = []
    for line in jobfile.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        model_match = re.search(r"\bmodel_loc=(\S+)", line)
        version_match = re.search(r"\bdata_version=(\S+)", line)
        if not model_match or not version_match:
            raise ValueError(f"Could not parse Stage 3 jobfile line: {line}")
        model_loc = model_match.group(1)
        if model_loc not in model_lookup:
            raise ValueError(f"Model missing from stage manifest: {model_loc}")
        tag_match = re.search(r"\bIRB_PREDS_TAG=(\S+)", line)
        if not tag_match or tag_match.group(1) != expected_preds_tag:
            raise ValueError(
                f"Unexpected prediction tag in {jobfile}: "
                f"{tag_match.group(1) if tag_match else None!r}"
            )
        task = dict(model_lookup[model_loc])
        task["data_version"] = version_match.group(1)
        tasks.append(task)
    if len(tasks) != expected_runs:
        raise ValueError(
            f"Expected {expected_runs} tasks in {jobfile}, found {len(tasks)}"
        )
    if len({str(task["model_loc"]) for task in tasks}) != expected_runs:
        raise ValueError(f"Duplicate model locations in {jobfile}")
    return tasks


def validate_outcome_contract(
    spec: ExperimentSpec,
    binary_outcomes: tuple[str, ...],
    regression_outcomes: tuple[str, ...],
) -> None:
    expected_binary = 16 if spec.experiment == "exp3" else 17
    if spec.experiment == "exp3" and "icu_admission" in binary_outcomes:
        raise ValueError("Exp3 must exclude degenerate icu_admission")
    if len(binary_outcomes) != expected_binary or len(regression_outcomes) != 22:
        raise ValueError(
            f"Unexpected {spec.experiment} outcome contract: "
            f"{len(binary_outcomes)} binary / {len(regression_outcomes)} regression"
        )


def expected_metric_row_count(
    spec: ExperimentSpec,
    binary_outcomes: tuple[str, ...],
    regression_outcomes: tuple[str, ...],
) -> int:
    return spec.expected_runs * (
        len(binary_outcomes) * 4 + len(regression_outcomes) * 4
    )


def collect_experiment(
    run_root: pathlib.Path,
    tokenized_root: pathlib.Path,
    spec: ExperimentSpec,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[pathlib.Path]]:
    stage3_root = run_root / spec.stage3_dir
    manifest_path = stage3_root / f"{spec.experiment}_stage3_manifest.json"
    manifest = load_json(manifest_path)
    tasks = manifest["tasks"]
    model_lookup = {str(task["model_loc"]): task for task in tasks}
    if len(model_lookup) != spec.expected_runs:
        raise ValueError(
            f"Expected {spec.expected_runs} unique {spec.experiment} stage models"
        )
    suffix = str(manifest.get("preds_tag_suffix", ""))
    if suffix and not suffix.startswith("-"):
        suffix = f"-{suffix}"
    binary_tag = f"ml4h-{spec.experiment}-binary{suffix}"
    regression_tag = f"ml4h-{spec.experiment}-regression{suffix}"
    binary_outcomes = tuple(str(value) for value in manifest["binary_outcomes"])
    regression_outcomes = tuple(str(value) for value in manifest["regression_outcomes"])
    validate_outcome_contract(spec, binary_outcomes, regression_outcomes)
    binary_jobfile = stage3_root / f"{spec.experiment}_stage3_binary.jobfile"
    regression_jobfile = stage3_root / f"{spec.experiment}_stage3_regression.jobfile"
    binary_tasks = stage3_task_list(
        binary_jobfile,
        model_lookup,
        expected_runs=spec.expected_runs,
        expected_preds_tag=binary_tag,
    )
    regression_tasks = stage3_task_list(
        regression_jobfile,
        model_lookup,
        expected_runs=spec.expected_runs,
        expected_preds_tag=regression_tag,
    )

    registry_rows: list[dict[str, object]] = []
    training_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    artifact_hashes: dict[str, str] = {}
    seen_models: set[str] = set()
    input_paths = [
        manifest_path,
        binary_jobfile,
        regression_jobfile,
    ]

    for task in binary_tasks:
        model_loc = str(task["model_loc"])
        if model_loc not in seen_models:
            config = experiment_config_parts(spec, str(task["config_id"]), task)
            run_values, curve = read_training_record(
                model_loc,
                str(task["backbone"]),
                int(task["seed"]),
                str(task["config_id"]),
            )
            curve["experiment"] = spec.experiment
            registry_rows.append(
                {
                    "experiment": spec.experiment,
                    "config_id": task["config_id"],
                    "config_label": experiment_config_label(
                        spec, str(task["config_id"])
                    ),
                    "backbone": task["backbone"],
                    "seed": int(task["seed"]),
                    "model_loc": model_loc,
                    "model_stem": artifact_stem(model_loc),
                    "test_cohort_rows": spec.test_rows,
                    **config,
                    **run_values,
                }
            )
            training_frames.append(curve)
            input_paths.extend(
                [
                    pathlib.Path(run_values["run_record"]),
                    pathlib.Path(run_values["loss_curve"]),
                ]
            )
            seen_models.add(model_loc)
        metric_rows.extend(
            load_prediction_metrics(
                task=task,
                kind="binary",
                spec=spec,
                tokenized_root=tokenized_root,
                preds_tag=binary_tag,
                expected_outcomes=binary_outcomes,
                artifact_hashes=artifact_hashes,
            )
        )

    for task in regression_tasks:
        metric_rows.extend(
            load_prediction_metrics(
                task=task,
                kind="regression",
                spec=spec,
                tokenized_root=tokenized_root,
                preds_tag=regression_tag,
                expected_outcomes=regression_outcomes,
                artifact_hashes=artifact_hashes,
            )
        )

    registry = pd.DataFrame(registry_rows)
    metrics = pd.DataFrame(metric_rows)
    training = pd.concat(training_frames, ignore_index=True)
    if (
        registry["config_id"].nunique() != len(spec.config_order)
        or registry["backbone"].nunique() != 2
        or registry["seed"].nunique() != 3
        or len(registry) != spec.expected_runs
    ):
        raise ValueError(
            f"{spec.experiment} registry does not have the expected "
            f"{len(spec.config_order)} × 2 × 3 design"
        )
    if set(registry["config_id"]) != set(spec.config_order):
        raise ValueError(f"Unexpected configuration set for {spec.experiment}")
    expected_metric_rows = expected_metric_row_count(
        spec, binary_outcomes, regression_outcomes
    )
    if len(metrics) != expected_metric_rows:
        raise ValueError(
            f"Expected {expected_metric_rows} {spec.experiment} metric rows, "
            f"found {len(metrics)}"
        )

    coverage_rows = [
        {
            "experiment": spec.experiment,
            "expected_runs": spec.expected_runs,
            "observed_runs": len(registry),
            "expected_binary_outcomes_per_run": len(binary_outcomes),
            "expected_regression_outcomes_per_run": len(regression_outcomes),
            "observed_metric_rows": len(metrics),
            "test_cohort_rows": spec.test_rows,
            "status": "complete",
        }
    ]
    return registry, metrics, training, pd.DataFrame(coverage_rows), input_paths


def style() -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 10,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


class FigureWriter:
    def __init__(
        self,
        output_root: pathlib.Path,
        *,
        input_hashes: dict[str, str],
        coverage: pd.DataFrame,
    ) -> None:
        self.output_root = output_root
        self.input_hashes = input_hashes
        self.coverage = coverage
        self.index: list[dict[str, object]] = []

    def write(
        self,
        fig: plt.Figure,
        *,
        section: str,
        stem: str,
        source: pd.DataFrame,
        description: str,
    ) -> None:
        section_dir = self.output_root / section
        section_dir.mkdir(parents=True, exist_ok=True)
        sources_dir = self.output_root / "sources"
        metadata_dir = self.output_root / "metadata"
        sources_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)
        pdf = section_dir / f"{stem}.pdf"
        png = section_dir / f"{stem}.png"
        source_path = sources_dir / f"{stem}.csv"
        metadata_path = metadata_dir / f"{stem}.json"
        fig.savefig(pdf, bbox_inches="tight")
        fig.savefig(png, dpi=240, bbox_inches="tight")
        plt.close(fig)
        source.to_csv(source_path, index=False)
        metadata_path.write_text(
            json.dumps(
                {
                    "created_at": utc_now(),
                    "description": description,
                    "rows": len(source),
                    "columns": list(source.columns),
                    "input_hashes": self.input_hashes,
                    "coverage": self.coverage.to_dict(orient="records"),
                    "outputs": {
                        "pdf": str(pdf),
                        "png": str(png),
                        "source": str(source_path),
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        self.index.append(
            {
                "section": section,
                "stem": stem,
                "description": description,
                "pdf": str(pdf.relative_to(self.output_root)),
                "png": str(png.relative_to(self.output_root)),
                "source": str(source_path.relative_to(self.output_root)),
                "metadata": str(metadata_path.relative_to(self.output_root)),
            }
        )


def metric_summary(metrics: pd.DataFrame, metric: str) -> pd.DataFrame:
    selected = metrics.loc[metrics["metric"] == metric].copy()
    return (
        selected.groupby(
            ["config_id", "config_label", "backbone", "seed"], as_index=False
        )["point"]
        .mean()
        .rename(columns={"point": "mean_outcome_metric"})
    )


def plot_overview(registry: pd.DataFrame, writer: FigureWriter) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 6.3), sharex=True, sharey=True)
    configs = list(CONFIG_ORDER)
    for row, backbone in enumerate(BACKBONE_ORDER):
        for col, seed in enumerate(SEED_ORDER):
            ax = axes[row, col]
            subset = registry[
                (registry["backbone"] == backbone) & (registry["seed"] == seed)
            ]
            present = set(subset["config_id"])
            matrix = np.array([int(config in present) for config in configs]).reshape(3, 4)
            ax.imshow(
                matrix,
                cmap=ListedColormap(["#FEE2E2", "#DCFCE7"]),
                vmin=0,
                vmax=1,
                aspect="auto",
            )
            for index, config in enumerate(configs):
                y, x = divmod(index, 4)
                ax.text(
                    x,
                    y,
                    config_label(config).replace(" ", "\n"),
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="#111827",
                )
            ax.set_title(f"{backbone.title()} · seed {seed}")
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("Exp1 design matrix: 12 configurations in every architecture × seed cell")
    fig.tight_layout()
    writer.write(
        fig,
        section="01_overview",
        stem="exp1__overview__design-matrix",
        source=registry.sort_values(["backbone", "seed", "config_id"]),
        description="Coverage map for all 72 Exp1 model runs.",
    )


def plot_summary(metrics: pd.DataFrame, writer: FigureWriter) -> None:
    source = pd.concat(
        [
            metric_summary(metrics, "roc_auc").assign(summary_metric="AUROC"),
            metric_summary(metrics, "spearman_rho").assign(
                summary_metric="Spearman rho"
            ),
        ],
        ignore_index=True,
    )
    fig, axes = plt.subplots(2, 2, figsize=(14.2, 8.4), sharex=True)
    x = np.arange(len(CONFIG_ORDER))
    for row, summary_metric in enumerate(("AUROC", "Spearman rho")):
        for col, backbone in enumerate(BACKBONE_ORDER):
            ax = axes[row, col]
            subset = source[
                (source["summary_metric"] == summary_metric)
                & (source["backbone"] == backbone)
            ]
            for seed_index, seed in enumerate(SEED_ORDER):
                values = (
                    subset.loc[subset["seed"] == seed]
                    .set_index("config_id")
                    .reindex(CONFIG_ORDER)["mean_outcome_metric"]
                )
                ax.scatter(
                    x + (seed_index - 1) * 0.12,
                    values,
                    color=BACKBONE_COLORS[backbone],
                    alpha=0.42,
                    s=24,
                    label=f"seed {seed}" if row == 0 and col == 0 else None,
                )
            mean = (
                subset.groupby("config_id")["mean_outcome_metric"]
                .mean()
                .reindex(CONFIG_ORDER)
            )
            spread = (
                subset.groupby("config_id")["mean_outcome_metric"]
                .agg(lambda values: values.max() - values.min())
                .reindex(CONFIG_ORDER)
            )
            ax.errorbar(
                x,
                mean,
                yerr=spread / 2,
                color=BACKBONE_COLORS[backbone],
                marker="o",
                linewidth=1.2,
                capsize=2,
                label="seed mean ± half range"
                if row == 0 and col == 0
                else None,
            )
            ax.set_title(f"{backbone.title()} · {summary_metric}")
            ax.grid(axis="y", alpha=0.25)
            if row == 0:
                ax.set_ylim(0.5, 1.0)
            else:
                ax.set_ylim(bottom=0.0)
    for ax in axes[-1]:
        ax.set_xticks(x)
        ax.set_xticklabels(
            [config_label(config) for config in CONFIG_ORDER],
            rotation=55,
            ha="right",
        )
    axes[0, 0].legend(loc="lower right")
    fig.suptitle(
        "Exp1 outcome-averaged performance: every seed is visible; bars show seed range"
    )
    fig.tight_layout()
    writer.write(
        fig,
        section="02_summary",
        stem="exp1__summary__architecture-stratified-performance",
        source=source,
        description=(
            "Architecture-stratified outcome-average performance with all three "
            "seed points and architecture-specific seed ranges."
        ),
    )


def plot_heatmaps(metrics: pd.DataFrame, writer: FigureWriter, metric: str) -> None:
    subset = metrics.loc[metrics["metric"] == metric].copy()
    task_label = "binary" if metric == "roc_auc" else "regression"
    outcomes = sorted(subset["outcome"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(16.0, 10.0), constrained_layout=True)
    vmin, vmax = float(subset["point"].min()), float(subset["point"].max())
    for row, backbone in enumerate(BACKBONE_ORDER):
        architecture = subset.loc[subset["backbone"] == backbone]
        mean = (
            architecture.groupby(["config_id", "outcome"])["point"]
            .mean()
            .unstack("outcome")
            .reindex(index=CONFIG_ORDER, columns=outcomes)
        )
        spread = (
            architecture.groupby(["config_id", "outcome"])["point"]
            .agg(lambda values: values.max() - values.min())
            .unstack("outcome")
            .reindex(index=CONFIG_ORDER, columns=outcomes)
        )
        image = axes[row, 0].imshow(
            mean.to_numpy(), aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax
        )
        axes[row, 0].set_title(f"{backbone.title()} mean {metric}")
        axes[row, 1].imshow(
            spread.to_numpy(),
            aspect="auto",
            cmap="magma",
            vmin=0,
            vmax=max(float(spread.max().max()), 1e-6),
        )
        axes[row, 1].set_title(f"{backbone.title()} seed range")
        for col in range(2):
            axes[row, col].set_yticks(np.arange(len(CONFIG_ORDER)))
            axes[row, col].set_yticklabels(
                [config_label(config) for config in CONFIG_ORDER], fontsize=6
            )
            axes[row, col].set_xticks(np.arange(len(outcomes)))
            axes[row, col].set_xticklabels(
                [outcome_label(outcome) for outcome in outcomes],
                rotation=60,
                ha="right",
                fontsize=6,
            )
    fig.colorbar(image, ax=axes[:, 0], shrink=0.8, label=metric)
    fig.suptitle(
        f"Exp1 {task_label} outcome heatmap: mean performance and seed disagreement"
    )
    writer.write(
        fig,
        section="02_summary",
        stem=f"exp1__summary__{task_label}-outcome-heatmap",
        source=subset,
        description=(
            f"{task_label.title()} outcome means and seed ranges for every "
            "configuration and architecture."
        ),
    )


def plot_primary_outcomes(
    metrics: pd.DataFrame,
    writer: FigureWriter,
    *,
    metric: str,
    outcomes: tuple[str, ...],
    task_label: str,
) -> None:
    subset = metrics[
        (metrics["metric"] == metric) & metrics["outcome"].isin(outcomes)
    ].copy()
    fig, axes = plt.subplots(1, len(outcomes), figsize=(15.0, 8.5), sharey=True)
    y = np.arange(len(CONFIG_ORDER))[::-1]
    for ax, outcome in zip(axes, outcomes):
        data = subset.loc[subset["outcome"] == outcome]
        for backbone, offset, marker in (("llama", -0.12, "o"), ("qwen", 0.12, "s")):
            architecture = data.loc[data["backbone"] == backbone]
            for seed_index, seed in enumerate(SEED_ORDER):
                values = (
                    architecture.loc[architecture["seed"] == seed]
                    .set_index("config_id")
                    .reindex(CONFIG_ORDER)["point"]
                )
                ax.scatter(
                    values,
                    y + offset + (seed_index - 1) * 0.035,
                    s=18,
                    marker=marker,
                    color=BACKBONE_COLORS[backbone],
                    alpha=0.38,
                )
            summary = architecture.groupby("config_id")["point"].agg(["mean", "min", "max"])
            summary = summary.reindex(CONFIG_ORDER)
            ax.errorbar(
                summary["mean"],
                y + offset,
                xerr=np.vstack(
                    [summary["mean"] - summary["min"], summary["max"] - summary["mean"]]
                ),
                fmt=marker,
                color=BACKBONE_COLORS[backbone],
                markersize=4,
                linewidth=0.8,
                capsize=1.5,
                label=backbone.title() if outcome == outcomes[0] else None,
            )
        ax.set_title(outcome_label(outcome))
        ax.grid(axis="x", alpha=0.25)
        ax.set_xlabel("AUROC" if metric == "roc_auc" else r"Spearman $\rho$")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([config_label(config) for config in CONFIG_ORDER])
    axes[0].legend(loc="lower right")
    fig.suptitle(
        f"Exp1 primary {task_label} outcomes: architecture means, seed ranges, and seed points"
    )
    fig.tight_layout()
    writer.write(
        fig,
        section="02_summary",
        stem=f"exp1__summary__primary-{task_label}-outcomes",
        source=subset,
        description=(
            f"Primary {task_label} outcome forest plots retaining every seed and "
            "both architectures."
        ),
    )


def plot_atlas(metrics: pd.DataFrame, writer: FigureWriter) -> None:
    for backbone in BACKBONE_ORDER:
        for seed in SEED_ORDER:
            subset = metrics[
                (metrics["backbone"] == backbone) & (metrics["seed"] == seed)
            ]
            fig, axes = plt.subplots(2, 1, figsize=(17.0, 9.2), constrained_layout=True)
            for ax, metric, title in (
                (axes[0], "roc_auc", "Binary outcomes · AUROC"),
                (axes[1], "spearman_rho", r"Regression outcomes · Spearman $\rho$"),
            ):
                data = subset.loc[subset["metric"] == metric]
                outcomes = sorted(data["outcome"].unique())
                matrix = (
                    data.pivot(index="config_id", columns="outcome", values="point")
                    .reindex(index=CONFIG_ORDER, columns=outcomes)
                )
                image = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="viridis")
                ax.set_title(title)
                ax.set_yticks(np.arange(len(CONFIG_ORDER)))
                ax.set_yticklabels([config_label(config) for config in CONFIG_ORDER])
                ax.set_xticks(np.arange(len(outcomes)))
                ax.set_xticklabels(
                    [outcome_label(outcome) for outcome in outcomes],
                    rotation=55,
                    ha="right",
                    fontsize=6,
                )
                fig.colorbar(image, ax=ax, shrink=0.75)
            fig.suptitle(f"Exp1 audit atlas · {backbone.title()} · seed {seed}")
            writer.write(
                fig,
                section="03_seed_architecture_atlas",
                stem=f"exp1__atlas__{backbone}__seed-{seed}",
                source=subset,
                description=(
                    "All 12 configuration runs for one architecture and one seed."
                ),
            )


def plot_stability(metrics: pd.DataFrame, writer: FigureWriter) -> None:
    source = pd.concat(
        [
            metric_summary(metrics, "roc_auc").assign(summary_metric="AUROC"),
            metric_summary(metrics, "spearman_rho").assign(
                summary_metric="Spearman rho"
            ),
        ],
        ignore_index=True,
    )
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.8), sharex=True)
    x = np.arange(len(CONFIG_ORDER))
    for ax, summary_metric in zip(axes, ("AUROC", "Spearman rho")):
        data = source.loc[source["summary_metric"] == summary_metric]
        for backbone in BACKBONE_ORDER:
            architecture = data.loc[data["backbone"] == backbone]
            for seed in SEED_ORDER:
                values = (
                    architecture.loc[architecture["seed"] == seed]
                    .set_index("config_id")
                    .reindex(CONFIG_ORDER)["mean_outcome_metric"]
                )
                ax.plot(
                    x,
                    values,
                    color=BACKBONE_COLORS[backbone],
                    alpha=0.25,
                    linewidth=0.8,
                )
            spread = (
                architecture.groupby("config_id")["mean_outcome_metric"]
                .agg(lambda values: values.max() - values.min())
                .reindex(CONFIG_ORDER)
            )
            ax.plot(
                x,
                spread,
                marker="o",
                color=BACKBONE_COLORS[backbone],
                label=f"{backbone.title()} seed range",
            )
        ax.set_title(f"Seed stability · {summary_metric}")
        ax.set_ylabel("Range across three seeds")
        ax.grid(axis="y", alpha=0.25)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [config_label(config) for config in CONFIG_ORDER],
            rotation=55,
            ha="right",
        )
        ax.legend()
    fig.tight_layout()
    writer.write(
        fig,
        section="04_stability_and_interactions",
        stem="exp1__stability__seed-range",
        source=source,
        description="Configuration-level seed ranges within each architecture.",
    )

    interactions: list[dict[str, object]] = []
    for metric in ("roc_auc", "spearman_rho"):
        data = metric_summary(metrics, metric)
        for config_id in CONFIG_ORDER:
            for seed in SEED_ORDER:
                pair = data[
                    (data["config_id"] == config_id) & (data["seed"] == seed)
                ].set_index("backbone")["mean_outcome_metric"]
                interactions.append(
                    {
                        "metric": metric,
                        "config_id": config_id,
                        "config_label": config_label(config_id),
                        "seed": seed,
                        "qwen_minus_llama": pair["qwen"] - pair["llama"],
                    }
                )
    interaction = pd.DataFrame(interactions)
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.8), sharex=True)
    for ax, metric in zip(axes, ("roc_auc", "spearman_rho")):
        data = interaction.loc[interaction["metric"] == metric]
        for seed in SEED_ORDER:
            values = (
                data.loc[data["seed"] == seed]
                .set_index("config_id")
                .reindex(CONFIG_ORDER)["qwen_minus_llama"]
            )
            ax.scatter(x, values, s=24, alpha=0.6, label=f"seed {seed}")
        average = data.groupby("config_id")["qwen_minus_llama"].mean().reindex(CONFIG_ORDER)
        ax.plot(x, average, color="#222222", linewidth=1.1, label="seed mean")
        ax.axhline(0, color="#777777", linewidth=0.8)
        ax.set_title(f"Architecture interaction · {metric}")
        ax.set_ylabel("Qwen − Llama")
        ax.grid(axis="y", alpha=0.25)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [config_label(config) for config in CONFIG_ORDER],
            rotation=55,
            ha="right",
        )
        ax.legend()
    fig.tight_layout()
    writer.write(
        fig,
        section="04_stability_and_interactions",
        stem="exp1__interaction__qwen-minus-llama",
        source=interaction,
        description="Matched-seed architecture differences in outcome-averaged performance.",
    )


def plot_training(training: pd.DataFrame, registry: pd.DataFrame, writer: FigureWriter) -> None:
    eval_rows = training.loc[training["eval_loss"].notna()].copy()
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 7.5), sharex=True, sharey=True)
    for row, backbone in enumerate(BACKBONE_ORDER):
        for col, seed in enumerate(SEED_ORDER):
            ax = axes[row, col]
            data = eval_rows[
                (eval_rows["backbone"] == backbone) & (eval_rows["seed"] == seed)
            ]
            grouped = data.groupby("progress_fraction")["eval_loss"].agg(["median", "min", "max"])
            ax.fill_between(
                grouped.index,
                grouped["min"],
                grouped["max"],
                color=BACKBONE_COLORS[backbone],
                alpha=0.16,
            )
            ax.plot(
                grouped.index,
                grouped["median"],
                color=BACKBONE_COLORS[backbone],
                linewidth=1.4,
            )
            ax.set_title(f"{backbone.title()} · seed {seed}")
            ax.grid(alpha=0.25)
            ax.set_xlim(0, 1)
    for ax in axes[-1]:
        ax.set_xlabel("Fraction of achieved training steps")
    for ax in axes[:, 0]:
        ax.set_ylabel("Evaluation loss")
    fig.suptitle("Exp1 training trajectories: median and configuration range")
    fig.tight_layout()
    writer.write(
        fig,
        section="05_training_diagnostics",
        stem="exp1__training__eval-loss-trajectories",
        source=eval_rows,
        description=(
            "Evaluation-loss trajectories faceted by backbone and seed; the band "
            "spans the 12 configurations."
        ),
    )

    exposure = registry.copy()
    exposure["achieved_fraction"] = exposure["global_step"] / exposure["max_steps"]
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 7.0), sharex=True, sharey=True)
    for row, backbone in enumerate(BACKBONE_ORDER):
        for col, seed in enumerate(SEED_ORDER):
            ax = axes[row, col]
            data = exposure[
                (exposure["backbone"] == backbone) & (exposure["seed"] == seed)
            ].set_index("config_id").reindex(CONFIG_ORDER)
            ax.barh(
                np.arange(len(CONFIG_ORDER)),
                data["achieved_fraction"],
                color=BACKBONE_COLORS[backbone],
                alpha=0.8,
            )
            ax.set_title(f"{backbone.title()} · seed {seed}")
            ax.set_xlim(0, 1.02)
            ax.grid(axis="x", alpha=0.25)
    for ax in axes[:, 0]:
        ax.set_yticks(np.arange(len(CONFIG_ORDER)))
        ax.set_yticklabels([config_label(config) for config in CONFIG_ORDER], fontsize=6)
    fig.suptitle("Exp1 achieved training exposure")
    fig.tight_layout()
    writer.write(
        fig,
        section="05_training_diagnostics",
        stem="exp1__training__achieved-exposure",
        source=exposure,
        description="Achieved training-step fraction for every architecture, seed, and configuration.",
    )


def write_docs(
    output_root: pathlib.Path,
    *,
    writer: FigureWriter,
    coverage: pd.DataFrame,
    registry: pd.DataFrame,
) -> None:
    figure_index = pd.DataFrame(writer.index)
    figure_index.to_csv(output_root / "figure_index.csv", index=False)
    coverage_headers = list(coverage.columns)
    coverage_markdown = [
        "| " + " | ".join(coverage_headers) + " |",
        "| " + " | ".join("---" for _ in coverage_headers) + " |",
    ]
    for row in coverage.itertuples(index=False, name=None):
        coverage_markdown.append(
            "| " + " | ".join(str(value) for value in row) + " |"
        )
    lines = [
        "# ML4H figure bundle",
        "",
        "This ML4H figure bundle is independent of `ML4H2026`.",
        "",
        "## Coverage",
        "",
        *coverage_markdown,
        "",
        "## Figure index",
        "",
    ]
    for row in writer.index:
        lines.append(
            f"- [{row['stem']}]({row['pdf']}) — {row['description']} "
            f"([source]({row['source']}), [metadata]({row['metadata']}))"
    )
    lines.extend(["", "## Run dimensions", ""])
    if registry.empty:
        lines.extend(["No completed run registry was requested.", ""])
    else:
        for experiment, subset in registry.groupby("experiment", sort=True):
            spec = EXPERIMENT_SPECS[experiment]
        lines.extend(
            [
                    f"- {spec.display_name}: {len(subset)} runs",
                    f"  - Configurations: {subset['config_id'].nunique()}",
                    f"  - Test cohort: {int(subset['test_cohort_rows'].iloc[0]):,} stays",
                    f"  - Backbones: {', '.join(BACKBONE_ORDER)}",
                    f"  - Seeds: {', '.join(str(seed) for seed in SEED_ORDER)}",
                ]
            )
        lines.append("")
    (output_root / "README.md").write_text("\n".join(lines), encoding="utf-8")
    (output_root / "PROVISIONAL.md").write_text(
        "\n".join(
            [
                "# ML4H figure status",
                "",
                "Current status: provisional experiment deliverables.",
                "They are not copied into or synchronized with ML4H2026.",
                "Every image has an adjacent source table and metadata record.",
                "Seed dots and seed ranges represent trained-model variation; they are",
                "not bootstrap confidence intervals.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def seed_interval_source(metrics: pd.DataFrame) -> pd.DataFrame:
    """Collapse the three seeds into a mean point with a min-max interval."""
    selected = metrics[metrics["metric"].isin(["roc_auc", "spearman_rho"])].copy()
    grouped = (
        selected.groupby(
            ["backbone", "config_id", "task_type", "outcome", "metric"], as_index=False
        )
        .agg(
            point=("point", "mean"),
            lo=("point", "min"),
            hi=("point", "max"),
            n_seeds=("seed", "nunique"),
        )
    )
    if not (grouped["n_seeds"] == 3).all():
        raise ValueError("Every ML4H-ready point must aggregate exactly three seeds")
    grouped["regime"] = grouped["config_id"].map(MLHC_REGIME_BY_CONFIG)
    grouped["fusion"] = np.where(
        grouped["config_id"].str.contains("fusedTrue"), "Fused", "Unfused"
    )
    grouped["panel"] = np.where(grouped["metric"] == "roc_auc", "binary", "regression")
    grouped["outcome_label"] = grouped["outcome"].map(outcome_label)
    grouped["backbone_label"] = grouped["backbone"].map(BACKBONE_DISPLAY)
    return grouped


def _forest_row_banding(ax, base_y: np.ndarray) -> None:
    for index in range(len(base_y)):
        if index % 2:
            ax.axhspan(
                base_y[index] - 0.5,
                base_y[index] + 0.5,
                color=FOREST_BAND_COLOR,
                zorder=0,
            )
        if index:
            ax.axhline(
                base_y[index] + 0.5,
                color=FOREST_SEPARATOR_COLOR,
                linewidth=0.55,
                alpha=0.65,
                zorder=1,
            )


def _plot_ml4h_forest_panel(
    ax,
    source: pd.DataFrame,
    *,
    outcome_order: tuple[str, ...],
    panel_title: str,
    xlabel: str,
    x_lo_floor: float | None,
) -> None:
    base_y = np.arange(len(outcome_order), dtype=float)[::-1]
    n_handles = len(CONFIG_ORDER)
    offsets = np.linspace(
        (n_handles - 1) * FOREST_HANDLE_SPREAD,
        -(n_handles - 1) * FOREST_HANDLE_SPREAD,
        n_handles,
    )
    points: list[float] = []
    for outcome_index, outcome in enumerate(outcome_order):
        rows = source[source["outcome"] == outcome]
        for handle_index, config_id in enumerate(CONFIG_ORDER):
            row = rows[rows["config_id"] == config_id]
            if row.empty:
                continue
            record = row.iloc[0]
            color = MLHC_REGIME_COLORS[record["regime"]]
            fused = record["fusion"] == "Fused"
            y = base_y[outcome_index] + offsets[handle_index]
            points.extend([float(record["lo"]), float(record["hi"])])
            ax.hlines(
                y,
                record["lo"],
                record["hi"],
                color=color,
                linewidth=0.8,
                alpha=0.95 if fused else 0.55,
                zorder=3,
            )
            ax.scatter(
                record["point"],
                y,
                color=color,
                marker="s" if fused else "o",
                s=14,
                edgecolor="#222222",
                linewidth=0.25,
                alpha=0.95 if fused else 0.55,
                zorder=4,
            )
    ax.set_yticks(base_y)
    ax.set_yticklabels([outcome_label(outcome) for outcome in outcome_order], fontsize=6.8)
    ax.set_title(panel_title, fontsize=8.5, pad=4)
    ax.set_xlabel(xlabel, fontsize=7.5)
    ax.tick_params(axis="x", labelsize=6.8, length=2)
    ax.set_ylim(base_y.min() - 0.6, base_y.max() + 0.6)
    if points:
        low, high = float(np.min(points)), float(np.max(points))
        pad = 0.04 * max(high - low, 1e-3)
        left = low - pad
        if x_lo_floor is not None:
            left = max(left, x_lo_floor)
        ax.set_xlim(left, high + pad)
    _forest_row_banding(ax, base_y)
    ax.grid(axis="x", alpha=0.20, linewidth=0.5)


def build_ml4h_ready_outcome_forests(source: pd.DataFrame, writer: FigureWriter) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 19.0))
    for row_index, backbone in enumerate(BACKBONE_ORDER):
        for col_index, (panel, outcome_order, xlabel, floor) in enumerate(
            (
                ("binary", ML4H_BINARY_ORDER, "AUROC", 0.52),
                ("regression", ML4H_REGRESSION_ORDER, r"Spearman $\rho$", None),
            )
        ):
            subset = source[
                (source["backbone"] == backbone) & (source["panel"] == panel)
            ]
            _plot_ml4h_forest_panel(
                axes[row_index, col_index],
                subset,
                outcome_order=outcome_order,
                panel_title=(
                    f"{BACKBONE_DISPLAY[backbone]} · "
                    f"{'Binary outcomes' if panel == 'binary' else 'Regression outcomes'}"
                ),
                xlabel=xlabel,
                x_lo_floor=floor,
            )
    color_handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            color=color,
            markerfacecolor=color,
            markeredgecolor="#222222",
            linewidth=0,
            markersize=4.5,
            label=label,
        )
        for label, color in MLHC_REGIME_COLORS.items()
    ]
    marker_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            color="#4B5563",
            markerfacecolor="#4B5563",
            markeredgecolor="#222222",
            linewidth=0,
            markersize=4.5,
            alpha=alpha,
            label=label,
        )
        for label, marker, alpha in (("Unfused", "o", 0.55), ("Fused", "s", 0.95))
    ]
    fig.tight_layout(rect=[0, 0, 1, 0.945])
    fig.legend(
        handles=color_handles,
        title="Numeric tokenization regime (color)",
        loc="upper left",
        bbox_to_anchor=(0.02, 0.985),
        ncol=3,
        fontsize=7,
        title_fontsize=7.5,
        frameon=False,
    )
    fig.legend(
        handles=marker_handles,
        title="Tokenization (marker)",
        loc="upper right",
        bbox_to_anchor=(0.98, 0.985),
        ncol=2,
        fontsize=7,
        title_fontsize=7.5,
        frameon=False,
    )
    fig.suptitle(
        "Experiment 1 outcome sweep across architectures. "
        "Points are means over three training seeds; whiskers span the seed minimum to maximum.",
        y=0.999,
        fontsize=9.5,
    )
    writer.write(
        fig,
        section=ML4H_READY_SECTION,
        stem="exp1_outcome_forests",
        source=source,
        description=(
            "MLHC-style Exp1 outcome forest with architecture rows; seed mean point "
            "and seed min-max whisker across all 39 outcomes."
        ),
    )


def build_ml4h_ready_axis_summary_bars(
    source: pd.DataFrame, writer: FigureWriter
) -> None:
    frames: list[pd.DataFrame] = []
    for axis, column in (
        ("Fusion", "fusion"),
        ("Granularity", "granularity"),
        ("Anchoring", "anchoring"),
    ):
        working = source.copy()
        working["granularity"] = working["regime"].str.extract(
            r"^(Deciles|Ventiles|Trentiles|Centiles)"
        )
        working["anchoring"] = np.where(
            working["regime"].str.contains("reference-range", regex=False),
            "Clinical bins",
            "Population bins",
        )
        if axis == "Anchoring":
            working = working[working["granularity"].isin(["Ventiles", "Trentiles"])]
        aggregated = (
            working.groupby(["panel", "backbone", column], as_index=False)
            .agg(point=("point", "mean"), lo=("lo", "mean"), hi=("hi", "mean"))
            .rename(columns={column: "level"})
        )
        aggregated["axis"] = axis
        frames.append(aggregated)
    summary = pd.concat(frames, ignore_index=True)

    axis_levels = {
        "Fusion": ["Unfused", "Fused"],
        "Granularity": ["Deciles", "Ventiles", "Trentiles", "Centiles"],
        "Anchoring": ["Population bins", "Clinical bins"],
    }
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.6))
    for row_index, (panel, ylabel, floor) in enumerate(
        (
            ("binary", "Mean AUROC across outcomes", 0.5),
            ("regression", r"Mean Spearman $\rho$ across outcomes", 0.0),
        )
    ):
        for col_index, (axis, levels) in enumerate(axis_levels.items()):
            ax = axes[row_index, col_index]
            data = summary[(summary["panel"] == panel) & (summary["axis"] == axis)]
            x = np.arange(len(levels), dtype=float)
            width = 0.36
            for offset, backbone in zip((-width / 2, width / 2), BACKBONE_ORDER):
                values = (
                    data[data["backbone"] == backbone]
                    .set_index("level")
                    .reindex(levels)
                )
                ax.bar(
                    x + offset,
                    values["point"],
                    width=width,
                    color=BACKBONE_COLORS[backbone],
                    alpha=0.92,
                    edgecolor="#FFFFFF",
                    label=BACKBONE_DISPLAY[backbone] if row_index == 0 and col_index == 0 else None,
                )
                ax.errorbar(
                    x + offset,
                    values["point"],
                    yerr=np.vstack(
                        [
                            values["point"] - values["lo"],
                            values["hi"] - values["point"],
                        ]
                    ),
                    fmt="none",
                    ecolor="#2F343A",
                    elinewidth=0.8,
                    capsize=2,
                )
            ax.set_xticks(x)
            ax.set_xticklabels([level.replace(" ", "\n") for level in levels], fontsize=7)
            ax.set_ylim(bottom=floor)
            ax.grid(axis="y", alpha=0.25, color="#E5E7EB")
            if row_index == 0:
                ax.set_title(axis, fontsize=9.2)
            if col_index == 0:
                ax.set_ylabel(ylabel, fontsize=8.4)
    axes[0, 0].legend(loc="lower right", fontsize=7)
    fig.suptitle(
        "Experiment 1 axis comparisons by architecture. "
        "Bars are outcome means of the seed mean; whiskers are outcome means of the seed range.",
        fontsize=9.5,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    writer.write(
        fig,
        section=ML4H_READY_SECTION,
        stem="exp1_axis_summary_bars",
        source=summary,
        description=(
            "MLHC-style Exp1 axis summary bars for fusion, granularity, and anchoring, "
            "grouped by architecture."
        ),
    )


def build_ml4h_ready_training_curves(
    training: pd.DataFrame, writer: FigureWriter
) -> None:
    evaluations = training.loc[training["eval_loss"].notna()].copy()
    evaluations["regime"] = evaluations["config_id"].map(MLHC_REGIME_BY_CONFIG)
    evaluations["fusion"] = np.where(
        evaluations["config_id"].str.contains("fusedTrue"), "Fused", "Unfused"
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), sharey=True)
    for ax, backbone in zip(axes, BACKBONE_ORDER):
        subset = evaluations[evaluations["backbone"] == backbone]
        for config_id in CONFIG_ORDER:
            config_rows = subset[subset["config_id"] == config_id]
            if config_rows.empty:
                continue
            grouped = config_rows.groupby("progress_fraction")["eval_loss"].agg(
                ["mean", "min", "max"]
            )
            color = MLHC_REGIME_COLORS[MLHC_REGIME_BY_CONFIG[config_id]]
            fused = "fusedTrue" in config_id
            ax.fill_between(
                grouped.index,
                grouped["min"],
                grouped["max"],
                color=color,
                alpha=0.12,
                linewidth=0,
            )
            ax.plot(
                grouped.index,
                grouped["mean"],
                color=color,
                linewidth=1.2,
                linestyle="-" if fused else "--",
                alpha=0.95 if fused else 0.7,
            )
        ax.set_title(BACKBONE_DISPLAY[backbone], fontsize=9.2)
        ax.set_xlabel("Fraction of achieved training steps", fontsize=8.4)
        ax.grid(alpha=0.25)
        ax.set_xlim(0, 1)
    axes[0].set_ylabel("Validation loss", fontsize=8.4)
    legend_handles = [
        Line2D([0], [0], color=color, linewidth=1.4, label=label)
        for label, color in MLHC_REGIME_COLORS.items()
    ] + [
        Line2D([0], [0], color="#4B5563", linewidth=1.4, linestyle="--", label="Unfused"),
        Line2D([0], [0], color="#4B5563", linewidth=1.4, linestyle="-", label="Fused"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=4,
        fontsize=7,
        frameon=False,
    )
    fig.suptitle(
        "Experiment 1 validation loss. Lines are seed means; bands span the seed range.",
        y=0.855,
        fontsize=9.5,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.80])
    writer.write(
        fig,
        section=ML4H_READY_SECTION,
        stem="exp1_training_validation_loss",
        source=evaluations,
        description=(
            "MLHC-style validation-loss curves faceted by architecture with seed-range bands."
        ),
    )


def _generic_stem(spec: ExperimentSpec, suffix: str) -> str:
    return f"{spec.experiment}__{suffix}"


def _generic_primary_outcomes(
    metrics: pd.DataFrame, metric: str
) -> tuple[str, ...]:
    preferred = PRIMARY_BINARY if metric == "roc_auc" else PRIMARY_REGRESSION
    available = set(metrics.loc[metrics["metric"] == metric, "outcome"])
    return tuple(outcome for outcome in preferred if outcome in available)


def plot_generic_overview(
    registry: pd.DataFrame, writer: FigureWriter, spec: ExperimentSpec
) -> None:
    configs = list(spec.config_order)
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 6.3), sharex=True, sharey=True)
    for row, backbone in enumerate(BACKBONE_ORDER):
        for col, seed in enumerate(SEED_ORDER):
            ax = axes[row, col]
            present = set(
                registry.loc[
                    (registry["backbone"] == backbone) & (registry["seed"] == seed),
                    "config_id",
                ]
            )
            values = np.array([int(config in present) for config in configs])
            ax.bar(
                np.arange(len(configs)),
                values,
                color=[experiment_config_color(spec, config) for config in configs],
                edgecolor="#FFFFFF",
                linewidth=0.4,
            )
            ax.set_title(f"{backbone.title()} · seed {seed}")
            ax.set_ylim(-0.05, 1.05)
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["Missing", "Present"])
            ax.set_xticks(np.arange(len(configs)))
            ax.set_xticklabels(
                [experiment_config_label(spec, config) for config in configs],
                rotation=55,
                ha="right",
                fontsize=5.6,
            )
    fig.suptitle(
        f"{spec.display_name} design coverage: every configuration in every architecture × seed cell"
    )
    fig.tight_layout()
    writer.write(
        fig,
        section="01_overview",
        stem=_generic_stem(spec, "overview__design-matrix"),
        source=registry,
        description=(
            f"Coverage map for all {spec.expected_runs} {spec.display_name.lower()} model runs."
        ),
    )


def plot_generic_summary(
    metrics: pd.DataFrame, writer: FigureWriter, spec: ExperimentSpec
) -> None:
    source = pd.concat(
        [
            metric_summary(metrics, "roc_auc").assign(summary_metric="AUROC"),
            metric_summary(metrics, "spearman_rho").assign(
                summary_metric="Spearman rho"
            ),
        ],
        ignore_index=True,
    )
    configs = list(spec.config_order)
    x = np.arange(len(configs))
    fig, axes = plt.subplots(2, 2, figsize=(14.2, 8.4), sharex=True)
    for row, summary_metric in enumerate(("AUROC", "Spearman rho")):
        for col, backbone in enumerate(BACKBONE_ORDER):
            ax = axes[row, col]
            subset = source[
                (source["summary_metric"] == summary_metric)
                & (source["backbone"] == backbone)
            ]
            for seed_index, seed in enumerate(SEED_ORDER):
                values = (
                    subset.loc[subset["seed"] == seed]
                    .set_index("config_id")
                    .reindex(configs)["mean_outcome_metric"]
                )
                ax.scatter(
                    x + (seed_index - 1) * 0.12,
                    values,
                    color=BACKBONE_COLORS[backbone],
                    alpha=0.42,
                    s=24,
                    label=f"seed {seed}" if row == 0 and col == 0 else None,
                )
            mean = subset.groupby("config_id")["mean_outcome_metric"].mean().reindex(configs)
            spread = (
                subset.groupby("config_id")["mean_outcome_metric"]
                .agg(lambda values: values.max() - values.min())
                .reindex(configs)
            )
            ax.errorbar(
                x,
                mean,
                yerr=spread / 2,
                color=BACKBONE_COLORS[backbone],
                marker="o",
                linewidth=1.2,
                capsize=2,
                label="seed mean ± half range"
                if row == 0 and col == 0
                else None,
            )
            ax.set_title(f"{backbone.title()} · {summary_metric}")
            ax.grid(axis="y", alpha=0.25)
            if summary_metric == "AUROC":
                ax.set_ylim(0.5, 1.0)
            else:
                ax.set_ylim(bottom=0.0)
    for ax in axes[-1]:
        ax.set_xticks(x)
        ax.set_xticklabels(
            [experiment_config_label(spec, config) for config in configs],
            rotation=55,
            ha="right",
            fontsize=6,
        )
    axes[0, 0].legend(loc="lower right")
    fig.suptitle(
        f"{spec.display_name} outcome-averaged performance: every seed is visible; bars show seed range"
    )
    fig.tight_layout()
    writer.write(
        fig,
        section="02_summary",
        stem=_generic_stem(spec, "summary__architecture-stratified-performance"),
        source=source,
        description=(
            "Architecture-stratified outcome-average performance with all three "
            "seed points and architecture-specific seed ranges."
        ),
    )


def plot_generic_heatmaps(
    metrics: pd.DataFrame,
    writer: FigureWriter,
    spec: ExperimentSpec,
    metric: str,
) -> None:
    subset = metrics.loc[metrics["metric"] == metric].copy()
    task_label = "binary" if metric == "roc_auc" else "regression"
    configs = list(spec.config_order)
    outcomes = list(
        (
            ML4H_BINARY_ORDER if task_label == "binary" else ML4H_REGRESSION_ORDER
        )
    )
    outcomes = [outcome for outcome in outcomes if outcome in set(subset["outcome"])]
    fig, axes = plt.subplots(2, 2, figsize=(16.0, 10.0), constrained_layout=True)
    vmin, vmax = float(subset["point"].min()), float(subset["point"].max())
    for row, backbone in enumerate(BACKBONE_ORDER):
        architecture = subset.loc[subset["backbone"] == backbone]
        mean = (
            architecture.groupby(["config_id", "outcome"])["point"]
            .mean()
            .unstack("outcome")
            .reindex(index=configs, columns=outcomes)
        )
        spread = (
            architecture.groupby(["config_id", "outcome"])["point"]
            .agg(lambda values: values.max() - values.min())
            .unstack("outcome")
            .reindex(index=configs, columns=outcomes)
        )
        image = axes[row, 0].imshow(
            mean.to_numpy(), aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax
        )
        axes[row, 0].set_title(f"{backbone.title()} mean {metric}")
        axes[row, 1].imshow(
            spread.to_numpy(),
            aspect="auto",
            cmap="magma",
            vmin=0,
            vmax=max(float(spread.max().max()), 1e-6),
        )
        axes[row, 1].set_title(f"{backbone.title()} seed range")
        for col in range(2):
            axes[row, col].set_yticks(np.arange(len(configs)))
            axes[row, col].set_yticklabels(
                [experiment_config_label(spec, config) for config in configs],
                fontsize=5.8,
            )
            axes[row, col].set_xticks(np.arange(len(outcomes)))
            axes[row, col].set_xticklabels(
                [outcome_label(outcome) for outcome in outcomes],
                rotation=60,
                ha="right",
                fontsize=5.8,
            )
    fig.colorbar(image, ax=axes[:, 0], shrink=0.8, label=metric)
    fig.suptitle(
        f"{spec.display_name} {task_label} outcome heatmap: mean performance and seed disagreement"
    )
    writer.write(
        fig,
        section="02_summary",
        stem=_generic_stem(spec, f"summary__{task_label}-outcome-heatmap"),
        source=subset,
        description=(
            f"{task_label.title()} outcome means and seed ranges for every "
            "configuration and architecture."
        ),
    )


def plot_generic_primary_outcomes(
    metrics: pd.DataFrame,
    writer: FigureWriter,
    spec: ExperimentSpec,
    *,
    metric: str,
) -> None:
    outcomes = _generic_primary_outcomes(metrics, metric)
    task_label = "binary" if metric == "roc_auc" else "regression"
    subset = metrics[
        (metrics["metric"] == metric) & metrics["outcome"].isin(outcomes)
    ].copy()
    configs = list(spec.config_order)
    fig, axes = plt.subplots(1, len(outcomes), figsize=(15.0, 8.5), sharey=True)
    axes = np.atleast_1d(axes)
    y = np.arange(len(configs))[::-1]
    for ax, outcome in zip(axes, outcomes):
        data = subset.loc[subset["outcome"] == outcome]
        for backbone, offset, marker in (("llama", -0.12, "o"), ("qwen", 0.12, "s")):
            architecture = data.loc[data["backbone"] == backbone]
            for seed_index, seed in enumerate(SEED_ORDER):
                values = (
                    architecture.loc[architecture["seed"] == seed]
                    .set_index("config_id")
                    .reindex(configs)["point"]
                )
                ax.scatter(
                    values,
                    y + offset + (seed_index - 1) * 0.035,
                    s=18,
                    marker=marker,
                    color=BACKBONE_COLORS[backbone],
                    alpha=0.38,
                )
            summary = architecture.groupby("config_id")["point"].agg(["mean", "min", "max"])
            summary = summary.reindex(configs)
            ax.errorbar(
                summary["mean"],
                y + offset,
                xerr=np.vstack(
                    [summary["mean"] - summary["min"], summary["max"] - summary["mean"]]
                ),
                fmt=marker,
                color=BACKBONE_COLORS[backbone],
                markersize=4,
                linewidth=0.8,
                capsize=1.5,
                label=backbone.title() if outcome == outcomes[0] else None,
            )
        ax.set_title(outcome_label(outcome))
        ax.grid(axis="x", alpha=0.25)
        ax.set_xlabel("AUROC" if metric == "roc_auc" else r"Spearman $\rho$")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(
        [experiment_config_label(spec, config) for config in configs], fontsize=6
    )
    axes[0].legend(loc="lower right")
    fig.suptitle(
        f"{spec.display_name} primary {task_label} outcomes: architecture means, seed ranges, and seed points"
    )
    fig.tight_layout()
    writer.write(
        fig,
        section="02_summary",
        stem=_generic_stem(spec, f"summary__primary-{task_label}-outcomes"),
        source=subset,
        description=(
            f"Primary {task_label} outcome forest plots retaining every seed and "
            "both architectures."
        ),
    )


def plot_generic_atlas(
    metrics: pd.DataFrame, writer: FigureWriter, spec: ExperimentSpec
) -> None:
    configs = list(spec.config_order)
    for backbone in BACKBONE_ORDER:
        for seed in SEED_ORDER:
            subset = metrics[
                (metrics["backbone"] == backbone) & (metrics["seed"] == seed)
            ]
            fig, axes = plt.subplots(2, 1, figsize=(17.0, 9.2), constrained_layout=True)
            for ax, metric, title in (
                (axes[0], "roc_auc", "Binary outcomes · AUROC"),
                (axes[1], "spearman_rho", r"Regression outcomes · Spearman $\rho$"),
            ):
                data = subset.loc[subset["metric"] == metric]
                outcomes = list(
                    (
                        ML4H_BINARY_ORDER
                        if metric == "roc_auc"
                        else ML4H_REGRESSION_ORDER
                    )
                )
                outcomes = [outcome for outcome in outcomes if outcome in set(data["outcome"])]
                matrix = (
                    data.pivot(index="config_id", columns="outcome", values="point")
                    .reindex(index=configs, columns=outcomes)
                )
                image = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="viridis")
                ax.set_title(title)
                ax.set_yticks(np.arange(len(configs)))
                ax.set_yticklabels(
                    [experiment_config_label(spec, config) for config in configs],
                    fontsize=6,
                )
                ax.set_xticks(np.arange(len(outcomes)))
                ax.set_xticklabels(
                    [outcome_label(outcome) for outcome in outcomes],
                    rotation=55,
                    ha="right",
                    fontsize=6,
                )
                fig.colorbar(image, ax=ax, shrink=0.75)
            fig.suptitle(f"{spec.display_name} audit atlas · {backbone.title()} · seed {seed}")
            writer.write(
                fig,
                section="03_seed_architecture_atlas",
                stem=_generic_stem(spec, f"atlas__{backbone}__seed-{seed}"),
                source=subset,
                description=(
                    f"All {len(configs)} configuration runs for one architecture and one seed."
                ),
            )


def plot_generic_stability(
    metrics: pd.DataFrame, writer: FigureWriter, spec: ExperimentSpec
) -> None:
    source = pd.concat(
        [
            metric_summary(metrics, "roc_auc").assign(summary_metric="AUROC"),
            metric_summary(metrics, "spearman_rho").assign(
                summary_metric="Spearman rho"
            ),
        ],
        ignore_index=True,
    )
    configs = list(spec.config_order)
    x = np.arange(len(configs))
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.8), sharex=True)
    for ax, summary_metric in zip(axes, ("AUROC", "Spearman rho")):
        data = source.loc[source["summary_metric"] == summary_metric]
        for backbone in BACKBONE_ORDER:
            architecture = data.loc[data["backbone"] == backbone]
            for seed in SEED_ORDER:
                values = (
                    architecture.loc[architecture["seed"] == seed]
                    .set_index("config_id")
                    .reindex(configs)["mean_outcome_metric"]
                )
                ax.plot(
                    x,
                    values,
                    color=BACKBONE_COLORS[backbone],
                    alpha=0.25,
                    linewidth=0.8,
                )
            spread = (
                architecture.groupby("config_id")["mean_outcome_metric"]
                .agg(lambda values: values.max() - values.min())
                .reindex(configs)
            )
            ax.plot(
                x,
                spread,
                marker="o",
                color=BACKBONE_COLORS[backbone],
                label=f"{backbone.title()} seed range",
            )
        ax.set_title(f"Seed stability · {summary_metric}")
        ax.set_ylabel("Range across three seeds")
        ax.grid(axis="y", alpha=0.25)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [experiment_config_label(spec, config) for config in configs],
            rotation=55,
            ha="right",
            fontsize=6,
        )
        ax.legend()
    fig.tight_layout()
    writer.write(
        fig,
        section="04_stability_and_interactions",
        stem=_generic_stem(spec, "stability__seed-range"),
        source=source,
        description="Configuration-level seed ranges within each architecture.",
    )

    interactions: list[dict[str, object]] = []
    for metric in ("roc_auc", "spearman_rho"):
        data = metric_summary(metrics, metric)
        for config_id in configs:
            for seed in SEED_ORDER:
                pair = data[
                    (data["config_id"] == config_id) & (data["seed"] == seed)
                ].set_index("backbone")["mean_outcome_metric"]
                interactions.append(
                    {
                        "metric": metric,
                        "config_id": config_id,
                        "config_label": experiment_config_label(spec, config_id),
                        "seed": seed,
                        "qwen_minus_llama": pair["qwen"] - pair["llama"],
                    }
                )
    interaction = pd.DataFrame(interactions)
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.8), sharex=True)
    for ax, metric in zip(axes, ("roc_auc", "spearman_rho")):
        data = interaction.loc[interaction["metric"] == metric]
        for seed in SEED_ORDER:
            values = (
                data.loc[data["seed"] == seed]
                .set_index("config_id")
                .reindex(configs)["qwen_minus_llama"]
            )
            ax.scatter(x, values, s=24, alpha=0.6, label=f"seed {seed}")
        average = data.groupby("config_id")["qwen_minus_llama"].mean().reindex(configs)
        ax.plot(x, average, color="#222222", linewidth=1.1, label="seed mean")
        ax.axhline(0, color="#777777", linewidth=0.8)
        ax.set_title(f"Architecture interaction · {metric}")
        ax.set_ylabel("Qwen − Llama")
        ax.grid(axis="y", alpha=0.25)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [experiment_config_label(spec, config) for config in configs],
            rotation=55,
            ha="right",
            fontsize=6,
        )
        ax.legend()
    fig.tight_layout()
    writer.write(
        fig,
        section="04_stability_and_interactions",
        stem=_generic_stem(spec, "interaction__qwen-minus-llama"),
        source=interaction,
        description="Matched-seed architecture differences in outcome-averaged performance.",
    )


def plot_generic_training(
    training: pd.DataFrame,
    registry: pd.DataFrame,
    writer: FigureWriter,
    spec: ExperimentSpec,
) -> None:
    configs = list(spec.config_order)
    eval_rows = training.loc[training["eval_loss"].notna()].copy()
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 7.5), sharex=True, sharey=True)
    for row, backbone in enumerate(BACKBONE_ORDER):
        for col, seed in enumerate(SEED_ORDER):
            ax = axes[row, col]
            data = eval_rows[
                (eval_rows["backbone"] == backbone) & (eval_rows["seed"] == seed)
            ]
            grouped = data.groupby("progress_fraction")["eval_loss"].agg(["median", "min", "max"])
            ax.fill_between(
                grouped.index,
                grouped["min"],
                grouped["max"],
                color=BACKBONE_COLORS[backbone],
                alpha=0.16,
            )
            ax.plot(
                grouped.index,
                grouped["median"],
                color=BACKBONE_COLORS[backbone],
                linewidth=1.4,
            )
            ax.set_title(f"{backbone.title()} · seed {seed}")
            ax.grid(alpha=0.25)
            ax.set_xlim(0, 1)
    for ax in axes[-1]:
        ax.set_xlabel("Fraction of achieved training steps")
    for ax in axes[:, 0]:
        ax.set_ylabel("Evaluation loss")
    fig.suptitle(f"{spec.display_name} training trajectories: median and configuration range")
    fig.tight_layout()
    writer.write(
        fig,
        section="05_training_diagnostics",
        stem=_generic_stem(spec, "training__eval-loss-trajectories"),
        source=eval_rows,
        description=(
            "Evaluation-loss trajectories faceted by backbone and seed; the band "
            "spans the experiment configurations."
        ),
    )

    exposure = registry.copy()
    exposure["achieved_fraction"] = exposure["global_step"] / exposure["max_steps"]
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 7.0), sharex=True, sharey=True)
    for row, backbone in enumerate(BACKBONE_ORDER):
        for col, seed in enumerate(SEED_ORDER):
            ax = axes[row, col]
            data = exposure[
                (exposure["backbone"] == backbone) & (exposure["seed"] == seed)
            ].set_index("config_id").reindex(configs)
            ax.barh(
                np.arange(len(configs)),
                data["achieved_fraction"],
                color=BACKBONE_COLORS[backbone],
                alpha=0.8,
            )
            ax.set_title(f"{backbone.title()} · seed {seed}")
            ax.set_xlim(0, 1.02)
            ax.grid(axis="x", alpha=0.25)
    for ax in axes[:, 0]:
        ax.set_yticks(np.arange(len(configs)))
        ax.set_yticklabels(
            [experiment_config_label(spec, config) for config in configs], fontsize=6
        )
    fig.suptitle(f"{spec.display_name} achieved training exposure")
    fig.tight_layout()
    writer.write(
        fig,
        section="05_training_diagnostics",
        stem=_generic_stem(spec, "training__achieved-exposure"),
        source=exposure,
        description="Achieved training-step fraction for every architecture, seed, and configuration.",
    )


def generic_seed_interval_source(
    metrics: pd.DataFrame, spec: ExperimentSpec
) -> pd.DataFrame:
    selected = metrics[metrics["metric"].isin(["roc_auc", "spearman_rho"])].copy()
    grouped = (
        selected.groupby(
            ["backbone", "config_id", "config_label", "task_type", "outcome", "metric"],
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
        raise ValueError(
            f"Every {spec.experiment} point must aggregate exactly three seeds"
        )
    grouped["panel"] = np.where(
        grouped["metric"] == "roc_auc", "binary", "regression"
    )
    grouped["outcome_label"] = grouped["outcome"].map(outcome_label)
    grouped["backbone_label"] = grouped["backbone"].map(BACKBONE_DISPLAY)
    return grouped


def _generic_forest_panel(
    ax,
    source: pd.DataFrame,
    *,
    spec: ExperimentSpec,
    outcome_order: tuple[str, ...],
    panel_title: str,
    xlabel: str,
    x_lo_floor: float | None,
) -> None:
    base_y = np.arange(len(outcome_order), dtype=float)[::-1]
    configs = list(spec.config_order)
    offsets = np.linspace(
        (len(configs) - 1) * FOREST_HANDLE_SPREAD,
        -(len(configs) - 1) * FOREST_HANDLE_SPREAD,
        len(configs),
    )
    points: list[float] = []
    for outcome_index, outcome in enumerate(outcome_order):
        rows = source[source["outcome"] == outcome]
        for config_index, config_id in enumerate(configs):
            row = rows[rows["config_id"] == config_id]
            if row.empty:
                continue
            record = row.iloc[0]
            y = base_y[outcome_index] + offsets[config_index]
            points.extend([float(record["lo"]), float(record["hi"])])
            color = experiment_config_color(spec, config_id)
            ax.hlines(
                y,
                record["lo"],
                record["hi"],
                color=color,
                linewidth=0.8,
                alpha=0.9,
                zorder=3,
            )
            ax.scatter(
                record["point"],
                y,
                color=color,
                marker=experiment_config_marker(spec, config_id),
                s=14,
                edgecolor="#222222",
                linewidth=0.25,
                zorder=4,
            )
    ax.set_yticks(base_y)
    ax.set_yticklabels([outcome_label(outcome) for outcome in outcome_order], fontsize=6.8)
    ax.set_title(panel_title, fontsize=8.5, pad=4)
    ax.set_xlabel(xlabel, fontsize=7.5)
    ax.tick_params(axis="x", labelsize=6.8, length=2)
    ax.set_ylim(base_y.min() - 0.6, base_y.max() + 0.6)
    if points:
        low, high = float(np.min(points)), float(np.max(points))
        pad = 0.04 * max(high - low, 1e-3)
        left = max(low - pad, x_lo_floor) if x_lo_floor is not None else low - pad
        ax.set_xlim(left, high + pad)
    _forest_row_banding(ax, base_y)
    ax.grid(axis="x", alpha=0.20, linewidth=0.5)


def build_generic_outcome_forests(
    source: pd.DataFrame, writer: FigureWriter, spec: ExperimentSpec
) -> None:
    binary_order = tuple(
        outcome for outcome in ML4H_BINARY_ORDER if outcome in set(source["outcome"])
    )
    regression_order = tuple(
        outcome
        for outcome in ML4H_REGRESSION_ORDER
        if outcome in set(source["outcome"])
    )
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 19.0))
    for row_index, backbone in enumerate(BACKBONE_ORDER):
        for col_index, (panel, outcome_order, xlabel, floor) in enumerate(
            (
                ("binary", binary_order, "AUROC", 0.52),
                ("regression", regression_order, r"Spearman $\rho$", None),
            )
        ):
            subset = source[
                (source["backbone"] == backbone) & (source["panel"] == panel)
            ]
            _generic_forest_panel(
                axes[row_index, col_index],
                subset,
                spec=spec,
                outcome_order=outcome_order,
                panel_title=(
                    f"{BACKBONE_DISPLAY[backbone]} · "
                    f"{'Binary outcomes' if panel == 'binary' else 'Regression outcomes'}"
                ),
                xlabel=xlabel,
                x_lo_floor=floor,
            )
    handles = [
        Line2D(
            [0],
            [0],
            marker=experiment_config_marker(spec, config),
            color=experiment_config_color(spec, config),
            markerfacecolor=experiment_config_color(spec, config),
            markeredgecolor="#222222",
            linewidth=0,
            markersize=4.5,
            label=experiment_config_label(spec, config),
        )
        for config in spec.config_order
    ]
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.legend(
        handles=handles,
        title="Configuration",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=min(4, len(handles)),
        fontsize=6.5,
        title_fontsize=7,
        frameon=False,
    )
    fig.suptitle(
        f"{spec.display_name} outcome sweep. Points are means over three training seeds; "
        "whiskers span the seed minimum to maximum.",
        y=0.999,
        fontsize=9.5,
    )
    writer.write(
        fig,
        section=ML4H_READY_SECTION,
        stem=_generic_stem(spec, "outcome_forests"),
        source=source,
        description=(
            f"{spec.display_name} outcome forest with architecture rows; seed mean "
            "points and seed min-max whiskers."
        ),
    )


def generic_axis_summary_source(
    metrics: pd.DataFrame, spec: ExperimentSpec
) -> pd.DataFrame:
    selected = metrics[metrics["metric"].isin(["roc_auc", "spearman_rho"])].copy()
    seed_summary = (
        selected.groupby(
            [
                "backbone",
                "config_id",
                "task_type",
                "outcome",
                "metric",
                "quantizer",
                "anchoring",
                "fusion",
                "representation",
                "temporal",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(point=("point", "mean"), lo=("point", "min"), hi=("point", "max"))
    )
    seed_summary["panel"] = np.where(
        seed_summary["metric"] == "roc_auc", "binary", "regression"
    )
    frames: list[pd.DataFrame] = []
    if spec.experiment == "exp1":
        work = seed_summary.copy()
        work["granularity"] = work["quantizer"].str.replace("iles", "iles", regex=False).str.title()
        work["anchoring_level"] = np.where(
            work["anchoring"] == "none", "Population bins", "Clinical bins"
        )
        frames.extend(
            [
                work.assign(axis="Fusion", level=work["fusion"].str.title()),
                work.assign(axis="Granularity", level=work["granularity"]),
                work[
                    work["quantizer"].isin(["ventiles", "trentiles"])
                ].assign(axis="Anchoring", level=work["anchoring_level"]),
            ]
        )
    elif spec.experiment == "exp2":
        representation_labels = {
            "discrete": "Discrete",
            "soft": "Soft",
            "xval": "xVal",
            "xval_affine": "xVal-affine",
        }
        temporal_labels = {
            "event_order": "Event order",
            "time_tokens": "Time tokens",
            "time_rope": "Time RoPE",
        }
        frames.extend(
            [
                seed_summary.assign(
                    axis="Value encoder",
                    level=seed_summary["representation"].map(representation_labels),
                ),
                seed_summary.assign(
                    axis="Temporal encoding",
                    level=seed_summary["temporal"].map(temporal_labels),
                ),
            ]
        )
    elif spec.experiment == "exp3":
        work = seed_summary.copy()
        work["level"] = np.where(
            work["config_id"].str.startswith("meds_icu_"),
            "Native MIMIC",
            "CLIF harmonized",
        )
        frames.append(work.assign(axis="Vocabulary arm"))
    else:
        raise ValueError(f"Unsupported experiment: {spec.experiment}")
    merged = pd.concat(frames, ignore_index=True)
    return (
        merged.groupby(["panel", "backbone", "axis", "level"], as_index=False)
        .agg(point=("point", "mean"), lo=("lo", "mean"), hi=("hi", "mean"))
    )


def build_generic_axis_summary_bars(
    metrics: pd.DataFrame, writer: FigureWriter, spec: ExperimentSpec
) -> None:
    summary = generic_axis_summary_source(metrics, spec)
    axis_order = list(dict.fromkeys(summary["axis"]))
    fig, axes = plt.subplots(
        2, len(axis_order), figsize=(4.2 * len(axis_order), 7.6), squeeze=False
    )
    for row_index, (panel, ylabel, floor) in enumerate(
        (
            ("binary", "Mean AUROC across outcomes", 0.5),
            ("regression", r"Mean Spearman $\rho$ across outcomes", 0.0),
        )
    ):
        for col_index, axis in enumerate(axis_order):
            ax = axes[row_index, col_index]
            data = summary[
                (summary["panel"] == panel) & (summary["axis"] == axis)
            ]
            levels = list(dict.fromkeys(data["level"]))
            x = np.arange(len(levels), dtype=float)
            width = 0.36
            for offset, backbone in zip((-width / 2, width / 2), BACKBONE_ORDER):
                values = data[data["backbone"] == backbone].set_index("level").reindex(levels)
                ax.bar(
                    x + offset,
                    values["point"],
                    width=width,
                    color=BACKBONE_COLORS[backbone],
                    alpha=0.92,
                    edgecolor="#FFFFFF",
                    label=BACKBONE_DISPLAY[backbone]
                    if row_index == 0 and col_index == 0
                    else None,
                )
                ax.errorbar(
                    x + offset,
                    values["point"],
                    yerr=np.vstack(
                        [
                            values["point"] - values["lo"],
                            values["hi"] - values["point"],
                        ]
                    ),
                    fmt="none",
                    ecolor="#2F343A",
                    elinewidth=0.8,
                    capsize=2,
                )
            ax.set_xticks(x)
            ax.set_xticklabels([level.replace(" ", "\n") for level in levels], fontsize=7)
            ax.set_ylim(bottom=floor)
            ax.grid(axis="y", alpha=0.25, color="#E5E7EB")
            if row_index == 0:
                ax.set_title(axis, fontsize=9.2)
            if col_index == 0:
                ax.set_ylabel(ylabel, fontsize=8.4)
    axes[0, 0].legend(loc="lower right", fontsize=7)
    fig.suptitle(
        f"{spec.display_name} axis comparisons by architecture. Bars are outcome means; "
        "whiskers are means of three-seed ranges.",
        fontsize=9.5,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    writer.write(
        fig,
        section=ML4H_READY_SECTION,
        stem=_generic_stem(spec, "axis_summary_bars"),
        source=summary,
        description=(
            f"{spec.display_name} axis summary bars grouped by architecture."
        ),
    )


def build_generic_training_validation_loss(
    training: pd.DataFrame, writer: FigureWriter, spec: ExperimentSpec
) -> None:
    evaluations = training.loc[training["eval_loss"].notna()].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), sharey=True)
    for ax, backbone in zip(axes, BACKBONE_ORDER):
        subset = evaluations[evaluations["backbone"] == backbone]
        for config_id in spec.config_order:
            config_rows = subset[subset["config_id"] == config_id]
            if config_rows.empty:
                continue
            grouped = config_rows.groupby("progress_fraction")["eval_loss"].agg(
                ["mean", "min", "max"]
            )
            color = experiment_config_color(spec, config_id)
            ax.fill_between(
                grouped.index,
                grouped["min"],
                grouped["max"],
                color=color,
                alpha=0.12,
                linewidth=0,
            )
            ax.plot(
                grouped.index,
                grouped["mean"],
                color=color,
                linewidth=1.2,
                linestyle="-" if experiment_config_marker(spec, config_id) != "o" else "--",
                alpha=0.9,
            )
        ax.set_title(BACKBONE_DISPLAY[backbone], fontsize=9.2)
        ax.set_xlabel("Fraction of achieved training steps", fontsize=8.4)
        ax.grid(alpha=0.25)
        ax.set_xlim(0, 1)
    axes[0].set_ylabel("Validation loss", fontsize=8.4)
    handles = [
        Line2D(
            [0],
            [0],
            color=experiment_config_color(spec, config),
            linewidth=1.4,
            label=experiment_config_label(spec, config),
        )
        for config in spec.config_order
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=min(4, len(handles)),
        fontsize=6.4,
        frameon=False,
    )
    fig.suptitle(
        f"{spec.display_name} validation loss. Lines are seed means; bands span the seed range.",
        y=0.855,
        fontsize=9.5,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.80])
    writer.write(
        fig,
        section=ML4H_READY_SECTION,
        stem=_generic_stem(spec, "training_validation_loss"),
        source=evaluations,
        description=(
            "Validation-loss curves faceted by architecture with seed-range bands."
        ),
    )


def render_generic_suite(
    registry: pd.DataFrame,
    metrics: pd.DataFrame,
    training: pd.DataFrame,
    writer: FigureWriter,
    spec: ExperimentSpec,
) -> None:
    """Render the 19-panel audit suite for Exp2 and Exp3."""
    plot_generic_overview(registry, writer, spec)
    plot_generic_summary(metrics, writer, spec)
    plot_generic_heatmaps(metrics, writer, spec, "roc_auc")
    plot_generic_heatmaps(metrics, writer, spec, "spearman_rho")
    plot_generic_primary_outcomes(metrics, writer, spec, metric="roc_auc")
    plot_generic_primary_outcomes(metrics, writer, spec, metric="spearman_rho")
    plot_generic_atlas(metrics, writer, spec)
    plot_generic_stability(metrics, writer, spec)
    plot_generic_training(training, registry, writer, spec)
    seed_source = generic_seed_interval_source(metrics, spec)
    build_generic_outcome_forests(seed_source, writer, spec)
    build_generic_axis_summary_bars(metrics, writer, spec)
    build_generic_training_validation_loss(training, writer, spec)


def ensure_layout(output_root: pathlib.Path) -> None:
    for directory in (
        "00_data",
        "01_overview",
        "02_summary",
        "03_seed_architecture_atlas",
        "04_stability_and_interactions",
        "05_training_diagnostics",
        "06_appendix",
        ML4H_READY_SECTION,
        "sources",
        "metadata",
    ):
        (output_root / directory).mkdir(parents=True, exist_ok=True)


def generate(
    *,
    run_root: pathlib.Path,
    tokenized_root: pathlib.Path,
    output_root: pathlib.Path,
    experiments: set[str],
    dry_run: bool,
) -> dict[str, object]:
    input_paths: list[pathlib.Path] = []
    coverage_frames: list[pd.DataFrame] = []
    registry_frames: list[pd.DataFrame] = []
    metrics_frames: list[pd.DataFrame] = []
    training_frames: list[pd.DataFrame] = []
    collected: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}

    for experiment in sorted(experiments):
        spec = EXPERIMENT_SPECS[experiment]
        registry, metrics, training, coverage, inputs = collect_experiment(
            run_root, tokenized_root, spec
        )
        collected[experiment] = (registry, metrics, training)
        registry_frames.append(registry)
        metrics_frames.append(metrics)
        training_frames.append(training)
        coverage_frames.append(coverage)
        input_paths.extend(inputs)

    registry = pd.concat(registry_frames, ignore_index=True)
    metrics = pd.concat(metrics_frames, ignore_index=True)
    training = pd.concat(training_frames, ignore_index=True)
    coverage = pd.concat(coverage_frames, ignore_index=True)
    summary = {
        "run_counts": {
            experiment: int(len(collected[experiment][0]))
            for experiment in sorted(collected)
        },
        "metric_rows": len(metrics),
        "training_rows": len(training),
        "coverage": coverage.to_dict(orient="records"),
    }
    if dry_run:
        return summary

    ensure_layout(output_root)
    registry.to_parquet(output_root / "00_data/run_registry.parquet", index=False)
    metrics.to_parquet(output_root / "00_data/metrics_long.parquet", index=False)
    training.to_parquet(output_root / "00_data/training_long.parquet", index=False)
    coverage.to_csv(output_root / "00_data/coverage.csv", index=False)
    registry.to_csv(output_root / "00_data/run_registry.csv", index=False)

    input_hashes = {
        str(path): sha256(path)
        for path in sorted(set(input_paths), key=str)
        if path.is_file()
    }
    input_hashes[str(pathlib.Path(__file__).resolve())] = sha256(
        pathlib.Path(__file__).resolve()
    )
    writer = FigureWriter(
        output_root, input_hashes=input_hashes, coverage=coverage
    )
    style()
    if "exp1" in experiments:
        exp1_registry, exp1_metrics, exp1_training = collected["exp1"]
        plot_overview(exp1_registry, writer)
        plot_summary(exp1_metrics, writer)
        plot_heatmaps(exp1_metrics, writer, "roc_auc")
        plot_heatmaps(exp1_metrics, writer, "spearman_rho")
        plot_primary_outcomes(
            exp1_metrics,
            writer,
            metric="roc_auc",
            outcomes=PRIMARY_BINARY,
            task_label="binary",
        )
        plot_primary_outcomes(
            exp1_metrics,
            writer,
            metric="spearman_rho",
            outcomes=PRIMARY_REGRESSION,
            task_label="regression",
        )
        plot_atlas(exp1_metrics, writer)
        plot_stability(exp1_metrics, writer)
        plot_training(exp1_training, exp1_registry, writer)
        ml4h_ready_source = seed_interval_source(exp1_metrics)
        build_ml4h_ready_outcome_forests(ml4h_ready_source, writer)
        build_ml4h_ready_axis_summary_bars(ml4h_ready_source, writer)
        build_ml4h_ready_training_curves(exp1_training, writer)
    for experiment in ("exp2", "exp3"):
        if experiment not in experiments:
            continue
        exp_registry, exp_metrics, exp_training = collected[experiment]
        render_generic_suite(
            exp_registry,
            exp_metrics,
            exp_training,
            writer,
            EXPERIMENT_SPECS[experiment],
        )
    expected_figure_count = 19 * len(experiments)
    if len(writer.index) != expected_figure_count:
        raise ValueError(
            f"Expected {expected_figure_count} figures, wrote {len(writer.index)}"
        )
    write_docs(
        output_root,
        writer=writer,
        coverage=coverage,
        registry=registry,
    )

    manifest = {
        "created_at": utc_now(),
        "generator": str(pathlib.Path(__file__).resolve()),
        "requested_experiments": sorted(experiments),
        "summary": summary,
        "input_hashes": input_hashes,
        "figure_count": len(writer.index),
        "figures": writer.index,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=pathlib.Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument(
        "--tokenized-root", type=pathlib.Path, default=DEFAULT_TOKENIZED_ROOT
    )
    parser.add_argument(
        "--output-root", type=pathlib.Path, default=DEFAULT_OUTPUT_ROOT
    )
    parser.add_argument(
        "--experiments",
        default="all",
        help="Comma-separated subset of exp1, exp2, exp3, or all.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print coverage without writing output files.",
    )
    args = parser.parse_args()
    selected = (
        {"exp1", "exp2", "exp3"}
        if args.experiments == "all"
        else {item.strip() for item in args.experiments.split(",") if item.strip()}
    )
    unknown = selected - {"exp1", "exp2", "exp3"}
    if unknown:
        parser.error(f"Unknown experiments: {sorted(unknown)}")
    summary = generate(
        run_root=args.run_root.expanduser().resolve(),
        tokenized_root=args.tokenized_root.expanduser().resolve(),
        output_root=args.output_root.expanduser().resolve(),
        experiments=selected,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
