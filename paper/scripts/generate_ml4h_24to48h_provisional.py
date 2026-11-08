#!/usr/bin/env python3
"""Build the 24-to-48-hour metric tables from Stage 3 prediction files."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn import metrics as skl_metrics

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "../fms-ehrs"))
sys.path.insert(0, str(REPO_ROOT))

from fms_ehrs.framework.artifacts import model_artifact_stem
from pipeline.provisional_24to48_task_registry import (
    EXPECTED_TASK_COUNTS,
    OUTCOMES_FILENAME,
    PROVISIONAL_BINARY_TASKS,
    PROVISIONAL_EXP3_BINARY_TASKS,
    PROVISIONAL_REGRESSION_TASKS,
)


DEFAULT_RUN_ROOT = REPO_ROOT / "outputs/runs/ml4h_2026"
DEFAULT_TOKENIZED_ROOT = (
    REPO_ROOT / "outputs/runs/tokenized/mimiciv-3.1_meds_70-10-20"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs/runs/ml4h_2026/metrics"

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
REPRESENTATIONS = ("discrete", "soft", "xval", "xval_affine")
TEMPORAL_MODES = ("event_order", "time_tokens", "time_rope")


@dataclass(frozen=True)
class ExperimentSpec:
    """Manifest contract for one study in the benchmark."""

    experiment: str
    display_name: str
    stage3_dir: str
    expected_runs: int
    test_rows: int
    config_order: tuple[str, ...]


EXPERIMENT_SPECS = {
    "exp1": ExperimentSpec(
        experiment="exp1",
        display_name="Experiment 1: numeric tokenization",
        stage3_dir="stage3_24to48h_exp1",
        expected_runs=72,
        test_rows=85_057,
        config_order=CONFIG_ORDER,
    ),
    "exp2": ExperimentSpec(
        experiment="exp2",
        display_name="Experiment 2: representation mechanics",
        stage3_dir="stage3_24to48h_exp2",
        expected_runs=72,
        test_rows=85_057,
        config_order=EXP2_CONFIG_ORDER,
    ),
    "exp3": ExperimentSpec(
        experiment="exp3",
        display_name="Experiment 3: semantic harmonization",
        stage3_dir="stage3_24to48h_exp3",
        expected_runs=12,
        test_rows=16_529,
        config_order=EXP3_CONFIG_ORDER,
    ),
}
SPECS = {
    name: {"expected_runs": spec.expected_runs, "test_rows": spec.test_rows}
    for name, spec in EXPERIMENT_SPECS.items()
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    for representation in sorted(REPRESENTATIONS, key=len, reverse=True):
        marker = f"_{representation}_"
        if marker in config_id:
            temporal = config_id.split(marker, maxsplit=1)[1]
            if temporal in TEMPORAL_MODES:
                return representation, temporal
    raise ValueError(f"Could not parse Exp2 configuration: {config_id}")


def experiment_config_parts(
    spec: ExperimentSpec, config_id: str, task: dict | None = None
) -> dict[str, object]:
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
            data_format = (
                "meds_icu" if config_id.startswith("meds_icu_") else "meds_clif"
            )
        return {
            "quantizer": "deciles",
            "anchoring": "none",
            "fusion": "unfused",
            "fused": False,
            "representation": "discrete",
            "temporal": "time_tokens",
            "data_format": data_format,
            "source": (
                "Native MIMIC" if data_format == "meds_icu" else "CLIF harmonized"
            ),
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
    return model_artifact_stem(Path(model_loc))


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


def load_prediction_metrics(
    *,
    task: dict,
    kind: str,
    spec: ExperimentSpec,
    tokenized_root: Path,
    preds_tag: str,
    expected_outcomes: tuple[str, ...],
    artifact_hashes: dict[str, str],
) -> list[dict[str, object]]:
    model_loc = str(task["model_loc"])
    stem = artifact_stem(model_loc)
    prefix = "logistic_regression" if kind == "binary" else "reg_ridge_regression"
    data_dir = Path(task.get("data_dir", tokenized_root))
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


def outcome_contract(experiment: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    binary = (
        PROVISIONAL_EXP3_BINARY_TASKS
        if experiment == "exp3"
        else PROVISIONAL_BINARY_TASKS
    )
    return binary, PROVISIONAL_REGRESSION_TASKS


def artifact_path(
    *,
    task: dict,
    kind: str,
    preds_tag: str,
    shared_tokenized_root: Path,
) -> Path:
    tokenized_root = Path(task.get("data_dir", shared_tokenized_root))
    prefix = "logistic_regression" if kind == "binary" else "reg_ridge_regression"
    return (
        tokenized_root
        / f"{task['data_version']}-tokenized"
        / "test"
        / f"{prefix}-preds-{preds_tag}-{model_artifact_stem(Path(task['model_loc']))}.pkl"
    )


def label_rows(
    *,
    experiment: str,
    task: dict,
    kind: str,
    outcomes: tuple[str, ...],
    preds_tag: str,
    shared_tokenized_root: Path,
) -> tuple[list[dict[str, object]], Path]:
    artifact = artifact_path(
        task=task,
        kind=kind,
        preds_tag=preds_tag,
        shared_tokenized_root=shared_tokenized_root,
    )
    with artifact.open("rb") as handle:
        payload = pickle.load(handle)
    rows: list[dict[str, object]] = []
    for outcome in outcomes:
        raw_labels = np.asarray(payload["labels"][outcome], dtype=float)
        predictions = np.asarray(payload["predictions"][outcome])
        qualifier = np.asarray(payload["qualifiers"][outcome], dtype=bool)
        labels = raw_labels
        if len(predictions) != len(raw_labels):
            if len(qualifier) != len(raw_labels) or int(qualifier.sum()) != len(predictions):
                raise ValueError(f"Cannot align labels and predictions in {artifact}")
            labels = raw_labels[qualifier]
        labeled = np.asarray(labels, dtype=float)
        finite = np.isfinite(labeled)
        rows.append(
            {
                "experiment": experiment,
                "config_id": task["config_id"],
                "backbone": task["backbone"],
                "seed": int(task["seed"]),
                "data_version": task["data_version"],
                "task_type": kind,
                "outcome": outcome,
                "test_cohort_rows": len(raw_labels),
                "n_labeled": int(finite.sum()),
                "n_unlabeled": int((~finite).sum()),
                "n_positive": (
                    int((labeled[finite] > 0).sum()) if kind == "binary" else np.nan
                ),
                "source_artifact": str(artifact),
            }
        )
    return rows, artifact


def selected_specs(experiments: tuple[str, ...]) -> dict[str, dict[str, int]]:
    if not experiments:
        raise ValueError("At least one experiment is required.")
    unknown = [name for name in experiments if name not in SPECS]
    if unknown:
        raise ValueError(f"Unsupported experiments: {', '.join(unknown)}")
    ordered = tuple(name for name in ("exp1", "exp2", "exp3") if name in experiments)
    return {name: SPECS[name] for name in ordered}


def expected_metric_rows(experiments: tuple[str, ...]) -> int:
    return sum(
        spec["expected_runs"]
        * (
            len(outcome_contract(experiment)[0]) * 4
            + len(outcome_contract(experiment)[1]) * 4
        )
        for experiment, spec in selected_specs(experiments).items()
    )


def collect(
    *,
    run_root: Path,
    shared_tokenized_root: Path,
    experiments: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str]]:
    registry_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    label_coverage_rows: list[dict[str, object]] = []
    artifact_hashes: dict[str, str] = {}

    for experiment, contract in selected_specs(experiments).items():
        spec = EXPERIMENT_SPECS[experiment]
        stage_dir = run_root / spec.stage3_dir
        manifest_path = stage_dir / f"{experiment}_24to48h_stage3_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        tasks = manifest["tasks"]
        if len(tasks) != contract["expected_runs"]:
            raise ValueError(
                f"{experiment}: expected {contract['expected_runs']} tasks, "
                f"found {len(tasks)}."
            )
        binary_outcomes, regression_outcomes = outcome_contract(experiment)
        if tuple(manifest["binary_outcomes"]) != binary_outcomes:
            raise ValueError(f"{experiment}: unexpected binary outcome contract.")
        if tuple(manifest["regression_outcomes"]) != regression_outcomes:
            raise ValueError(f"{experiment}: unexpected regression outcome contract.")
        if int(manifest["expected_outcome_count"]) != EXPECTED_TASK_COUNTS[experiment]:
            raise ValueError(f"{experiment}: unexpected total outcome count.")
        if manifest["outcomes_filename"] != OUTCOMES_FILENAME:
            raise ValueError(f"{experiment}: wrong outcome parquet.")

        tags = manifest["preds_tags"]
        for task in tasks:
            parts = experiment_config_parts(spec, str(task["config_id"]), task)
            registry_rows.append(
                {
                    "experiment": experiment,
                    "config_id": task["config_id"],
                    "config_label": experiment_config_label(
                        spec, str(task["config_id"])
                    ),
                    "backbone": task["backbone"],
                    "seed": int(task["seed"]),
                    "model_loc": task["model_loc"],
                    "model_stem": artifact_stem(task["model_loc"]),
                    "data_version": task["data_version"],
                    "test_cohort_rows": contract["test_rows"],
                    **parts,
                }
            )
            for kind, outcomes in (
                ("binary", binary_outcomes),
                ("regression", regression_outcomes),
            ):
                metric_rows.extend(
                    load_prediction_metrics(
                        task=task,
                        kind=kind,
                        spec=spec,
                        tokenized_root=shared_tokenized_root,
                        preds_tag=tags[kind],
                        expected_outcomes=outcomes,
                        artifact_hashes=artifact_hashes,
                    )
                )
                coverage, artifact = label_rows(
                    experiment=experiment,
                    task=task,
                    kind=kind,
                    outcomes=outcomes,
                    preds_tag=tags[kind],
                    shared_tokenized_root=shared_tokenized_root,
                )
                label_coverage_rows.extend(coverage)
                artifact_hashes.setdefault(str(artifact), sha256(artifact))
        artifact_hashes[str(manifest_path)] = sha256(manifest_path)

    registry = pd.DataFrame(registry_rows)
    metrics = pd.DataFrame(metric_rows)
    coverage = pd.DataFrame(label_coverage_rows)
    if registry.duplicated(["experiment", "model_loc"]).any():
        raise ValueError("Duplicate model rows in run registry.")
    return registry, metrics, coverage, artifact_hashes


def generate(
    *,
    run_root: Path,
    shared_tokenized_root: Path,
    output_root: Path,
    dry_run: bool,
    experiments: tuple[str, ...] = ("exp1", "exp2", "exp3"),
) -> dict[str, object]:
    specs = selected_specs(experiments)
    registry, metrics, label_coverage, _ = collect(
        run_root=run_root,
        shared_tokenized_root=shared_tokenized_root,
        experiments=tuple(specs),
    )
    expected_rows = expected_metric_rows(tuple(specs))
    if len(metrics) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} metric rows, found {len(metrics)}."
        )
    summary = {
        "analysis": "24to48h",
        "requested_experiments": list(specs),
        "run_counts": registry.groupby("experiment").size().to_dict(),
        "metric_rows": len(metrics),
        "expected_metric_rows": expected_rows,
        "outcome_counts": {
            name: EXPECTED_TASK_COUNTS[name] for name in specs
        },
    }
    if dry_run:
        return summary

    output_root.mkdir(parents=True, exist_ok=True)
    registry.to_parquet(output_root / "run_registry.parquet", index=False)
    registry.to_csv(output_root / "run_registry.csv", index=False)
    metrics.to_parquet(output_root / "metrics_long.parquet", index=False)
    metrics.to_csv(output_root / "metrics_long.csv", index=False)
    label_coverage.to_parquet(output_root / "label_coverage_long.parquet", index=False)
    label_coverage.to_csv(output_root / "label_coverage_long.csv", index=False)

    per_task = (
        label_coverage.groupby(["experiment", "task_type", "outcome"], as_index=False)
        .agg(
            test_rows=("test_cohort_rows", "median"),
            labeled=("n_labeled", "median"),
            unlabeled=("n_unlabeled", "median"),
            positives=("n_positive", "median"),
        )
    )
    per_task.to_csv(output_root / "outcome_coverage_summary.csv", index=False)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument(
        "--shared-tokenized-root", type=Path, default=DEFAULT_TOKENIZED_ROOT
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=tuple(SPECS),
        default=list(SPECS),
        help="Subset of experiments to aggregate. Default is all three.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    summary = generate(
        run_root=args.run_root.expanduser().resolve(),
        shared_tokenized_root=args.shared_tokenized_root.expanduser().resolve(),
        output_root=args.output_root.expanduser().resolve(),
        dry_run=args.dry_run,
        experiments=tuple(args.experiments),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
