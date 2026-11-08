#!/usr/bin/env python3
"""Render the isolated ML4H 24-to-48-hour provisional evaluation bundle."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "../fms-ehrs"))
sys.path.insert(0, str(REPO_ROOT))

from fms_ehrs.framework.artifacts import model_artifact_stem
from pipeline.provisional_24to48_task_registry import (
    EXPECTED_TASK_COUNTS,
    HORIZON_END_HOURS,
    HORIZON_START_HOURS,
    OUTCOMES_FILENAME,
    PROVISIONAL_BINARY_TASKS,
    PROVISIONAL_EXP3_BINARY_TASKS,
    PROVISIONAL_REGRESSION_TASKS,
    TASK_SEMANTICS,
)


DEFAULT_RUN_ROOT = REPO_ROOT / "outputs/runs/ml4h_2026"
DEFAULT_TOKENIZED_ROOT = (
    REPO_ROOT / "outputs/runs/tokenized/mimiciv-3.1_meds_70-10-20"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT.parent / "Archive/ml4h-dump/08_ML4H_provisional"
FIGURE_SECTIONS = (
    "00_data",
    "01_representation_summary",
    "02_outcome_forests",
    "03_label_coverage",
    "sources",
    "metadata",
)
SPECS = {
    "exp1": {"expected_runs": 72, "test_rows": 85_057},
    "exp2": {"expected_runs": 72, "test_rows": 85_057},
    "exp3": {"expected_runs": 12, "test_rows": 16_529},
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_primary_renderer():
    path = REPO_ROOT / "paper/scripts/generate_ml4h_figures.py"
    spec = importlib.util.spec_from_file_location("ml4h_primary_renderer", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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


FIGURES_PER_EXPERIMENT = 4


def expected_figure_count(experiments: tuple[str, ...]) -> int:
    return FIGURES_PER_EXPERIMENT * len(selected_specs(experiments))


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
    renderer,
    experiments: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str]]:
    registry_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    label_coverage_rows: list[dict[str, object]] = []
    artifact_hashes: dict[str, str] = {}

    for experiment, contract in selected_specs(experiments).items():
        stage_dir = run_root / f"stage3_24to48h_{experiment}"
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
        spec = renderer.EXPERIMENT_SPECS[experiment]
        for task in tasks:
            parts = renderer.experiment_config_parts(spec, str(task["config_id"]), task)
            registry_rows.append(
                {
                    "experiment": experiment,
                    "config_id": task["config_id"],
                    "config_label": renderer.experiment_config_label(
                        spec, str(task["config_id"])
                    ),
                    "backbone": task["backbone"],
                    "seed": int(task["seed"]),
                    "model_loc": task["model_loc"],
                    "model_stem": renderer.artifact_stem(task["model_loc"]),
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
                    renderer.load_prediction_metrics(
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
        raise ValueError("Duplicate model rows in provisional registry.")
    return registry, metrics, coverage, artifact_hashes


def output_path(output_root: Path, section: str, stem: str, extension: str) -> Path:
    return output_root / section / f"{stem}.{extension}"


def write_figure(
    *,
    output_root: Path,
    index: list[dict[str, object]],
    figure: plt.Figure,
    section: str,
    stem: str,
    source: pd.DataFrame,
    description: str,
    input_hashes: dict[str, str],
) -> None:
    source_path = output_path(output_root, "sources", stem, "csv")
    metadata_path = output_path(output_root, "metadata", stem, "json")
    pdf_path = output_path(output_root, section, stem, "pdf")
    png_path = output_path(output_root, section, stem, "png")
    source.to_csv(source_path, index=False)
    source_hash = sha256(source_path)
    metadata = {
        "analysis": "24to48h",
        "description": description,
        "source": str(source_path.relative_to(output_root)),
        "source_sha256": source_hash,
        "input_hashes": input_hashes,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, bbox_inches="tight", dpi=220)
    plt.close(figure)
    index.append(
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


def plot_representation_summary(
    *,
    experiment: str,
    metrics: pd.DataFrame,
    output_root: Path,
    index: list[dict[str, object]],
    input_hashes: dict[str, str],
) -> None:
    selected = metrics.loc[
        ((metrics["task_type"] == "binary") & (metrics["metric"] == "roc_auc"))
        | (
            (metrics["task_type"] == "regression")
            & (metrics["metric"] == "spearman_rho")
        )
    ].copy()
    summary = (
        selected.groupby(["config_label", "task_type", "metric"], as_index=False)["point"]
        .mean()
        .pivot(index="config_label", columns="metric", values="point")
        .fillna(np.nan)
        .sort_index()
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, max(3.5, 0.35 * len(summary))))
    for axis, metric, title in (
        (axes[0], "roc_auc", "Mean AUROC across binary outcomes"),
        (axes[1], "spearman_rho", "Mean Spearman rho across regression outcomes"),
    ):
        values = summary.get(metric, pd.Series(dtype=float)).sort_values()
        axis.barh(values.index, values.values, color="#4E79A7")
        axis.set_title(title)
        axis.set_xlabel(metric.replace("_", " "))
        axis.grid(axis="x", alpha=0.25)
    fig.suptitle(f"{experiment.upper()} — representation summary, outcomes 24–48h")
    fig.tight_layout()
    write_figure(
        output_root=output_root,
        index=index,
        figure=fig,
        section="01_representation_summary",
        stem=f"{experiment}_representation_summary",
        source=selected,
        description="Configuration-average discrimination and regression association for bounded future outcomes.",
        input_hashes=input_hashes,
    )


def plot_outcome_forest(
    *,
    experiment: str,
    metrics: pd.DataFrame,
    output_root: Path,
    index: list[dict[str, object]],
    input_hashes: dict[str, str],
) -> None:
    selected = metrics.loc[
        ((metrics["task_type"] == "binary") & (metrics["metric"] == "roc_auc"))
        | (
            (metrics["task_type"] == "regression")
            & (metrics["metric"] == "spearman_rho")
        )
    ].copy()
    grouped = (
        selected.groupby(["task_type", "outcome", "outcome_label", "metric"], as_index=False)
        .agg(mean=("point", "mean"), low=("point", "min"), high=("point", "max"))
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 11), sharex=False)
    for axis, kind, metric, title in (
        (axes[0], "binary", "roc_auc", "Binary outcome AUROC"),
        (axes[1], "regression", "spearman_rho", "Regression outcome Spearman rho"),
    ):
        panel = grouped.loc[
            (grouped["task_type"] == kind) & (grouped["metric"] == metric)
        ].sort_values("mean")
        y = np.arange(len(panel))
        axis.hlines(y, panel["low"], panel["high"], color="#9ECAE1", linewidth=3)
        axis.scatter(panel["mean"], y, color="#08519C", s=28, zorder=3)
        axis.set_yticks(y)
        axis.set_yticklabels(panel["outcome_label"], fontsize=7)
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.25)
    fig.suptitle(
        f"{experiment.upper()} — outcome-level performance, 24–48h follow-up\n"
        "point: mean across model runs; line: observed model-run range"
    )
    fig.tight_layout()
    write_figure(
        output_root=output_root,
        index=index,
        figure=fig,
        section="02_outcome_forests",
        stem=f"{experiment}_outcome_forest",
        source=grouped,
        description="Outcome-level performance across all configurations, architectures, and seeds.",
        input_hashes=input_hashes,
    )


def plot_seed_interval_forests(
    *,
    experiment: str,
    metrics: pd.DataFrame,
    output_root: Path,
    index: list[dict[str, object]],
    input_hashes: dict[str, str],
    renderer,
) -> None:
    spec = renderer.EXPERIMENT_SPECS[experiment]
    source = renderer.generic_seed_interval_source(metrics, spec)
    binary_order = tuple(
        outcome
        for outcome in renderer.ML4H_BINARY_ORDER
        if outcome in set(source["outcome"])
    )
    regression_order = tuple(
        outcome
        for outcome in renderer.ML4H_REGRESSION_ORDER
        if outcome in set(source["outcome"])
    )
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 19.0))
    for row_index, backbone in enumerate(renderer.BACKBONE_ORDER):
        for col_index, (panel, outcome_order, xlabel, floor) in enumerate(
            (
                ("binary", binary_order, "AUROC", 0.52),
                ("regression", regression_order, r"Spearman $\rho$", None),
            )
        ):
            subset = source[
                (source["backbone"] == backbone) & (source["panel"] == panel)
            ]
            renderer._generic_forest_panel(
                axes[row_index, col_index],
                subset,
                spec=spec,
                outcome_order=outcome_order,
                panel_title=(
                    f"{renderer.BACKBONE_DISPLAY[backbone]} · "
                    f"{'Binary outcomes' if panel == 'binary' else 'Regression outcomes'}"
                ),
                xlabel=xlabel,
                x_lo_floor=floor,
            )
    handles = [
        Line2D(
            [0],
            [0],
            marker=renderer.experiment_config_marker(spec, config),
            color=renderer.experiment_config_color(spec, config),
            markerfacecolor=renderer.experiment_config_color(spec, config),
            markeredgecolor="#222222",
            linewidth=0,
            markersize=4.5,
            label=renderer.experiment_config_label(spec, config),
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
        f"{spec.display_name}, 24–48h follow-up. "
        "Points are means over three training seeds; whiskers span the seed minimum to maximum.",
        y=0.999,
        fontsize=9.5,
    )
    write_figure(
        output_root=output_root,
        index=index,
        figure=fig,
        section="02_outcome_forests",
        stem=f"{experiment}__outcome_forests",
        source=source,
        description=(
            f"{spec.display_name} 24–48h outcome forest with architecture rows; "
            "seed-mean points and seed min-max whiskers by configuration."
        ),
        input_hashes=input_hashes,
    )


def plot_label_coverage(
    *,
    experiment: str,
    label_coverage: pd.DataFrame,
    output_root: Path,
    index: list[dict[str, object]],
    input_hashes: dict[str, str],
) -> None:
    summary = (
        label_coverage.groupby(["task_type", "outcome"], as_index=False)
        .agg(
            test_rows=("test_cohort_rows", "median"),
            labeled=("n_labeled", "median"),
            positives=("n_positive", "median"),
        )
    )
    summary["labeled_fraction"] = summary["labeled"] / summary["test_rows"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 11))
    for axis, kind, title in (
        (axes[0], "binary", "Binary label coverage"),
        (axes[1], "regression", "Regression label coverage"),
    ):
        panel = summary.loc[summary["task_type"] == kind].sort_values("labeled_fraction")
        y = np.arange(len(panel))
        axis.barh(y, panel["labeled_fraction"], color="#59A14F")
        axis.set_yticks(y)
        axis.set_yticklabels(panel["outcome"], fontsize=7)
        axis.set_xlim(0, 1)
        axis.set_xlabel("fraction with a finite follow-up label")
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.25)
    fig.suptitle(f"{experiment.upper()} — 24–48h label availability")
    fig.tight_layout()
    write_figure(
        output_root=output_root,
        index=index,
        figure=fig,
        section="03_label_coverage",
        stem=f"{experiment}_label_coverage",
        source=summary,
        description="Per-outcome label availability; unmeasured physiology remains unlabeled.",
        input_hashes=input_hashes,
    )


def write_docs(
    *,
    output_root: Path,
    figure_index: pd.DataFrame,
    metrics: pd.DataFrame,
    label_coverage: pd.DataFrame,
    experiments: tuple[str, ...],
) -> None:
    per_task = (
        label_coverage.groupby(["experiment", "task_type", "outcome"], as_index=False)
        .agg(
            test_rows=("test_cohort_rows", "median"),
            labeled=("n_labeled", "median"),
            unlabeled=("n_unlabeled", "median"),
            positives=("n_positive", "median"),
        )
    )
    per_task.to_csv(output_root / "00_data/outcome_coverage_summary.csv", index=False)
    semantics = "\n".join(
        f"- `{task}`: {meaning}" for task, meaning in sorted(TASK_SEMANTICS.items())
    )
    figure_lines = "\n".join(
        f"- [{row.stem}]({row.pdf}) — {row.description}"
        for row in figure_index.itertuples(index=False)
    )
    metrics_rows = len(metrics)
    (output_root / "README.md").write_text(
        "\n".join(
            [
                "# ML4H provisional 24–48h evaluation",
                "",
                "This bundle is separate from the primary ML4H analysis and is not a replacement for `07_ML4H_ready`.",
                "",
                "## Outcome contract",
                "",
                f"- Features: model representations from the first {HORIZON_START_HOURS:g} hours.",
                f"- Follow-up window: `(admission + {HORIZON_START_HOURS:g}h, admission + {HORIZON_END_HOURS:g}h]`.",
                "- Recorded procedural/discrete events absent during follow-up are negative, including after early discharge.",
                "- Lab/vital-derived binary labels and all regression labels are missing when no qualifying follow-up measurement exists; metrics use finite labels only.",
                "- `length_of_stay` and `icu_length_of_stay` regression tasks are intentionally omitted.",
                "- Exp1/Exp2 each contain 37 tasks; Exp3 contains 36 after excluding degenerate ICU admission.",
                f"- Included experiments in this render: {', '.join(experiments)}.",
                *(
                    [
                        "- Exp1 is omitted from this render because repaired first-24h features are not ready yet."
                    ]
                    if "exp1" not in experiments
                    else []
                ),
                "",
                "## Horizon-specific endpoint semantics",
                "",
                semantics,
                "",
                "## Contents",
                "",
                f"- {metrics_rows:,} metric rows in `00_data/metrics_long.parquet`.",
                "- `00_data/outcome_coverage_summary.csv` reports labeled and positive counts per outcome.",
                "- Every figure includes a source CSV and metadata file with input hashes.",
                "",
                "## Figure index",
                "",
                figure_lines,
                "",
            ]
        ),
        encoding="utf-8",
    )
    (output_root / "PROVISIONAL.md").write_text(
        "\n".join(
            [
                "# Provisional status",
                "",
                "These results are exploratory bounded-horizon probes. They do not alter or supersede the primary ML4H figures.",
                "Interpret measurement-derived targets with their outcome-specific labeled cohorts.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def generate(
    *,
    run_root: Path,
    shared_tokenized_root: Path,
    output_root: Path,
    dry_run: bool,
    experiments: tuple[str, ...] = ("exp1", "exp2", "exp3"),
) -> dict[str, object]:
    specs = selected_specs(experiments)
    renderer = load_primary_renderer()
    registry, metrics, label_coverage, input_hashes = collect(
        run_root=run_root,
        shared_tokenized_root=shared_tokenized_root,
        renderer=renderer,
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
        "figure_count": expected_figure_count(tuple(specs)),
    }
    if dry_run:
        return summary

    for section in FIGURE_SECTIONS:
        (output_root / section).mkdir(parents=True, exist_ok=True)
    registry.to_parquet(output_root / "00_data/run_registry.parquet", index=False)
    registry.to_csv(output_root / "00_data/run_registry.csv", index=False)
    metrics.to_parquet(output_root / "00_data/metrics_long.parquet", index=False)
    metrics.to_csv(output_root / "00_data/metrics_long.csv", index=False)
    label_coverage.to_parquet(output_root / "00_data/label_coverage_long.parquet", index=False)
    label_coverage.to_csv(output_root / "00_data/label_coverage_long.csv", index=False)

    input_hashes[str(Path(__file__).resolve())] = sha256(Path(__file__).resolve())
    index: list[dict[str, object]] = []
    for experiment in specs:
        subset_metrics = metrics.loc[metrics["experiment"] == experiment].copy()
        subset_coverage = label_coverage.loc[
            label_coverage["experiment"] == experiment
        ].copy()
        plot_representation_summary(
            experiment=experiment,
            metrics=subset_metrics,
            output_root=output_root,
            index=index,
            input_hashes=input_hashes,
        )
        plot_outcome_forest(
            experiment=experiment,
            metrics=subset_metrics,
            output_root=output_root,
            index=index,
            input_hashes=input_hashes,
        )
        plot_seed_interval_forests(
            experiment=experiment,
            metrics=subset_metrics,
            output_root=output_root,
            index=index,
            input_hashes=input_hashes,
            renderer=renderer,
        )
        plot_label_coverage(
            experiment=experiment,
            label_coverage=subset_coverage,
            output_root=output_root,
            index=index,
            input_hashes=input_hashes,
        )

    if len(index) != expected_figure_count(tuple(specs)):
        raise ValueError(
            f"Expected {expected_figure_count(tuple(specs))} figures, wrote {len(index)}."
        )
    figure_index = pd.DataFrame(index)
    figure_index.to_csv(output_root / "figure_index.csv", index=False)
    write_docs(
        output_root=output_root,
        figure_index=figure_index,
        metrics=metrics,
        label_coverage=label_coverage,
        experiments=tuple(specs),
    )
    manifest = {
        "created_at": utc_now(),
        "generator": str(Path(__file__).resolve()),
        "summary": summary,
        "figure_count": len(index),
        "figures": index,
        "input_hashes": input_hashes,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
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
        help="Subset of experiments to render. Default is all three.",
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
