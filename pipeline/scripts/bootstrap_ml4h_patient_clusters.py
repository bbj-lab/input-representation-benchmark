#!/usr/bin/env python3
"""Patient-clustered bootstrap intervals for printed ML4H family differences.

Each Slurm task handles one printed table column. A draw samples test patients
with replacement and includes all admissions from each sampled patient.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
UTILITY_DIR = REPO_ROOT / "utilities" / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(UTILITY_DIR))

import bootstrap_ml4h_family_effects as family_effects  # noqa: E402
from count_hospitalizations_by_patient_visits import (  # noqa: E402
    unique_stay_pairs,
)


DEFAULT_METRICS = (
    REPO_ROOT / "outputs/runs/ml4h_2026/metrics/metrics_long.parquet"
)
DEFAULT_MEDS_RAW = (
    REPO_ROOT / "benchmarks/mimic-meds-extraction/data/meds/data/raw"
)
DEFAULT_WORK_ROOT = (
    REPO_ROOT / "outputs/runs/ml4h_2026/patient_cluster_bootstrap"
)
DEFAULT_OUTPUT = DEFAULT_WORK_ROOT / "patient_cluster_intervals.parquet"
BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 42
ALPHA = 0.05
# Prediction artifacts use float32, while metrics_long stores more precision.
POINT_TOLERANCE = 5e-5
TASK_COUNT = family_effects.TASK_COUNT

_BOOTSTRAP_CONTEXT: dict[str, object] | None = None


def task_specs() -> list[dict[str, str | int]]:
    """Return the 17 printed contrast columns."""
    return family_effects.task_specs()


def subject_map_path(work_root: Path) -> Path:
    """Return the default admission-to-patient map path."""
    return work_root / "hadm_to_subject.parquet"


def load_subject_map(path: Path) -> pd.DataFrame:
    """Load and validate one patient id for each admission id."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing admission-to-patient map: {path}")
    frame = pd.read_parquet(path, columns=["admission_id", "subject_id"])
    if frame.empty:
        raise ValueError(f"Admission-to-patient map is empty: {path}")
    if frame[["admission_id", "subject_id"]].isna().any().any():
        raise ValueError(f"Admission-to-patient map contains null values: {path}")
    frame["admission_id"] = frame["admission_id"].astype(str)
    frame["subject_id"] = frame["subject_id"].astype(str)
    if frame["admission_id"].duplicated().any():
        raise ValueError(f"Admission-to-patient map has duplicate admissions: {path}")
    return frame


def build_subject_map(*, meds_raw: Path, output_path: Path) -> dict[str, object]:
    """Scan MEDS shards and write one patient id for each admission."""
    if output_path.is_file():
        frame = load_subject_map(output_path)
        return {
            "status": "skipped",
            "output": str(output_path),
            "n_admissions": int(len(frame)),
            "n_patients": int(frame["subject_id"].nunique()),
        }

    shard_paths = sorted(meds_raw.glob("*/*.parquet"))
    if not shard_paths:
        raise FileNotFoundError(f"No MEDS parquet shards under {meds_raw}")
    mapping = unique_stay_pairs(shard_paths)
    frame = pd.DataFrame(
        {
            "admission_id": [str(value) for value in mapping],
            "subject_id": [str(mapping[value]) for value in mapping],
        }
    )
    frame = frame.sort_values("admission_id", kind="stable").reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.parent / f"{output_path.name}.{os.getpid()}.tmp"
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, output_path)
    return {
        "status": "wrote",
        "output": str(output_path),
        "n_admissions": int(len(frame)),
        "n_patients": int(frame["subject_id"].nunique()),
        "n_shards": int(len(shard_paths)),
    }


def align_subject_ids(
    admission_ids: np.ndarray,
    subject_map: pd.DataFrame,
) -> np.ndarray:
    """Align patient ids to the admission order in prediction artifacts."""
    lookup = subject_map.set_index("admission_id")["subject_id"]
    requested = pd.Index(np.asarray(admission_ids).astype(str))
    aligned = lookup.reindex(requested)
    if aligned.isna().any():
        missing = requested[aligned.isna()].unique()
        example = ", ".join(map(str, missing[:5]))
        raise ValueError(
            f"Patient ids are missing for {len(missing)} admissions. "
            f"Examples: {example}"
        )
    return aligned.astype(str).to_numpy()


def cluster_bootstrap_indices(
    cluster_ids: np.ndarray,
    sampled_clusters: np.ndarray,
) -> np.ndarray:
    """Expand a patient draw to all admission rows in each sampled cluster."""
    cluster_ids = np.asarray(cluster_ids).astype(str)
    sampled_clusters = np.asarray(sampled_clusters).astype(str)
    clusters, inverse = np.unique(cluster_ids, return_inverse=True)
    positions = np.searchsorted(clusters, sampled_clusters)
    valid = positions < len(clusters)
    valid &= clusters[np.minimum(positions, len(clusters) - 1)] == sampled_clusters
    if not np.all(valid):
        unknown = np.unique(sampled_clusters[~valid])
        raise ValueError(f"Unknown sampled patient ids: {unknown[:5].tolist()}")
    multiplicity = np.bincount(positions, minlength=len(clusters))
    row_multiplicity = multiplicity[inverse]
    return np.repeat(
        np.arange(len(cluster_ids), dtype=np.int64),
        row_multiplicity,
    )


def draw_cluster_indices(
    cluster_inverse: np.ndarray,
    n_clusters: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw patients with replacement and expand their admission rows."""
    sampled = rng.integers(0, n_clusters, size=n_clusters)
    multiplicity = np.bincount(sampled, minlength=n_clusters)
    return np.repeat(
        np.arange(len(cluster_inverse), dtype=np.int64),
        multiplicity[cluster_inverse],
    )


def add_cluster_ids(
    context: dict[str, object],
    subject_map: pd.DataFrame,
) -> dict[str, object]:
    """Attach aligned patient cluster ids to one contrast context."""
    subject_ids = align_subject_ids(context["admission_ids"], subject_map)
    clusters, inverse = np.unique(subject_ids, return_inverse=True)
    context["subject_ids"] = subject_ids
    context["cluster_inverse"] = inverse.astype(np.int64, copy=False)
    context["n_clusters"] = int(len(clusters))
    return context


def family_delta_with_indices(
    context: dict[str, object],
    family: dict[str, object],
    indices: np.ndarray,
) -> float:
    """Recompute one family difference on a patient bootstrap sample."""
    task_type = str(family["task_type"])
    outcomes = tuple(family["outcomes"])
    labels = context["labels"]
    valid = context["valid"]
    predictions = context["predictions"]
    config_ids = context["config_ids"]
    run_keys = tuple(context["run_keys"])
    baseline_config = tuple(config_ids["baseline"])[0]
    level_config = tuple(config_ids["level"])[0]

    model_values: list[float] = []
    for run_key in run_keys:
        outcome_level: list[float] = []
        outcome_base: list[float] = []
        for outcome in outcomes:
            sampled_valid = valid[outcome][indices]
            y_true = labels[outcome][indices][sampled_valid]
            level_all = predictions["level"][level_config][run_key][outcome]
            base_all = predictions["baseline"][baseline_config][run_key][outcome]
            level_score = family_effects.metric_value(
                task_type,
                y_true,
                level_all[indices][sampled_valid],
            )
            base_score = family_effects.metric_value(
                task_type,
                y_true,
                base_all[indices][sampled_valid],
            )
            if not np.isfinite(level_score) or not np.isfinite(base_score):
                return float("nan")
            outcome_level.append(level_score)
            outcome_base.append(base_score)
        model_values.append(
            float(np.mean(outcome_level) - np.mean(outcome_base))
        )
    return float(np.mean(model_values))


def bootstrap_one(seed: int) -> np.ndarray:
    """Compute all eight family differences for one patient draw."""
    if _BOOTSTRAP_CONTEXT is None:
        raise RuntimeError("Patient bootstrap worker context is unavailable")
    rng = np.random.default_rng(seed)
    indices = draw_cluster_indices(
        _BOOTSTRAP_CONTEXT["cluster_inverse"],
        int(_BOOTSTRAP_CONTEXT["n_clusters"]),
        rng,
    )
    values = [
        family_delta_with_indices(_BOOTSTRAP_CONTEXT, family, indices)
        for family in _BOOTSTRAP_CONTEXT["families"]
    ]
    return np.asarray(values, dtype=float)


def shard_paths(
    work_root: Path,
    task_index: int,
) -> tuple[Path, Path, Path]:
    """Return result, marker, and temporary paths for one task."""
    stem = f"{task_index:03d}"
    return (
        work_root / "shards" / f"{stem}.parquet",
        work_root / "shards" / f"{stem}.done.json",
        work_root / "tmp" / f"{stem}.{os.getpid()}.parquet",
    )


def run_task(
    *,
    task_index: int,
    metrics_path: Path,
    map_path: Path,
    work_root: Path,
    n_bootstrap: int,
    seed: int,
    workers: int,
) -> dict[str, object]:
    """Run one printed contrast column."""
    tasks = task_specs()
    if task_index < 0 or task_index >= len(tasks):
        raise IndexError(task_index)
    task = tasks[task_index]
    shard, done, temporary = shard_paths(work_root, task_index)
    if shard.is_file() and done.is_file():
        return {
            "status": "skipped",
            "task_index": task_index,
            "shard": str(shard),
        }

    metrics = pd.read_parquet(metrics_path)
    context = family_effects.load_column_context(metrics, task)
    context = add_cluster_ids(context, load_subject_map(map_path))

    full_indices = np.arange(int(context["n_rows"]), dtype=np.int64)
    reproduced = np.asarray(
        [
            family_delta_with_indices(context, family, full_indices)
            for family in context["families"]
        ]
    )
    expected = np.asarray(
        [float(family["point"]) for family in context["families"]]
    )
    point_gap = float(np.max(np.abs(reproduced - expected)))
    if point_gap > POINT_TOLERANCE:
        raise ValueError(
            f"Task {task_index}: full-sample point gap is {point_gap:.3g}"
        )

    seeds = [
        int(item.generate_state(1, dtype=np.uint64)[0])
        for item in np.random.SeedSequence(seed).spawn(n_bootstrap)
    ]
    global _BOOTSTRAP_CONTEXT
    _BOOTSTRAP_CONTEXT = context
    if workers > 1:
        with mp.get_context("fork").Pool(processes=workers) as pool:
            draws = pool.map(
                bootstrap_one,
                seeds,
                chunksize=max(1, n_bootstrap // (workers * 8)),
            )
    else:
        draws = [bootstrap_one(draw_seed) for draw_seed in seeds]

    values = np.vstack(draws)
    accepted = np.isfinite(values).all(axis=1)
    values = values[accepted]
    minimum_accepted = max(1, int(n_bootstrap * 0.99))
    if len(values) < minimum_accepted:
        raise ValueError(
            f"Task {task_index}: accepted {len(values)}/{n_bootstrap} draws"
        )

    records = []
    for index, family in enumerate(context["families"]):
        interval_low = float(np.quantile(values[:, index], ALPHA / 2))
        interval_high = float(
            np.quantile(values[:, index], 1.0 - ALPHA / 2)
        )
        records.append(
            {
                **task,
                "role": "family",
                "task_type": family["task_type"],
                "family": family["family"],
                "point": float(family["point"]),
                "cluster_ci_low": interval_low,
                "cluster_ci_high": interval_high,
                "cluster_ci_contains_zero": bool(
                    interval_low <= 0.0 <= interval_high
                ),
                "bootstrap_n": int(n_bootstrap),
                "bootstrap_accepted": int(len(values)),
                "bootstrap_seed": int(seed),
                "n_models": 6,
                "n_outcomes": int(len(family["outcomes"])),
                "n_patients": int(context["n_clusters"]),
                "n_admissions": int(context["n_rows"]),
                "resampling_unit": "test_patient",
                "aggregation": "family_outcomes_then_six_models",
            }
        )

    frame = pd.DataFrame(records)
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, shard)
    summary = {
        "status": "wrote",
        "task_index": task_index,
        "experiment": task["experiment"],
        "condition": task["condition"],
        "level": task["level"],
        "n_patients": int(context["n_clusters"]),
        "n_admissions": int(context["n_rows"]),
        "bootstrap_accepted": int(len(values)),
        "point_gap": point_gap,
        "shard": str(shard),
    }
    done.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def write_summary(table: pd.DataFrame, destination: Path) -> None:
    """Write a compact text summary of patient-cluster intervals."""
    lines = [
        "experiment\tcondition\tlevel\ttask_type\tfamily\tpoint\tci_low\tci_high\tcontains_zero"
    ]
    for row in table.itertuples():
        lines.append(
            "\t".join(
                [
                    str(row.experiment),
                    str(row.condition),
                    str(row.level),
                    str(row.task_type),
                    str(row.family),
                    f"{row.point:.6f}",
                    f"{row.cluster_ci_low:.6f}",
                    f"{row.cluster_ci_high:.6f}",
                    str(bool(row.cluster_ci_contains_zero)).lower(),
                ]
            )
        )
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def concat_tasks(
    *,
    work_root: Path,
    output_path: Path,
    metrics_path: Path,
    n_bootstrap: int,
    seed: int,
) -> dict[str, object]:
    """Merge all task shards and validate printed point estimates."""
    frames: list[pd.DataFrame] = []
    for task in task_specs():
        shard, done, _ = shard_paths(work_root, int(task["task_index"]))
        if not shard.is_file() or not done.is_file():
            raise FileNotFoundError(f"Incomplete patient bootstrap task: {shard}")
        frames.append(pd.read_parquet(shard))
    table = pd.concat(frames, ignore_index=True)
    if len(table) != family_effects.contrasts.ROW_COUNT:
        raise ValueError(
            f"Expected {family_effects.contrasts.ROW_COUNT} rows, found {len(table)}"
        )
    if int(table["task_index"].nunique()) != TASK_COUNT:
        raise ValueError(f"Expected {TASK_COUNT} tasks")
    if not table["bootstrap_n"].eq(n_bootstrap).all():
        raise ValueError("Bootstrap count changed across tasks")
    if not table["bootstrap_seed"].eq(seed).all():
        raise ValueError("Bootstrap seed changed across tasks")

    metrics = pd.read_parquet(metrics_path)
    expected_points = [
        family_effects.contrasts.unrounded_table_delta(
            metrics,
            experiment=str(row.experiment),
            condition_key=str(row.condition),
            level_label=str(row.level),
            task_type=str(row.task_type),
            family=str(row.family),
        )
        for row in table.itertuples()
    ]
    if not np.allclose(
        table["point"].to_numpy(),
        np.asarray(expected_points),
        atol=1e-12,
    ):
        raise ValueError("Stored points do not match printed family differences")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = work_root / "tmp" / f"{output_path.name}.{os.getpid()}"
    table.to_parquet(temporary, index=False)
    os.replace(temporary, output_path)
    summary_path = output_path.with_suffix(".txt")
    write_summary(table, summary_path)
    metadata = {
        "analysis": "ml4h_patient_cluster_family_difference_bootstrap",
        "n_column_tasks": TASK_COUNT,
        "n_family_tests": int(len(table)),
        "bootstrap_n": int(n_bootstrap),
        "bootstrap_seed": int(seed),
        "resampling_unit": "test_patient",
        "paired_across": [
            "baseline_and_comparison",
            "eight_families",
            "six_models",
        ],
        "aggregation": "family_outcomes_then_six_models",
        "n_intervals_containing_zero": int(
            table["cluster_ci_contains_zero"].sum()
        ),
        "output": str(output_path),
        "summary": str(summary_path),
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("map", "list", "run", "concat"),
        required=True,
    )
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--meds-raw", type=Path, default=DEFAULT_MEDS_RAW)
    parser.add_argument("--subject-map", type=Path, default=None)
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--task-index", type=int, default=None)
    parser.add_argument("--bootstrap-n", type=int, default=BOOTSTRAP_N)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--workers", type=int, default=4)
    return parser


def main() -> int:
    """Run one patient-cluster bootstrap mode."""
    args = build_parser().parse_args()
    metrics = args.metrics.expanduser().resolve()
    meds_raw = args.meds_raw.expanduser().resolve()
    work_root = args.work_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    map_path = (
        args.subject_map.expanduser().resolve()
        if args.subject_map is not None
        else subject_map_path(work_root)
    )
    for directory in (work_root / "shards", work_root / "tmp", work_root / "logs"):
        directory.mkdir(parents=True, exist_ok=True)

    if args.mode == "map":
        result = build_subject_map(
            meds_raw=meds_raw,
            output_path=map_path,
        )
    elif args.mode == "list":
        result = {"n_tasks": TASK_COUNT, "tasks": task_specs()}
    elif args.mode == "run":
        if args.task_index is None:
            raise ValueError("--task-index is required for run mode")
        result = run_task(
            task_index=int(args.task_index),
            metrics_path=metrics,
            map_path=map_path,
            work_root=work_root,
            n_bootstrap=int(args.bootstrap_n),
            seed=int(args.bootstrap_seed),
            workers=int(args.workers),
        )
    else:
        result = concat_tasks(
            work_root=work_root,
            output_path=output,
            metrics_path=metrics,
            n_bootstrap=int(args.bootstrap_n),
            seed=int(args.bootstrap_seed),
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
