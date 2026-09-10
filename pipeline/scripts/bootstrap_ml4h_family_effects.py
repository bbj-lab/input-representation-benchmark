#!/usr/bin/env python3
"""Paired permutation tests for ML4H family-level configuration effects.

Each Slurm task is one printed Δ column. One admission-swap draw recomputes
all eight family-mean Δs on the full test set.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn import metrics as skl_metrics


REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURE_SCRIPT_DIR = REPO_ROOT / "paper/scripts"
sys.path.insert(0, str(FIGURE_SCRIPT_DIR))
import generate_ml4h_submission_previews as previews
import ml4h_main_table_contrasts as contrasts


DEFAULT_METRICS = REPO_ROOT / "outputs/runs/ml4h_2026/metrics/metrics_long.parquet"
DEFAULT_WORK_ROOT = (
    REPO_ROOT / "outputs/runs/ml4h_2026/family_effect_bootstrap"
)
DEFAULT_OUTPUT = DEFAULT_WORK_ROOT / "family_effect_intervals.parquet"
BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 42
TASK_COUNT = contrasts.COLUMN_COUNT


_PERMUTATION_CONTEXT: dict[str, object] | None = None


def task_specs() -> list[dict[str, str | int]]:
    tasks = contrasts.column_specs()
    if len(tasks) != TASK_COUNT:
        raise ValueError(f"Expected {TASK_COUNT} column tasks, found {len(tasks)}")
    return tasks


def find_condition(experiment: str, key: str):
    return contrasts.find_condition(experiment, key)


def find_level(condition, label: str):
    return contrasts.find_level(condition, label)


def select_config_ids(
    metrics: pd.DataFrame,
    *,
    experiment: str,
    task_type: str,
    filters: dict[str, str],
) -> tuple[str, ...]:
    rows = metrics.loc[
        (metrics["experiment"] == experiment)
        & (metrics["task_type"] == task_type)
        & (metrics["metric"] == previews.TASK_METRIC[task_type])
    ].copy()
    rows["schema"] = np.where(
        rows["config_id"].astype(str).str.startswith("meds_icu_"),
        "Native MIMIC",
        np.where(
            rows["config_id"].astype(str).str.startswith("meds_clif_"),
            "CLIF harmonized",
            "",
        ),
    )
    rows = previews.filter_rows(rows, filters)
    configs = tuple(sorted(rows["config_id"].astype(str).unique()))
    if len(configs) != 1:
        raise ValueError(
            f"{experiment}/{task_type}/{filters}: expected 1 config, found {configs}"
        )
    return configs


def artifact_table(
    metrics: pd.DataFrame,
    *,
    experiment: str,
    task_type: str,
    config_ids: tuple[str, ...],
) -> pd.DataFrame:
    rows = metrics.loc[
        (metrics["experiment"] == experiment)
        & (metrics["task_type"] == task_type)
        & (metrics["config_id"].isin(config_ids)),
        ["config_id", "backbone", "seed", "source_artifact"],
    ].drop_duplicates()
    if rows.duplicated(["config_id", "backbone", "seed"]).any():
        raise ValueError(f"Multiple artifacts found for {experiment}/{task_type}")
    expected = len(config_ids) * 6
    if len(rows) != expected:
        raise ValueError(
            f"{experiment}/{task_type}: expected {expected} artifacts, found {len(rows)}"
        )
    return rows


def load_pickle(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_timeline_ids(artifact: Path) -> np.ndarray:
    timeline = artifact.parent / "tokens_timelines.parquet"
    if not timeline.is_file():
        raise FileNotFoundError(f"Missing admission-id table for {artifact}: {timeline}")
    import pyarrow.parquet as pq

    schema_names = set(pq.ParquetFile(timeline).schema_arrow.names)
    column = next(
        (
            name
            for name in ("hadm_id", "hospitalization_id", "subject_id")
            if name in schema_names
        ),
        None,
    )
    if column is None:
        raise ValueError(f"No admission-id column in {timeline}")
    ids = pd.read_parquet(timeline, columns=[column])[column].astype(str).to_numpy()
    if len(ids) == 0 or len(np.unique(ids)) != len(ids):
        raise ValueError(f"Admission ids in {timeline} are empty or duplicated")
    return ids


def permutation_to(source_ids: np.ndarray, target_ids: np.ndarray) -> np.ndarray:
    if source_ids.shape != target_ids.shape:
        raise ValueError("Admission-id lengths differ across configs")
    if np.array_equal(source_ids, target_ids):
        return np.arange(len(target_ids), dtype=np.int64)
    if set(source_ids) != set(target_ids):
        missing = set(target_ids) - set(source_ids)
        extra = set(source_ids) - set(target_ids)
        raise ValueError(
            f"Admission ids differ across configs "
            f"(missing={len(missing)}, extra={len(extra)})"
        )
    index = {value: position for position, value in enumerate(source_ids)}
    return np.fromiter(
        (index[value] for value in target_ids),
        dtype=np.int64,
        count=len(target_ids),
    )


def full_arrays(payload: dict, outcome: str) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(payload["labels"][outcome], dtype=float)
    predictions = np.asarray(payload["predictions"][outcome], dtype=float)
    qualifiers = np.asarray(payload["qualifiers"][outcome], dtype=bool)
    if predictions.shape == labels.shape:
        full_predictions = predictions
    else:
        if qualifiers.shape != labels.shape or int(qualifiers.sum()) != len(predictions):
            raise ValueError(f"Cannot align prediction rows for {outcome}")
        full_predictions = np.full(labels.shape, np.nan, dtype=float)
        full_predictions[qualifiers] = predictions
    return labels, full_predictions


def family_outcomes(
    metrics: pd.DataFrame,
    *,
    experiment: str,
    task_type: str,
    family: str,
) -> tuple[str, ...]:
    present = set(
        metrics.loc[
            (metrics["experiment"] == experiment) & (metrics["task_type"] == task_type),
            "outcome",
        ].astype(str)
    )
    outcomes = tuple(
        outcome
        for outcome, assigned_family in previews.OUTCOME_FAMILIES.items()
        if assigned_family == family and outcome in present
    )
    if not outcomes:
        raise ValueError(f"No outcomes found for {experiment}/{task_type}/{family}")
    return outcomes


def load_column_context(
    metrics: pd.DataFrame,
    task: dict[str, str | int],
) -> dict[str, object]:
    experiment = str(task["experiment"])
    condition = find_condition(experiment, str(task["condition"]))
    level = find_level(condition, str(task["level"]))
    baseline_filters = contrasts.level_baseline_filters(condition, level)
    config_ids = {
        "baseline": select_config_ids(
            metrics,
            experiment=experiment,
            task_type="binary",
            filters=baseline_filters,
        ),
        "level": select_config_ids(
            metrics,
            experiment=experiment,
            task_type="binary",
            filters=level.filters,
        ),
    }
    for side, filters in (("baseline", baseline_filters), ("level", level.filters)):
        regression_ids = select_config_ids(
            metrics,
            experiment=experiment,
            task_type="regression",
            filters=filters,
        )
        if regression_ids != config_ids[side]:
            raise ValueError(
                f"{experiment}/{side}: binary and regression config ids differ"
            )

    families: list[dict[str, object]] = []
    for task_type, family in contrasts.LEAF_ORDER:
        families.append(
            {
                "task_type": task_type,
                "family": family,
                "outcomes": family_outcomes(
                    metrics,
                    experiment=experiment,
                    task_type=task_type,
                    family=family,
                ),
                "point": contrasts.unrounded_table_delta(
                    metrics,
                    experiment=experiment,
                    condition_key=condition.key,
                    level_label=level.label,
                    task_type=task_type,
                    family=family,
                ),
            }
        )

    labels_by_outcome: dict[str, np.ndarray] = {}
    valid_by_outcome: dict[str, np.ndarray] = {}
    predictions: dict[
        str,
        dict[str, dict[tuple[str, int], dict[str, np.ndarray]]],
    ] = {"baseline": {}, "level": {}}
    run_keys: set[tuple[str, int]] | None = None
    canonical_ids: np.ndarray | None = None

    for task_type in contrasts.TASK_ORDER:
        type_outcomes = tuple(
            outcome
            for family in families
            if family["task_type"] == task_type
            for outcome in family["outcomes"]
        )
        for side in ("baseline", "level"):
            artifacts = artifact_table(
                metrics,
                experiment=experiment,
                task_type=task_type,
                config_ids=config_ids[side],
            )
            side_run_keys = {
                (str(row.backbone), int(row.seed)) for row in artifacts.itertuples()
            }
            if len(side_run_keys) != 6:
                raise ValueError(f"{experiment}/{task_type}/{side}: expected six model keys")
            if run_keys is None:
                run_keys = side_run_keys
            elif side_run_keys != run_keys:
                raise ValueError(f"{experiment}: model keys differ across sides or task types")

            for row in artifacts.itertuples():
                config_id = str(row.config_id)
                run_key = (str(row.backbone), int(row.seed))
                artifact = Path(str(row.source_artifact))
                payload = load_pickle(artifact)
                row_ids = load_timeline_ids(artifact)
                if canonical_ids is None:
                    canonical_ids = row_ids
                    permute = np.arange(len(row_ids), dtype=np.int64)
                else:
                    permute = permutation_to(row_ids, canonical_ids)
                config_store = predictions[side].setdefault(config_id, {})
                run_store = config_store.setdefault(run_key, {})
                for outcome in type_outcomes:
                    labels, full_predictions = full_arrays(payload, outcome)
                    if labels.shape[0] != len(permute):
                        raise ValueError(
                            f"{experiment}/{outcome}: label rows {labels.shape[0]} "
                            f"!= timeline rows {len(permute)}"
                        )
                    labels = labels[permute]
                    full_predictions = full_predictions[permute]
                    valid = np.isfinite(labels) & np.isfinite(full_predictions)
                    if outcome not in labels_by_outcome:
                        labels_by_outcome[outcome] = labels.astype(np.float64, copy=False)
                        valid_by_outcome[outcome] = valid
                    else:
                        reference = labels_by_outcome[outcome]
                        if labels.shape != reference.shape or not np.allclose(
                            labels,
                            reference,
                            equal_nan=True,
                        ):
                            raise ValueError(
                                f"{experiment}/{outcome}: labels are not aligned across configs"
                            )
                        if not np.array_equal(valid, valid_by_outcome[outcome]):
                            raise ValueError(
                                f"{experiment}/{outcome}: qualifying rows differ across configs"
                            )
                    run_store[outcome] = full_predictions.astype(np.float32, copy=False)

    assert run_keys is not None
    cohort_sizes = {len(values) for values in labels_by_outcome.values()}
    if len(cohort_sizes) != 1:
        raise ValueError(f"{experiment}: outcomes have different cohort rows")

    return {
        "task": task,
        "families": families,
        "n_rows": cohort_sizes.pop(),
        "labels": labels_by_outcome,
        "valid": valid_by_outcome,
        "predictions": predictions,
        "config_ids": config_ids,
        "run_keys": tuple(sorted(run_keys)),
    }


def metric_value(task_type: str, y_true: np.ndarray, y_score: np.ndarray) -> float:
    if task_type == "binary":
        if np.unique(y_true).size < 2:
            return float("nan")
        return float(skl_metrics.roc_auc_score(y_true, y_score))
    return float(stats.spearmanr(y_true, y_score).statistic)


def family_delta_with_swap(
    context: dict[str, object],
    family: dict[str, object],
    swap: np.ndarray,
) -> float:
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
            keep = valid[outcome]
            y_true = labels[outcome][keep]
            pred_level = predictions["level"][level_config][run_key][outcome]
            pred_base = predictions["baseline"][baseline_config][run_key][outcome]
            y_level = np.where(swap, pred_base, pred_level)[keep]
            y_base = np.where(swap, pred_level, pred_base)[keep]
            value_level = metric_value(task_type, y_true, y_level)
            value_base = metric_value(task_type, y_true, y_base)
            if not np.isfinite(value_level) or not np.isfinite(value_base):
                return float("nan")
            outcome_level.append(value_level)
            outcome_base.append(value_base)
        model_values.append(float(np.mean(outcome_level) - np.mean(outcome_base)))
    return float(np.mean(model_values))


def permute_one(seed: int) -> np.ndarray:
    if _PERMUTATION_CONTEXT is None:
        raise RuntimeError("Permutation worker context is unavailable")
    rng = np.random.default_rng(seed)
    n_rows = int(_PERMUTATION_CONTEXT["n_rows"])
    swap = rng.random(n_rows) < 0.5
    deltas = [
        family_delta_with_swap(_PERMUTATION_CONTEXT, family, swap)
        for family in _PERMUTATION_CONTEXT["families"]
    ]
    return np.asarray(deltas, dtype=float)


def shard_paths(work_root: Path, task_index: int) -> tuple[Path, Path, Path]:
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
    work_root: Path,
    n_bootstrap: int,
    seed: int,
    workers: int,
) -> dict[str, object]:
    tasks = task_specs()
    if task_index < 0 or task_index >= len(tasks):
        raise IndexError(task_index)
    task = tasks[task_index]
    shard, done, temporary = shard_paths(work_root, task_index)
    if shard.is_file() and done.is_file():
        return {"status": "skipped", "task_index": task_index, "shard": str(shard)}

    metrics = pd.read_parquet(metrics_path)
    context = load_column_context(metrics, task)
    seeds = [
        int(item.generate_state(1, dtype=np.uint64)[0])
        for item in np.random.SeedSequence(seed).spawn(n_bootstrap)
    ]
    global _PERMUTATION_CONTEXT
    _PERMUTATION_CONTEXT = context
    if workers > 1:
        with mp.get_context("fork").Pool(processes=workers) as pool:
            draws = pool.map(
                permute_one, seeds, chunksize=max(1, n_bootstrap // (workers * 8))
            )
    else:
        draws = [permute_one(draw_seed) for draw_seed in seeds]
    values = np.vstack(draws)
    accepted = np.isfinite(values).all(axis=1)
    values = values[accepted]
    minimum_accepted = max(1, int(n_bootstrap * 0.99))
    if len(values) < minimum_accepted:
        raise ValueError(
            f"Task {task_index}: accepted {len(values)}/{n_bootstrap} permutation draws"
        )
    records = []
    for index, family in enumerate(context["families"]):
        observed = float(family["point"])
        records.append(
            {
                **task,
                "role": "family",
                "task_type": family["task_type"],
                "family": family["family"],
                "point": observed,
                "p_value": contrasts.permutation_pvalue(observed, values[:, index]),
                "permutation_n": int(n_bootstrap),
                "permutation_accepted": int(len(values)),
                "permutation_seed": int(seed),
                "n_models": 6,
                "n_outcomes": int(len(family["outcomes"])),
                "resampling_unit": "test_admission",
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
        "n_rows": len(records),
        "permutation_accepted": int(len(values)),
        "shard": str(shard),
    }
    done.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def concat_tasks(
    *,
    work_root: Path,
    output_path: Path,
    metrics_path: Path,
    n_bootstrap: int,
    seed: int,
) -> dict[str, object]:
    frames: list[pd.DataFrame] = []
    for task in task_specs():
        shard, done, _ = shard_paths(work_root, int(task["task_index"]))
        if not shard.is_file() or not done.is_file():
            raise FileNotFoundError(f"Incomplete family permutation task: {shard}")
        frames.append(pd.read_parquet(shard))
    table = pd.concat(frames, ignore_index=True)
    if len(table) != contrasts.ROW_COUNT:
        raise ValueError(f"Expected {contrasts.ROW_COUNT} family rows, found {len(table)}")
    if int(table["task_index"].nunique()) != TASK_COUNT:
        raise ValueError(f"Expected {TASK_COUNT} column tasks, found {table['task_index'].nunique()}")
    if not table["permutation_n"].eq(n_bootstrap).all():
        raise ValueError("Permutation count changed across tasks")
    if not table["permutation_seed"].eq(seed).all():
        raise ValueError("Permutation seed changed across tasks")
    metrics = pd.read_parquet(metrics_path)
    expected_points = [
        contrasts.unrounded_table_delta(
            metrics,
            experiment=str(row.experiment),
            condition_key=str(row.condition),
            level_label=str(row.level),
            task_type=str(row.task_type),
            family=str(row.family),
        )
        for row in table.itertuples()
    ]
    if not np.allclose(table["point"].to_numpy(), np.asarray(expected_points), atol=1e-12):
        raise ValueError("Stored points do not match unrounded main-table Δs")
    if "p_value" not in table.columns or table["p_value"].isna().any():
        raise ValueError("Every family row needs a two-sided permutation p-value")
    table = contrasts.attach_within_table_bh(table)
    temporary = work_root / "tmp" / f"{output_path.name}.{os.getpid()}.parquet"
    table.to_parquet(temporary, index=False)
    os.replace(temporary, output_path)
    metadata = {
        "analysis": "ml4h_24to48h_family_effect_permutation",
        "n_column_tasks": TASK_COUNT,
        "n_family_tests": contrasts.LEAF_COUNT,
        "n_rows": contrasts.ROW_COUNT,
        "permutation_n": n_bootstrap,
        "permutation_seed": seed,
        "resampling_unit": "test_admission",
        "paired_across": [
            "baseline_and_comparison",
            "eight_families_on_shared_swaps",
            "six_models",
        ],
        "aggregation": "family_outcomes_then_six_models",
        "temporal_contrast": "discrete_values_only",
        "multiple_testing": {
            "method": "within_table_bh",
            "q": contrasts.BH_Q,
        },
        "n_bh_significant": int(table["bh_significant"].sum()),
        "output": str(output_path),
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("list", "run", "concat"), required=True)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--task-index", type=int, default=None)
    parser.add_argument("--bootstrap-n", type=int, default=BOOTSTRAP_N)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--workers", type=int, default=4)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    metrics = args.metrics.expanduser().resolve()
    work_root = args.work_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    for directory in (work_root / "shards", work_root / "tmp", work_root / "logs"):
        directory.mkdir(parents=True, exist_ok=True)
    if args.mode == "list":
        result = {"n_tasks": TASK_COUNT, "tasks": task_specs()}
    elif args.mode == "run":
        if args.task_index is None:
            raise ValueError("--task-index is required for run mode")
        result = run_task(
            task_index=int(args.task_index),
            metrics_path=metrics,
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
