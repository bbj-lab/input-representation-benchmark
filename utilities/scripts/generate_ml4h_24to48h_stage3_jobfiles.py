#!/usr/bin/env python3
"""Generate validated provisional 24-to-48-hour Stage-3 probe jobfiles."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "fms-ehrs"))
from fms_ehrs.framework.artifacts import (
    assert_unique_model_stems,
    feature_filename,
    model_artifact_stem,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline.provisional_24to48_task_registry import (
    EXPECTED_TASK_COUNTS,
    OUTCOMES_FILENAME,
    PROVISIONAL_BINARY_TASKS,
    PROVISIONAL_EXP3_BINARY_TASKS,
    PROVISIONAL_REGRESSION_TASKS,
)


DEFAULT_SHARED_TOKENIZED_ROOT = Path(
    "outputs/runs/tokenized/mimiciv-3.1_meds_70-10-20"
)
EXPERIMENT_CONFIG = {
    "exp1": {
        "expected_runs": 72,
        "stage2_manifest": Path(
            "outputs/runs/ml4h_2026/stage2_repaired_test/exp1_extract_manifest.json"
        ),
    },
    "exp2": {
        "expected_runs": 72,
        "stage2_manifest": Path(
            "outputs/runs/ml4h_2026/stage2_exp2/exp2_extract_manifest.json"
        ),
    },
    "exp3": {
        "expected_runs": 12,
        "stage2_manifest": Path(
            "outputs/runs/ml4h_2026/stage2_exp3/exp3_extract_manifest.json"
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=tuple(EXPERIMENT_CONFIG), required=True)
    parser.add_argument("--stage2-manifest", type=Path, default=None)
    parser.add_argument(
        "--shared-tokenized-root",
        type=Path,
        default=DEFAULT_SHARED_TOKENIZED_ROOT,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to outputs/runs/ml4h_2026/stage3_24to48h_<experiment>.",
    )
    parser.add_argument("--fms-ehrs-home", type=Path, default=Path("../fms-ehrs"))
    parser.add_argument(
        "--expected-runs",
        type=int,
        default=None,
        help="Override the Stage-2 task count check (default: experiment spec).",
    )
    parser.add_argument(
        "--skip-feature-validation",
        action="store_true",
        help="Write jobfiles before afterok-gated feature/outcome materialization.",
    )
    return parser.parse_args()


def outcomes_for(experiment: str, task_kind: str) -> tuple[str, ...]:
    if task_kind == "binary":
        return (
            PROVISIONAL_EXP3_BINARY_TASKS
            if experiment == "exp3"
            else PROVISIONAL_BINARY_TASKS
        )
    if task_kind == "regression":
        return PROVISIONAL_REGRESSION_TASKS
    raise ValueError(f"Unknown task kind: {task_kind}")


def task_tokenized_dir(
    *,
    experiment: str,
    task: dict,
    shared_tokenized_root: Path,
) -> Path:
    root = (
        Path(task["data_dir"]).expanduser().resolve()
        if experiment == "exp3"
        else shared_tokenized_root
    )
    return root / f"{task['data_version']}-tokenized"


def is_valid_feature_file(path: Path, expected_shape: tuple[int, int]) -> bool:
    if not path.is_file():
        return False
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        valid = array.shape == expected_shape and array.dtype == np.float16
        del array
        return valid
    except (OSError, ValueError):
        return False


def validate_task(
    *,
    experiment: str,
    task: dict,
    shared_tokenized_root: Path,
    outcomes: tuple[str, ...],
) -> None:
    model_loc = Path(task["model_loc"])
    config_path = model_loc / "config.json"
    if not config_path.is_file():
        raise SystemExit(f"Missing model config: {config_path}")
    hidden_size = int(json.loads(config_path.read_text(encoding="utf-8"))["hidden_size"])
    tokenized_dir = task_tokenized_dir(
        experiment=experiment,
        task=task,
        shared_tokenized_root=shared_tokenized_root,
    )
    missing: list[str] = []
    for split in ("train", "val", "test"):
        split_dir = tokenized_dir / split
        timeline_path = split_dir / "tokens_timelines.parquet"
        outcome_path = split_dir / OUTCOMES_FILENAME
        feature_path = split_dir / feature_filename(model_artifact_stem(model_loc))
        if not timeline_path.is_file():
            missing.append(str(timeline_path))
            continue
        if not outcome_path.is_file():
            missing.append(str(outcome_path))
            continue
        n_rows = pq.ParquetFile(timeline_path).metadata.num_rows
        if not is_valid_feature_file(feature_path, (n_rows, hidden_size)):
            missing.append(str(feature_path))
        outcome_parquet = pq.ParquetFile(outcome_path)
        if outcome_parquet.metadata.num_rows != n_rows:
            missing.append(
                f"{outcome_path} (has {outcome_parquet.metadata.num_rows} rows; "
                f"timelines have {n_rows})"
            )
        absent = sorted(set(outcomes) - set(outcome_parquet.schema_arrow.names))
        if absent:
            missing.append(f"{outcome_path} (missing {', '.join(absent)})")
    if missing:
        raise SystemExit(
            f"Cannot schedule provisional Stage 3 for {model_loc}; invalid inputs:\n  - "
            + "\n  - ".join(missing)
        )


def command(
    *,
    experiment: str,
    task: dict,
    shared_tokenized_root: Path,
    fms_ehrs_home: Path,
    task_kind: str,
) -> str:
    if task_kind == "binary":
        classifier = "logistic_regression"
        task_type = "classification"
        extra_args = (
            "--outcomes "
            + " ".join(outcomes_for(experiment, task_kind))
            + f" --outcomes_parquet {OUTCOMES_FILENAME}"
            + " --no-tune_logreg_C --logreg_C_grid [0.01]"
        )
    else:
        classifier = "ridge_regression"
        task_type = "regression"
        extra_args = (
            "--outcomes "
            + " ".join(outcomes_for(experiment, task_kind))
            + f" --outcomes_parquet {OUTCOMES_FILENAME}"
            + " --ridge_alpha_grid [100]"
        )
    data_dir = (
        Path(task["data_dir"]).expanduser().resolve()
        if experiment == "exp3"
        else shared_tokenized_root
    )
    environment = {
        "data_dir": str(data_dir),
        "data_version": task["data_version"],
        "model_loc": task["model_loc"],
        "FMS_EHRS_HOME": str(fms_ehrs_home),
        "IRB_PREDS_TAG": f"ml4h-{experiment}-24to48h-{task_kind}",
        "IRB_STAGE3_CLASSIFIER": classifier,
        "IRB_STAGE3_TASK_TYPE": task_type,
        "IRB_XFER_PRED_ARGS": extra_args,
    }
    return " ".join(
        f"{key}={shlex.quote(str(value))}" for key, value in environment.items()
    ) + " bash slurm/ref_qse/10_xfer_rep_based_preds.sh"


def write_jobfile(path: Path, *, experiment: str, kind: str, commands: list[str]) -> None:
    path.write_text(
        "\n".join(
            [
                f"# ML4H {experiment} provisional 24-to-48-hour {kind} probes",
                "# Existing 0-to-24h feature matrices; fixed C=0.01 / alpha=100.",
                *commands,
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    experiment = args.experiment
    config = EXPERIMENT_CONFIG[experiment]
    stage2_manifest = (
        args.stage2_manifest or config["stage2_manifest"]
    ).expanduser().resolve()
    shared_tokenized_root = args.shared_tokenized_root.expanduser().resolve()
    fms_ehrs_home = args.fms_ehrs_home.expanduser().resolve()
    if not fms_ehrs_home.is_dir():
        raise SystemExit(f"FMS-EHRs home does not exist: {fms_ehrs_home}")
    expected_runs = args.expected_runs or config["expected_runs"]
    source_manifest = json.loads(stage2_manifest.read_text(encoding="utf-8"))
    tasks = source_manifest["tasks"]
    if len(tasks) != expected_runs:
        raise SystemExit(
            f"Expected {expected_runs} {experiment} Stage-2 tasks; "
            f"found {len(tasks)}."
        )
    assert_unique_model_stems(task["model_loc"] for task in tasks)

    output_dir = (
        args.output_dir
        or Path(f"outputs/runs/ml4h_2026/stage3_24to48h_{experiment}")
    ).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_feature_validation:
        for task in tasks:
            for kind in ("binary", "regression"):
                validate_task(
                    experiment=experiment,
                    task=task,
                    shared_tokenized_root=shared_tokenized_root,
                    outcomes=outcomes_for(experiment, kind),
                )

    for kind in ("binary", "regression"):
        commands = [
            command(
                experiment=experiment,
                task=task,
                shared_tokenized_root=shared_tokenized_root,
                fms_ehrs_home=fms_ehrs_home,
                task_kind=kind,
            )
            for task in tasks
        ]
        write_jobfile(
            output_dir / f"{experiment}_24to48h_{kind}.jobfile",
            experiment=experiment,
            kind=kind,
            commands=commands,
        )
    binary_outcomes = outcomes_for(experiment, "binary")
    regression_outcomes = outcomes_for(experiment, "regression")
    (output_dir / f"{experiment}_24to48h_stage3_manifest.json").write_text(
        json.dumps(
            {
                "analysis": "24to48h",
                "experiment": experiment,
                "task_count_per_family": len(tasks),
                "binary_outcomes": list(binary_outcomes),
                "regression_outcomes": list(regression_outcomes),
                "outcomes_filename": OUTCOMES_FILENAME,
                "expected_outcome_count": EXPECTED_TASK_COUNTS[experiment],
                "stage2_manifest": str(stage2_manifest),
                "fixed_probes": {"logreg_C": 0.01, "ridge_alpha": 100},
                "preds_tags": {
                    "binary": f"ml4h-{experiment}-24to48h-binary",
                    "regression": f"ml4h-{experiment}-24to48h-regression",
                },
                "tasks": tasks,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote provisional {experiment} Stage-3 jobfiles for {len(tasks)} models "
        f"({len(binary_outcomes)} binary + {len(regression_outcomes)} regression outcomes)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
