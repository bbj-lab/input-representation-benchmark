#!/usr/bin/env python3
"""Generate validated ML4H Exp3 Stage-3 binary and regression probe jobfiles.

Exp3 uses the ICU-hospitalization cohort H_ICU, where ``icu_admission`` is
degenerate. Binary probes therefore drop that endpoint and keep
``prolonged_icu_stay`` (38 of 39 ML4H outcomes). Fixed probes: logistic C=0.01,
ridge alpha=100.
"""

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
from pipeline.task_registry import BINARY_TASKS, REGRESSION_TASKS

# H_ICU makes icu_admission degenerate; keep prolonged_icu_stay instead.
EXP3_BINARY_TASKS = tuple(task for task in BINARY_TASKS if task != "icu_admission")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage2-manifest",
        type=Path,
        default=Path("outputs/runs/ml4h_2026/stage2_exp3/exp3_extract_manifest.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/runs/ml4h_2026/stage3_exp3"),
    )
    parser.add_argument(
        "--fms-ehrs-home",
        type=Path,
        default=Path("../fms-ehrs"),
    )
    parser.add_argument(
        "--preds-tag-suffix",
        default="",
        help="Suffix appended to Stage-3 prediction and probe artifact tags.",
    )
    return parser.parse_args()


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


def validate_task(*, task: dict, outcomes: tuple[str, ...]) -> None:
    model_loc = Path(task["model_loc"])
    config_path = model_loc / "config.json"
    if not config_path.is_file():
        raise SystemExit(f"Missing model config: {config_path}")
    hidden_size = int(json.loads(config_path.read_text(encoding="utf-8"))["hidden_size"])
    data_dir = Path(task["data_dir"])
    tokenized = data_dir / f"{task['data_version']}-tokenized"
    missing: list[str] = []
    for split in ("train", "val", "test"):
        split_dir = tokenized / split
        token_path = split_dir / "tokens_timelines.parquet"
        outcomes_path = split_dir / "tokens_timelines_extended_outcomes.parquet"
        feature_path = split_dir / feature_filename(model_artifact_stem(model_loc))
        if not token_path.is_file():
            missing.append(str(token_path))
            continue
        if not outcomes_path.is_file():
            missing.append(str(outcomes_path))
            continue
        n_rows = pq.ParquetFile(token_path).metadata.num_rows
        if not is_valid_feature_file(feature_path, (n_rows, hidden_size)):
            missing.append(str(feature_path))
        outcome_rows = pq.ParquetFile(outcomes_path).metadata.num_rows
        if outcome_rows != n_rows:
            missing.append(
                f"{outcomes_path} (has {outcome_rows} rows; timelines have {n_rows})"
            )
        columns = set(pq.ParquetFile(outcomes_path).schema_arrow.names)
        absent_outcomes = sorted(set(outcomes) - columns)
        if absent_outcomes:
            missing.append(f"{outcomes_path} (missing {', '.join(absent_outcomes)})")
    if missing:
        raise SystemExit(
            f"Cannot schedule Stage 3 for {model_loc}; missing or invalid inputs:\n  - "
            + "\n  - ".join(missing)
        )


def command(
    *,
    task: dict,
    fms_ehrs_home: Path,
    task_kind: str,
    preds_tag_suffix: str,
) -> str:
    if task_kind == "binary":
        classifier = "logistic_regression"
        task_type = "classification"
        outcomes = EXP3_BINARY_TASKS
        extra_args = (
            "--outcomes "
            + " ".join(outcomes)
            + " --outcomes_parquet tokens_timelines_extended_outcomes.parquet"
            + " --no-tune_logreg_C --logreg_C_grid [0.01]"
        )
    elif task_kind == "regression":
        classifier = "ridge_regression"
        task_type = "regression"
        outcomes = REGRESSION_TASKS
        extra_args = (
            "--outcomes "
            + " ".join(outcomes)
            + " --outcomes_parquet tokens_timelines_extended_outcomes.parquet"
            + " --ridge_alpha_grid [100]"
        )
    else:
        raise ValueError(f"Unknown Stage-3 task kind: {task_kind}")

    environment = {
        "data_dir": str(Path(task["data_dir"])),
        "data_version": task["data_version"],
        "model_loc": task["model_loc"],
        "FMS_EHRS_HOME": str(fms_ehrs_home),
        "IRB_PREDS_TAG": f"ml4h-exp3-{task_kind}{preds_tag_suffix}",
        "IRB_STAGE3_CLASSIFIER": classifier,
        "IRB_STAGE3_TASK_TYPE": task_type,
        "IRB_XFER_PRED_ARGS": extra_args,
    }
    return " ".join(f"{key}={shlex.quote(value)}" for key, value in environment.items()) + (
        " bash slurm/ref_qse/10_xfer_rep_based_preds.sh"
    )


def write_jobfile(path: Path, commands: list[str]) -> None:
    path.write_text(
        "\n".join(
            [
                "# ML4H Exp3 Stage-3 probes",
                "# One task per checkpoint; outcomes are evaluated serially within the task.",
                "# Binary drops degenerate icu_admission; keeps prolonged_icu_stay.",
                "# Fixed probes: logistic C=0.01; ridge alpha=100.",
                *commands,
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    stage2_manifest = args.stage2_manifest.resolve()
    output_dir = args.output_dir.resolve()
    fms_ehrs_home = args.fms_ehrs_home.resolve()
    if not fms_ehrs_home.is_dir():
        raise SystemExit(f"FMS-EHRs home does not exist: {fms_ehrs_home}")
    preds_tag_suffix = args.preds_tag_suffix
    if preds_tag_suffix and not preds_tag_suffix.startswith("-"):
        preds_tag_suffix = f"-{preds_tag_suffix}"

    manifest = json.loads(stage2_manifest.read_text(encoding="utf-8"))
    tasks = manifest["tasks"]
    if len(tasks) != 12:
        raise SystemExit(f"Expected 12 Stage-2 tasks; found {len(tasks)}.")
    assert_unique_model_stems(task["model_loc"] for task in tasks)

    output_dir.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        validate_task(task=task, outcomes=EXP3_BINARY_TASKS)
        validate_task(task=task, outcomes=REGRESSION_TASKS)

    for task_kind in ("binary", "regression"):
        commands = [
            command(
                task=task,
                fms_ehrs_home=fms_ehrs_home,
                task_kind=task_kind,
                preds_tag_suffix=preds_tag_suffix,
            )
            for task in tasks
        ]
        write_jobfile(output_dir / f"exp3_stage3_{task_kind}.jobfile", commands)

    (output_dir / "exp3_stage3_manifest.json").write_text(
        json.dumps(
            {
                "task_count_per_family": len(tasks),
                "binary_outcomes": list(EXP3_BINARY_TASKS),
                "regression_outcomes": list(REGRESSION_TASKS),
                "dropped_binary_outcomes": ["icu_admission"],
                "stage2_manifest": str(stage2_manifest),
                "preds_tag_suffix": preds_tag_suffix,
                "fixed_probes": {"logreg_C": 0.01, "ridge_alpha": 100},
                "tasks": tasks,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote Stage-3 jobfiles for {len(tasks)} models to {output_dir}")
    print(f"  binary outcomes: {len(EXP3_BINARY_TASKS)} (dropped icu_admission)")
    print(f"  regression outcomes: {len(REGRESSION_TASKS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
