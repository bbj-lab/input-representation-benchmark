#!/usr/bin/env python3
"""Generate validated, route-sharded Stage-2 extraction jobfiles for ML4H Exp2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "fms-ehrs"))
from fms_ehrs.framework.artifacts import assert_unique_model_stems

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline.run_experiments import get_data_version, make_exp2_configs


ROUTES = ("h200", "rtx_zhou", "rtx_bbj", "rtx_yang")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("outputs/runs/ml4h_2026/models"),
    )
    parser.add_argument(
        "--tokenized-root",
        type=Path,
        default=Path("outputs/runs/tokenized/mimiciv-3.1_meds_70-10-20"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/runs/ml4h_2026/stage2_exp2"),
    )
    parser.add_argument(
        "--fms-ehrs-home",
        type=Path,
        default=Path("../fms-ehrs"),
    )
    parser.add_argument("--h200-count", type=int, default=48)
    parser.add_argument("--rtx-zhou-count", type=int, default=8)
    parser.add_argument("--rtx-bbj-count", type=int, default=8)
    parser.add_argument("--rtx-yang-count", type=int, default=8)
    parser.add_argument(
        "--only-representation",
        default=None,
        help="If set, generate extraction tasks for this representation only.",
    )
    parser.add_argument(
        "--expected-tasks",
        type=int,
        default=None,
        help=(
            "Expected number of extraction tasks after filtering. Defaults to 72, "
            "or 18 when --only-representation=xval_affine."
        ),
    )
    return parser.parse_args()


def validate_run(*, model_loc: Path, tokenized_root: Path, data_version: str) -> None:
    required = [model_loc / "config.json"]
    weights = model_loc / "model.safetensors"
    weights_bin = model_loc / "pytorch_model.bin"
    if not weights.is_file() and not weights_bin.is_file():
        required.append(weights)
    data_dir = tokenized_root / f"{data_version}_first_24h-tokenized"
    required.extend(
        data_dir / split / name
        for split in ("train", "val", "test")
        for name in ("tokens_timelines.parquet",)
    )
    required.append(data_dir / "train" / "vocab.gzip")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(
            "Cannot schedule extraction; required model or evaluation data is missing:\n  - "
            + "\n  - ".join(missing)
        )


def command(*, tokenized_root: Path, data_version: str, model_loc: Path, fms_ehrs_home: Path) -> str:
    return (
        f"data_dir={tokenized_root} "
        f"data_version={data_version}_first_24h "
        f"model_loc={model_loc} "
        f"FMS_EHRS_HOME={fms_ehrs_home} "
        "IRB_STAGE2_NPROC_PER_NODE=1 "
        "IRB_STAGE2_BATCH_SZ=8 "
        "bash slurm/ref_qse/09_extract_reps.sh"
    )


def write_jobfile(path: Path, tasks: list[dict]) -> None:
    lines = [
        "# ML4H Exp2 Stage-2 hidden-state extraction",
        "# One task per model; one GPU per task; deterministic task order.",
        *(task["command"] for task in tasks),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    route_counts = {
        "h200": args.h200_count,
        "rtx_zhou": args.rtx_zhou_count,
        "rtx_bbj": args.rtx_bbj_count,
        "rtx_yang": args.rtx_yang_count,
    }
    if any(count < 0 for count in route_counts.values()):
        raise SystemExit("Route task counts must be non-negative.")

    models_dir = args.models_dir.resolve()
    tokenized_root = args.tokenized_root.resolve()
    output_dir = args.output_dir.resolve()
    fms_ehrs_home = args.fms_ehrs_home.resolve()
    if not fms_ehrs_home.is_dir():
        raise SystemExit(f"FMS-EHRs home does not exist: {fms_ehrs_home}")

    configs = make_exp2_configs(
        include_discrete=True,
        discrete_only=False,
        include_soft=True,
        include_continuous=True,
        include_affine_continuous=True,
        quantizer="deciles",
        clinical_anchoring="none",
        fused_discrete=False,
    )
    if args.only_representation is not None:
        configs = [
            config
            for config in configs
            if config.representation == args.only_representation
        ]
        if not configs:
            raise SystemExit(
                f"No Exp2 configs matched representation={args.only_representation!r}."
            )
    expected_tasks = args.expected_tasks
    if expected_tasks is None:
        expected_tasks = 18 if args.only_representation == "xval_affine" else 72

    tasks: list[dict] = []
    for backbone in ("llama", "qwen"):
        for seed in (42, 43, 44):
            for config in configs:
                data_version = get_data_version(config)
                run_dir = (
                    models_dir
                    / f"ml4h_{backbone}_exp2_{config.config_id}-"
                    f"{config.representation}-{config.temporal}-s{seed}"
                )
                model_loc = run_dir / f"model-{config.representation}-{config.temporal}"
                validate_run(
                    model_loc=model_loc,
                    tokenized_root=tokenized_root,
                    data_version=data_version,
                )
                tasks.append(
                    {
                        "task_index": len(tasks),
                        "backbone": backbone,
                        "seed": seed,
                        "config_id": config.config_id,
                        "representation": config.representation,
                        "temporal": config.temporal,
                        "data_version": f"{data_version}_first_24h",
                        "model_loc": str(model_loc),
                        "command": command(
                            tokenized_root=tokenized_root,
                            data_version=data_version,
                            model_loc=model_loc,
                            fms_ehrs_home=fms_ehrs_home,
                        ),
                    }
                )

    if len(tasks) != expected_tasks:
        raise SystemExit(
            f"Expected {expected_tasks} Exp2 extraction tasks; generated {len(tasks)}."
        )
    assert_unique_model_stems(task["model_loc"] for task in tasks)
    if sum(route_counts.values()) != len(tasks):
        raise SystemExit(
            "Route task counts must total "
            f"{len(tasks)}; received "
            + ", ".join(f"{route}={route_counts[route]}" for route in ROUTES)
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jobfile(output_dir / "exp2_extract_all.jobfile", tasks)
    start = 0
    route_manifest: dict[str, list[int]] = {}
    for route in ROUTES:
        end = start + route_counts[route]
        route_tasks = tasks[start:end]
        write_jobfile(output_dir / f"exp2_extract_{route}.jobfile", route_tasks)
        route_manifest[route] = [task["task_index"] for task in route_tasks]
        start = end

    manifest = {
        "task_count": len(tasks),
        "nproc_per_task": 1,
        "batch_size_per_gpu": 8,
        "route_counts": route_counts,
        "route_task_indices": route_manifest,
        "tasks": tasks,
    }
    (output_dir / "exp2_extract_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(tasks)} validated extraction tasks to {output_dir}")
    for route in ROUTES:
        print(f"  {route}: {route_counts[route]} tasks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
