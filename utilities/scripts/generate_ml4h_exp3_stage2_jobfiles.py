#!/usr/bin/env python3
"""Generate validated, route-sharded Stage-2 extraction jobfiles for ML4H Exp3."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "fms-ehrs"))
from fms_ehrs.framework.artifacts import assert_unique_model_stems

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline.run_experiments import get_data_version, make_exp3_configs


ROUTES = ("h200", "rtx_zhou", "rtx_bbj", "rtx_yang")

ARM_TO_DATA_DIR = {
    "meds_icu": "outputs/runs/exp3/meds_icu",
    "meds_clif": "outputs/runs/exp3/meds_clif",
}

ARM_TO_MODEL_PREFIX = {
    "meds_icu": "meds_icu_native",
    "meds_clif": "meds_clif",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("outputs/runs/ml4h_2026/models"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/runs/ml4h_2026/stage2_exp3"),
    )
    parser.add_argument(
        "--fms-ehrs-home",
        type=Path,
        default=Path("../fms-ehrs"),
    )
    parser.add_argument("--h200-count", type=int, default=8)
    parser.add_argument("--rtx-zhou-count", type=int, default=1)
    parser.add_argument("--rtx-bbj-count", type=int, default=3)
    parser.add_argument("--rtx-yang-count", type=int, default=0)
    return parser.parse_args()


def validate_run(*, model_loc: Path, data_dir: Path, data_version: str) -> None:
    required = [model_loc / "config.json"]
    weights = model_loc / "model.safetensors"
    weights_bin = model_loc / "pytorch_model.bin"
    if not weights.is_file() and not weights_bin.is_file():
        required.append(weights)
    tokenized = data_dir / f"{data_version}_first_24h-tokenized"
    required.extend(
        tokenized / split / name
        for split in ("train", "val", "test")
        for name in ("tokens_timelines.parquet",)
    )
    required.append(tokenized / "train" / "vocab.gzip")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(
            "Cannot schedule extraction; required model or evaluation data is missing:\n  - "
            + "\n  - ".join(missing)
        )


def command(*, data_dir: Path, data_version: str, model_loc: Path, fms_ehrs_home: Path) -> str:
    return (
        f"data_dir={data_dir} "
        f"data_version={data_version}_first_24h "
        f"model_loc={model_loc} "
        f"FMS_EHRS_HOME={fms_ehrs_home} "
        "IRB_STAGE2_NPROC_PER_NODE=1 "
        "IRB_STAGE2_BATCH_SZ=8 "
        "bash slurm/ref_qse/09_extract_reps.sh"
    )


def write_jobfile(path: Path, tasks: list[dict]) -> None:
    lines = [
        "# ML4H Exp3 Stage-2 hidden-state extraction",
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
    output_dir = args.output_dir.resolve()
    fms_ehrs_home = args.fms_ehrs_home.resolve()
    if not fms_ehrs_home.is_dir():
        raise SystemExit(f"FMS-EHRs home does not exist: {fms_ehrs_home}")

    configs = make_exp3_configs()
    tasks: list[dict] = []
    for backbone in ("llama", "qwen"):
        for seed in (42, 43, 44):
            for config in configs:
                arm = "meds_icu" if config.data_format == "meds_icu" else "meds_clif"
                data_dir = Path(ARM_TO_DATA_DIR[arm]).resolve()
                data_version = get_data_version(config)
                model_prefix = ARM_TO_MODEL_PREFIX[arm]
                run_dir = (
                    models_dir
                    / f"ml4h_{backbone}_exp3_{model_prefix}-"
                    f"{config.representation}-{config.temporal}-s{seed}"
                )
                model_loc = run_dir / f"model-{config.representation}-{config.temporal}"
                validate_run(
                    model_loc=model_loc,
                    data_dir=data_dir,
                    data_version=data_version,
                )
                tasks.append(
                    {
                        "task_index": len(tasks),
                        "backbone": backbone,
                        "seed": seed,
                        "config_id": config.config_id,
                        "data_format": config.data_format,
                        "data_dir": str(data_dir),
                        "data_version": f"{data_version}_first_24h",
                        "model_loc": str(model_loc),
                        "command": command(
                            data_dir=data_dir,
                            data_version=data_version,
                            model_loc=model_loc,
                            fms_ehrs_home=fms_ehrs_home,
                        ),
                    }
                )

    if len(tasks) != 12:
        raise SystemExit(f"Expected 12 Exp3 extraction tasks; generated {len(tasks)}.")
    assert_unique_model_stems(task["model_loc"] for task in tasks)
    if sum(route_counts.values()) != len(tasks):
        raise SystemExit(
            "Route task counts must total 12; received "
            + ", ".join(f"{route}={route_counts[route]}" for route in ROUTES)
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jobfile(output_dir / "exp3_extract_all.jobfile", tasks)
    start = 0
    route_manifest: dict[str, list[int]] = {}
    for route in ROUTES:
        end = start + route_counts[route]
        route_tasks = tasks[start:end]
        write_jobfile(output_dir / f"exp3_extract_{route}.jobfile", route_tasks)
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
    (output_dir / "exp3_extract_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(tasks)} validated extraction tasks to {output_dir}")
    for route in ROUTES:
        print(f"  {route}: {route_counts[route]} tasks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
