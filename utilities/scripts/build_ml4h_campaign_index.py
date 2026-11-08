#!/usr/bin/env python3
"""Build a task-index for ML4H Stage-1 jobfiles (array id → arm metadata)."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _parse_exp1(line: str, task_id: int) -> dict:
    config_id = re.search(r"config_id=(\S+)", line)
    data_version = re.search(r"data_version=(\S+)", line)
    seed = re.search(r"\bseed=(\d+)", line)
    model = re.search(r"IRB_MODEL_NAME=(\S+)", line)
    prefix = re.search(r"IRB_MODEL_VERSION_PREFIX=(\S+)", line)
    backbone = "llama" if prefix and "llama" in prefix.group(1) else "qwen"
    return {
        "task_id": task_id,
        "experiment": "exp1",
        "backbone": backbone,
        "model_name": model.group(1) if model else None,
        "model_version_prefix": prefix.group(1) if prefix else None,
        "config_id": config_id.group(1) if config_id else None,
        "data_version": data_version.group(1) if data_version else None,
        "seed": int(seed.group(1)) if seed else None,
        "command": line,
    }


def _parse_exp2(line: str, task_id: int) -> dict:
    config_id = re.search(r"--config_id (\S+)", line)
    data_version = re.search(r"--data_version (\S+)", line)
    representation = re.search(r"--representation (\S+)", line)
    temporal = re.search(r"--temporal (\S+)", line)
    num_bins = re.search(r"--num_bins (\S+)", line)
    seed = re.search(r"--seed (\d+)", line)
    model = re.search(r"IRB_MODEL_NAME=(\S+)", line)
    prefix = re.search(r"IRB_MODEL_VERSION_PREFIX=(\S+)", line)
    backbone = "llama" if prefix and "llama" in prefix.group(1) else "qwen"
    return {
        "task_id": task_id,
        "experiment": "exp2",
        "backbone": backbone,
        "model_name": model.group(1) if model else None,
        "model_version_prefix": prefix.group(1) if prefix else None,
        "config_id": config_id.group(1) if config_id else None,
        "data_version": data_version.group(1) if data_version else None,
        "representation": representation.group(1) if representation else None,
        "temporal": temporal.group(1) if temporal else None,
        "num_bins": int(num_bins.group(1)) if num_bins else None,
        "seed": int(seed.group(1)) if seed else None,
        "command": line,
    }


def index_jobfile(path: Path, experiment: str) -> list[dict]:
    if not path.is_file():
        return []
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    parser = _parse_exp1 if experiment == "exp1" else _parse_exp2
    return [parser(line, task_id) for task_id, line in enumerate(lines)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generated-dir",
        type=Path,
        default=Path("slurm/generated/ml4h_2026"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/runs/ml4h_2026/submissions/campaign_task_index.json"),
    )
    args = parser.parse_args()

    exp1_path = args.generated_dir / "02_exp1_stage1.jobfile"
    exp2_path = args.generated_dir / "03_exp2_stage1.jobfile"
    held_exp2 = args.generated_dir / "held_until_exp1" / "03_exp2_stage1.jobfile"
    if not exp2_path.is_file() and held_exp2.is_file():
        exp2_path = held_exp2

    exp1 = index_jobfile(exp1_path, "exp1")
    exp2 = index_jobfile(exp2_path, "exp2")
    payload = {
        "exp1": {
            "jobfile": str(exp1_path),
            "n_tasks": len(exp1),
            "array_spec": f"0-{len(exp1) - 1}" if exp1 else "",
            "tasks": exp1,
        },
        "exp2": {
            "jobfile": str(exp2_path) if exp2_path.is_file() else None,
            "status": "held_until_exp1" if "held_until_exp1" in str(exp2_path) else "active",
            "n_tasks": len(exp2),
            "array_spec": f"0-{len(exp2) - 1}" if exp2 else "",
            "tasks": exp2,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Compact TSV for grepping in the terminal.
    tsv = args.output.with_suffix(".tsv")
    with tsv.open("w", encoding="utf-8") as handle:
        handle.write(
            "experiment\ttask_id\tbackbone\tseed\tconfig_id\trepresentation\ttemporal\n"
        )
        for experiment, block in (("exp1", exp1), ("exp2", exp2)):
            for row in block:
                handle.write(
                    f"{experiment}\t{row['task_id']}\t{row['backbone']}\t{row.get('seed')}\t"
                    f"{row.get('config_id')}\t{row.get('representation', '')}\t"
                    f"{row.get('temporal', '')}\n"
                )

    print(f"Wrote {args.output} ({len(exp1)} Exp1 + {len(exp2)} Exp2 tasks)")
    print(f"Wrote {tsv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
