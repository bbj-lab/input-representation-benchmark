#!/usr/bin/env python3
"""Report Slurm state and local live telemetry for ML4H training runs.

Examples
--------
python utilities/scripts/monitor_ml4h_runs.py --jobs 13234494,13234495
python utilities/scripts/monitor_ml4h_runs.py --runs-root outputs/runs/ml4h_2026
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


def _last_json_line(path: Path) -> dict[str, Any] | None:
    latest = None
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                latest = json.loads(line)
            except json.JSONDecodeError:
                continue
    return latest


def _scheduler_state(job_ids: str) -> str:
    command = [
        "squeue",
        f"--jobs={job_ids}",
        "--format=%.18i %.2t %.10M %.10P %.18R",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        return result.stderr.strip() or "squeue returned no data."
    return result.stdout.rstrip()


def _progress_row(path: Path, event: dict[str, Any]) -> str:
    metrics = event.get("metrics", {})
    progress = event.get("progress_fraction")
    progress_text = f"{100 * progress:.1f}%" if progress is not None else "unknown"
    loss = metrics.get("loss", metrics.get("train_loss", "—"))
    eval_loss = metrics.get("eval_loss", "—")
    step = event.get("global_step", 0)
    max_steps = event.get("max_steps", "—")
    return (
        f"{path.parent.name}\n"
        f"  event={event.get('event')} step={step}/{max_steps} progress={progress_text} "
        f"epoch={event.get('epoch')} loss={loss} eval_loss={eval_loss} "
        f"rate={event.get('steps_per_second')} steps/s "
        f"at={event.get('timestamp_utc')}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("outputs/runs/ml4h_2026"),
    )
    parser.add_argument(
        "--jobs",
        help="Comma-separated Slurm job or array IDs to query.",
    )
    parser.add_argument("--json", action="store_true", help="Emit telemetry as JSON.")
    args = parser.parse_args()

    if args.jobs:
        print("Scheduler:")
        print(_scheduler_state(args.jobs))

    telemetry = []
    for path in sorted(args.runs_root.glob("**/progress.jsonl")):
        event = _last_json_line(path)
        if event is not None:
            telemetry.append((path, event))

    if args.json:
        print(json.dumps([event for _, event in telemetry], indent=2, sort_keys=True))
        return 0

    print("\nLocal trainer telemetry:")
    if not telemetry:
        print("No progress.jsonl files yet.")
        return 0
    for path, event in telemetry:
        print(_progress_row(path, event))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
