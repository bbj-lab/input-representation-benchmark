#!/usr/bin/env python
"""Check that every Exp2 configuration index is active or completed exactly once.

A failed array task is not coverage: its index must appear in an active rescue
array. Completed tasks are retained as coverage so routine monitoring does not
mistake successful runs for missing work.
"""

from __future__ import annotations

import collections
import subprocess
import sys

N_CONFIGS = 72
CAMPAIGN_START = "2026-08-03T00:00:00"


def _run(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout


def _fmt(indices: set[int]) -> str:
    spans: list[list[int]] = []
    for value in sorted(indices):
        if spans and value == spans[-1][1] + 1:
            spans[-1][1] = value
        else:
            spans.append([value, value])
    return ", ".join(f"{a}-{b}" if a != b else str(a) for a, b in spans)


def main() -> int:
    live: dict[str, set[int]] = collections.defaultdict(set)
    meta: dict[str, tuple[str, str]] = {}
    for line in _run(
        ["squeue", "-r", "-h", "-u", "ihlee", "-o", "%i|%j|%T|%P|%q"]
    ).splitlines():
        if not line.strip():
            continue
        job_id, name, _state, partition, qos = line.split("|")
        array, _, task = job_id.rpartition("_")
        if not task.isdigit() or "exp2" not in name:
            continue
        live[array].add(int(task))
        meta[array] = (name, f"{partition} / {qos}")

    completed: set[int] = set()
    for line in _run(
        [
            "sacct",
            "-X",
            "-n",
            "-P",
            "-S",
            CAMPAIGN_START,
            "-u",
            "ihlee",
            "--format=JobID,JobName,State",
        ]
    ).splitlines():
        if not line.strip():
            continue
        job_id, name, state = line.split("|")
        array, _, task = job_id.rpartition("_")
        if "ml4h-exp2" not in name or not task.isdigit():
            continue
        if state == "COMPLETED":
            completed.add(int(task))

    if not live and not completed:
        print("no active or completed Exp2 configurations found")
        return 1

    coverage: collections.Counter[int] = collections.Counter()
    print(f"{'array':<12}{'route':<22}{'n':>4}  active indices")
    for array in sorted(live):
        coverage.update(live[array])
        name, route = meta[array]
        print(f"{array:<12}{name:<22}{len(live[array]):>4}  [{_fmt(live[array])}]")

    coverage.update(completed)

    missing = sorted(set(range(N_CONFIGS)) - set(coverage))
    duplicated = sorted(index for index, count in coverage.items() if count > 1)
    print(f"\ncompleted indices   : [{_fmt(completed)}]")
    print(f"covered configurations: {sum(coverage.values())}/{N_CONFIGS}")
    print(f"missing             : {missing or 'NONE'}")
    print(f"duplicated          : {duplicated or 'NONE'}")

    ok = not missing and not duplicated
    print(
        "\nRESULT:",
        "all 72 configurations are active or completed exactly once"
        if ok
        else "ACTION NEEDED — coverage is not exact",
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
