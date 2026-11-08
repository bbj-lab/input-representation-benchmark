#!/usr/bin/env python3
"""Print the space-separated probe names for one registry task group."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline.task_registry import task_names


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_type", choices=("binary", "regression"))
    args = parser.parse_args()
    print(" ".join(task_names(args.task_type)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
