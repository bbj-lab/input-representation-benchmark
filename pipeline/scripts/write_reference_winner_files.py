#!/usr/bin/env python3
"""
Write the benchmark-pinned winner config_id files used by gated downstream jobs.

These winners are the current reference decisions documented in the benchmark
README for the fixed 24-model benchmark, not a generic winner-selection rule
for arbitrary future experiments.
"""

from __future__ import annotations

import argparse
from pathlib import Path


REFERENCE_EXP1_OVERALL_WINNER = "meds_deciles_none_fusedTrue_discrete_time_tokens"
REFERENCE_EXP1_UNFUSED_WINNER = "meds_deciles_none_fusedFalse_discrete_time_tokens"


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Write benchmark-pinned Exp1 winner config_id files."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs"),
        help="Directory where winner files should be written.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    _write(output_dir / "exp1_winner_config_id.txt", REFERENCE_EXP1_OVERALL_WINNER)
    _write(
        output_dir / "exp1_unfused_winner_config_id.txt",
        REFERENCE_EXP1_UNFUSED_WINNER,
    )
    print(f"Wrote {output_dir / 'exp1_winner_config_id.txt'}")
    print(f"Wrote {output_dir / 'exp1_unfused_winner_config_id.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
