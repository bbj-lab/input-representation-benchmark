#!/usr/bin/env python3
"""Create missing representation_mechanics.pt files for legacy xVal runs.

Usage:
    python3 pipeline/scripts/create_xval_rep_mechanics.py
    python3 pipeline/scripts/create_xval_rep_mechanics.py --dry-run
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


CONFIGS = [
    {
        "path": "outputs/runs/models/exp2_meds_deciles_none_fusedFalse_xval_time_tokens-xval-time_tokens-s42/model-xval-time_tokens/representation_mechanics.pt",
        "representation": "xval",
        "temporal": "time_tokens",
    },
    {
        "path": "outputs/runs/models/exp2_meds_deciles_none_fusedFalse_xval_time_rope-xval-time_rope-s42/model-xval-time_rope/representation_mechanics.pt",
        "representation": "xval",
        "temporal": "time_rope",
    },
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo_root",
        type=Path,
        default=Path("."),
        help="Benchmark repository root (default: current directory).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print target paths without writing files.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()

    for cfg in CONFIGS:
        target_path = repo_root / cfg["path"]
        rep_state = {
            "representation": cfg["representation"],
            "temporal": cfg["temporal"],
            "num_bins": 10,
            "time_rope_scaling": 60.0,
            "value_encoder_state": None,  # xVal does not use a separate value encoder.
        }
        if args.dry_run:
            print(f"[dry-run] would create: {target_path}")
            continue

        target_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(rep_state, target_path)
        print(f"Created: {target_path}")

    if args.dry_run:
        print("Done (dry-run).")
    else:
        print("Done. All representation_mechanics.pt files created.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
