#!/usr/bin/env python3
"""
Backfill stable Exp1 best-model aliases in the current canonical artifact tree.

Fresh Exp1 runs already create `-hp-<data_version>` aliases via `tune_model.py`.
This script only materializes the same aliases for the existing historical run
tree so Stage 2/3 helpers can use the stable path convention uniformly.
"""

from __future__ import annotations

from pathlib import Path


RUN_MAP = {
    "meds_deciles_none_fusedFalse_discrete_time_tokens": ("deciles_none_unfused_time_tokens", "run-0"),
    "meds_deciles_none_fusedTrue_discrete_time_tokens": ("deciles_none_fused_time_tokens", "run-1"),
    "meds_ventiles_none_fusedFalse_discrete_time_tokens": ("ventiles_none_unfused_time_tokens", "run-1"),
    "meds_ventiles_none_fusedTrue_discrete_time_tokens": ("ventiles_none_fused_time_tokens", "run-2"),
    "meds_ventiles_5-10-5_fusedFalse_discrete_time_tokens": ("ventiles_5-10-5_unfused_time_tokens", "run-1"),
    "meds_ventiles_5-10-5_fusedTrue_discrete_time_tokens": ("ventiles_5-10-5_fused_time_tokens", "run-0"),
    "meds_trentiles_none_fusedFalse_discrete_time_tokens": ("trentiles_none_unfused_time_tokens", "run-1"),
    "meds_trentiles_none_fusedTrue_discrete_time_tokens": ("trentiles_none_fused_time_tokens", "run-2"),
    "meds_trentiles_10-10-10_fusedFalse_discrete_time_tokens": ("trentiles_10-10-10_unfused_time_tokens", "run-1"),
    "meds_trentiles_10-10-10_fusedTrue_discrete_time_tokens": ("trentiles_10-10-10_fused_time_tokens", "run-0"),
    "meds_centiles_none_fusedFalse_discrete_time_tokens": ("centiles_none_unfused_time_tokens", "run-0"),
    "meds_centiles_none_fusedTrue_discrete_time_tokens": ("centiles_none_fused_time_tokens", "run-0"),
}


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    model_root = repo_root / "artifacts" / "runs" / "models"

    for config_id, (data_version, run_dir) in RUN_MAP.items():
        source = model_root / f"exp1_{config_id}-s42" / run_dir / "checkpoint-9000"
        alias = model_root / f"exp1_{config_id}-s42-hp-{data_version}"
        if not source.exists():
            raise FileNotFoundError(f"Missing source checkpoint for alias: {source}")
        if alias.is_symlink():
            current = alias.resolve(strict=False)
            if current == source and current.exists():
                continue
            alias.unlink()
        elif alias.exists():
            raise FileExistsError(f"Alias path exists and is not a symlink: {alias}")
        alias.symlink_to(source, target_is_directory=True)
        print(f"Created {alias} -> {source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
