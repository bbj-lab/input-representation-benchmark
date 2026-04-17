#!/usr/bin/env python3
"""Merge outcome-sharded family stats into the standard family output layout."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _bh_adjust(pvals: list[float]) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    n = p.size
    if n == 0:
        return np.asarray([], dtype=float)
    order = np.argsort(np.isnan(p).astype(int), kind="stable")
    ranked = p[order]
    adjusted = np.full(n, np.nan, dtype=float)
    running = 1.0
    for idx in range(n - 1, -1, -1):
        value = ranked[idx]
        if np.isnan(value):
            continue
        rank = idx + 1
        candidate = value * n / rank
        running = min(running, candidate)
        adjusted[idx] = running
    out = np.full(n, np.nan, dtype=float)
    out[order] = np.clip(adjusted, 0.0, 1.0)
    return out


def _collect_family_paths(shard_root: Path, family_name: str) -> tuple[list[Path], list[Path]]:
    metrics_paths: list[Path] = []
    pairwise_paths: list[Path] = []
    for shard_dir in sorted(p for p in shard_root.iterdir() if p.is_dir()):
        family_dir = shard_dir / family_name
        metrics_path = family_dir / f"{family_name}-metrics.csv"
        pairwise_path = family_dir / f"{family_name}-pairwise.csv"
        if metrics_path.exists():
            metrics_paths.append(metrics_path)
        if pairwise_path.exists():
            pairwise_paths.append(pairwise_path)
    return metrics_paths, pairwise_paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard_root", type=Path, required=True)
    parser.add_argument("--family_name", type=str, required=True)
    parser.add_argument("--out_root", type=Path, required=True)
    parser.add_argument("--expected_shards", type=int, default=None)
    parser.add_argument("--fdr", choices=["bh", "none"], default="bh")
    args = parser.parse_args()

    shard_root = args.shard_root.expanduser().resolve()
    out_root = args.out_root.expanduser().resolve()
    out_dir = out_root / args.family_name
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_paths, pairwise_paths = _collect_family_paths(shard_root, args.family_name)
    if not metrics_paths:
        raise FileNotFoundError(f"No shard metrics found for {args.family_name} under {shard_root}")
    if not pairwise_paths:
        raise FileNotFoundError(f"No shard pairwise tables found for {args.family_name} under {shard_root}")
    if args.expected_shards is not None:
        if len(metrics_paths) != args.expected_shards or len(pairwise_paths) != args.expected_shards:
            raise ValueError(
                f"Expected {args.expected_shards} shards for {args.family_name}, "
                f"found metrics={len(metrics_paths)} pairwise={len(pairwise_paths)}."
            )

    metrics_df = pd.concat((pd.read_csv(path) for path in metrics_paths), ignore_index=True)
    metrics_df = metrics_df.sort_values(
        ["outcome", "metric", "handle"],
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)

    pairwise_df = pd.concat((pd.read_csv(path) for path in pairwise_paths), ignore_index=True)
    pairwise_df = pairwise_df.sort_values(
        ["outcome", "metric", "handle0", "handle1"],
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)
    if not pairwise_df.empty:
        pairwise_df["p_adj"] = np.nan
        if args.fdr == "bh":
            for _, idxs in pairwise_df.groupby("metric", sort=False).groups.items():
                row_idx = list(idxs)
                pvals = pairwise_df.loc[row_idx, "p_raw"].astype(float).tolist()
                pairwise_df.loc[row_idx, "p_adj"] = _bh_adjust(pvals)
        else:
            pairwise_df["p_adj"] = pairwise_df["p_raw"]

    metrics_path = out_dir / f"{args.family_name}-metrics.csv"
    pairwise_path = out_dir / f"{args.family_name}-pairwise.csv"
    metrics_df.to_csv(metrics_path, index=False)
    pairwise_df.to_csv(pairwise_path, index=False)

    print(f"Wrote {metrics_path}")
    print(f"Wrote {pairwise_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
