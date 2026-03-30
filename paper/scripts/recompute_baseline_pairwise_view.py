#!/usr/bin/env python3
"""Create a baseline-centered pairwise summary from the combined audit CSV.

This helper keeps the existing all-pairs audit intact, but writes a paper-facing
combined pairwise table where Experiment 1 is restricted to comparisons against
the shared unfused discrete decile baseline. Experiments 2 and 3 are already
baseline-centered in the current rerun chain, so their rows are passed through
unchanged.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IN = (
    ROOT
    / "artifacts"
    / "runs"
    / "statistics"
    / "paper_audit_20260316_idaligned_fullstats"
    / "all_family_pairwise.csv"
)
DEFAULT_OUT = (
    ROOT
    / "artifacts"
    / "runs"
    / "statistics"
    / "paper_audit_20260316_idaligned_fullstats"
    / "all_family_pairwise_baseline.csv"
)
EXP1_BASELINE = "deciles_unfused"


def _bh_adjust(pvals: list[float]) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    n = p.size
    if n == 0:
        return np.asarray([], dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.empty(n, dtype=float)
    running = 1.0
    for idx in range(n - 1, -1, -1):
        rank = idx + 1
        candidate = ranked[idx] * n / rank
        running = min(running, candidate)
        adjusted[idx] = running
    out = np.empty(n, dtype=float)
    out[order] = np.clip(adjusted, 0.0, 1.0)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairwise_csv", type=Path, default=DEFAULT_IN)
    parser.add_argument("--out_csv", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--exp1_baseline", type=str, default=EXP1_BASELINE)
    args = parser.parse_args()

    pairwise = pd.read_csv(args.pairwise_csv)

    exp1_mask = pairwise["family_name"].astype(str).str.startswith("exp1_")
    exp1_keep = exp1_mask & (
        (pairwise["handle0"] == args.exp1_baseline)
        | (pairwise["handle1"] == args.exp1_baseline)
    )
    out = pairwise[(~exp1_mask) | exp1_keep].copy()

    exp1_rows = out["family_name"].astype(str).str.startswith("exp1_")
    if exp1_rows.any():
        out.loc[exp1_rows, "p_adj"] = np.nan
        grouped = out[exp1_rows].groupby(["family_name", "metric"], sort=False)
        for _, idxs in grouped.groups.items():
            row_idx = list(idxs)
            pvals = out.loc[row_idx, "p_raw"].astype(float).tolist()
            out.loc[row_idx, "p_adj"] = _bh_adjust(pvals)

    out_path = args.out_csv.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(
        f"Wrote baseline-centered pairwise view to {out_path} "
        f"({len(out)} rows from {len(pairwise)} source rows)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
