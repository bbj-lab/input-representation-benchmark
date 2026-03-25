#!/usr/bin/env python3
"""
Evaluation-time retokenization maxlen analysis.

Goal
----
When comparing fused vs unfused tokenization regimes, differences in truncation at a fixed
`max_padded_len` can create an unfair advantage: fused sequences may simply fit more events
in the same context window. This script computes token-length distribution statistics and
truncation/coverage curves to support a principled choice of `eval_max_padded_len` for
Exp1 Stage0E (evaluation retokenization).

Inputs
------
- A tokenized dataset directory produced by Stage0 tokenization, e.g.
  ${DATA_DIR}/deciles_none_unfused_time_tokens_first_24h-tokenized/
  containing split subdirs train/val/test with tokens_timelines.parquet.

Outputs
-------
- Per-version length quantiles for the `tokens` column (untruncated list length).
- For candidate max lengths L, truncation rates and retained-token fractions under the
  padding/truncation rule used by fms-ehrs Tokenizer21 (keep first L-1 tokens and append TRUNC).
- Optional fused↔unfused paired summaries if version names follow the run_experiments.py naming.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pyarrow as pa
import pyarrow.parquet as pq


@dataclass(frozen=True)
class LengthStats:
    n: int
    mean: float
    q: dict[int, float]
    max_len: int


def _normalize_version_dir(base: Path, version: str) -> Path:
    """
    Accept either:
    - version directory name (e.g., 'deciles_none_unfused_time_tokens_first_24h-tokenized')
    - data_version base (e.g., 'deciles_none_unfused_time_tokens_first_24h')
    and return the directory that contains train/val/test.
    """
    v = version
    cand = base / v
    if cand.exists():
        return cand
    if not v.endswith("-tokenized"):
        cand = base / f"{v}-tokenized"
        if cand.exists():
            return cand
    raise FileNotFoundError(f"Could not resolve version directory under base={base}: version={version}")


def _iter_token_lengths(parquet_path: Path, *, batch_size: int = 8192) -> Iterable[int]:
    """
    Stream token lengths from a Parquet file without materializing the full column.

    Assumes the file has a list-typed column named 'tokens' (as written by fms-ehrs tokenization).
    """
    pf = pq.ParquetFile(parquet_path)
    for batch in pf.iter_batches(columns=["tokens"], batch_size=batch_size):
        arr: pa.Array = batch.column(0)
        # list-lengths are stored efficiently; value_lengths() avoids round-tripping to Python lists.
        for x in arr.value_lengths().to_pylist():
            yield int(x)


def _quantile(sorted_vals: Sequence[int], q: float) -> float:
    """
    Linear interpolation quantile for sorted integer values.
    q in [0, 1].
    """
    if not sorted_vals:
        return float("nan")
    if q <= 0:
        return float(sorted_vals[0])
    if q >= 1:
        return float(sorted_vals[-1])
    k = (len(sorted_vals) - 1) * q
    f = math.floor(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return float(sorted_vals[f])
    return float(sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f))


def compute_length_stats(lengths: list[int], *, quantiles: Sequence[float]) -> LengthStats:
    lengths_sorted = sorted(int(x) for x in lengths)
    n = len(lengths_sorted)
    mean = float(sum(lengths_sorted) / n) if n else float("nan")
    q = {int(round(p * 100)): _quantile(lengths_sorted, p) for p in quantiles}
    max_len = int(lengths_sorted[-1]) if n else 0
    return LengthStats(n=n, mean=mean, q=q, max_len=max_len)


def truncation_curve(lengths: list[int], *, max_lens: Sequence[int]) -> dict[int, dict[str, float]]:
    """
    For each candidate max_padded_len L:
    - trunc_rate: P(seq_len > L)
    - retained_frac_mean: E[min(seq_len, L-1)/seq_len] with convention 0 if seq_len==0
      (Tokenizer21 appends TRUNC, so the number of *original* tokens retained is L-1 when truncated)
    """
    n = len(lengths)
    out: dict[int, dict[str, float]] = {}
    if n == 0:
        for L in max_lens:
            out[int(L)] = {"trunc_rate": float("nan"), "retained_frac_mean": float("nan")}
        return out

    for L in max_lens:
        L = int(L)
        if L <= 1:
            raise ValueError(f"max_padded_len must be >=2 (got {L})")
        trunc = 0
        retained_sum = 0.0
        for s in lengths:
            if s > L:
                trunc += 1
                retained_sum += (L - 1) / s
            elif s > 0:
                retained_sum += 1.0
        out[L] = {
            "trunc_rate": float(trunc / n),
            "retained_frac_mean": float(retained_sum / n),
        }
    return out


def _infer_fused_pair(version: str) -> str | None:
    """
    Best-effort pairing helper for run_experiments.py-style versions:
      <quantizer>_<anchoring>_<fused|unfused>_<temporal>...
    """
    if "_unfused_" in version:
        return version.replace("_unfused_", "_fused_", 1)
    if "_fused_" in version:
        return version.replace("_fused_", "_unfused_", 1)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base",
        type=Path,
        required=True,
        help="Base directory containing <version>-tokenized directories (e.g., $DATA_DIR).",
    )
    ap.add_argument(
        "--versions",
        type=str,
        nargs="+",
        required=True,
        help="One or more tokenized versions (dir name or data_version base).",
    )
    ap.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Which split to use for length distribution (default: train).",
    )
    ap.add_argument(
        "--max_lens",
        type=int,
        nargs="+",
        default=[2048, 4096, 8192, 16384],
        help="Candidate eval max_padded_len values to evaluate.",
    )
    ap.add_argument(
        "--quantiles",
        type=float,
        nargs="+",
        default=[0.5, 0.9, 0.95, 0.99, 0.999],
        help="Quantiles to report for seq_len distribution.",
    )
    ap.add_argument(
        "--pair_fused_unfused",
        action="store_true",
        help="If set, attempt to compute fused↔unfused paired deltas (based on version naming).",
    )
    ap.add_argument(
        "--json_out",
        type=Path,
        default=None,
        help="Optional path to write a JSON report.",
    )
    args = ap.parse_args()

    base = Path(args.base).expanduser().resolve()
    versions = list(args.versions)
    max_lens = [int(x) for x in args.max_lens]
    quantiles = [float(x) for x in args.quantiles]

    report: dict[str, object] = {
        "base": str(base),
        "split": args.split,
        "max_lens": max_lens,
        "quantiles": quantiles,
        "versions": {},
        "paired": {},
    }

    lengths_by_version: dict[str, list[int]] = {}
    for v in versions:
        vdir = _normalize_version_dir(base, v)
        parquet_path = vdir / args.split / "tokens_timelines.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(f"Missing tokens_timelines.parquet: {parquet_path}")
        lengths = list(_iter_token_lengths(parquet_path))
        lengths_by_version[v] = lengths
        ls = compute_length_stats(lengths, quantiles=quantiles)
        curve = truncation_curve(lengths, max_lens=max_lens)
        report["versions"][v] = {
            "version_dir": str(vdir),
            "n": ls.n,
            "mean": ls.mean,
            "max": ls.max_len,
            "quantiles": {str(k): v for k, v in ls.q.items()},
            "curve": {str(k): v for k, v in curve.items()},
        }

    if args.pair_fused_unfused:
        paired: dict[str, object] = {}
        for v in versions:
            other = _infer_fused_pair(v)
            if other is None or other not in versions:
                continue
            # Only compute in one direction to avoid duplicates.
            if v > other:
                continue
            a = lengths_by_version[v]
            b = lengths_by_version[other]
            paired_key = f"{v}__vs__{other}"
            paired[paired_key] = {
                "trunc_rate_delta": {
                    # delta = trunc_rate(v) - trunc_rate(other)
                    str(L): truncation_curve(a, max_lens=[L])[L]["trunc_rate"]
                    - truncation_curve(b, max_lens=[L])[L]["trunc_rate"]
                    for L in max_lens
                },
                "retained_frac_mean_delta": {
                    str(L): truncation_curve(a, max_lens=[L])[L]["retained_frac_mean"]
                    - truncation_curve(b, max_lens=[L])[L]["retained_frac_mean"]
                    for L in max_lens
                },
            }
        report["paired"] = paired

    # Human-readable summary
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.json_out is not None:
        outp = Path(args.json_out).expanduser().resolve()
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

