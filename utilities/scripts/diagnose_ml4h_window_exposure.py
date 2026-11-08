#!/usr/bin/env python3
"""Quantify Exp2 window-cap exposure, non-pad retention, and TL_CONT positions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def required_windows(length: int, *, max_seq_length: int) -> int:
    if length <= max_seq_length:
        return 1
    remaining = length - max_seq_length
    # TL_CONT windows cover (L-1) raw tokens after the first window.
    return 1 + int(math.ceil(remaining / max(1, max_seq_length - 1)))


def _sample_indices(n: int, limit: int | None, seed: int) -> np.ndarray:
    if limit is None or limit >= n:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=limit, replace=False))


def diagnose_token_lengths(
    parquet_path: Path,
    *,
    max_seq_length: int,
    max_windows: int,
    sample_limit: int | None,
    seed: int,
) -> dict:
    table = pq.read_table(parquet_path, columns=["tokens"])
    n = table.num_rows
    idxs = _sample_indices(n, sample_limit, seed)
    tokens = table.column("tokens")
    lengths = np.asarray([len(tokens[int(i)].as_py()) for i in idxs], dtype=np.int64)
    req = np.asarray(
        [required_windows(int(v), max_seq_length=max_seq_length) for v in lengths],
        dtype=np.int64,
    )
    truncated = int((req > max_windows).sum())
    # Approximate retained non-pad fraction under the window cap: each kept
    # window contributes up to max_seq_length slots, but only raw tokens count.
    retained_tokens = []
    for length, windows in zip(lengths.tolist(), req.tolist()):
        kept = min(int(windows), max_windows)
        if kept <= 1:
            retained_tokens.append(min(length, max_seq_length))
        else:
            first = max_seq_length
            later = (kept - 1) * (max_seq_length - 1)
            retained_tokens.append(min(length, first + later))
    retained_arr = np.asarray(retained_tokens, dtype=np.float64)
    return {
        "parquet": str(parquet_path),
        "admissions_sampled": int(len(idxs)),
        "admissions_total": int(n),
        "token_length": {
            "mean": float(lengths.mean()),
            "p50": float(np.median(lengths)),
            "p95": float(np.percentile(lengths, 95)),
            "max": int(lengths.max()),
        },
        "required_windows": {
            "mean": float(req.mean()),
            "p50": float(np.median(req)),
            "p95": float(np.percentile(req, 95)),
            "max": int(req.max()),
        },
        "admissions_exceeding_window_cap": truncated,
        "fraction_exceeding_window_cap": truncated / len(idxs) if len(idxs) else None,
        "approx_raw_token_retention_under_cap": {
            "mean": float((retained_arr / np.maximum(lengths, 1)).mean()),
            "p50": float(np.median(retained_arr / np.maximum(lengths, 1))),
        },
        "max_windows_per_admission": max_windows,
        "max_seq_length": max_seq_length,
    }


def diagnose_times(
    parquet_path: Path,
    *,
    sample_limit: int | None,
    seed: int,
    seconds_per_position: float,
) -> dict:
    pf = pq.ParquetFile(parquet_path)
    names = set(pf.schema_arrow.names)
    time_col = None
    for candidate in ("times", "padded_times", "time"):
        if candidate in names:
            time_col = candidate
            break
    if time_col is None:
        return {"available": False, "reason": f"no time column in {sorted(names)}"}

    table = pq.read_table(parquet_path, columns=[time_col])
    n = table.num_rows
    idxs = _sample_indices(n, sample_limit, seed)
    col = table.column(time_col)
    cont_anchor_seconds: list[float] = []
    for i in idxs:
        times = col[int(i)].as_py()
        if not times or len(times) < 2:
            continue
        # Approximate first TL_CONT-prefixed window: after the first 4096 tokens
        # the next window starts near the time of token index 4095 when using
        # non-overlapping stride with CONT. Report late-timeline anchors.
        if len(times) <= 4096:
            continue
        t0 = times[0]
        t_anchor = times[4095]
        try:
            # times may be epoch ms ints or already-relative floats
            if isinstance(t0, (int, float)) and isinstance(t_anchor, (int, float)):
                if t_anchor > 1e11:  # epoch ms
                    rel = (float(t_anchor) - float(t0)) / 1000.0
                else:
                    rel = float(t_anchor) - float(t0)
                if rel >= 0:
                    cont_anchor_seconds.append(rel)
        except Exception:
            continue
    arr = np.asarray(cont_anchor_seconds, dtype=np.float64)
    return {
        "available": True,
        "time_column": time_col,
        "multi_window_admissions_sampled": int(arr.size),
        "tl_cont_anchor_seconds": {
            "mean": float(arr.mean()) if arr.size else None,
            "p50": float(np.median(arr)) if arr.size else None,
            "p95": float(np.percentile(arr, 95)) if arr.size else None,
            "max": float(arr.max()) if arr.size else None,
        },
        "tl_cont_anchor_positions_at_spp": {
            "seconds_per_position": seconds_per_position,
            "mean": float((arr / seconds_per_position).mean()) if arr.size else None,
            "p95": float(np.percentile(arr / seconds_per_position, 95)) if arr.size else None,
            "max": float((arr / seconds_per_position).max()) if arr.size else None,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("outputs/runs/tokenized/mimiciv-3.1_meds_70-10-20"),
    )
    parser.add_argument(
        "--data-version",
        default="deciles_none_unfused_time_rope",
    )
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--max-windows-per-admission", type=int, default=128)
    parser.add_argument("--seconds-per-position", type=float, default=30.0)
    parser.add_argument("--sample-limit", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/runs/ml4h_2026/calibration/window_exposure.json"),
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[2]
    data_dir = args.data_dir if args.data_dir.is_absolute() else repo / args.data_dir
    output = args.output if args.output.is_absolute() else repo / args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    parquet_path = (
        data_dir / f"{args.data_version}-tokenized" / "train" / "tokens_timelines.parquet"
    )
    if not parquet_path.is_file():
        raise SystemExit(f"Missing tokenized parquet: {parquet_path}")

    gate_errors: list[str] = []
    raw_stats = diagnose_token_lengths(
        parquet_path,
        max_seq_length=args.max_seq_length,
        max_windows=args.max_windows_per_admission,
        sample_limit=args.sample_limit,
        seed=args.seed,
    )
    time_stats = diagnose_times(
        parquet_path,
        sample_limit=args.sample_limit,
        seed=args.seed,
        seconds_per_position=args.seconds_per_position,
    )

    frac = raw_stats["fraction_exceeding_window_cap"]
    if frac is None:
        gate_errors.append("Failed to estimate window-cap exposure.")
    elif frac > 0.25:
        gate_errors.append(
            "More than 25% of sampled admissions exceed the 128-window cap; "
            "report this as a material exposure limit."
        )

    retention = raw_stats["approx_raw_token_retention_under_cap"]["mean"]
    if retention is None:
        gate_errors.append("Could not estimate retained token fraction under the window cap.")
    elif retention < 0.05:
        gate_errors.append(
            "Approximate retained raw-token fraction under the window cap is <5%."
        )

    report = {
        "data_dir": str(data_dir),
        "data_version": args.data_version,
        "raw_admission_length_stats": raw_stats,
        "tl_cont_time_stats": time_stats,
        "gate_errors": gate_errors,
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "gate_errors": gate_errors, "summary": {
        "fraction_exceeding_window_cap": frac,
        "approx_retention_mean": retention,
    }}, indent=2))
    return 0 if not gate_errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
