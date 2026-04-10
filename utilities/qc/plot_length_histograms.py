#!/usr/bin/env python3
"""
Plot token-length distribution histograms for tokenized EHR timelines.

This script is designed for the benchmark decision: choosing
- Exp1 packed training `max_seq_length`
- Exp2/Exp3 padded (per-admission aligned) window length (formerly `max_padded_len`)

We intentionally plot *token list length* from the Parquet `tokens` column (untruncated),
because:
- quantization granularity does not change token count materially
- fused vs unfused does change token count
- 24h-cut vs full-timeline does change token count

Inputs
------
Each "version" should be a directory containing split subdirs (train/val/test) with:
  tokens_timelines.parquet
and typically follows naming:
  <data_version>-tokenized/
  <data_version>_first_24h-tokenized/

Example usage
-------------
python qc/plot_length_histograms.py \
  --base /path/to/MEDS_events_dir \
  --versions deciles_none_unfused_time_tokens deciles_none_fused_time_tokens \
            deciles_none_unfused_time_tokens_first_24h deciles_none_fused_time_tokens_first_24h \
  --split train \
  --out_pdf "methods/ML4H 2025 Proceedings Template/figures/length_histograms.pdf"
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq


def _pretty_version_label(version: str) -> str:
    """Human-readable label derived from a data-version string."""
    v = version
    if v.endswith("-tokenized"):
        v = v[: -len("-tokenized")]

    fused = "fused" if ("fused" in v and "unfused" not in v) else "unfused"
    cut = "first 24h" if "first_24h" in v else "full timeline"

    if "time_tokens" in v and "time_rope" not in v:
        temporal = "time-spacing tokens"
    elif "time_rope" in v:
        temporal = "no time-spacing tokens"
    else:
        temporal = None

    if temporal is not None:
        return f"{temporal} — {cut}"
    return f"{fused} — {cut}"


def _normalize_version_dir(base: Path, version: str) -> Path:
    cand = base / version
    if cand.exists():
        return cand
    if not version.endswith("-tokenized"):
        cand = base / f"{version}-tokenized"
        if cand.exists():
            return cand
    raise FileNotFoundError(f"Could not resolve version under base={base}: {version}")


def _iter_token_lengths(parquet_path: Path, *, batch_size: int = 8192) -> Iterable[int]:
    pf = pq.ParquetFile(parquet_path)
    for batch in pf.iter_batches(columns=["tokens"], batch_size=batch_size):
        arr = batch.column(0)
        for x in arr.value_lengths().to_pylist():
            yield int(x)


@dataclass(frozen=True)
class LengthSummary:
    n: int
    max_len: int
    frac_le_1024: float
    frac_le_2048: float
    frac_le_4096: float


def summarize_lengths(parquet_path: Path) -> LengthSummary:
    n = 0
    max_len = 0
    le_1024 = 0
    le_2048 = 0
    le_4096 = 0
    for L in _iter_token_lengths(parquet_path):
        n += 1
        if L > max_len:
            max_len = L
        if L <= 1024:
            le_1024 += 1
        if L <= 2048:
            le_2048 += 1
        if L <= 4096:
            le_4096 += 1
    denom = float(n) if n else 1.0
    return LengthSummary(
        n=n,
        max_len=max_len,
        frac_le_1024=le_1024 / denom,
        frac_le_2048=le_2048 / denom,
        frac_le_4096=le_4096 / denom,
    )


def histogram_counts(parquet_path: Path, bins: np.ndarray) -> np.ndarray:
    counts = np.zeros(len(bins) - 1, dtype=np.int64)
    for L in _iter_token_lengths(parquet_path):
        # Bin assignment: right-open intervals [bins[i], bins[i+1])
        j = int(np.searchsorted(bins, L, side="right") - 1)
        if 0 <= j < counts.size:
            counts[j] += 1
        elif j >= counts.size:
            # Put out-of-range values into the last bin
            counts[-1] += 1
    return counts


def _default_bins(max_len: int) -> np.ndarray:
    # Hand-chosen bins to highlight the decisions 1024/2048/4096 while still showing tails.
    anchors = [0, 64, 128, 256, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576, 32768]
    if max_len <= anchors[-1]:
        anchors.append(max_len + 1)
        return np.array(sorted(set(anchors)), dtype=np.int64)
    # Extend bins roughly doubling until we exceed max_len
    b = anchors[:]
    v = b[-1]
    while v < max_len:
        v = int(v * 1.5)
        b.append(v)
    b.append(max_len + 1)
    return np.array(sorted(set(b)), dtype=np.int64)


def _bins_for_publication(*, xmax: int, max_len: int) -> np.ndarray:
    """Bins chosen for interpretability under heavy right tails.

    We use fine-grained bins in [0, xmax] and a single overflow bin [xmax, max_len+1)
    so tail mass is aggregated (not hidden).
    """
    xmax = int(xmax)
    max_len = int(max_len)
    if xmax <= 0:
        raise ValueError(f"xmax must be positive, got {xmax}")
    if max_len < 0:
        raise ValueError(f"max_len must be non-negative, got {max_len}")

    step = 64
    edges = list(range(0, xmax + 1, step))
    if edges[-1] != xmax:
        edges.append(xmax)
    overflow_right = max(max_len + 1, xmax + 1)
    edges.append(overflow_right)
    return np.array(sorted(set(edges)), dtype=np.int64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True, help="Base directory containing tokenized versions.")
    ap.add_argument("--versions", type=str, nargs="+", required=True, help="Exactly 4 versions (unfused/fused × full/24h).")
    ap.add_argument("--labels", type=str, nargs="*", default=None, help="Optional explicit labels for each version panel (must match --versions count).")
    ap.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    ap.add_argument("--out_pdf", type=Path, required=True)
    ap.add_argument("--title", type=str, default="Token-length distributions (tokens column)")
    ap.add_argument(
        "--xmax",
        type=int,
        default=6000,
        help=(
            "Max x-axis value for plotting. Values above xmax are aggregated into an overflow bin "
            "to keep heavy tails interpretable."
        ),
    )
    args = ap.parse_args()

    base = args.base.expanduser().resolve()
    versions = list(args.versions)
    if len(versions) != 4:
        raise ValueError(f"--versions must provide exactly 4 entries, got {len(versions)}")
    explicit_labels = args.labels
    if explicit_labels is not None and len(explicit_labels) != len(versions):
        raise ValueError(f"--labels must match --versions count ({len(versions)}), got {len(explicit_labels)}")

    # Resolve parquet paths and summaries
    paths = []
    summaries = {}
    max_overall = 0
    for v in versions:
        vdir = _normalize_version_dir(base, v)
        p = vdir / args.split / "tokens_timelines.parquet"
        if not p.exists():
            raise FileNotFoundError(f"Missing tokens_timelines.parquet: {p}")
        paths.append((v, p))
        s = summarize_lengths(p)
        summaries[v] = s
        max_overall = max(max_overall, s.max_len)

    xmax = int(args.xmax)
    bins = _bins_for_publication(xmax=xmax, max_len=max_overall)

    # Plot 2x2 panel in the same order as versions given
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        pass

    fig, axes = plt.subplots(2, 2, figsize=(8.5, 7.5), constrained_layout=True)
    axes = axes.flatten()
    for panel_idx, (ax, (v, p)) in enumerate(zip(axes, paths)):
        counts = histogram_counts(p, bins=bins)
        # Normalize to probability mass
        prob = counts / max(summaries[v].n, 1)
        # Use bar with bin widths. The final bin is an overflow bin [xmax, ∞).
        widths = bins[1:] - bins[:-1]
        ax.bar(
            bins[:-1],
            prob,
            width=widths,
            align="edge",
            edgecolor="white",
            linewidth=0.5,
            color="#4C72B0",
            alpha=0.85,
        )

        panel_letter = "ABCD"[panel_idx] if panel_idx < 4 else ""
        label = explicit_labels[panel_idx] if explicit_labels is not None else _pretty_version_label(v)
        ax.set_title(f"{panel_letter}. {label}", fontsize=12, loc="left")
        ax.set_xlabel("Tokenized length (tokens per admission)")
        ax.set_ylabel("Probability mass")
        ax.grid(True, axis="y", alpha=0.25)

        # Overlay an empirical CDF (computed from exact histogram counts).
        cdf = np.cumsum(prob)
        ax2 = ax.twinx()
        ax2.step(bins[1:], cdf, where="post", color="#DD8452", linewidth=1.25, alpha=0.95)
        ax2.set_ylim(0.0, 1.0)
        ax2.set_yticks([0.0, 0.5, 1.0])
        ax2.set_ylabel("CDF", color="#DD8452")
        ax2.tick_params(axis="y", labelcolor="#DD8452")

        # Mark decision thresholds
        for thr in (1024, 2048, 4096):
            ax.axvline(thr, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.axvline(xmax, color="black", linewidth=0.9, linestyle=":", alpha=0.75)

        s = summaries[v]
        overflow_mass = float(prob[bins[:-1] >= xmax].sum())
        ax.text(
            0.98,
            0.98,
            "\n".join(
                [
                    f"n={s.n:,}",
                    f"P(L≤1024)={s.frac_le_1024:.3f}",
                    f"P(L≤2048)={s.frac_le_2048:.3f}",
                    f"P(L≤4096)={s.frac_le_4096:.6f}",
                    f"P(L>4096)={1.0 - s.frac_le_4096:.6f}",
                    f"P(L>{xmax})={overflow_mass:.6f}",
                    f"max={s.max_len:,}",
                ]
            ),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.75", alpha=0.92),
        )
        ax.set_xlim(0, xmax)

    fig.suptitle(args.title, fontsize=14)
    out_pdf = args.out_pdf.expanduser().resolve()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    print(f"Wrote {out_pdf}")


if __name__ == "__main__":
    main()

