from __future__ import annotations

from datetime import datetime, timedelta
import gzip
import importlib.util
from pathlib import Path
import pickle
import sys

import numpy as np
import polars as pl
import pytest


def _load_plot_module():
    script_path = (
        Path(__file__).resolve().parents[3]
        / "utilities"
        / "qc"
        / "plot_length_histograms.py"
    )
    spec = importlib.util.spec_from_file_location("plot_length_histograms", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _write_vocab(vocab_path: Path, *, trunc_id: int = 99) -> None:
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(vocab_path, "wb") as f:
        pickle.dump({"lookup": {"PAD": 0, "TRUNC": trunc_id}, "reverse": {}, "aux": {}}, f)


def test_recovers_untruncated_24h_lengths_when_first_24h_parquet_is_trunc_capped(
    tmp_path: Path,
) -> None:
    mod = _load_plot_module()
    trunc_id = 99
    t0 = datetime(2026, 1, 1, 0, 0, 0)

    # Stay 1: 4,200 tokens inside first 24h plus 300 tokens after 24h -> 24h length = 4,200
    times_1 = [t0 + timedelta(seconds=10 * i) for i in range(4200)] + [
        t0 + timedelta(hours=25, seconds=i) for i in range(300)
    ]
    tokens_1 = [10] * 4500

    # Stay 2: 500 tokens all inside first 24h -> 24h length = 500
    times_2 = [t0 + timedelta(minutes=i) for i in range(500)]
    tokens_2 = [11] * 500

    full_dir = tmp_path / "deciles_none_unfused_time_rope-tokenized" / "train"
    full_dir.mkdir(parents=True)
    _write_vocab(full_dir / "vocab.gzip", trunc_id=trunc_id)
    pl.DataFrame(
        {
            "hadm_id": [1, 2],
            "tokens": [tokens_1, tokens_2],
            "times": [times_1, times_2],
        }
    ).write_parquet(full_dir / "tokens_timelines.parquet")

    # Truncated first-24h parquet (mimics truncate_only / Stage0E cap at 4,096 with TRUNC)
    cut_dir = tmp_path / "deciles_none_unfused_time_rope_first_24h-tokenized" / "train"
    cut_dir.mkdir(parents=True)
    _write_vocab(cut_dir / "vocab.gzip", trunc_id=trunc_id)
    tokens_1_capped = [10] * 4095 + [trunc_id]
    pl.DataFrame(
        {
            "hadm_id": [1, 2],
            "tokens": [tokens_1_capped, tokens_2],
        }
    ).write_parquet(cut_dir / "tokens_timelines.parquet")

    summary = mod.summarize_lengths(cut_dir / "tokens_timelines.parquet")
    assert summary.n == 2
    assert summary.max_len == 4200
    assert summary.frac_le_1024 == pytest.approx(0.5)
    assert summary.frac_le_4096 == pytest.approx(0.5)

    bins = np.array([0, 1024, 4096, 5000], dtype=np.int64)
    counts = mod.histogram_counts(cut_dir / "tokens_timelines.parquet", bins)
    assert counts.tolist() == [1, 0, 1]


def test_raises_if_trunc_capped_parquet_has_no_untruncated_pair(tmp_path: Path) -> None:
    mod = _load_plot_module()
    trunc_id = 99
    cut_dir = tmp_path / "orphan_first_24h-tokenized" / "train"
    cut_dir.mkdir(parents=True)
    _write_vocab(cut_dir / "vocab.gzip", trunc_id=trunc_id)
    pl.DataFrame({"hadm_id": [1], "tokens": [[10] * 4095 + [trunc_id]]}).write_parquet(
        cut_dir / "tokens_timelines.parquet"
    )

    with pytest.raises(ValueError, match="TRUNC-capped"):
        mod.summarize_lengths(cut_dir / "tokens_timelines.parquet")
