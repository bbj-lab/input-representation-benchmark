#!/usr/bin/env python3
"""
Compute token-length distribution statistics for tokenized timelines.
"""

from pathlib import Path
import argparse
import gzip
import pickle
import statistics
import pyarrow.parquet as pq


def percentile(sorted_vals, p):
    if not sorted_vals:
        return None
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return float(sorted_vals[f])
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return float(d0 + d1)


def stats_for(path: Path):
    pf = pq.ParquetFile(path)
    lengths = []
    for batch in pf.iter_batches(columns=["tokens"], batch_size=4096):
        arr = batch.column(0)
        lengths.extend(arr.value_lengths().to_pylist())
    lengths.sort()
    count = len(lengths)
    mean = statistics.fmean(lengths) if lengths else 0.0
    median = statistics.median(lengths) if lengths else 0.0
    p90 = percentile(lengths, 90)
    p95 = percentile(lengths, 95)
    p99 = percentile(lengths, 99)
    maxv = lengths[-1] if lengths else 0
    le_1024 = sum(1 for x in lengths if x <= 1024) / count if count else 0.0
    le_2048 = sum(1 for x in lengths if x <= 2048) / count if count else 0.0
    return {
        "count": count,
        "mean": mean,
        "median": median,
        "p90": p90,
        "p95": p95,
        "p99": p99,
        "max": maxv,
        "le_1024": le_1024,
        "le_2048": le_2048,
    }


def load_vocab_size(vocab_path: Path):
    if not vocab_path.exists():
        return None
    with gzip.open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    lookup = vocab.get("lookup")
    return len(lookup) if isinstance(lookup, dict) else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="Tokenized dataset base dir")
    parser.add_argument("--version", required=True, help="Tokenized version dir")
    args = parser.parse_args()

    base = Path(args.base)
    version = args.version
    train_path = base / version / "train" / "tokens_timelines.parquet"
    val_path = base / version / "val" / "tokens_timelines.parquet"
    test_path = base / version / "test" / "tokens_timelines.parquet"
    vocab_path = base / version / "train" / "vocab.gzip"

    print("version", version)
    print("vocab_size", load_vocab_size(vocab_path))
    for split, path in [("train", train_path), ("val", val_path), ("test", test_path)]:
        if not path.exists():
            print(split, "missing", path)
            continue
        print("computing", split)
        print(split, stats_for(path))


if __name__ == "__main__":
    main()
