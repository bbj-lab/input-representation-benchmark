#!/usr/bin/env python3
"""
Inspect shortest/longest tokenized timelines and summarize content.
"""

from pathlib import Path
import argparse
import gzip
import pickle
import math
import heapq
import pyarrow.parquet as pq


def load_vocab_reverse(vocab_path: Path):
    with gzip.open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    return vocab.get("reverse", {})


def top_k_extremes(path: Path, k: int):
    pf = pq.ParquetFile(path)
    small = []  # max-heap for smallest
    large = []  # min-heap for largest
    for batch in pf.iter_batches(columns=["hadm_id", "seq_len"], batch_size=8192):
        hadm = batch.column(0).to_pylist()
        seq = batch.column(1).to_pylist()
        for h, s in zip(hadm, seq):
            if s is None:
                continue
            if len(small) < k:
                heapq.heappush(small, (-s, h))
            elif s < -small[0][0]:
                heapq.heapreplace(small, (-s, h))
            if len(large) < k:
                heapq.heappush(large, (s, h))
            elif s > large[0][0]:
                heapq.heapreplace(large, (s, h))
    smallest = sorted([(-s, h) for s, h in small])
    largest = sorted(large, reverse=True)
    return smallest, largest


def summarize_timelines(path: Path, vocab_path: Path, hadm_ids):
    pf = pq.ParquetFile(path)
    rev = load_vocab_reverse(vocab_path)
    want = set(hadm_ids)
    results = {}
    for batch in pf.iter_batches(columns=["hadm_id", "tokens", "numeric_values", "seq_len"], batch_size=2048):
        hadm = batch.column(0).to_pylist()
        tokens = batch.column(1).to_pylist()
        nums = batch.column(2).to_pylist()
        seqlen = batch.column(3).to_pylist()
        for h, t, n, s in zip(hadm, tokens, nums, seqlen):
            if h in want:
                non_nan = 0
                if n is not None:
                    for v in n:
                        if v is not None and not math.isnan(v):
                            non_nan += 1
                tok_strings = [rev.get(tok, f"ID{tok}") for tok in t]
                results[h] = {
                    "seq_len": s,
                    "unique_tokens": len(set(t)),
                    "numeric_values_non_nan": non_nan,
                    "numeric_values_frac": (non_nan / s) if s else 0.0,
                    "first_20": tok_strings[:20],
                    "last_20": tok_strings[-20:],
                }
                if len(results) == len(want):
                    return results
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="Tokenized dataset base dir")
    parser.add_argument("--version", required=True, help="Tokenized version dir")
    parser.add_argument("--k", type=int, default=5, help="Number of extremes")
    args = parser.parse_args()

    base = Path(args.base)
    version = args.version
    train_path = base / version / "train" / "tokens_timelines.parquet"
    vocab_path = base / version / "train" / "vocab.gzip"

    smallest, largest = top_k_extremes(train_path, args.k)
    print("smallest", smallest)
    print("largest", largest)

    hadm_ids = [h for _, h in smallest] + [h for _, h in largest]
    summaries = summarize_timelines(train_path, vocab_path, hadm_ids)
    for h in sorted(summaries, key=lambda x: summaries[x]["seq_len"]):
        r = summaries[h]
        print("hadm_id", h)
        print("  seq_len", r["seq_len"])
        print("  unique_tokens", r["unique_tokens"])
        print("  numeric_values_non_nan", r["numeric_values_non_nan"])
        print("  numeric_values_frac", f"{r['numeric_values_frac']:.3f}")
        print("  first_20", r["first_20"])
        print("  last_20", r["last_20"])


if __name__ == "__main__":
    main()
