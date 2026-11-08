#!/usr/bin/env python3
"""Compare Exp3 native MIMIC and CLIF token volumes on the training split."""

from __future__ import annotations

import gzip
import json
import pickle
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path("/gpfs/data/bbj-lab/users/daniel/input-representation-benchmark")
ARMS = {
    "native_mimic": ROOT / "outputs/runs/exp3/meds_icu",
    "clif_2_1_0": ROOT / "outputs/runs/exp3/meds_clif",
}
SCOPES = {
    "first_24h_representation_extraction":
        "deciles_none_unfused_time_tokens_first_24h-tokenized",
    "full_hospitalization_pretraining":
        "deciles_none_unfused_time_tokens-tokenized",
}
MAX_SEQUENCE_LENGTH = 4096
MAX_WINDOWS_PER_ADMISSION = 128


def vocab_size(path: Path) -> int | None:
    vocab_path = path / "train" / "vocab.gzip"
    if not vocab_path.is_file():
        return None
    with gzip.open(vocab_path, "rb") as handle:
        vocab = pickle.load(handle)
    lookup = vocab.get("lookup")
    return len(lookup) if isinstance(lookup, dict) else None


def token_totals(
    parquet_path: Path,
) -> tuple[dict[str, object], list[int]]:
    table = pq.ParquetFile(parquet_path)
    n = 0
    total = 0
    lengths: list[int] = []
    for batch in table.iter_batches(columns=["tokens"], batch_size=4096):
        lens = batch.column(0).value_lengths().to_pylist()
        n += len(lens)
        total += int(sum(lens))
        lengths.extend(int(x) for x in lens)
    lengths.sort()
    median = float(lengths[len(lengths) // 2]) if lengths else 0.0
    if lengths and len(lengths) % 2 == 0:
        median = 0.5 * (lengths[len(lengths) // 2 - 1] + lengths[len(lengths) // 2])
    return (
        {
            "n_admissions": n,
            "total_tokens": total,
            "mean_tokens": (total / n) if n else 0.0,
            "median_tokens": median,
        },
        lengths,
    )


def extraction_input_totals(lengths: list[int]) -> dict[str, int]:
    return {
        "non_padding_tokens_after_4096_cap": sum(
            min(length, MAX_SEQUENCE_LENGTH) for length in lengths
        ),
        "admissions_truncated_at_4096": sum(
            length > MAX_SEQUENCE_LENGTH for length in lengths
        ),
        "padded_input_positions": len(lengths) * MAX_SEQUENCE_LENGTH,
    }


def pretraining_window_totals(
    lengths: list[int],
    *,
    total_tokens: int,
) -> dict[str, int]:
    windows = 0
    capped_admissions = 0
    raw_tokens_retained = 0

    for length in lengths:
        starts = [0]
        if length > MAX_SEQUENCE_LENGTH:
            starts.extend(
                range(
                    MAX_SEQUENCE_LENGTH,
                    length,
                    MAX_SEQUENCE_LENGTH - 1,
                )
            )
        if len(starts) > MAX_WINDOWS_PER_ADMISSION:
            capped_admissions += 1
            last = len(starts) - 1
            starts = [
                starts[
                    (i * last) // (MAX_WINDOWS_PER_ADMISSION - 1)
                ]
                for i in range(MAX_WINDOWS_PER_ADMISSION)
            ]

        for window_index, start in enumerate(starts):
            take = MAX_SEQUENCE_LENGTH - (window_index > 0)
            raw_tokens_retained += min(take, max(0, length - start))
        windows += len(starts)

    continuation_markers = windows - len(lengths)
    return {
        "windows_per_epoch": windows,
        "admissions_reaching_128_window_cap": capped_admissions,
        "raw_tokens_retained_per_epoch": raw_tokens_retained,
        "raw_tokens_omitted_by_window_cap_per_epoch":
            total_tokens - raw_tokens_retained,
        "continuation_markers_per_epoch": continuation_markers,
        "non_padding_tokens_per_epoch":
            raw_tokens_retained + continuation_markers,
        "padded_input_positions_per_epoch": windows * MAX_SEQUENCE_LENGTH,
        "padded_input_positions_over_five_epochs":
            windows * MAX_SEQUENCE_LENGTH * 5,
    }


def main() -> None:
    scopes: dict[str, object] = {}
    out: dict[str, object] = {
        "schema_version": 2,
        "split": "train",
        "counted_column": "tokens",
        "notes": [
            "Counts include serialized timeline and special tokens in the variable-length tokens column.",
            "Counts exclude padding and TL_CONT markers added later by the training window loader.",
            "Representation extraction uses one 4096-position padded or truncated input per admission.",
            "Pretraining splits full hospitalizations into contiguous 4096-position windows; later windows begin with TL_CONT.",
        ],
        "scopes": scopes,
    }
    for scope, suffix in SCOPES.items():
        records: dict[str, dict[str, object]] = {}
        for name, root in ARMS.items():
            base = root / suffix
            parquet = base / "train" / "tokens_timelines.parquet"
            rec, lengths = token_totals(parquet)
            if scope == "first_24h_representation_extraction":
                rec["model_input"] = extraction_input_totals(lengths)
            else:
                rec["model_input"] = pretraining_window_totals(
                    lengths,
                    total_tokens=int(rec["total_tokens"]),
                )
            rec["vocab_size"] = vocab_size(base)
            rec["path"] = str(parquet)
            records[name] = rec
        native = int(records["native_mimic"]["total_tokens"])
        clif = int(records["clif_2_1_0"]["total_tokens"])
        comparison: dict[str, object] = dict(records)
        comparison["clif_over_native"] = (clif / native) if native else None
        comparison["native_minus_clif"] = native - clif
        scopes[scope] = comparison

    dest = ROOT / "outputs/runs/ml4h_2026/metrics/exp3_train_token_totals.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
