#!/usr/bin/env python3
"""One-batch integration checks for every Exp2 representation (no full-dataset map)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / ".." / "fms-ehrs"))
from fms_ehrs.framework.dataset import _windowed_padded_examples  # noqa: E402
from fms_ehrs.framework.model_wrapper import create_representation_model  # noqa: E402
from fms_ehrs.framework.vocabulary import Vocabulary  # noqa: E402
from fms_ehrs.scripts.train_representation import RepresentationDataCollator  # noqa: E402
from transformers import AutoModelForCausalLM, LlamaConfig  # noqa: E402


REPS = ("discrete", "soft", "xval")


def _data_version(representation: str) -> str:
    if representation.startswith("xval"):
        return "deciles_none_unfused_time_rope_numencxval"
    return "deciles_none_unfused_time_rope"


def _load_rows(data_dir: Path, data_version: str, *, n: int) -> tuple[Vocabulary, list[dict]]:
    split_dir = data_dir / f"{data_version}-tokenized" / "train"
    vocab = Vocabulary().load(split_dir / "vocab.gzip")
    parquet_path = split_dir / "tokens_timelines.parquet"
    pf = pq.ParquetFile(parquet_path)
    names = set(pf.schema_arrow.names)
    wanted = ["tokens"]
    for key in ("times", "padded_times", "numeric_values", "values"):
        if key in names:
            wanted.append(key)
    rows: list[dict] = []
    for batch in pf.iter_batches(batch_size=max(n, 8), columns=wanted):
        table = batch
        for i in range(min(n - len(rows), table.num_rows)):
            tokens = table.column("tokens")[i].as_py()
            times = None
            for key in ("times", "padded_times"):
                if key in table.column_names:
                    times = table.column(key)[i].as_py()
                    break
            numeric = None
            for key in ("numeric_values", "values"):
                if key in table.column_names:
                    numeric = table.column(key)[i].as_py()
                    break
            rows.append({"tokens": tokens, "times": times, "numeric_values": numeric})
        if len(rows) >= n:
            break
    return vocab, rows


def _windows_as_features(
    rows: list[dict],
    *,
    vocab: Vocabulary,
    max_seq_length: int,
    needs_numeric: bool,
    batch_size: int,
) -> list[dict]:
    pad_id = int(vocab("PAD"))
    cont_id = int(vocab("TL_CONT")) if "TL_CONT" in vocab.lookup else None
    features: list[dict] = []
    for row in rows:
        # Use a short window for the smoke forward while still exercising
        # the same windowing helpers used in training.
        out = _windowed_padded_examples(
            tokens=row["tokens"],
            times=row["times"],
            numeric_values=row["numeric_values"] if needs_numeric else None,
            window_len=min(128, max_seq_length),
            window_stride=min(128, max_seq_length),
            pad_id=pad_id,
            cont_id=cont_id,
            max_windows=2,
        )
        for j, ids in enumerate(out["input_ids"]):
            feat = {
                "input_ids": torch.tensor(ids, dtype=torch.long),
            }
            if needs_numeric and out.get("numeric_values") is not None:
                feat["numeric_values"] = torch.tensor(
                    out["numeric_values"][j], dtype=torch.float32
                )
            feat["relative_times_seconds"] = torch.tensor(
                out["relative_times_seconds"][j], dtype=torch.float32
            )
            features.append(feat)
            if len(features) >= batch_size:
                return features
    return features


def _build_tiny_model(vocab, representation: str):
    config = LlamaConfig(
        vocab_size=len(vocab),
        bos_token_id=vocab("TL_START"),
        eos_token_id=vocab("TL_END"),
        pad_token_id=vocab("PAD"),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=128,
        rope_theta=10000.0,
    )
    base = AutoModelForCausalLM.from_config(config)
    return create_representation_model(
        base_model=base,
        vocab=vocab,
        representation=representation,
        temporal="time_rope",
        num_bins=10,
        seconds_per_position=30.0,
        numeric_loss_weight=1.0,
    )


def check_representation(
    *,
    data_dir: Path,
    representation: str,
    batch_size: int,
    max_seq_length: int,
) -> dict:
    needs_numeric = representation in ("soft", "xval")
    vocab, rows = _load_rows(
        data_dir, _data_version(representation), n=max(4, batch_size)
    )
    features = _windows_as_features(
        rows,
        vocab=vocab,
        max_seq_length=max_seq_length,
        needs_numeric=needs_numeric,
        batch_size=batch_size,
    )
    if len(features) < 1:
        return {
            "representation": representation,
            "ok": False,
            "errors": ["no window features produced"],
        }

    collator = RepresentationDataCollator(
        pad_token_id=int(vocab("PAD")),
        include_numeric_values=needs_numeric,
        include_times=True,
    )
    batch = collator(features[:batch_size])
    errors: list[str] = []
    warnings: list[str] = []
    required = ["input_ids", "attention_mask", "labels", "relative_times_seconds"]
    if needs_numeric:
        required.append("numeric_values")
    for key in required:
        if key not in batch:
            errors.append(f"missing tensor {key}")
    if "input_ids" in batch and int(batch["input_ids"].shape[1]) > max_seq_length:
        errors.append("sequence length exceeds max_seq_length")
    if "relative_times_seconds" in batch:
        times = batch["relative_times_seconds"]
        # Training clamps negatives to 0 in the temporal encoder; record only.
        if (times < 0).any():
            warnings.append(
                "raw relative_times_seconds contain negatives; model path clamps to >=0"
            )
        if not torch.isfinite(times.nan_to_num(0.0)).all():
            errors.append("relative_times_seconds non-finite after nan_to_num")
    if needs_numeric and "numeric_values" in batch and torch.isinf(batch["numeric_values"]).any():
        errors.append("numeric_values contains inf")

    model = _build_tiny_model(vocab, representation)
    model.eval()
    with torch.no_grad():
        out = model(**batch)
    if isinstance(out, dict):
        loss = out.get("loss")
    else:
        loss = getattr(out, "loss", None)
    if loss is None or not torch.isfinite(loss):
        errors.append(f"forward loss is missing or non-finite ({loss})")

    loss_value = None
    if loss is not None and torch.is_tensor(loss) and torch.isfinite(loss):
        loss_value = float(loss.detach().cpu())

    return {
        "representation": representation,
        "batch_size": min(batch_size, len(features)),
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "shapes": {
            key: list(batch[key].shape)
            for key in batch
            if torch.is_tensor(batch[key])
        },
        "loss": loss_value,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("outputs/runs/tokenized/mimiciv-3.1_meds_70-10-20"),
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/runs/ml4h_2026/calibration/representation_batch_checks.json"
        ),
    )
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    data_dir = args.data_dir if args.data_dir.is_absolute() else repo / args.data_dir
    output = args.output if args.output.is_absolute() else repo / args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    results = []
    gate_errors: list[str] = []
    for representation in REPS:
        result = check_representation(
            data_dir=data_dir,
            representation=representation,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length,
        )
        results.append(result)
        if not result["ok"]:
            gate_errors.append(f"{representation}: " + "; ".join(result["errors"]))

    report = {
        "batch_size": args.batch_size,
        "results": results,
        "gate_errors": gate_errors,
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "gate_errors": gate_errors}, indent=2))
    return 0 if not gate_errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
