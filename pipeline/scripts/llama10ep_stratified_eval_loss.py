#!/usr/bin/env python3
"""Compute Llama10ep validation loss stratified by sequence/window metadata."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import sys
from collections import defaultdict

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM

FMS_ROOT = pathlib.Path("/gpfs/data/bbj-lab/users/daniel/fms-ehrs")
if str(FMS_ROOT) not in sys.path:
    sys.path.insert(0, str(FMS_ROOT))

from fms_ehrs.framework.dataset import compute_relative_times_hours  # noqa: E402
from fms_ehrs.framework.model_wrapper import create_representation_model  # noqa: E402
from fms_ehrs.framework.vocabulary import Vocabulary  # noqa: E402


def _load_representation_meta(model_loc: pathlib.Path) -> dict | None:
    rep_path = model_loc / "representation_mechanics.pt"
    if rep_path.exists():
        return torch.load(rep_path, map_location="cpu", weights_only=False)
    return None


def _infer_representation_from_path(model_loc: pathlib.Path) -> tuple[str, str]:
    parts = model_loc.name.split("-", 1)
    if len(parts) < 2:
        return "discrete", "time_tokens"
    remainder = parts[1]
    for rep in ("xval_affine", "soft", "xval", "discrete"):
        if remainder.startswith(rep):
            rest = remainder[len(rep):]
            return rep, rest[1:] if rest.startswith("-") else "time_tokens"
    return "discrete", "time_tokens"


def _window_starts(n_tokens: int, *, window_len: int, window_stride: int, cont_id: int | None, max_windows: int | None) -> list[int]:
    if n_tokens == 0:
        starts = [0]
    elif cont_id is not None and window_stride == window_len:
        starts = [0]
        if n_tokens > window_len:
            starts.extend(list(range(window_len, n_tokens, window_len - 1)))
    else:
        starts = list(range(0, n_tokens, window_stride))

    if max_windows is not None and len(starts) > int(max_windows):
        m = int(max_windows)
        if m == 1:
            starts = [starts[0]]
        else:
            last = len(starts) - 1
            starts = [starts[(i * last) // (m - 1)] for i in range(m)]
    return starts


def _make_windows(
    *,
    hadm_id: int | None,
    tokens: list[int],
    times: list | None,
    pad_id: int,
    cont_id: int | None,
    window_len: int,
    window_stride: int,
    max_windows: int | None,
) -> list[dict]:
    n = len(tokens)
    starts = _window_starts(
        n,
        window_len=window_len,
        window_stride=window_stride,
        cont_id=cont_id,
        max_windows=max_windows,
    )

    t0 = None
    if times:
        for ts in times:
            if ts is not None:
                t0 = ts
                break

    rows = []
    n_windows = len(starts)
    for w, start in enumerate(starts):
        if n > 0 and start >= n:
            break
        use_cont = (w > 0) and (cont_id is not None)
        take = window_len - (1 if use_cont else 0)
        sl = tokens[start : start + take]
        win_tokens = ([cont_id] + sl) if use_cont else list(sl)
        raw_token_count = len(sl)
        if len(win_tokens) < window_len:
            win_tokens = win_tokens + [pad_id] * (window_len - len(win_tokens))
        else:
            win_tokens = win_tokens[:window_len]

        rel_times = None
        if times is not None:
            slt = times[start : start + take]
            win_times = ([None] + list(slt)) if use_cont else list(slt)
            if len(win_times) < window_len:
                win_times = win_times + [None] * (window_len - len(win_times))
            else:
                win_times = win_times[:window_len]
            rel_times = compute_relative_times_hours(win_times, t0=t0)

        if n_windows == 1:
            pos_class = "single"
        elif w == 0:
            pos_class = "first"
        elif w == n_windows - 1:
            pos_class = "last"
        else:
            pos_class = "middle"

        rows.append(
            {
                "hadm_id": hadm_id,
                "seq_len": n,
                "window_index": w,
                "n_windows": n_windows,
                "window_start": start,
                "window_position_class": pos_class,
                "raw_token_count": raw_token_count,
                "input_ids": win_tokens,
                "relative_times": rel_times,
            }
        )
        if n > 0 and (start + take) >= n:
            break
    return rows


def _seq_len_bin(seq_len: int) -> str:
    if seq_len <= 512:
        return "<=512"
    if seq_len <= 1024:
        return "513-1024"
    if seq_len <= 2048:
        return "1025-2048"
    if seq_len <= 4096:
        return "2049-4096"
    return ">4096"


def _load_model(model_loc: pathlib.Path, vocab: Vocabulary, device: torch.device):
    rep_meta = _load_representation_meta(model_loc)
    if rep_meta is not None:
        representation = rep_meta["representation"]
        temporal = rep_meta["temporal"]
    else:
        representation, temporal = _infer_representation_from_path(model_loc)

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(model_loc, torch_dtype=dtype)
    needs_wrapper = not (representation == "discrete" and temporal == "time_tokens")
    if needs_wrapper:
        kwargs = {}
        if rep_meta is not None:
            kwargs["num_bins"] = rep_meta.get("num_bins", 10)
            kwargs["time_rope_scaling"] = rep_meta.get("time_rope_scaling", 60.0)
        model = create_representation_model(
            base_model=base_model,
            vocab=vocab,
            representation=representation,
            temporal=temporal,
            **kwargs,
        )
    else:
        model = base_model
    return model.to(device).eval(), representation, temporal


def _aggregate(rows: list[dict], keys: list[str]) -> list[dict]:
    agg = defaultdict(lambda: {"loss_sum": 0.0, "token_count": 0, "window_count": 0})
    for row in rows:
        k = tuple(row[key] for key in keys)
        agg[k]["loss_sum"] += row["loss_sum"]
        agg[k]["token_count"] += row["token_count"]
        agg[k]["window_count"] += 1
    out = []
    for k, vals in sorted(agg.items()):
        item = {key: val for key, val in zip(keys, k)}
        item.update(vals)
        item["mean_loss"] = vals["loss_sum"] / max(vals["token_count"], 1)
        item["perplexity"] = math.exp(item["mean_loss"])
        out.append(item)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=pathlib.Path, required=True)
    parser.add_argument("--data-version", default="deciles_none_unfused_time_rope")
    parser.add_argument("--model-loc", type=pathlib.Path, required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--window-stride", type=int, default=4096)
    parser.add_argument("--max-windows-per-admission", type=int, default=128)
    parser.add_argument("--max-admissions", type=int, default=None)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    split_dir = args.data_dir / f"{args.data_version}-tokenized" / "val"
    parquet_path = split_dir / "tokens_timelines.parquet"
    vocab = Vocabulary().load(args.data_dir / f"{args.data_version}-tokenized" / "train" / "vocab.gzip")
    pad_id = int(vocab("PAD"))
    cont_id = int(vocab("TL_CONT")) if "TL_CONT" in vocab.lookup else None
    stop_ids = {pad_id, int(vocab("TRUNC")), int(vocab("TL_END"))}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)
    model, representation, temporal = _load_model(args.model_loc, vocab, device)

    per_window_rows = []
    batch_windows = []

    def flush_batch() -> None:
        if not batch_windows:
            return
        input_ids = torch.tensor([w["input_ids"] for w in batch_windows], dtype=torch.long, device=device)
        fwd = {"input_ids": input_ids}
        if temporal == "time_rope":
            fwd["relative_times"] = torch.tensor(
                [w["relative_times"] for w in batch_windows],
                dtype=torch.float32,
                device=device,
            )
        with torch.inference_mode():
            outputs = model.forward(**fwd)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
        shift_logits = logits[:, :-1, :].contiguous()
        targets = input_ids[:, 1:].contiguous()
        ce = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            targets.view(-1),
            reduction="none",
            ignore_index=pad_id,
        ).view(targets.shape)
        valid = targets != pad_id
        for stop_id in stop_ids:
            valid = valid & (targets != stop_id)

        for i, meta in enumerate(batch_windows):
            count = int(valid[i].sum().item())
            loss_sum = float((ce[i] * valid[i].float()).sum().item())
            mean_loss = loss_sum / max(count, 1)
            per_window_rows.append(
                {
                    "seed": args.seed,
                    "hadm_id": meta["hadm_id"],
                    "seq_len": meta["seq_len"],
                    "seq_len_bin": _seq_len_bin(meta["seq_len"]),
                    "window_index": meta["window_index"],
                    "n_windows": meta["n_windows"],
                    "window_start": meta["window_start"],
                    "window_position_class": meta["window_position_class"],
                    "raw_token_count": meta["raw_token_count"],
                    "token_count": count,
                    "loss_sum": loss_sum,
                    "mean_loss": mean_loss,
                    "perplexity": math.exp(mean_loss),
                    "model_loc": str(args.model_loc),
                    "representation": representation,
                    "temporal": temporal,
                }
            )
        batch_windows.clear()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    pf = pq.ParquetFile(parquet_path)
    seen = 0
    for record_batch in tqdm(
        pf.iter_batches(batch_size=128, columns=["hadm_id", "tokens", "times"]),
        desc=f"seed {args.seed} val windows",
    ):
        batch = record_batch.to_pydict()
        for hadm_id, tokens, times in zip(batch["hadm_id"], batch["tokens"], batch["times"]):
            if args.max_admissions is not None and seen >= args.max_admissions:
                break
            seen += 1
            windows = _make_windows(
                hadm_id=hadm_id,
                tokens=list(tokens) if tokens is not None else [],
                times=list(times) if times is not None else None,
                pad_id=pad_id,
                cont_id=cont_id,
                window_len=args.max_seq_length,
                window_stride=args.window_stride,
                max_windows=args.max_windows_per_admission,
            )
            for window in windows:
                batch_windows.append(window)
                if len(batch_windows) >= args.batch_size:
                    flush_batch()
        if args.max_admissions is not None and seen >= args.max_admissions:
            break
    flush_batch()

    exact_path = args.out_dir / f"llama10ep_s{args.seed}_stratified_eval_loss_per_window.csv"
    with exact_path.open("w", newline="") as f:
        fieldnames = list(per_window_rows[0].keys()) if per_window_rows else [
            "seed", "hadm_id", "seq_len", "seq_len_bin", "window_index", "n_windows",
            "window_start", "window_position_class", "raw_token_count", "token_count",
            "loss_sum", "mean_loss", "perplexity", "model_loc", "representation", "temporal",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_window_rows)

    for name, keys in {
        "by_seq_len_bin": ["seed", "seq_len_bin"],
        "by_window_position": ["seed", "window_position_class"],
        "by_window_index": ["seed", "window_index"],
        "by_seq_len_bin_and_window_position": ["seed", "seq_len_bin", "window_position_class"],
    }.items():
        rows = _aggregate(per_window_rows, keys)
        with (args.out_dir / f"llama10ep_s{args.seed}_stratified_eval_loss_{name}.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else keys)
            writer.writeheader()
            writer.writerows(rows)

    metadata = {
        "seed": args.seed,
        "model_loc": str(args.model_loc),
        "data_dir": str(args.data_dir),
        "data_version": args.data_version,
        "split": "val",
        "n_admissions_seen": seen,
        "n_windows": len(per_window_rows),
        "batch_size": args.batch_size,
        "max_seq_length": args.max_seq_length,
        "window_stride": args.window_stride,
        "max_windows_per_admission": args.max_windows_per_admission,
        "device": str(device),
        "representation": representation,
        "temporal": temporal,
    }
    (args.out_dir / f"llama10ep_s{args.seed}_stratified_eval_loss_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
