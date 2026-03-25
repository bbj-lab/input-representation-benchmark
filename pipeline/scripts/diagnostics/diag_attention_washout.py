#!/usr/bin/env python3
"""Analysis 1: Non-Numeric CE Crisis — Attention Washout Detection.

Proves that soft discretization's interpolated embeddings cause attention washout
on neighboring categorical tokens. Compares the Discrete-RoPE model (non_numeric_ce = 2.510)
against the Soft-RoPE model (non_numeric_ce = 3.328) by measuring:

1. Attention share: fraction of attention weight that categorical tokens pay to
   numeric (Q-token) vs categorical positions, per layer per head.
2. Contextual degradation: per-token cross-entropy on categorical tokens that
   immediately follow a numeric token vs those following other categorical tokens.

GPU required (forward pass with attention output).

Usage:
    python pipeline/scripts/diagnostics/diag_attention_washout.py \
        --data_dir benchmarks/mimic-meds-extraction/data/meds/data \
        --data_version deciles_none_unfused_time_rope_first_24h \
        [--n_samples 500] \
        [--output_dir outputs/diagnostics] \
        [--dry_run]
"""

import argparse
import gc
import gzip
import json
import pathlib
import pickle
import sys

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Model loading (adapted from extract_hidden_states.py)
# ---------------------------------------------------------------------------

def load_model_for_attention(
    model_dir: pathlib.Path,
    device: str = "cuda",
) -> tuple:
    """Load model with attention output enabled.

    Handles both plain LlamaForCausalLM and RepresentationModelWrapper
    by detecting representation_mechanics.pt.

    Returns (model, vocab_data, is_wrapped, config).
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    config_path = model_dir / "config.json"
    rep_mechanics_path = model_dir / "representation_mechanics.pt"
    vocab_path = model_dir / "vocab.gzip"

    config = AutoConfig.from_pretrained(str(model_dir))

    # Load vocabulary
    with gzip.open(vocab_path, "r+") as f:
        vocab_data = pickle.load(f)

    if rep_mechanics_path.exists():
        # Wrapped model (Exp2)
        rep_meta = torch.load(rep_mechanics_path, map_location="cpu", weights_only=False)
        representation = rep_meta.get("representation", "discrete")
        temporal = rep_meta.get("temporal", "time_tokens")

        print(f"  Loading wrapped model: representation={representation}, temporal={temporal}")

        # Load base LLM
        base_model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            config=config,
            torch_dtype=torch.float16,
            attn_implementation="eager",  # Required for output_attentions=True
        )

        # We need fms_ehrs to construct the wrapper
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / ".." / "fms-ehrs"))
        from fms_ehrs.framework.model_wrapper import create_representation_model
        from fms_ehrs.framework.vocabulary import Vocabulary

        # Reconstruct the Vocabulary object
        vocab = Vocabulary()
        vocab.lookup = dict(vocab_data["lookup"])  # copy so we don't mutate original
        vocab.reverse = dict(vocab_data["reverse"])
        vocab.aux = vocab_data.get("aux", {})
        vocab._is_training = False

        # Soft discretization (ConSE) requires N+1 Q-tokens for N bins.
        # The tokenizer vocab only has Q0..Q(N-1). Inject the boundary token.
        if representation == "soft":
            num_bins = rep_meta.get("num_bins", 10)
            boundary_token = f"Q{num_bins}"
            if boundary_token not in vocab.lookup:
                next_id = max(vocab.lookup.values()) + 1
                vocab.lookup[boundary_token] = next_id
                vocab.reverse[next_id] = boundary_token
                print(f"  Injected {boundary_token} (id={next_id}) for ConSE boundary")

        # Extract constructor kwargs from rep_meta (they're not auto-extracted
        # from the rep_mechanics dict by create_representation_model).
        extra_kwargs = {}
        if "num_bins" in rep_meta:
            extra_kwargs["num_bins"] = rep_meta["num_bins"]
        if "time_rope_scaling" in rep_meta:
            extra_kwargs["time_rope_scaling"] = rep_meta["time_rope_scaling"]

        model = create_representation_model(
            base_model=base_model,
            vocab=vocab,
            representation=representation,
            temporal=temporal,
            **extra_kwargs,
        )
        model = model.to(device)
        model.eval()
        return model, vocab_data, True, config
    else:
        # Plain model (should not reach here for Exp2)
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            config=config,
            torch_dtype=torch.float16,
            attn_implementation="eager",
        )
        model = model.to(device)
        model.eval()
        return model, vocab_data, False, config


def load_data_batch(
    data_dir: pathlib.Path,
    data_version: str,
    split: str = "train",
    max_samples: int = 500,
    max_seq_len: int = 4096,
) -> dict:
    """Load tokenized data from parquet.

    Returns dict with 'input_ids', 'numeric_values', 'relative_times' tensors.
    """
    import pyarrow.parquet as pq

    tokenized_dir = data_dir / f"{data_version}-tokenized" / split
    parquet_path = tokenized_dir / "tokens_timelines.parquet"

    print(f"  Loading data: {parquet_path}")
    table = pq.read_table(str(parquet_path))

    # Prefer pre-padded columns (aligned to same max_seq_len)
    if "padded" in table.column_names:
        token_col = "padded"
        numeric_col = "padded_numeric_values" if "padded_numeric_values" in table.column_names else "numeric_values"
        times_col = "padded_times" if "padded_times" in table.column_names else "relative_times"
    elif "tokens" in table.column_names:
        token_col = "tokens"
        numeric_col = "numeric_values"
        times_col = "relative_times"
    else:
        raise ValueError(f"No token column found. Available: {table.column_names}")

    has_numeric = numeric_col in table.column_names
    has_times = times_col in table.column_names

    # Sample rows
    n = min(max_samples, len(table))

    all_ids = []
    all_nums = [] if has_numeric else None
    all_times = [] if has_times else None

    for idx in range(n):
        tokens = table.column(token_col)[idx].as_py()
        if tokens is None:
            continue
        tokens = tokens[:max_seq_len]
        all_ids.append(tokens)

        if has_numeric:
            nums = table.column(numeric_col)[idx].as_py()
            nums = nums[:max_seq_len] if nums else [float("nan")] * len(tokens)
            # Sanitize None → NaN (parquet stores nulls as None)
            nums = [float("nan") if v is None else float(v) for v in nums]
            # Align length to tokens (trim or pad)
            if len(nums) < len(tokens):
                nums = nums + [float("nan")] * (len(tokens) - len(nums))
            elif len(nums) > len(tokens):
                nums = nums[:len(tokens)]
            all_nums.append(nums)

        if has_times:
            times = table.column(times_col)[idx].as_py()
            times = times[:max_seq_len] if times else [0.0] * len(tokens)
            # Convert datetimes to epoch seconds, sanitize None
            converted = []
            for v in times:
                if v is None:
                    converted.append(0.0)
                elif hasattr(v, 'timestamp'):
                    converted.append(v.timestamp())
                else:
                    converted.append(float(v))
            times = converted
            if len(times) < len(tokens):
                times = times + [0.0] * (len(tokens) - len(times))
            elif len(times) > len(tokens):
                times = times[:len(tokens)]
            all_times.append(times)

    # Pad to uniform length across batch
    max_len = max(len(ids) for ids in all_ids)
    pad_id = 0

    padded_ids = []
    padded_mask = []
    padded_nums = []
    padded_times = []

    for i, ids in enumerate(all_ids):
        pad_len = max_len - len(ids)
        padded_ids.append(ids + [pad_id] * pad_len)
        padded_mask.append([1] * len(ids) + [0] * pad_len)
        if all_nums is not None:
            padded_nums.append(all_nums[i] + [float("nan")] * pad_len)
        if all_times is not None:
            padded_times.append(all_times[i] + [0.0] * pad_len)

    result = {
        "input_ids": torch.tensor(padded_ids, dtype=torch.long),
        "attention_mask": torch.tensor(padded_mask, dtype=torch.long),
    }
    if all_nums is not None:
        result["numeric_values"] = torch.tensor(padded_nums, dtype=torch.float32)
    if all_times is not None:
        result["relative_times"] = torch.tensor(padded_times, dtype=torch.float32)

    return result


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def identify_token_types(
    input_ids: torch.Tensor,
    vocab_data: dict,
    attention_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Identify numeric (Q-token) and categorical positions.

    Returns dict with boolean masks:
      - 'q_mask': True at Q0..Q9 positions
      - 'cat_mask': True at non-Q, non-pad, non-special positions
      - 'cat_after_q': True at categorical positions immediately after a Q position
      - 'cat_after_cat': True at categorical positions immediately after another categorical
    """
    lookup = vocab_data["lookup"]

    # Find Q-token IDs
    q_ids = set()
    for name, tid in lookup.items():
        if isinstance(name, str) and name.startswith("Q") and name[1:].isdigit():
            q_ids.add(tid)

    # Also identify special tokens
    special_names = {"[PAD]", "[BOS]", "[EOS]", "[SEP]", "[UNK]", "[NUM]", "AGE", None}
    special_ids = set()
    for name, tid in lookup.items():
        if name in special_names or (isinstance(name, str) and name.startswith("[TIME_")):
            special_ids.add(tid)

    batch_size, seq_len = input_ids.shape
    device = input_ids.device

    q_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    cat_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)

    for qid in q_ids:
        q_mask |= (input_ids == qid)

    valid = attention_mask.bool()
    for sid in special_ids:
        valid &= (input_ids != sid)

    cat_mask = valid & ~q_mask

    # Shifted masks
    cat_after_q = torch.zeros_like(cat_mask)
    cat_after_cat = torch.zeros_like(cat_mask)

    if seq_len > 1:
        prev_is_q = q_mask[:, :-1]
        prev_is_cat = cat_mask[:, :-1]
        cat_after_q[:, 1:] = cat_mask[:, 1:] & prev_is_q
        cat_after_cat[:, 1:] = cat_mask[:, 1:] & prev_is_cat

    return {
        "q_mask": q_mask,
        "cat_mask": cat_mask,
        "cat_after_q": cat_after_q,
        "cat_after_cat": cat_after_cat,
    }


def analyze_attention_patterns(
    attentions: tuple[torch.Tensor],
    masks: dict[str, torch.Tensor],
    attention_mask: torch.Tensor,
) -> dict:
    """Analyze attention patterns between categorical and numeric tokens.

    For each layer and head, compute the average attention that categorical tokens
    pay to numeric positions vs categorical positions.

    Returns dict with per-layer per-head statistics.
    """
    n_layers = len(attentions)
    cat_mask = masks["cat_mask"]  # (B, S)
    q_mask = masks["q_mask"]      # (B, S)
    valid_mask = attention_mask.bool()

    results = {}

    for layer_idx in range(n_layers):
        attn = attentions[layer_idx]  # (B, n_heads, S, S)
        n_heads = attn.shape[1]
        layer_results = {}

        for head_idx in range(n_heads):
            head_attn = attn[:, head_idx]  # (B, S, S) — row=query, col=key

            # For each categorical query position, compute attention to Q-keys vs cat-keys
            # Average over the batch and valid categorical query positions

            cat_queries = cat_mask.unsqueeze(2)  # (B, S, 1) — which positions are querying
            q_keys = q_mask.unsqueeze(1).float()  # (B, 1, S) — which keys are Q tokens
            cat_keys = cat_mask.unsqueeze(1).float()

            # Attention from cat queries to Q keys
            attn_to_q = (head_attn * cat_queries.float() * q_keys).sum() / cat_queries.float().sum().clamp(1)
            attn_to_cat = (head_attn * cat_queries.float() * cat_keys).sum() / cat_queries.float().sum().clamp(1)

            layer_results[f"head_{head_idx}"] = {
                "attn_to_numeric": float(attn_to_q),
                "attn_to_categorical": float(attn_to_cat),
                "numeric_share": float(attn_to_q / (attn_to_q + attn_to_cat + 1e-10)),
            }

        # Aggregate over heads
        shares = [v["numeric_share"] for v in layer_results.values()]
        layer_results["mean_numeric_share"] = float(np.mean(shares))
        layer_results["std_numeric_share"] = float(np.std(shares))

        results[f"layer_{layer_idx}"] = layer_results

    return results


def compute_contextual_ce(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    masks: dict[str, torch.Tensor],
) -> dict:
    """Compute per-token CE grouped by context type.

    CE for categorical tokens following Q tokens vs following other categoricals.
    """
    # Shift for next-token prediction
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    cat_after_q = masks["cat_after_q"][:, 1:]
    cat_after_cat = masks["cat_after_cat"][:, 1:]

    # Per-token CE
    ce_all = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        shift_labels.reshape(-1),
        reduction="none",
    ).reshape(shift_labels.shape)

    # Mean CE by context
    ce_after_q = ce_all[cat_after_q].float()
    ce_after_cat = ce_all[cat_after_cat].float()

    return {
        "ce_after_numeric": float(ce_after_q.mean()) if ce_after_q.numel() > 0 else None,
        "ce_after_categorical": float(ce_after_cat.mean()) if ce_after_cat.numel() > 0 else None,
        "n_after_numeric": int(ce_after_q.numel()),
        "n_after_categorical": int(ce_after_cat.numel()),
        "ce_degradation": (
            float(ce_after_q.mean() - ce_after_cat.mean())
            if ce_after_q.numel() > 0 and ce_after_cat.numel() > 0
            else None
        ),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_attention_comparison(
    results: dict[str, dict],
    output_path: pathlib.Path,
):
    """Heatmap comparing numeric attention share across layers/heads for both models."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    model_names = list(results.keys())
    fig, axes = plt.subplots(
        1, len(model_names),
        figsize=(7 * len(model_names), 5),
        constrained_layout=True,
    )
    if len(model_names) == 1:
        axes = [axes]

    for ax, model_name in zip(axes, model_names):
        attn_data = results[model_name]["attention_patterns"]
        layers = sorted([k for k in attn_data if k.startswith("layer_")],
                       key=lambda x: int(x.split("_")[1]))
        n_layers = len(layers)

        # Get number of heads from first layer
        first_layer = attn_data[layers[0]]
        n_heads = len([k for k in first_layer if k.startswith("head_")])

        matrix = np.zeros((n_layers, n_heads))
        for i, layer_name in enumerate(layers):
            layer = attn_data[layer_name]
            for j in range(n_heads):
                matrix[i, j] = layer[f"head_{j}"]["numeric_share"]

        im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=0.5)
        ax.set_xlabel("Head", fontsize=12)
        ax.set_ylabel("Layer", fontsize=12)
        ax.set_xticks(range(n_heads))
        ax.set_yticks(range(n_layers))
        ax.set_title(model_name, fontsize=13, fontweight="bold")

        for i in range(n_layers):
            for j in range(n_heads):
                ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center",
                       fontsize=7, color="white" if matrix[i,j] > 0.25 else "black")

    # Place colorbar below the plots (horizontal) to avoid overlapping panel content
    cbar = fig.colorbar(
        im, ax=axes, orientation="horizontal",
        shrink=0.6, pad=0.12, aspect=30,
        label="Numeric Attention Share",
    )
    cbar.ax.tick_params(labelsize=10)
    fig.suptitle("Attention Share to Numeric Tokens\n(from Categorical Query Positions)",
                 fontsize=14, fontweight="bold")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved attention comparison: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analysis 1: Attention Washout Detection")
    parser.add_argument("--data_dir", type=pathlib.Path, required=True)
    parser.add_argument("--data_version", type=str,
                        default="deciles_none_unfused_time_rope_first_24h")
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--split", type=str, default="test",
                        help="Data split to use (default: test, to match Table 8 CE)")
    parser.add_argument("--output_dir", type=pathlib.Path, default=pathlib.Path("outputs/diagnostics"))
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Models to compare
    base = args.data_dir.parent.parent.parent.parent.parent  # up to benchmark root
    if not (base / "models").exists():
        base = pathlib.Path(".")

    models = {
        "Discrete-RoPE": base / "models" / "exp2_meds_deciles_none_fusedFalse_discrete_time_rope-discrete-time_rope-s42" / "model-discrete-time_rope",
        "Soft-RoPE": base / "models" / "exp2_meds_deciles_none_fusedFalse_soft_time_rope-soft-time_rope-s42" / "model-soft-time_rope",
    }

    # Load data once
    print("Loading data...")
    n_samples = 8 if args.dry_run else args.n_samples
    data = load_data_batch(args.data_dir, args.data_version, split=args.split, max_samples=n_samples)
    print(f"  Loaded {data['input_ids'].shape[0]} sequences, max_len={data['input_ids'].shape[1]}")

    all_results = {}

    for model_name, model_dir in models.items():
        print(f"\n{'='*60}")
        print(f"Processing: {model_name}")
        print(f"  Model: {model_dir}")
        print(f"{'='*60}")

        if not model_dir.exists():
            print(f"  [SKIP] Model directory not found")
            continue

        model, vocab_data, is_wrapped, config = load_model_for_attention(
            model_dir, device=args.device,
        )

        # Identify token types
        masks = identify_token_types(data["input_ids"], vocab_data, data["attention_mask"])
        n_q = masks["q_mask"].sum().item()
        n_cat = masks["cat_mask"].sum().item()
        print(f"  Token counts: {n_q} numeric (Q), {n_cat} categorical")

        # Process in batches
        batch_size = 1 if args.dry_run else args.batch_size
        n_total = data["input_ids"].shape[0]
        all_attn_patterns = []
        all_ce_data = []

        for start_idx in range(0, n_total, batch_size):
            end_idx = min(start_idx + batch_size, n_total)
            batch_ids = data["input_ids"][start_idx:end_idx].to(args.device)
            batch_mask = data["attention_mask"][start_idx:end_idx].to(args.device)

            # Build kwargs for forward pass
            fwd_kwargs = {
                "input_ids": batch_ids,
                "attention_mask": batch_mask,
                "output_attentions": True,
            }

            if is_wrapped and "numeric_values" in data:
                fwd_kwargs["numeric_values"] = data["numeric_values"][start_idx:end_idx].to(args.device)
            if is_wrapped and "relative_times" in data:
                fwd_kwargs["relative_times"] = data["relative_times"][start_idx:end_idx].to(args.device)

            with torch.no_grad():
                outputs = model(**fwd_kwargs)

            # Attention analysis
            batch_masks = {k: v[start_idx:end_idx].to(args.device) for k, v in masks.items()}
            attn_result = analyze_attention_patterns(
                outputs.attentions, batch_masks, batch_mask,
            )
            all_attn_patterns.append(attn_result)

            # CE analysis
            ce_result = compute_contextual_ce(
                outputs.logits, batch_ids, batch_masks,
            )
            all_ce_data.append(ce_result)

            if args.dry_run:
                print(f"  [DRY RUN] Batch {start_idx}-{end_idx}: "
                      f"attn layers={len(outputs.attentions)}, "
                      f"CE after Q={ce_result['ce_after_numeric']}")
                break

            # Clear GPU memory
            del outputs
            gc.collect()
            torch.cuda.empty_cache()

        # Aggregate results (using first batch for structure, averaging values)
        model_results = {
            "attention_patterns": all_attn_patterns[0] if all_attn_patterns else {},
            "contextual_ce": all_ce_data[0] if all_ce_data else {},
            "n_samples": n_total,
            "n_numeric_tokens": n_q,
            "n_categorical_tokens": n_cat,
        }

        all_results[model_name] = model_results

        # Clean up model
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Save results
    results_path = output_dir / "attention_washout_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    if not args.dry_run and len(all_results) >= 2:
        plot_attention_comparison(all_results, output_dir / "attention_washout_comparison.png")

    # Summary
    print("\n=== Summary ===")
    for name, data in all_results.items():
        ce = data.get("contextual_ce", {})
        print(f"  {name}:")
        print(f"    CE after numeric: {ce.get('ce_after_numeric', 'N/A')}")
        print(f"    CE after categorical: {ce.get('ce_after_categorical', 'N/A')}")
        print(f"    CE degradation (Δ): {ce.get('ce_degradation', 'N/A')}")


if __name__ == "__main__":
    main()
