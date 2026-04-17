#!/usr/bin/env python3
"""Analysis 2: xVal Zero-Out Proof — Hidden State L2 Norms vs |z|.

Proves the xVal failure mechanism: when a numeric value equals the population median
for its code, the z-score z = (v - μ) / σ ≈ 0, which scales the embedding to near-zero,
effectively zeroing out the information at that position.

Method:
1. Forward pass through xVal model with hook functions at layers 0, 4, 7.
2. At each [NUM] token position, record:
   - The z-score computed by _normalize_by_code_id
   - The L2 norm of the hidden state
   - The scale factor applied to the embedding
3. Scatter plot: |z| vs ||h||_2 per layer. Expect a "trench" near z ≈ 0.

GPU required.

Usage:
    python pipeline/scripts/diagnostics/diag_xval_zero_out.py \
        --data_dir benchmarks/mimic-meds-extraction/data/meds/data \
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


# ---------------------------------------------------------------------------
# Model loading (xVal specific)
# ---------------------------------------------------------------------------

def load_xval_model(
    model_dir: pathlib.Path,
    device: str = "cuda",
):
    """Load xVal wrapped model.

    Returns (model, vocab_data, rep_meta).
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(str(model_dir))
    rep_mechanics_path = model_dir / "representation_mechanics.pt"
    vocab_path = model_dir / "vocab.gzip"

    with gzip.open(vocab_path, "r+") as f:
        vocab_data = pickle.load(f)

    rep_meta = torch.load(rep_mechanics_path, map_location="cpu", weights_only=False)

    print(f"  representation: {rep_meta.get('representation')}")
    print(f"  temporal: {rep_meta.get('temporal')}")

    base_model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), config=config, torch_dtype=torch.float16,
    )

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / ".." / "fms-ehrs"))
    from fms_ehrs.framework.model_wrapper import create_representation_model
    from fms_ehrs.framework.vocabulary import Vocabulary

    vocab = Vocabulary()
    vocab.lookup = vocab_data["lookup"]
    vocab.reverse = vocab_data["reverse"]
    vocab.aux = vocab_data.get("aux", {})
    vocab._is_training = False

    model = create_representation_model(
        base_model=base_model,
        vocab=vocab,
        representation="xval",
        temporal=rep_meta.get("temporal", "time_tokens"),
        rep_mechanics=rep_meta,
    )
    model = model.to(device)
    model.eval()
    return model, vocab_data, rep_meta


def load_data_batch(
    data_dir: pathlib.Path,
    data_version: str,
    split: str = "train",
    max_samples: int = 500,
    max_seq_len: int = 4096,
) -> dict:
    """Load tokenized data with numeric values from parquet."""
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
    else:
        token_col = "tokens"
        numeric_col = "numeric_values"
        times_col = "relative_times"

    has_numeric = numeric_col in table.column_names
    has_times = times_col in table.column_names

    if not has_numeric:
        raise ValueError("xVal analysis requires numeric_values in data")

    n = min(max_samples, len(table))

    all_ids, all_nums, all_times = [], [], []
    for idx in range(n):
        tokens = table.column(token_col)[idx].as_py()
        if tokens is None:
            continue
        tokens = tokens[:max_seq_len]
        all_ids.append(tokens)

        nums = table.column(numeric_col)[idx].as_py()
        nums = nums[:max_seq_len] if nums else [float("nan")] * len(tokens)
        # Sanitize None → NaN (parquet stores nulls as None)
        nums = [float("nan") if v is None else float(v) for v in nums]
        # Align length to tokens
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

    max_len = max(len(ids) for ids in all_ids)

    padded_ids, padded_mask, padded_nums, padded_times = [], [], [], []
    for i, ids in enumerate(all_ids):
        pad_len = max_len - len(ids)
        padded_ids.append(ids + [0] * pad_len)
        padded_mask.append([1] * len(ids) + [0] * pad_len)
        padded_nums.append(all_nums[i] + [float("nan")] * pad_len)
        if all_times:
            padded_times.append(all_times[i] + [0.0] * pad_len)

    result = {
        "input_ids": torch.tensor(padded_ids, dtype=torch.long),
        "attention_mask": torch.tensor(padded_mask, dtype=torch.long),
        "numeric_values": torch.tensor(padded_nums, dtype=torch.float32),
    }
    if padded_times:
        result["relative_times"] = torch.tensor(padded_times, dtype=torch.float32)

    return result


# ---------------------------------------------------------------------------
# Hook-based hidden state extraction
# ---------------------------------------------------------------------------

class HiddenStateHook:
    """Hook to capture hidden states at specified layers."""

    def __init__(self):
        self.hidden_states = {}
        self.handles = []

    def register(self, model, layer_indices: list[int]):
        """Register forward hooks on specified LLaMA decoder layers."""
        # Navigate to the decoder layers in the wrapped model
        base_model = model
        if hasattr(model, "base_model"):
            base_model = model.base_model
        if hasattr(base_model, "model"):
            base_model = base_model.model

        layers = base_model.layers

        for idx in layer_indices:
            if idx >= len(layers):
                print(f"  [WARN] Layer {idx} out of range (total: {len(layers)})")
                continue

            handle = layers[idx].register_forward_hook(
                self._make_hook(idx)
            )
            self.handles.append(handle)

    def _make_hook(self, layer_idx: int):
        def hook_fn(module, input, output):
            # LlamaDecoderLayer output is (hidden_states, ...) or a tuple
            if isinstance(output, tuple):
                self.hidden_states[layer_idx] = output[0].detach().cpu()
            else:
                self.hidden_states[layer_idx] = output.detach().cpu()
        return hook_fn

    def clear(self):
        self.hidden_states = {}

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def compute_z_scores(
    model,
    input_ids: torch.Tensor,
    numeric_values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute z-scores and scale factors for [NUM] positions.

    Replicates the logic from XValModelWrapper.forward() to extract
    the exact z-scores and scale factors used during inference.

    Returns (z_scores, scale_factors, num_mask) — all on CPU.
    """
    device = input_ids.device

    # Navigate to xVal wrapper
    xval = model
    if hasattr(model, "xval_wrapper"):
        xval = model.xval_wrapper

    num_token_id = xval.num_token_id
    num_mask = (input_ids == num_token_id) & (~torch.isnan(numeric_values))
    if num_mask.size(1) > 0:
        num_mask[:, 0] = False

    # Code IDs (previous token)
    code_ids = torch.zeros_like(input_ids)
    code_ids[:, 1:] = input_ids[:, :-1]

    # Per-code stats availability
    has_stats = xval.has_stats_by_id.to(device=device)[code_ids]

    # Z-score normalization
    values_clean = numeric_values.nan_to_num(0.0)
    code_ids_long = code_ids.to(device=device, dtype=torch.long)
    means = xval.means_by_id.to(device)[code_ids_long]
    stds = xval.stds_by_id.to(device)[code_ids_long]

    z = (values_clean - means) / stds
    z = torch.clamp(z, -xval.clip_sigma, xval.clip_sigma)

    # Scale factors
    scale = torch.ones_like(z)
    active = num_mask & has_stats
    scale = torch.where(active, z, scale)

    return z.cpu(), scale.cpu(), num_mask.cpu()


def collect_norms_vs_z(
    hidden_states: dict[int, torch.Tensor],
    z_scores: torch.Tensor,
    scale_factors: torch.Tensor,
    num_mask: torch.Tensor,
) -> dict[int, dict]:
    """Collect L2 norms of hidden states at [NUM] positions along with z-scores.

    Returns dict mapping layer_idx -> {z_values, norms, scale_values}.
    """
    results = {}
    flat_mask = num_mask.cpu()

    for layer_idx, hs in hidden_states.items():
        # hs shape: (B, S, D)
        norms = torch.norm(hs.float(), dim=-1)  # (B, S)

        z_at_num = z_scores[flat_mask].numpy()
        norms_at_num = norms[flat_mask].numpy()
        scale_at_num = scale_factors[flat_mask].numpy()

        results[layer_idx] = {
            "z_values": z_at_num.tolist(),
            "norms": norms_at_num.tolist(),
            "scale_values": scale_at_num.tolist(),
        }

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_z_vs_norm(
    layer_data: dict[int, dict],
    model_name: str,
    output_path: pathlib.Path,
):
    """Scatter plot of |z| vs ||h||_2 per layer."""
    import matplotlib.pyplot as plt

    layer_items = sorted(layer_data.items())
    n_layers = len(layer_items)
    share_y = n_layers <= 2

    # Figure 9 discusses layers 0 and 7; when plotting two layers, use a taller,
    # lower-density layout so the axis labels remain legible in the PDF.
    if n_layers <= 2:
        fig, axes = plt.subplots(1, n_layers, figsize=(12.0, 6.8), sharey=share_y)
    else:
        fig_w = max(12.0, 5.2 * float(n_layers))
        fig, axes = plt.subplots(1, n_layers, figsize=(fig_w, 6.6), sharey=False)

    if n_layers == 1:
        axes = [axes]

    for idx, (ax, (layer_idx, data)) in enumerate(zip(axes, layer_items)):
        z = np.abs(np.array(data["z_values"]))
        norms = np.array(data["norms"])

        ax.scatter(z, norms, s=8, alpha=0.15, color="steelblue", rasterized=True)

        # Red zone near z ≈ 0
        near_zero = z < 0.5
        if near_zero.sum() > 0:
            ax.scatter(z[near_zero], norms[near_zero], s=12, alpha=0.3,
                       color="red", rasterized=True, label=f"|z| < 0.5 (n={near_zero.sum()})")

        # Mean norm in bins
        bins = np.linspace(0, z.max() + 0.1, 20)
        bin_means = []
        bin_centers = []
        for i in range(len(bins) - 1):
            mask = (z >= bins[i]) & (z < bins[i + 1])
            if mask.sum() > 5:
                bin_means.append(norms[mask].mean())
                bin_centers.append((bins[i] + bins[i + 1]) / 2)
        if bin_centers:
            ax.plot(bin_centers, bin_means, "k-o", markersize=7, linewidth=2.4,
                    label="Binned mean", zorder=10)

        ax.set_xlabel("|z| (absolute z-score)", fontsize=20)
        if idx == 0 or not share_y:
            ax.set_ylabel("||h||₂ (hidden state L2 norm)", fontsize=20)
        else:
            ax.set_ylabel("")
        ax.set_title(f"Layer {layer_idx}", fontsize=21, fontweight="bold")
        ax.tick_params(axis="both", labelsize=17)
        if idx == 0 or n_layers == 1:
            ax.legend(fontsize=16, loc="best")
        ax.grid(alpha=0.3)

    fig.tight_layout(pad=1.0)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved zero-out plot: {output_path}")


def plot_scale_distribution(
    all_scales: np.ndarray,
    all_z: np.ndarray,
    model_name: str,
    output_path: pathlib.Path,
):
    """Histogram of scale factors applied to [NUM] embeddings."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scale factor distribution
    ax = axes[0]
    ax.hist(all_scales, bins=100, color="steelblue", alpha=0.7, edgecolor="gray")
    ax.axvline(x=0, color="red", linestyle="--", linewidth=1.5, label="Zero")
    near_zero_pct = 100.0 * (np.abs(all_scales) < 0.1).sum() / len(all_scales)
    ax.set_xlabel("Scale Factor (z-score)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Scale Factor Distribution\n{near_zero_pct:.1f}% within ±0.1 of zero",
                 fontsize=13, fontweight="bold")
    ax.legend()

    # z-score distribution
    ax = axes[1]
    ax.hist(all_z, bins=100, color="coral", alpha=0.7, edgecolor="gray")
    ax.axvline(x=0, color="red", linestyle="--", linewidth=1.5)
    ax.set_xlabel("z-score", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Z-Score Distribution at [NUM] Positions", fontsize=13, fontweight="bold")

    fig.suptitle(f"xVal Multiplicative Scaling: {model_name}",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved scale distribution: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analysis 2: xVal Zero-Out Proof")
    parser.add_argument("--data_dir", type=pathlib.Path, required=True)
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--hook_layers", type=int, nargs="+", default=[0, 4, 7],
                        help="Layer indices to hook for hidden state extraction")
    parser.add_argument("--split", type=str, default="test",
                        help="Data split to use (default: test, to prove inference-time failure)")
    parser.add_argument("--output_dir", type=pathlib.Path, default=pathlib.Path("outputs/diagnostics"))
    parser.add_argument(
        "--model_names",
        nargs="+",
        default=None,
        help="Optional subset of xVal config names to run (e.g. xVal-TimeTokens xVal-Affine-TimeTokens)",
    )
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # xVal model configurations
    base = args.data_dir.parent.parent.parent.parent.parent
    if not (base / "models").exists():
        base = pathlib.Path(".")

    xval_configs = [
        {
            "name": "xVal-TimeTokens",
            "model_dir": base / "models" / "exp2_meds_deciles_none_fusedFalse_xval_time_tokens-xval-time_tokens-s42" / "model-xval-time_tokens",
            "data_version": "deciles_none_unfused_time_tokens_numencxval_first_24h",
        },
        {
            "name": "xVal-RoPE",
            "model_dir": base / "models" / "exp2_meds_deciles_none_fusedFalse_xval_time_rope-xval-time_rope-s42" / "model-xval-time_rope",
            "data_version": "deciles_none_unfused_time_rope_numencxval_first_24h",
        },
        {
            "name": "xVal-Affine-TimeTokens",
            "model_dir": base / "models" / "exp2_meds_deciles_none_fusedFalse_xval_affine_time_tokens-xval_affine-time_tokens-s42" / "model-xval_affine-time_tokens",
            "data_version": "deciles_none_unfused_time_tokens_numencxval_first_24h",
        },
        {
            "name": "xVal-Affine-RoPE",
            "model_dir": base / "models" / "exp2_meds_deciles_none_fusedFalse_xval_affine_time_rope-xval_affine-time_rope-s42" / "model-xval_affine-time_rope",
            "data_version": "deciles_none_unfused_time_rope_numencxval_first_24h",
        },
    ]

    if args.model_names is not None:
        requested = set(args.model_names)
        xval_configs = [cfg for cfg in xval_configs if cfg["name"] in requested]
        if not xval_configs:
            raise ValueError(f"No xVal configs matched --model_names={sorted(requested)}")

    all_results = {}

    for cfg in xval_configs:
        model_name = cfg["name"]
        model_dir = cfg["model_dir"]

        print(f"\n{'='*60}")
        print(f"Processing: {model_name}")
        print(f"  Model: {model_dir}")
        print(f"{'='*60}")

        if not model_dir.exists():
            print(f"  [SKIP] Model directory not found")
            continue

        # Load data
        n_samples = 8 if args.dry_run else args.n_samples
        data = load_data_batch(args.data_dir, cfg["data_version"], split=args.split, max_samples=n_samples)
        print(f"  Loaded {data['input_ids'].shape[0]} sequences")

        # Load model
        model, vocab_data, rep_meta = load_xval_model(model_dir, device=args.device)

        # Set up hooks
        hook = HiddenStateHook()
        hook.register(model, args.hook_layers)

        # Aggregate results
        agg_layer_data = {l: {"z_values": [], "norms": [], "scale_values": []}
                         for l in args.hook_layers}
        all_z_scores = []
        all_scale_factors = []

        batch_size = 1 if args.dry_run else args.batch_size
        n_total = data["input_ids"].shape[0]

        for start_idx in range(0, n_total, batch_size):
            end_idx = min(start_idx + batch_size, n_total)
            batch_ids = data["input_ids"][start_idx:end_idx].to(args.device)
            batch_mask = data["attention_mask"][start_idx:end_idx].to(args.device)
            batch_nums = data["numeric_values"][start_idx:end_idx].to(args.device)

            fwd_kwargs = {
                "input_ids": batch_ids,
                "attention_mask": batch_mask,
                "numeric_values": batch_nums,
            }
            if "relative_times" in data:
                fwd_kwargs["relative_times"] = data["relative_times"][start_idx:end_idx].to(args.device)

            # Compute z-scores before forward pass (replicate xVal logic)
            z_scores, scale_factors, num_mask = compute_z_scores(
                model, batch_ids, batch_nums,
            )

            hook.clear()
            with torch.no_grad():
                _ = model(**fwd_kwargs)

            # Collect norms
            layer_data = collect_norms_vs_z(
                hook.hidden_states, z_scores, scale_factors, num_mask,
            )

            for l, ld in layer_data.items():
                agg_layer_data[l]["z_values"].extend(ld["z_values"])
                agg_layer_data[l]["norms"].extend(ld["norms"])
                agg_layer_data[l]["scale_values"].extend(ld["scale_values"])

            all_z_scores.extend(z_scores[num_mask].numpy().tolist())
            all_scale_factors.extend(scale_factors[num_mask].numpy().tolist())

            if args.dry_run:
                n_num = num_mask.sum().item()
                print(f"  [DRY RUN] Batch {start_idx}-{end_idx}: "
                      f"{n_num} [NUM] positions, "
                      f"hook layers captured: {list(hook.hidden_states.keys())}")
                for l in sorted(hook.hidden_states.keys()):
                    ld = layer_data[l]
                    if ld["norms"]:
                        print(f"    Layer {l}: mean norm={np.mean(ld['norms']):.4f}, "
                              f"z range=[{np.min(ld['z_values']):.2f}, "
                              f"{np.max(ld['z_values']):.2f}]")
                break

            del _
            gc.collect()
            torch.cuda.empty_cache()

        hook.remove()

        model_results = {
            "n_num_positions": len(all_z_scores),
            "z_score_stats": {
                "mean": float(np.mean(all_z_scores)) if all_z_scores else None,
                "std": float(np.std(all_z_scores)) if all_z_scores else None,
                "pct_near_zero": float(100 * np.mean(np.abs(np.array(all_z_scores)) < 0.5))
                    if all_z_scores else None,
            },
        }

        # Per-layer summary
        for l in sorted(agg_layer_data.keys()):
            ld = agg_layer_data[l]
            if ld["norms"]:
                z_arr = np.abs(np.array(ld["z_values"]))
                n_arr = np.array(ld["norms"])
                near_zero = z_arr < 0.5
                model_results[f"layer_{l}"] = {
                    "mean_norm_all": float(n_arr.mean()),
                    "mean_norm_near_zero": float(n_arr[near_zero].mean()) if near_zero.sum() > 0 else None,
                    "mean_norm_far_zero": float(n_arr[~near_zero].mean()) if (~near_zero).sum() > 0 else None,
                    "norm_ratio": float(n_arr[near_zero].mean() / n_arr[~near_zero].mean())
                        if near_zero.sum() > 0 and (~near_zero).sum() > 0 else None,
                }

        all_results[model_name] = model_results

        if not args.dry_run:
            safe_name = model_name.lower().replace("-", "_")
            plot_z_vs_norm(
                agg_layer_data, model_name,
                output_dir / f"xval_zero_out_{safe_name}.png",
            )
            plot_scale_distribution(
                np.array(all_scale_factors), np.array(all_z_scores),
                model_name, output_dir / f"xval_scale_dist_{safe_name}.png",
            )

        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Save results
    results_path = output_dir / "xval_zero_out_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Summary
    print("\n=== Summary ===")
    for name, res in all_results.items():
        print(f"  {name}:")
        print(f"    [NUM] positions: {res['n_num_positions']}")
        z_stats = res.get("z_score_stats", {})
        print(f"    z-score mean: {z_stats.get('mean', 'N/A')}")
        print(f"    % near zero (|z|<0.5): {z_stats.get('pct_near_zero', 'N/A')}")
        for l in sorted(k for k in res.keys() if k.startswith("layer_")):
            ld = res[l]
            print(f"    {l}: norm ratio (near_zero/far): {ld.get('norm_ratio', 'N/A')}")


if __name__ == "__main__":
    main()
