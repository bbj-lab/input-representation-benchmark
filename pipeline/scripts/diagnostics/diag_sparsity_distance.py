#!/usr/bin/env python3
"""Analysis 4: Vocabulary Sparsity Proof — Distance-from-Initialization vs Token Frequency.

Proves that fine-grained quantization (e.g., Centiles with ~84K tokens) fails because
most rare tokens remain "frozen" near random initialization due to insufficient training
signal. Compares the Fused Centiles model (large vocab) against the Fused Deciles model
(small vocab) to visually demonstrate parameter starvation.

CPU-only. Loads embedding matrices from safetensors checkpoints, counts token frequencies
from parquet data, and produces scatter plots of ||W_learned - W_init||_2 vs log-frequency.

Usage:
    python pipeline/scripts/diagnostics/diag_sparsity_distance.py \
        --model_centiles artifacts/runs/models/exp1_meds_centiles_none_fusedTrue_discrete_time_tokens-s42/run-0/checkpoint-9000 \
        --model_deciles artifacts/runs/models/exp1_meds_deciles_none_fusedTrue_discrete_time_tokens-s42/run-0/checkpoint-9000 \
        --data_dir benchmarks/mimic-meds-extraction/data/meds/data \
        [--output_dir outputs/diagnostics] \
        [--dry_run]
"""

import argparse
import json
import pathlib

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_embeddings_safetensors(model_dir: pathlib.Path) -> np.ndarray:
    """Load embedding matrix from safetensors checkpoint."""
    from safetensors import safe_open
    safetensors_path = model_dir / "model.safetensors"
    with safe_open(str(safetensors_path), framework="pt", device="cpu") as f:
        embed_weight = f.get_tensor("model.embed_tokens.weight")
    return embed_weight.float().numpy()


def load_config(model_dir: pathlib.Path) -> dict:
    """Load model config.json."""
    config_path = model_dir / "config.json"
    with open(config_path) as f:
        return json.load(f)


def compute_expected_init_norm(hidden_size: int, initializer_range: float = 0.02) -> float:
    """Expected L2 norm of a random vector from N(0, initializer_range) in d dimensions.

    E[||w||_2] = initializer_range * sqrt(d)
    """
    return initializer_range * np.sqrt(hidden_size)


def compute_init_embeddings(vocab_size: int, hidden_size: int, seed: int = 42) -> np.ndarray:
    """Generate random initialization embeddings matching HF LlamaForCausalLM init.

    HF initialization: nn.Embedding weights ~ N(0, initializer_range=0.02)
    """
    rng = np.random.default_rng(seed)
    return rng.normal(0, 0.02, size=(vocab_size, hidden_size)).astype(np.float32)


def count_token_frequencies(data_dir: pathlib.Path, data_version: str) -> np.ndarray | None:
    """Count token frequencies from the training parquet file.

    Returns array of shape (max_token_id + 1,) with counts per token.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("  [WARN] pyarrow not available, using fallback frequency estimation")
        return None

    parquet_path = data_dir / f"{data_version}-tokenized" / "train" / "tokens_timelines.parquet"
    if not parquet_path.exists():
        print(f"  [WARN] Parquet not found: {parquet_path}")
        return None

    print(f"  Counting tokens from: {parquet_path}")
    table = pq.read_table(str(parquet_path))

    # The parquet has either 'padded' or 'tokens' column
    if "padded" in table.column_names:
        col_name = "padded"
    elif "tokens" in table.column_names:
        col_name = "tokens"
    else:
        print(f"  [WARN] No token column found. Available: {table.column_names}")
        return None

    col = table.column(col_name)
    # Flatten all sequences and count
    from collections import Counter
    counts: Counter = Counter()
    for row in col:
        tokens = row.as_py()
        if tokens:
            counts.update(tokens)

    if not counts:
        return None

    max_id = max(counts.keys())
    freq = np.zeros(max_id + 1, dtype=np.int64)
    for tid, c in counts.items():
        freq[tid] = c

    return freq


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_sparsity_scatter(
    distances: np.ndarray,
    frequencies: np.ndarray,
    title: str,
    output_path: pathlib.Path,
    vocab_size: int,
    highlight_threshold: float | None = None,
):
    """Scatter plot of ||W_learned - W_init||_2 vs log10(frequency + 1)."""
    import matplotlib.pyplot as plt

    log_freq = np.log10(frequencies + 1)

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.scatter(log_freq, distances, s=3, alpha=0.15, color="steelblue", rasterized=True)

    # Mark the expected initialization norm
    if highlight_threshold is not None:
        ax.axhline(y=highlight_threshold, color="red", linestyle="--", linewidth=1.5,
                   label=f"Expected init norm (≈{highlight_threshold:.2f})")

    # Count "frozen" tokens (distance < 2 * init_norm)
    if highlight_threshold is not None:
        frozen_mask = distances < (1.5 * highlight_threshold)
        n_frozen = frozen_mask.sum()
        pct_frozen = 100.0 * n_frozen / len(distances)
        ax.text(0.02, 0.95, f"Frozen tokens (<1.5× init): {n_frozen}/{len(distances)} ({pct_frozen:.1f}%)",
                transform=ax.transAxes, fontsize=10, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax.set_xlabel("log₁₀(Token Frequency + 1)", fontsize=13)
    ax.set_ylabel("||W_learned - W_init||₂", fontsize=13)
    ax.set_title(f"{title}\n(vocab_size = {vocab_size:,})", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved scatter plot: {output_path}")


def plot_combined_comparison(
    results: dict,
    output_path: pathlib.Path,
):
    """Side-by-side scatter comparison of two models."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    for ax, (model_name, data) in zip(axes, results.items()):
        log_freq = np.log10(data["frequencies"] + 1)
        ax.scatter(log_freq, data["distances"], s=3, alpha=0.15,
                   color="steelblue", rasterized=True)

        init_norm = data.get("expected_init_norm", None)
        if init_norm:
            ax.axhline(y=init_norm, color="red", linestyle="--", linewidth=1.5,
                       label=f"Expected init norm (≈{init_norm:.2f})")

            frozen_mask = data["distances"] < (1.5 * init_norm)
            n_frozen = frozen_mask.sum()
            pct = 100.0 * n_frozen / len(data["distances"])
            ax.text(0.02, 0.95,
                    f"Frozen: {n_frozen}/{len(data['distances'])} ({pct:.1f}%)",
                    transform=ax.transAxes, fontsize=10, verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        ax.set_xlabel("log₁₀(Token Frequency + 1)", fontsize=12)
        ax.set_ylabel("||W_learned − W_init||₂", fontsize=12)
        ax.set_title(f"{model_name}\n(vocab = {data['vocab_size']:,})",
                     fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

    fig.suptitle("Vocabulary Sparsity: Embedding Distance from Initialization",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved comparison plot: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analysis 4: Vocabulary Sparsity Proof")
    parser.add_argument(
        "--model_centiles", type=pathlib.Path, required=True,
        help="Path to Fused Centiles model checkpoint directory",
    )
    parser.add_argument(
        "--model_deciles", type=pathlib.Path, required=True,
        help="Path to Fused Deciles model checkpoint directory (comparison)",
    )
    parser.add_argument(
        "--data_dir", type=pathlib.Path,
        default=pathlib.Path("benchmarks/mimic-meds-extraction/data/meds/data"),
        help="Path to MEDS data directory (for token frequency counts)",
    )
    parser.add_argument(
        "--output_dir", type=pathlib.Path, default=pathlib.Path("outputs/diagnostics"),
    )
    parser.add_argument("--dry_run", action="store_true")

    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    results_all = {}

    for model_name, model_dir, data_version in [
        ("Fused Centiles", args.model_centiles,
         "centiles_none_fused_time_tokens"),
        ("Fused Deciles", args.model_deciles,
         "deciles_none_fused_time_tokens"),
    ]:
        print(f"\n{'='*60}")
        print(f"Processing: {model_name}")
        print(f"  Model dir: {model_dir}")
        print(f"{'='*60}")

        # Load config
        config = load_config(model_dir)
        vocab_size = config["vocab_size"]
        hidden_size = config["hidden_size"]
        init_range = config.get("initializer_range", 0.02)
        print(f"  Vocab size: {vocab_size}, Hidden size: {hidden_size}")

        # Load learned embeddings
        print("  Loading learned embeddings...")
        learned = load_embeddings_safetensors(model_dir)
        print(f"  Learned embeddings shape: {learned.shape}")

        # Compute analytical expected init norm
        expected_norm = compute_expected_init_norm(hidden_size, init_range)
        print(f"  Expected init norm: {expected_norm:.4f}")

        if args.dry_run:
            # Lightweight dry run: use L2 norms from origin as proxy
            # (avoids allocating a second 84K×1024 matrix)
            norms = np.linalg.norm(learned[:vocab_size], axis=1)
            # Frozen tokens have norms close to expected init norm
            frozen_mask = norms < (1.5 * expected_norm)
            print(f"\n  [DRY RUN] Embedding norms: mean={norms.mean():.4f}, "
                  f"median={np.median(norms):.4f}")
            print(f"  [DRY RUN] Frozen tokens (norm < 1.5× init): "
                  f"{frozen_mask.sum()}/{len(norms)} "
                  f"({100*frozen_mask.sum()/len(norms):.1f}%)")
            del learned
            continue

        # Compute distances from random init (full run only)
        print("  Generating random initialization for comparison...")
        init_embs = compute_init_embeddings(vocab_size, hidden_size)
        distances = np.linalg.norm(learned[:vocab_size] - init_embs, axis=1)
        del init_embs  # free memory immediately
        print(f"  Distance stats: mean={distances.mean():.4f}, "
              f"median={np.median(distances):.4f}, "
              f"min={distances.min():.4f}, max={distances.max():.4f}")

        # Count token frequencies
        print("  Counting token frequencies...")
        frequencies = count_token_frequencies(args.data_dir, data_version)
        if frequencies is None:
            print("  [FALLBACK] Generating uniform dummy frequencies")
            frequencies = np.ones(vocab_size, dtype=np.int64)
        else:
            # Pad or truncate to match vocab size
            if len(frequencies) < vocab_size:
                frequencies = np.pad(frequencies, (0, vocab_size - len(frequencies)))
            frequencies = frequencies[:vocab_size]

        results_all[model_name] = {
            "distances": distances,
            "frequencies": frequencies,
            "vocab_size": vocab_size,
            "expected_init_norm": expected_norm,
        }


        # Individual scatter plot
        safe_name = model_name.lower().replace(" ", "_")
        plot_sparsity_scatter(
            distances, frequencies,
            model_name,
            output_dir / f"sparsity_{safe_name}.png",
            vocab_size,
            highlight_threshold=expected_norm,
        )

    if not args.dry_run and len(results_all) == 2:
        plot_combined_comparison(results_all, output_dir / "sparsity_comparison.png")

    # Save numerical summary
    summary = {}
    for name, data in results_all.items():
        d = data["distances"]
        f = data["frequencies"]
        init_norm = data["expected_init_norm"]
        frozen = (d < 1.5 * init_norm).sum()
        zero_freq = (f == 0).sum()
        summary[name] = {
            "vocab_size": int(data["vocab_size"]),
            "expected_init_norm": float(init_norm),
            "mean_distance": float(d.mean()),
            "median_distance": float(np.median(d)),
            "n_frozen_tokens": int(frozen),
            "pct_frozen": float(100 * frozen / len(d)),
            "n_zero_freq_tokens": int(zero_freq),
        }

    results_path = output_dir / "sparsity_distance_results.json"
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
