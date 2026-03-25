#!/usr/bin/env python3
"""Analysis 3: Embedding geometry of shared and fused numeric encodings.

Extracts static learned embeddings for shared quantile tokens and code-specific fused
quantile tokens. Computes pairwise cosine similarity matrices and 2D PCA projections
to visualize ordinal structure (or lack thereof) in the embedding space.

Hypotheses:
  - Shared unfused discrete quantile tokens may learn folded manifolds because the same bins
    must serve many numeric streams.
  - Fused discrete tokens should reveal cleaner per-code ordinal geometry because each
    code has its own Q-bin embeddings.

CPU-only. No forward pass needed — works directly from saved weights.

Usage:
    python pipeline/scripts/diagnostics/diag_embedding_geometry.py \
        --model_discrete artifacts/runs/models/exp2_.../model-discrete-time_tokens \
        --vocab_discrete artifacts/runs/tokenized/.../train/vocab.gzip \
        [--model_fused artifacts/runs/models/exp1_.../checkpoint-9000] \
        [--vocab_fused artifacts/runs/tokenized/.../train/vocab.gzip] \
        [--measurements LAB_lab//50971//meq/l LAB_lab//51222//g/dl] \
        [--output_dir outputs/diagnostics] \
        [--dry_run]
"""

import argparse
import gzip
import json
import pathlib
import pickle
import sys

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Target lab variables for RSA analysis (MIMIC-IV labevents itemids)
DEFAULT_MEASUREMENTS = {
    "Potassium": "LAB_lab//50971//meq/l",
    "Glucose": "LAB_lab//50931//mg/dl",
    "Creatinine": "LAB_lab//50912//mg/dl",
    "Hemoglobin": "LAB_lab//51222//g/dl",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_vocab(vocab_path: pathlib.Path) -> dict:
    """Load vocabulary from gzip pickle (fms-ehrs format)."""
    with gzip.open(vocab_path, "r+") as f:
        return pickle.load(f)


def resolve_vocab_path(explicit_path: pathlib.Path | None, model_dir: pathlib.Path) -> pathlib.Path:
    """Resolve the vocabulary path for a model.

    Main model directories no longer always carry a colocated ``vocab.gzip``, so
    callers can pass the tokenized-data vocabulary explicitly. We keep the old fallback
    for backward compatibility.
    """
    if explicit_path is not None:
        return explicit_path
    fallback = model_dir / "vocab.gzip"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        f"Could not find vocab.gzip next to {model_dir}. Pass an explicit --vocab_* path "
        "from the matching tokenized dataset split."
    )


def load_embedding_weight(model_dir: pathlib.Path) -> torch.Tensor:
    """Load the token embedding matrix from either safetensors or PyTorch weights."""
    safetensors_path = model_dir / "model.safetensors"
    if safetensors_path.exists():
        from safetensors import safe_open

        with safe_open(str(safetensors_path), framework="pt", device="cpu") as f:
            return f.get_tensor("model.embed_tokens.weight")

    pytorch_path = model_dir / "pytorch_model.bin"
    if pytorch_path.exists():
        state_dict = torch.load(pytorch_path, map_location="cpu")
        for key in ("model.embed_tokens.weight", "embed_tokens.weight"):
            if key in state_dict:
                return state_dict[key]
        raise KeyError(
            f"Could not find token embeddings in {pytorch_path}; expected "
            "'model.embed_tokens.weight' or 'embed_tokens.weight'."
        )

    raise FileNotFoundError(
        f"Could not find model.safetensors or pytorch_model.bin in {model_dir}."
    )


def extract_discrete_bin_embeddings(
    model_dir: pathlib.Path,
    vocab_data: dict,
    num_bins: int = 10,
) -> dict[str, np.ndarray]:
    """Extract discrete quantile token embeddings from the model's embedding matrix.

    For each measurement code in the vocabulary, the discrete model has Q0..Q(num_bins-1)
    tokens. We extract the corresponding rows from the embedding matrix.

    Returns dict mapping measurement code -> (num_bins, embed_dim) array.
    """
    embed_weight = load_embedding_weight(model_dir)  # (vocab_size, embed_dim)

    lookup = vocab_data["lookup"]

    # Find Q token ids
    q_token_ids = []
    for i in range(num_bins):
        token_name = f"Q{i}"
        if token_name not in lookup:
            raise ValueError(f"Token {token_name} not found in vocabulary")
        q_token_ids.append(lookup[token_name])

    q_embeddings = embed_weight[q_token_ids].float().numpy()  # (num_bins, embed_dim)

    return {"shared_Q_tokens": q_embeddings}


def extract_fused_bin_embeddings(
    model_dir: pathlib.Path,
    vocab_data: dict,
    measurement_codes: list[str],
    num_bins: int = 10,
) -> dict[str, np.ndarray]:
    """Extract code-specific fused quantile embeddings from a fused discrete model."""
    embed_weight = load_embedding_weight(model_dir)

    lookup = vocab_data["lookup"]
    results = {}

    for code in measurement_codes:
        token_ids = []
        missing = []
        for i in range(num_bins):
            token_name = f"{code}_Q{i}"
            token_id = lookup.get(token_name)
            if token_id is None:
                missing.append(token_name)
                continue
            token_ids.append(token_id)

        if missing:
            print(f"  [WARN] Missing fused tokens for {code}: {', '.join(missing[:3])}"
                  + (" ..." if len(missing) > 3 else ""))
            continue

        results[code] = embed_weight[token_ids].float().numpy()

    return results


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity matrix for (n, d) embeddings."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normalized = embeddings / norms
    return normalized @ normalized.T


def pca_2d(embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project embeddings to 2D via PCA (mean-centered).

    Returns
    -------
    coords : np.ndarray
        Two-dimensional PCA coordinates.
    explained_variance_ratio : np.ndarray
        Fraction of variance explained by PC1 and PC2.
    """
    centered = embeddings - embeddings.mean(axis=0)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ Vt[:2].T  # (n, 2)
    denom = np.sum(S**2)
    if denom <= 0:
        evr = np.zeros(2, dtype=float)
    else:
        evr = (S[:2] ** 2) / denom
    return coords, evr


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_cosine_heatmap(
    sim_matrix: np.ndarray,
    title: str,
    output_path: pathlib.Path,
    num_bins: int = 10,
):
    """Plot cosine similarity heatmap."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(sim_matrix, cmap="RdYlBu_r", vmin=-1, vmax=1, aspect="equal")
    ax.set_xticks(range(num_bins))
    ax.set_yticks(range(num_bins))
    labels = [f"Bin {i}" for i in range(num_bins)]
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title(title, fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax, label="Cosine Similarity", shrink=0.8)

    # Annotate cells
    for i in range(num_bins):
        for j in range(num_bins):
            color = "white" if abs(sim_matrix[i, j]) > 0.5 else "black"
            ax.text(j, i, f"{sim_matrix[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color=color)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved heatmap: {output_path}")


def plot_pca_scatter(
    coords_dict: dict[str, tuple[np.ndarray, np.ndarray]],
    title: str,
    output_path: pathlib.Path,
    num_bins: int = 10,
):
    """Plot PCA scatter with bin labels."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(coords_dict), figsize=(7 * len(coords_dict), 6))
    if len(coords_dict) == 1:
        axes = [axes]

    cmap = plt.cm.RdYlGn_r  # Red for extreme bins, green for middle
    colors = [cmap(i / (num_bins - 1)) for i in range(num_bins)]

    for ax, (label, (coords, evr)) in zip(axes, coords_dict.items()):
        for i in range(num_bins):
            ax.scatter(coords[i, 0], coords[i, 1], color=colors[i], s=120,
                       zorder=5, edgecolors="black", linewidths=0.5)
            ax.annotate(f"Bin {i}", (coords[i, 0], coords[i, 1]),
                        textcoords="offset points", xytext=(8, 4), fontsize=9)

        # Connect bins in order to visualize the manifold shape
        ax.plot(coords[:, 0], coords[:, 1], "k--", alpha=0.3, linewidth=1)

        ax.set_xlabel(f"PC1 ({100 * evr[0]:.1f}% var)", fontsize=12)
        ax.set_ylabel(f"PC2 ({100 * evr[1]:.1f}% var)", fontsize=12)
        ax.set_title(label, fontsize=13, fontweight="bold")
        ax.grid(alpha=0.3)

    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved PCA scatter: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analysis 3: Embedding geometry diagnostics")
    parser.add_argument(
        "--model_discrete", type=pathlib.Path, required=True,
        help="Path to discrete model directory (e.g., model-discrete-time_rope)",
    )
    parser.add_argument(
        "--vocab_discrete", type=pathlib.Path,
        help="Path to discrete model vocab.gzip (typically from the matching tokenized split)",
    )
    parser.add_argument(
        "--model_fused", type=pathlib.Path,
        help="Optional fused discrete model directory for a code-specific comparison panel",
    )
    parser.add_argument(
        "--vocab_fused", type=pathlib.Path,
        help="Path to fused model vocab.gzip (required when --model_fused is used unless colocated)",
    )
    parser.add_argument(
        "--measurements", type=str, nargs="+",
        default=list(DEFAULT_MEASUREMENTS.values()),
        help="Lab-variable code names (e.g., LAB_lab//50971//meq/l)",
    )
    parser.add_argument(
        "--output_dir", type=pathlib.Path, default=pathlib.Path("outputs/diagnostics"),
        help="Output directory for figures and results",
    )
    parser.add_argument("--dry_run", action="store_true", help="Quick check; skip saving figures")
    parser.add_argument("--num_bins", type=int, default=10, help="Number of bins (deciles=10)")

    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build reverse name lookup
    code_to_name = {v: k for k, v in DEFAULT_MEASUREMENTS.items()}

    # ---- Load vocabularies ----
    discrete_vocab_path = resolve_vocab_path(args.vocab_discrete, args.model_discrete)
    print(f"Loading discrete vocab: {discrete_vocab_path}")
    discrete_vocab = load_vocab(discrete_vocab_path)

    fused_embs = {}
    if args.model_fused is not None:
        fused_vocab_path = resolve_vocab_path(args.vocab_fused, args.model_fused)
        print(f"Loading fused vocab: {fused_vocab_path}")
        fused_vocab = load_vocab(fused_vocab_path)
        print("\n=== Fused Model: Extracting code-specific quantile-token embeddings ===")
        fused_embs = extract_fused_bin_embeddings(
            args.model_fused, fused_vocab, args.measurements, num_bins=args.num_bins,
        )
        for code, emb in fused_embs.items():
            name = code_to_name.get(code, code)
            print(f"  {name}: shape {emb.shape}")

    # ---- Extract discrete embeddings ----
    print("\n=== Discrete Model: Extracting quantile-token embeddings ===")
    discrete_embs = extract_discrete_bin_embeddings(
        args.model_discrete, discrete_vocab, num_bins=args.num_bins,
    )
    shared_q = discrete_embs["shared_Q_tokens"]
    print(f"  Shared quantile-token embeddings shape: {shared_q.shape}")

    if args.dry_run:
        print("\n[DRY RUN] Skipping figure generation.")
        # Print summary statistics
        shared_sim = cosine_similarity_matrix(shared_q)
        print(f"\nDiscrete quantile-token cosine sim (corners):")
        print(f"  Bin0-Bin1: {shared_sim[0, 1]:.4f}")
        print(f"  Bin0-Bin9: {shared_sim[0, 9]:.4f}")
        print(f"  Bin8-Bin9: {shared_sim[8, 9]:.4f}")
        return

    # ---- Results storage ----
    results = {"discrete": {}, "fused": {}}

    # ---- Discrete: RSA + PCA (shared Q tokens) ----
    print("\n=== Discrete RSA + PCA (shared quantile tokens) ===")
    sim_discrete = cosine_similarity_matrix(shared_q)
    results["discrete"]["shared_Q_cosine_sim"] = sim_discrete.tolist()

    plot_cosine_heatmap(
        sim_discrete,
        "Discrete Encoding: Q-Token Cosine Similarity",
        output_dir / "discrete_cosine_heatmap.png",
        num_bins=args.num_bins,
    )

    pca_discrete = pca_2d(shared_q)

    # ---- Per-variable soft RSA + combined PCA ----
    for code in args.measurements:
        if code not in fused_embs:
            continue
        name = code_to_name.get(code, code)
        safe_name = name.lower().replace(" ", "_")

        print(f"\n=== {name}: shared vs fused comparison ===")
        sim_fused = cosine_similarity_matrix(fused_embs[code])
        results["fused"][code] = {
            "name": name,
            "cosine_sim": sim_fused.tolist(),
        }
        plot_cosine_heatmap(
            sim_fused,
            f"Fused discrete encoding: {name} cosine similarity",
            output_dir / f"fused_cosine_heatmap_{safe_name}.png",
            num_bins=args.num_bins,
        )

        coords_dict = {
            "Shared quantile tokens": pca_discrete,
            f"Fused quantile tokens ({name})": pca_2d(fused_embs[code]),
        }
        plot_pca_scatter(
            coords_dict,
            f"Embedding Geometry: {name}",
            output_dir / f"pca_comparison_{safe_name}.png",
            num_bins=args.num_bins,
        )

    # ---- U-shape metric ----
    # Quantify "U-shape": cosine(Bin0, Bin_last) vs cosine(Bin0, Bin_mid)
    n = args.num_bins
    mid = n // 2
    u_metric_discrete = float(sim_discrete[0, n - 1] - sim_discrete[0, mid])
    results["discrete"]["u_shape_metric"] = u_metric_discrete
    print(f"\nU-shape metric (Discrete): cos(B0,B{n-1}) - cos(B0,B{mid}) = {u_metric_discrete:.4f}")
    if u_metric_discrete > 0:
        print("  → U-shape DETECTED: extreme bins are more similar than expected")
    else:
        print("  → No strong U-shape: ordinal structure maintained")

    # ---- Save JSON results ----
    results_path = output_dir / "embedding_geometry_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
