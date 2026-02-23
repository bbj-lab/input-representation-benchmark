#!/usr/bin/env python3
"""Analysis 3: Geometric Proof of Ordinality — RSA and PCA of Discrete vs Soft Embeddings.

Extracts static learned embeddings for discrete quantile tokens and soft discretization
bin embeddings. Computes pairwise cosine similarity matrices and 2D PCA projections
to visualize ordinal structure (or lack thereof) in the embedding space.

Hypotheses:
  - Discrete models may learn "U-shaped" manifolds where extreme bins (BIN_1, BIN_10)
    are geometrically close due to similar clinical contexts (e.g., ICU transfer).
  - Soft discretization forces a linear, ordinal gradient by construction.

CPU-only. No forward pass needed — works directly from saved weights.

Usage:
    python scripts/diagnostics/diag_embedding_geometry.py \
        --model_discrete models/exp2_.../model-discrete-time_rope \
        --model_soft models/exp2_.../model-soft-time_rope \
        [--analytes LAB_lab//50971//meq/l LAB_lab//51222//g/dl] \
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

# Target analytes for RSA analysis (MIMIC-IV labevents itemids)
DEFAULT_ANALYTES = {
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


def extract_discrete_bin_embeddings(
    model_dir: pathlib.Path,
    vocab_data: dict,
    num_bins: int = 10,
) -> dict[str, np.ndarray]:
    """Extract discrete quantile token embeddings from the model's embedding matrix.

    For each analyte code in the vocabulary, the discrete model has Q0..Q(num_bins-1)
    tokens. We extract the corresponding rows from the embedding matrix.

    Returns dict mapping analyte code -> (num_bins, embed_dim) array.
    """
    from safetensors import safe_open

    safetensors_path = model_dir / "model.safetensors"
    with safe_open(str(safetensors_path), framework="pt", device="cpu") as f:
        embed_weight = f.get_tensor("model.embed_tokens.weight")  # (vocab_size, embed_dim)

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


def extract_soft_bin_embeddings(
    model_dir: pathlib.Path,
    vocab_data: dict,
    analyte_codes: list[str],
    num_bins: int = 10,
) -> dict[str, np.ndarray]:
    """Extract soft discretization embeddings for specific analytes.

    For soft encoding, we reconstruct the soft embedding at each decile midpoint
    by evaluating the SoftDiscretizationEncoder with the learned bin embeddings
    and per-code boundaries from vocab.aux.

    Returns dict mapping analyte code -> (num_bins, embed_dim) array.
    """
    rep_meta = torch.load(
        model_dir / "representation_mechanics.pt",
        map_location="cpu",
        weights_only=False,
    )
    venc_state = rep_meta["value_encoder_state"]
    bin_embeds = venc_state["bin_embeddings.weight"]  # (num_bins, embed_dim)

    aux = vocab_data.get("aux", {})
    results = {}

    for code in analyte_codes:
        if code not in aux:
            print(f"  [WARN] No aux boundaries for {code}, skipping soft embeddings")
            continue

        boundaries = torch.tensor(aux[code], dtype=torch.float32)  # (num_bins - 1,)

        # Compute embeddings at representative points:
        # - Below first boundary (pure BIN_0)
        # - Midpoint of each inter-boundary interval
        # - Above last boundary (pure BIN_last)
        points = []
        # Below first boundary
        points.append(boundaries[0] - 1.0)
        # Midpoints between consecutive boundaries
        for i in range(len(boundaries) - 1):
            points.append((boundaries[i] + boundaries[i + 1]) / 2.0)
        # Above last boundary
        points.append(boundaries[-1] + 1.0)

        embeddings = []
        for val in points:
            val_t = torch.tensor(val, dtype=torch.float32)
            emb = _compute_soft_embedding_standalone(val_t, boundaries, bin_embeds)
            embeddings.append(emb.float().numpy())

        results[code] = np.stack(embeddings)  # (num_bins, embed_dim)

    return results


def _compute_soft_embedding_standalone(
    value: torch.Tensor,
    boundaries: torch.Tensor,
    bin_embeddings_weight: torch.Tensor,
) -> torch.Tensor:
    """Standalone soft embedding computation (no nn.Module dependency)."""
    num_bins = bin_embeddings_weight.shape[0]
    n_boundaries = boundaries.numel()
    eff_bins = min(num_bins, n_boundaries + 1) if n_boundaries > 0 else 1

    bin_idx = torch.searchsorted(boundaries, value).clamp(0, eff_bins - 1)

    if bin_idx == 0:
        return bin_embeddings_weight[0]
    elif bin_idx >= eff_bins - 1:
        return bin_embeddings_weight[eff_bins - 1]

    lower_boundary = boundaries[bin_idx - 1]
    upper_boundary = boundaries[bin_idx]
    denom = upper_boundary - lower_boundary

    if denom.abs() < 1e-8:
        alpha = torch.tensor(0.5)
    else:
        alpha = ((value - lower_boundary) / denom).clamp(0.0, 1.0)

    lower_embed = bin_embeddings_weight[bin_idx - 1]
    upper_embed = bin_embeddings_weight[bin_idx]

    return (1.0 - alpha) * lower_embed + alpha * upper_embed


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity matrix for (n, d) embeddings."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normalized = embeddings / norms
    return normalized @ normalized.T


def pca_2d(embeddings: np.ndarray) -> np.ndarray:
    """Project embeddings to 2D via PCA (mean-centered)."""
    centered = embeddings - embeddings.mean(axis=0)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ Vt[:2].T  # (n, 2)


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
    coords_dict: dict[str, np.ndarray],
    title: str,
    output_path: pathlib.Path,
    num_bins: int = 10,
):
    """Plot PCA scatter with bin labels, comparing discrete vs soft."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(coords_dict), figsize=(7 * len(coords_dict), 6))
    if len(coords_dict) == 1:
        axes = [axes]

    cmap = plt.cm.RdYlGn_r  # Red for extreme bins, green for middle
    colors = [cmap(i / (num_bins - 1)) for i in range(num_bins)]

    for ax, (label, coords) in zip(axes, coords_dict.items()):
        for i in range(num_bins):
            ax.scatter(coords[i, 0], coords[i, 1], color=colors[i], s=120,
                       zorder=5, edgecolors="black", linewidths=0.5)
            ax.annotate(f"Bin {i}", (coords[i, 0], coords[i, 1]),
                        textcoords="offset points", xytext=(8, 4), fontsize=9)

        # Connect bins in order to visualize the manifold shape
        ax.plot(coords[:, 0], coords[:, 1], "k--", alpha=0.3, linewidth=1)

        ax.set_xlabel("PC1", fontsize=12)
        ax.set_ylabel("PC2", fontsize=12)
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
    parser = argparse.ArgumentParser(description="Analysis 3: Geometric Ordinality Proof")
    parser.add_argument(
        "--model_discrete", type=pathlib.Path, required=True,
        help="Path to discrete model directory (e.g., model-discrete-time_rope)",
    )
    parser.add_argument(
        "--model_soft", type=pathlib.Path, required=True,
        help="Path to soft model directory (e.g., model-soft-time_rope)",
    )
    parser.add_argument(
        "--analytes", type=str, nargs="+",
        default=list(DEFAULT_ANALYTES.values()),
        help="Analyte code names (e.g., LAB_lab//50971//meq/l)",
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
    code_to_name = {v: k for k, v in DEFAULT_ANALYTES.items()}

    # ---- Load vocabularies ----
    discrete_vocab_path = args.model_discrete / "vocab.gzip"
    soft_vocab_path = args.model_soft / "vocab.gzip"
    print(f"Loading discrete vocab: {discrete_vocab_path}")
    discrete_vocab = load_vocab(discrete_vocab_path)
    print(f"Loading soft vocab: {soft_vocab_path}")
    soft_vocab = load_vocab(soft_vocab_path)

    # ---- Extract discrete embeddings ----
    print("\n=== Discrete Model: Extracting Q-token embeddings ===")
    discrete_embs = extract_discrete_bin_embeddings(
        args.model_discrete, discrete_vocab, num_bins=args.num_bins,
    )
    shared_q = discrete_embs["shared_Q_tokens"]
    print(f"  Shared Q-token embeddings shape: {shared_q.shape}")

    # ---- Extract soft embeddings per analyte ----
    print("\n=== Soft Model: Computing per-analyte soft embeddings ===")
    soft_embs = extract_soft_bin_embeddings(
        args.model_soft, soft_vocab, args.analytes, num_bins=args.num_bins,
    )
    for code, emb in soft_embs.items():
        name = code_to_name.get(code, code)
        print(f"  {name}: shape {emb.shape}")

    if args.dry_run:
        print("\n[DRY RUN] Skipping figure generation.")
        # Print summary statistics
        shared_sim = cosine_similarity_matrix(shared_q)
        print(f"\nDiscrete Q-token cosine sim (corners):")
        print(f"  Bin0-Bin1: {shared_sim[0, 1]:.4f}")
        print(f"  Bin0-Bin9: {shared_sim[0, 9]:.4f}")
        print(f"  Bin8-Bin9: {shared_sim[8, 9]:.4f}")
        return

    # ---- Results storage ----
    results = {"discrete": {}, "soft": {}}

    # ---- Discrete: RSA + PCA (shared Q tokens) ----
    print("\n=== Discrete RSA + PCA (shared Q-tokens) ===")
    sim_discrete = cosine_similarity_matrix(shared_q)
    results["discrete"]["shared_Q_cosine_sim"] = sim_discrete.tolist()

    plot_cosine_heatmap(
        sim_discrete,
        "Discrete Encoding: Q-Token Cosine Similarity",
        output_dir / "discrete_cosine_heatmap.png",
        num_bins=args.num_bins,
    )

    pca_discrete = pca_2d(shared_q)

    # ---- Per-analyte soft RSA + combined PCA ----
    for code in args.analytes:
        if code not in soft_embs:
            continue
        name = code_to_name.get(code, code)
        safe_name = name.lower().replace(" ", "_")

        print(f"\n=== {name}: Soft vs Discrete comparison ===")
        soft_e = soft_embs[code]
        sim_soft = cosine_similarity_matrix(soft_e)
        results["soft"][code] = {
            "name": name,
            "cosine_sim": sim_soft.tolist(),
        }

        plot_cosine_heatmap(
            sim_soft,
            f"Soft Encoding: {name} Cosine Similarity",
            output_dir / f"soft_cosine_heatmap_{safe_name}.png",
            num_bins=args.num_bins,
        )

        # Combined PCA
        pca_soft = pca_2d(soft_e)
        plot_pca_scatter(
            {"Discrete (shared Q-tokens)": pca_discrete, f"Soft ({name})": pca_soft},
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
