#!/usr/bin/env python3
"""Exploratory multiscale potassium embedding-geometry grid.

This diagnostic compares potassium embedding geometry across three nominal
quantization granularities:

- deciles
- trentiles
- centiles

For each granularity, it plots:

1. shared unfused discrete quantile tokens
2. fused code-specific discrete tokens

The goal is to isolate the shared-versus-fused contrast as nominal granularity
increases, while also exposing how repeated rounded potassium values collapse the
effective number of realized high-resolution bins.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pipeline.scripts.diagnostics.diag_embedding_geometry import (
    extract_discrete_bin_embeddings,
    load_embedding_weight,
    load_vocab,
    pca_2d,
)


ANALYTE_CODE = "LAB_lab//50971//meq/l"
ANALYTE_NAME = "Potassium"


def _extract_fused_sparse(
    *,
    model_dir: Path,
    vocab_path: Path,
    code: str,
    num_bins: int,
) -> tuple[list[int], np.ndarray]:
    embed_weight = load_embedding_weight(model_dir)
    vocab = load_vocab(vocab_path)
    lookup = vocab["lookup"]
    present_bins: list[int] = []
    token_ids: list[int] = []
    for i in range(num_bins):
        token_id = lookup.get(f"{code}_Q{i}")
        if token_id is None:
            continue
        present_bins.append(i)
        token_ids.append(token_id)
    if not token_ids:
        raise ValueError(f"No fused tokens found for {code} in {vocab_path}")
    return present_bins, embed_weight[token_ids].float().numpy()


def _resolution_specs(repo_root: Path) -> list[dict[str, object]]:
    model_root = repo_root / "artifacts" / "runs" / "models"
    token_root = repo_root / "artifacts" / "runs" / "tokenized" / "mimiciv-3.1_meds_70-10-20"

    return [
        {
            "name": "Deciles",
            "data_version": "deciles_none",
            "num_bins": 10,
            "shared_model": model_root / "exp1_meds_deciles_none_fusedFalse_discrete_time_tokens-s42" / "run-0" / "checkpoint-9000",
            "fused_model": model_root / "exp1_meds_deciles_none_fusedTrue_discrete_time_tokens-s42" / "run-1" / "checkpoint-9000",
            "shared_vocab": token_root / "deciles_none_unfused_time_tokens_first_24h-tokenized" / "train" / "vocab.gzip",
            "fused_vocab": token_root / "deciles_none_fused_time_tokens_first_24h-tokenized" / "train" / "vocab.gzip",
            "annotate_bins": list(range(10)),
        },
        {
            "name": "Trentiles",
            "data_version": "trentiles_none",
            "num_bins": 30,
            "shared_model": model_root / "exp1_meds_trentiles_none_fusedFalse_discrete_time_tokens-s42" / "run-1" / "checkpoint-9000",
            "fused_model": model_root / "exp1_meds_trentiles_none_fusedTrue_discrete_time_tokens-s42" / "run-2" / "checkpoint-9000",
            "shared_vocab": token_root / "trentiles_none_unfused_time_tokens_first_24h-tokenized" / "train" / "vocab.gzip",
            "fused_vocab": token_root / "trentiles_none_fused_time_tokens_first_24h-tokenized" / "train" / "vocab.gzip",
            "annotate_bins": [0, 9, 14, 19, 29],
        },
        {
            "name": "Centiles",
            "data_version": "centiles_none",
            "num_bins": 100,
            "shared_model": model_root / "exp1_meds_centiles_none_fusedFalse_discrete_time_tokens-s42" / "run-0" / "checkpoint-9000",
            "fused_model": model_root / "exp1_meds_centiles_none_fusedTrue_discrete_time_tokens-s42" / "run-0" / "checkpoint-9000",
            "shared_vocab": token_root / "centiles_none_unfused_time_tokens_first_24h-tokenized" / "train" / "vocab.gzip",
            "fused_vocab": token_root / "centiles_none_fused_time_tokens_first_24h-tokenized" / "train" / "vocab.gzip",
            "annotate_bins": [0, 24, 49, 74, 99],
        },
    ]


def _add_panel(
    ax,
    *,
    coords: np.ndarray | None,
    evr: np.ndarray | None,
    title: str,
    color_positions: list[int] | None,
    annotate_bins: list[int] | None,
    note: str | None = None,
    placeholder: str | None = None,
) -> None:
    ax.set_title(title, fontsize=11, fontweight="bold")
    if coords is None or evr is None or color_positions is None:
        ax.axis("off")
        if placeholder is not None:
            ax.text(
                0.5,
                0.58,
                placeholder,
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                transform=ax.transAxes,
            )
        if note is not None:
            ax.text(
                0.5,
                0.34,
                note,
                ha="center",
                va="center",
                fontsize=10,
                transform=ax.transAxes,
            )
        return

    num_points = len(color_positions)
    cmap = plt.cm.RdYlGn_r
    max_color = max(max(color_positions), 1)
    colors = [cmap(i / max_color) for i in color_positions]
    size = 110 if num_points <= 10 else 42 if num_points <= 30 else 22
    line_alpha = 0.35 if num_points <= 10 else 0.22 if num_points <= 30 else 0.14

    for idx, (color, bin_id) in enumerate(zip(colors, color_positions)):
        ax.scatter(
            coords[idx, 0],
            coords[idx, 1],
            color=color,
            s=size,
            zorder=5,
            edgecolors="black",
            linewidths=0.3,
        )
    ax.plot(coords[:, 0], coords[:, 1], "k--", alpha=line_alpha, linewidth=0.8)

    bin_to_coord = {b: i for i, b in enumerate(color_positions)}
    for b in annotate_bins or []:
        if b not in bin_to_coord:
            continue
        idx = bin_to_coord[b]
        ax.annotate(
            f"Bin {b}",
            (coords[idx, 0], coords[idx, 1]),
            textcoords="offset points",
            xytext=(5, 3),
            fontsize=7 if num_points > 30 else 8,
        )

    ax.set_xlabel(f"PC1 ({100 * evr[0]:.1f}% var)", fontsize=9)
    ax.set_ylabel(f"PC2 ({100 * evr[1]:.1f}% var)", fontsize=9)
    ax.grid(alpha=0.25)
    if note:
        ax.text(
            0.03,
            0.04,
            note,
            transform=ax.transAxes,
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="0.8"),
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/diagnostics/embedding_geometry_exploratory"),
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    output_dir = (repo_root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    specs = _resolution_specs(repo_root)
    summary: dict[str, object] = {"measurement_code": ANALYTE_CODE, "rows": []}

    fig, axes = plt.subplots(3, 2, figsize=(13, 14))
    axes = axes.ravel()

    for row_idx, spec in enumerate(specs):
        name = str(spec["name"])
        num_bins = int(spec["num_bins"])
        annotate_bins = list(spec["annotate_bins"])

        shared_vocab = load_vocab(Path(spec["shared_vocab"]))
        shared_emb = extract_discrete_bin_embeddings(
            Path(spec["shared_model"]),
            shared_vocab,
            num_bins=num_bins,
        )["shared_Q_tokens"]
        shared_coords, shared_evr = pca_2d(shared_emb)
        aux = np.asarray(shared_vocab["aux"][ANALYTE_CODE], dtype=float)
        n_breakpoints = int(aux.size)
        n_unique_breakpoints = int(np.unique(aux).size)

        fused_present_bins, fused_emb = _extract_fused_sparse(
            model_dir=Path(spec["fused_model"]),
            vocab_path=Path(spec["fused_vocab"]),
            code=ANALYTE_CODE,
            num_bins=num_bins,
        )
        fused_coords, fused_evr = pca_2d(fused_emb)

        row_summary: dict[str, object] = {
            "name": name,
            "num_bins": num_bins,
            "n_breakpoints": n_breakpoints,
            "n_unique_breakpoints": n_unique_breakpoints,
            "shared_unfused_bins_present": num_bins,
            "fused_bins_present": len(fused_present_bins),
            "fused_bins_missing": [i for i in range(num_bins) if i not in fused_present_bins],
        }
        summary["rows"].append(row_summary)

        _add_panel(
            axes[row_idx * 2 + 0],
            coords=shared_coords,
            evr=shared_evr,
            title=f"{name}: shared unfused",
            color_positions=list(range(num_bins)),
            annotate_bins=annotate_bins,
        )

        fused_note = None
        if len(fused_present_bins) < num_bins:
            fused_note = (
                f"{len(fused_present_bins)}/{num_bins} fused bins present\n"
                f"{n_unique_breakpoints} unique breakpoints"
            )
        _add_panel(
            axes[row_idx * 2 + 1],
            coords=fused_coords,
            evr=fused_evr,
            title=f"{name}: fused discrete",
            color_positions=fused_present_bins,
            annotate_bins=[b for b in annotate_bins if b in fused_present_bins],
            note=fused_note,
        )

    fig.suptitle(
        "Exploratory potassium embedding geometry across quantization granularities",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.02,
        "Rounded potassium values make nominal high-resolution quantizers collapse: many centile breakpoints repeat, "
        "so effective fused resolution can be far smaller than the requested number of bins.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=[0.03, 0.05, 1.0, 0.96])

    fig_path = output_dir / "potassium_pca_deciles_trentiles_centiles_6panel.png"
    fig.savefig(fig_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    summary_path = output_dir / "potassium_pca_deciles_trentiles_centiles_6panel_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(fig_path)
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
