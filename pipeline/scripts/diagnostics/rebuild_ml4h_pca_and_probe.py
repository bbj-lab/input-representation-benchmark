#!/usr/bin/env python3
"""Rebuild ML4H centile PCA and clinical-boundary probe on best checkpoints."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

_DIAG_DIR = Path(__file__).resolve().parent
_IRB_HOME = _DIAG_DIR.parents[2]
if str(_DIAG_DIR) not in sys.path:
    sys.path.insert(0, str(_DIAG_DIR))

from checkpoint_selection import find_best_model_checkpoint
from diag_clinical_boundary_probe import main as probe_main
from diag_embedding_geometry import (
    extract_discrete_bin_embeddings,
    load_embedding_weight,
    load_vocab,
    pca_2d,
)

PCA_GRID_FIGSIZE = (12.4, 7.0)
PCA_GRID_TITLE_SIZE = 11.0
PCA_GRID_AXIS_LABEL_SIZE = 9.0
PCA_GRID_TICK_SIZE = 8.0
PCA_GRID_ANNOT_SIZE = 8.0
PCA_GRID_PANEL_NOTE_SIZE = 8.5
PCA_GRID_CBAR_LABEL_SIZE = 9.0
PCA_GRID_CBAR_TICK_SIZE = 8.0
PCA_GRID_SHARED_MARKER_SIZE = 20
PCA_GRID_FUSED_MARKER_SIZE = 20


def _pad_scatter_limits(ax, coords: np.ndarray, *, extra_bottom: bool = False) -> None:
    xmin, xmax = float(coords[:, 0].min()), float(coords[:, 0].max())
    ymin, ymax = float(coords[:, 1].min()), float(coords[:, 1].max())
    xspan = max(xmax - xmin, 1e-6)
    yspan = max(ymax - ymin, 1e-6)
    ax.set_xlim(xmin - 0.14 * xspan, xmax + 0.16 * xspan)
    bottom_pad = 0.24 * yspan if extra_bottom else 0.14 * yspan
    ax.set_ylim(ymin - bottom_pad, ymax + 0.15 * yspan)


def _bbox_overlap_area(a, b, pad: float = 0.0) -> float:
    x_overlap = min(a.x1 + pad, b.x1) - max(a.x0 - pad, b.x0)
    y_overlap = min(a.y1 + pad, b.y1) - max(a.y0 - pad, b.y0)
    if x_overlap <= 0.0 or y_overlap <= 0.0:
        return 0.0
    return float(x_overlap * y_overlap)


def _annotate_selected_bins(
    fig,
    ax,
    coords: np.ndarray,
    bins: list[int],
    selected_bins: set[int],
    *,
    badge_artist=None,
) -> None:
    renderer = fig.canvas.get_renderer()
    ax_bbox = ax.get_window_extent(renderer)
    badge_bbox = (
        badge_artist.get_window_extent(renderer).expanded(1.06, 1.18)
        if badge_artist is not None
        else None
    )
    pts_px = ax.transData.transform(coords)
    selected_indices = [idx for idx, b in enumerate(bins) if b in selected_bins]
    selected_set = set(selected_indices)

    def _crowding(idx: int) -> float:
        diffs = pts_px - pts_px[idx]
        d2 = np.sum(diffs * diffs, axis=1)
        d2[idx] = np.inf
        return float(np.min(d2))

    ordered_indices = sorted(selected_indices, key=_crowding)
    candidates = [
        (5, 4, "left", "bottom"),
        (-5, 4, "right", "bottom"),
        (5, -4, "left", "top"),
        (-5, -4, "right", "top"),
        (6, 0, "left", "center"),
        (-6, 0, "right", "center"),
        (0, 6, "center", "bottom"),
        (0, -6, "center", "top"),
        (10, 7, "left", "bottom"),
        (-10, 7, "right", "bottom"),
        (10, -7, "left", "top"),
        (-10, -7, "right", "top"),
        (12, 0, "left", "center"),
        (-12, 0, "right", "center"),
        (0, 11, "center", "bottom"),
        (0, -11, "center", "top"),
        (15, 10, "left", "bottom"),
        (-15, 10, "right", "bottom"),
        (15, -10, "left", "top"),
        (-15, -10, "right", "top"),
    ]
    placed_boxes = []
    pad_in = 3.5
    dot_r = 4.5

    for idx in ordered_indices:
        bin_id = bins[idx]
        best_cfg = candidates[0]
        best_penalty = float("inf")
        for dx, dy, ha, va in candidates:
            dist_sq = dx * dx + dy * dy
            ann = ax.annotate(
                f"Q{bin_id}",
                (coords[idx, 0], coords[idx, 1]),
                textcoords="offset points",
                xytext=(dx, dy),
                ha=ha,
                va=va,
                fontsize=PCA_GRID_ANNOT_SIZE,
                zorder=6,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.5},
            )
            box = ann.get_window_extent(renderer)
            ann.remove()

            penalty = 0.45 * dist_sq
            out_x = max(0.0, (ax_bbox.x0 + pad_in) - box.x0) + max(
                0.0, box.x1 - (ax_bbox.x1 - pad_in)
            )
            out_y = max(0.0, (ax_bbox.y0 + pad_in) - box.y0) + max(
                0.0, box.y1 - (ax_bbox.y1 - pad_in)
            )
            if out_x > 0.0 or out_y > 0.0:
                penalty += 1e6 + 1e4 * (out_x + out_y)
            if badge_bbox is not None:
                b_area = _bbox_overlap_area(box, badge_bbox, pad=3.0)
                if b_area > 0.0:
                    penalty += 1e6 + 1e3 * b_area
            for pbox in placed_boxes:
                ov = _bbox_overlap_area(box, pbox, pad=3.5)
                if ov > 0.0:
                    penalty += 5e5 + 1e3 * ov

            cx = 0.5 * (box.x0 + box.x1)
            cy = 0.5 * (box.y0 + box.y1)
            for j in range(len(bins)):
                if j == idx:
                    continue
                px, py = float(pts_px[j, 0]), float(pts_px[j, 1])
                if (box.x0 - dot_r) <= px <= (box.x1 + dot_r) and (
                    box.y0 - dot_r
                ) <= py <= (box.y1 + dot_r):
                    penalty += 8000.0 if j in selected_set else 650.0
                d2 = (px - cx) ** 2 + (py - cy) ** 2
                penalty += 35.0 / max(d2, 25.0)

            if penalty < best_penalty:
                best_penalty = penalty
                best_cfg = (dx, dy, ha, va)

        dx, dy, ha, va = best_cfg
        use_leader = (dx * dx + dy * dy) > 65
        final_ann = ax.annotate(
            f"Q{bin_id}",
            (coords[idx, 0], coords[idx, 1]),
            textcoords="offset points",
            xytext=(dx, dy),
            ha=ha,
            va=va,
            fontsize=PCA_GRID_ANNOT_SIZE,
            zorder=6,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.80, "pad": 0.5},
            arrowprops=(
                {
                    "arrowstyle": "-",
                    "color": "#666666",
                    "lw": 0.55,
                    "shrinkA": 0,
                    "shrinkB": 2.5,
                }
                if use_leader
                else None
            ),
        )
        placed_boxes.append(final_ann.get_window_extent(renderer))

PCA_MEASUREMENTS = [
    ("Potassium", "LAB_lab//50971//meq/l"),
    ("Hemoglobin", "LAB_lab//51222//g/dl"),
    ("Creatinine", "LAB_lab//50912//mg/dl"),
    ("Glucose", "LAB_lab//50931//mg/dl"),
    ("Sodium", "LAB_lab//50983//meq/l"),
    ("Heart rate", "VTL_vital//220045//bpm"),
]

UNFUSED_CENTILE = (
    "ml4h_llama_exp1_meds_centiles_none_fusedFalse_discrete_time_tokens-s42"
)
FUSED_CENTILE = (
    "ml4h_llama_exp1_meds_centiles_none_fusedTrue_discrete_time_tokens-s42"
)


def _selected_bin_labels(bins: list[int]) -> set[int]:
    if len(bins) <= 12:
        return set(bins)
    anchors = [0, 10, 25, 50, 75, 90, 99]
    selected = set()
    for anchor in anchors:
        nearest = min(bins, key=lambda b: abs(b - anchor))
        selected.add(nearest)
    return selected


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


def build_centile_pca_grid(
    *,
    models_dir: Path,
    token_root: Path,
    source_csv: Path,
    figure_pdf: Path,
    figure_png: Path,
) -> dict:
    num_bins = 100
    shared_model = find_best_model_checkpoint(models_dir / UNFUSED_CENTILE)
    fused_model = find_best_model_checkpoint(models_dir / FUSED_CENTILE)
    shared_vocab = (
        token_root / "centiles_none_unfused_time_tokens_first_24h-tokenized" / "train" / "vocab.gzip"
    )
    fused_vocab = (
        token_root / "centiles_none_fused_time_tokens_first_24h-tokenized" / "train" / "vocab.gzip"
    )

    shared_emb = extract_discrete_bin_embeddings(
        shared_model,
        load_vocab(shared_vocab),
        num_bins=num_bins,
    )["shared_Q_tokens"]
    shared_bins = list(range(num_bins))
    shared_coords, shared_evr = pca_2d(shared_emb)

    source_rows: list[dict[str, object]] = []
    fig = plt.figure(figsize=PCA_GRID_FIGSIZE)
    gs = fig.add_gridspec(
        2,
        4,
        height_ratios=[1.0, 1.0],
        width_ratios=[1.0, 1.0, 1.0, 1.0],
        left=0.06,
        right=0.99,
        top=0.93,
        bottom=0.22,
        wspace=0.38,
        hspace=0.38,
    )
    row0_col0 = gs[0, 0].get_position(fig)
    row1_col0 = gs[1, 0].get_position(fig)
    shared_ax = fig.add_axes(
        [
            row0_col0.x0 - 0.030,
            0.5 * (row0_col0.y0 + row1_col0.y0),
            row0_col0.width,
            row0_col0.height,
        ]
    )
    fused_axes = [
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[0, 3]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[1, 2]),
        fig.add_subplot(gs[1, 3]),
    ]
    cmap = plt.cm.RdYlGn_r

    shared_colors = [cmap(i / max(num_bins - 1, 1)) for i in shared_bins]
    shared_ax.scatter(
        shared_coords[:, 0],
        shared_coords[:, 1],
        c=shared_colors,
        s=PCA_GRID_SHARED_MARKER_SIZE,
        zorder=4,
        edgecolors="#222222",
        linewidths=0.3,
    )
    shared_ax.plot(shared_coords[:, 0], shared_coords[:, 1], "k--", alpha=0.25, linewidth=0.7)
    for idx, bin_id in enumerate(shared_bins):
        source_rows.append(
            {
                "measurement": "Shared centile vocabulary",
                "measurement_code": "shared_Q_tokens",
                "panel": "shared_centiles",
                "bin_id": bin_id,
                "pc1": shared_coords[idx, 0],
                "pc2": shared_coords[idx, 1],
                "pc1_var_explained": float(shared_evr[0]),
                "pc2_var_explained": float(shared_evr[1]),
                "model_dir": str(shared_model),
                "vocab_path": str(shared_vocab),
            }
        )
    _pad_scatter_limits(shared_ax, shared_coords, extra_bottom=False)
    shared_ax.set_title(
        "Shared centile tokens\n(one manifold for all numeric variables)",
        fontsize=PCA_GRID_TITLE_SIZE,
        fontweight="bold",
    )
    shared_ax.set_xlabel(f"PC1 ({100 * shared_evr[0]:.1f}% var)", fontsize=PCA_GRID_AXIS_LABEL_SIZE)
    shared_ax.set_ylabel(f"PC2 ({100 * shared_evr[1]:.1f}% var)", fontsize=PCA_GRID_AXIS_LABEL_SIZE)
    shared_ax.tick_params(axis="both", labelsize=PCA_GRID_TICK_SIZE)
    shared_ax.grid(alpha=0.22)

    realized = {}
    fused_panel_specs = []
    for ax, (measurement_name, measurement_code) in zip(fused_axes, PCA_MEASUREMENTS):
        fused_bins, fused_emb = _extract_fused_sparse(
            model_dir=fused_model,
            vocab_path=fused_vocab,
            code=measurement_code,
            num_bins=num_bins,
        )
        realized[measurement_name] = len(fused_bins)
        fused_coords, fused_evr = pca_2d(fused_emb)
        colors = [cmap(i / max(num_bins - 1, 1)) for i in fused_bins]
        ax.scatter(
            fused_coords[:, 0],
            fused_coords[:, 1],
            c=colors,
            s=PCA_GRID_FUSED_MARKER_SIZE,
            zorder=4,
            edgecolors="#222222",
            linewidths=0.3,
        )
        ax.plot(fused_coords[:, 0], fused_coords[:, 1], "k--", alpha=0.25, linewidth=0.6)
        for idx, bin_id in enumerate(fused_bins):
            source_rows.append(
                {
                    "measurement": measurement_name,
                    "measurement_code": measurement_code,
                    "panel": "fused_centiles",
                    "bin_id": bin_id,
                    "pc1": fused_coords[idx, 0],
                    "pc2": fused_coords[idx, 1],
                    "pc1_var_explained": float(fused_evr[0]),
                    "pc2_var_explained": float(fused_evr[1]),
                    "model_dir": str(fused_model),
                    "vocab_path": str(fused_vocab),
                }
            )
        _pad_scatter_limits(ax, fused_coords, extra_bottom=True)
        ax.set_title(measurement_name, fontsize=PCA_GRID_TITLE_SIZE, fontweight="bold")
        ax.set_xlabel(f"PC1 ({100 * fused_evr[0]:.1f}% var)", fontsize=PCA_GRID_AXIS_LABEL_SIZE)
        ax.set_ylabel(f"PC2 ({100 * fused_evr[1]:.1f}% var)", fontsize=PCA_GRID_AXIS_LABEL_SIZE)
        ax.tick_params(axis="both", labelsize=PCA_GRID_TICK_SIZE)
        ax.grid(alpha=0.22)
        badge_artist = ax.text(
            0.97,
            0.04,
            f"{len(fused_bins)} realized bins",
            transform=ax.transAxes,
            fontsize=PCA_GRID_PANEL_NOTE_SIZE,
            fontweight="semibold",
            va="bottom",
            ha="right",
            zorder=6,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.92, "pad": 1.8},
        )
        fused_panel_specs.append((ax, fused_coords, fused_bins, badge_artist))

    fig.canvas.draw()
    _annotate_selected_bins(
        fig,
        shared_ax,
        shared_coords,
        shared_bins,
        _selected_bin_labels(shared_bins),
        badge_artist=None,
    )
    for ax, fused_coords, fused_bins, badge_artist in fused_panel_specs:
        _annotate_selected_bins(
            fig,
            ax,
            fused_coords,
            fused_bins,
            _selected_bin_labels(fused_bins),
            badge_artist=badge_artist,
        )

    mappable = plt.cm.ScalarMappable(norm=Normalize(vmin=0, vmax=num_bins - 1), cmap=cmap)
    cbar_ax = fig.add_axes([0.12, 0.07, 0.76, 0.032])
    cbar = fig.colorbar(mappable, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Centile index (quantile bin id)", fontsize=PCA_GRID_CBAR_LABEL_SIZE)
    cbar.ax.tick_params(labelsize=PCA_GRID_CBAR_TICK_SIZE)
    source_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(source_rows).to_csv(source_csv, index=False)
    figure_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_pdf, bbox_inches="tight")
    fig.savefig(figure_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {
        "unfused_checkpoint": str(shared_model),
        "fused_checkpoint": str(fused_model),
        "realized_bins": realized,
    }


def _probe_summary(results: dict) -> dict:
    gran_acc = {}
    for gran, entries in results.items():
        if str(gran).startswith("_"):
            continue
        accs = [
            float(record["accuracy"])
            for record in entries.values()
            if record.get("accuracy") is not None
        ]
        named = {
            record["name"]: float(record["accuracy"])
            for record in entries.values()
            if record.get("accuracy") is not None
        }
        gran_acc[gran] = {
            "min": min(accs) if accs else None,
            "max": max(accs) if accs else None,
            "by_name": named,
        }
    return gran_acc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models_dir",
        type=Path,
        default=_IRB_HOME / "outputs/runs/ml4h_2026/models",
    )
    parser.add_argument(
        "--token_root",
        type=Path,
        default=_IRB_HOME / "outputs/runs/tokenized/mimiciv-3.1_meds_70-10-20",
    )
    parser.add_argument(
        "--metrics_dir",
        type=Path,
        default=_IRB_HOME / "outputs/runs/ml4h_2026/metrics",
    )
    parser.add_argument(
        "--paper_figures",
        type=Path,
        default=_IRB_HOME.parent / "ML4H2026/ML4H/figures",
    )
    parser.add_argument(
        "--probe_output_dir",
        type=Path,
        default=_IRB_HOME / "outputs/diagnostics_ml4h_clinical_boundary_probe",
    )
    parser.add_argument(
        "--pca_only",
        action="store_true",
        help="Rebuild only the centile PCA grid without rerunning the clinical-boundary probe.",
    )
    args = parser.parse_args()

    pca_info = build_centile_pca_grid(
        models_dir=args.models_dir,
        token_root=args.token_root,
        source_csv=args.metrics_dir / "pca_centile_geometry_grid_source.csv",
        figure_pdf=args.paper_figures / "pca_centile_geometry_grid.pdf",
        figure_png=args.paper_figures / "pca_centile_geometry_grid.png",
    )
    paper_source = args.paper_figures / "sources" / "pca_centile_geometry_grid_source.csv"
    paper_source.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.metrics_dir / "pca_centile_geometry_grid_source.csv", paper_source)
    if args.pca_only:
        summary_path = args.metrics_dir / "pca_probe_best_checkpoint_summary.json"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["pca"] = pca_info
            summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"pca": pca_info}, indent=2))
        return

    argv = sys.argv
    sys.argv = [
        "diag_clinical_boundary_probe.py",
        "--models_dir",
        str(args.models_dir),
        "--output_dir",
        str(args.probe_output_dir),
    ]
    try:
        probe_main()
    finally:
        sys.argv = argv

    probe_json = args.probe_output_dir / "clinical_boundary_probe_results.json"
    probe_png = args.probe_output_dir / "clinical_boundary_probe.png"
    shutil.copy2(probe_json, args.metrics_dir / "clinical_boundary_probe_results.json")
    shutil.copy2(probe_png, args.paper_figures / "clinical_boundary_probe.png")

    results = json.loads(probe_json.read_text())
    probe_meta_path = args.probe_output_dir / "clinical_boundary_probe_checkpoints.json"
    probe_meta = (
        json.loads(probe_meta_path.read_text()) if probe_meta_path.is_file() else {}
    )
    summary = {
        "checkpoint_rule": "trainer_state.best_model_checkpoint",
        "pca": pca_info,
        "probe": _probe_summary(results),
        "probe_checkpoints": probe_meta,
    }
    summary_path = args.metrics_dir / "pca_probe_best_checkpoint_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
