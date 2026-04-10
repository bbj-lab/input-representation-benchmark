#!/usr/bin/env python3
"""Analysis 5: Clinical Boundary Probe — Do Population Quantiles Learn Reference Ranges?

Tests whether the static bin embeddings of population-quantile models endogenously encode
clinical abnormality (normal vs abnormal) through contextual co-occurrence during
pretraining, even though explicit reference range boundaries were never provided.

Method: Extract the static quantile-token embeddings for selected lab tests, label each bin
as "normal" or "abnormal" based on whether the bin midpoint falls within the known MIMIC-IV
reference interval, then train a logistic regression probe on those embeddings.

CPU-only. Probes all population granularities (decile, ventile, trentile, centile).

Usage:
    python pipeline/scripts/diagnostics/diag_clinical_boundary_probe.py \
        [--models_dir models] \
        [--output_dir outputs/diagnostics] \
        [--dry_run]
"""

import argparse
import gzip
import json
import pathlib
import pickle

import numpy as np


# ---------------------------------------------------------------------------
# Reference ranges (MIMIC-IV median values from labevents ref_range_lower/upper)
# Source: benchmark-side lab reference-range augmentation step
# ---------------------------------------------------------------------------

MEASUREMENT_REFERENCE_RANGES = {
    "LAB_lab//50971//meq/l": {  # Potassium (serum)
        "name": "Potassium",
        "ref_lower": 3.5,
        "ref_upper": 5.0,
    },
    "LAB_lab//50931//mg/dl": {  # Glucose (serum)
        "name": "Glucose",
        "ref_lower": 70.0,
        "ref_upper": 100.0,
    },
    "LAB_lab//50912//mg/dl": {  # Creatinine
        "name": "Creatinine",
        "ref_lower": 0.6,
        "ref_upper": 1.2,
    },
    "LAB_lab//51222//g/dl": {  # Hemoglobin
        "name": "Hemoglobin",
        "ref_lower": 12.0,
        "ref_upper": 17.5,
    },
    "LAB_lab//50822//meq/l": {  # Potassium (blood gas)
        "name": "Potassium (BG)",
        "ref_lower": 3.5,
        "ref_upper": 5.0,
    },
    "LAB_lab//50809//mg/dl": {  # Glucose (blood gas)
        "name": "Glucose (BG)",
        "ref_lower": 70.0,
        "ref_upper": 100.0,
    },
    "LAB_lab//51003//ng/ml": {  # Troponin T
        "name": "Troponin T",
        "ref_lower": 0.0,
        "ref_upper": 0.04,
    },
}


# Models to probe: (label, model_dir_pattern, quantizer, n_bins)
EXP1_MODELS = [
    ("Deciles", "exp1_meds_deciles_none_fusedFalse_discrete_time_tokens-s42", "deciles", 10),
    ("Ventiles", "exp1_meds_ventiles_none_fusedFalse_discrete_time_tokens-s42", "ventiles", 20),
    ("Trentiles", "exp1_meds_trentiles_none_fusedFalse_discrete_time_tokens-s42", "trentiles", 30),
    ("Centiles", "exp1_meds_centiles_none_fusedFalse_discrete_time_tokens-s42", "centiles", 100),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_vocab(vocab_path: pathlib.Path) -> dict:
    """Load vocabulary from gzip pickle."""
    with gzip.open(vocab_path, "r+") as f:
        return pickle.load(f)


def find_best_checkpoint(model_dir: pathlib.Path) -> pathlib.Path:
    """Find the best checkpoint directory within an Exp1 model directory.

    Exp1 models have structure: run-0/checkpoint-NNNN/
    Returns the first checkpoint found.
    """
    for run_dir in sorted(model_dir.iterdir()):
        if run_dir.is_dir() and run_dir.name.startswith("run-"):
            for ckpt_dir in sorted(run_dir.iterdir()):
                if ckpt_dir.is_dir() and ckpt_dir.name.startswith("checkpoint-"):
                    return ckpt_dir
    raise FileNotFoundError(f"No checkpoint found in {model_dir}")


def extract_embeddings_for_measurement(
    model_dir: pathlib.Path,
    vocab_path: pathlib.Path,
    measurement_code: str,
    num_bins: int,
) -> tuple[np.ndarray, list[float]] | None:
    """Extract quantile token embeddings for a specific lab variable.

    In unfused tokenization, the vocab contains Q0..Q(n-1) tokens that are shared.
    The aux data stores per-code bin boundaries.

    Returns (embeddings, bin_midpoints) or None if the measurement is not in aux.
    """
    from safetensors import safe_open

    vocab_data = load_vocab(vocab_path)
    lookup = vocab_data["lookup"]
    aux = vocab_data.get("aux", {})

    if measurement_code not in aux:
        return None

    boundaries = aux[measurement_code]  # list of (num_bins - 1) boundary values

    # Compute bin midpoints
    midpoints = []
    # Below first boundary
    midpoints.append(boundaries[0] - (boundaries[1] - boundaries[0]) / 2
                     if len(boundaries) > 1 else boundaries[0] - 1.0)
    # Between consecutive boundaries
    for i in range(len(boundaries) - 1):
        midpoints.append((boundaries[i] + boundaries[i + 1]) / 2.0)
    # Above last boundary
    midpoints.append(boundaries[-1] + (boundaries[-1] - boundaries[-2]) / 2
                     if len(boundaries) > 1 else boundaries[-1] + 1.0)

    # Extract Q-token embeddings
    safetensors_path = model_dir / "model.safetensors"
    with safe_open(str(safetensors_path), framework="pt", device="cpu") as f:
        embed_weight = f.get_tensor("model.embed_tokens.weight")

    q_ids = []
    for i in range(min(num_bins, len(midpoints))):
        token_name = f"Q{i}"
        if token_name not in lookup:
            break
        q_ids.append(lookup[token_name])

    embeddings = embed_weight[q_ids].float().numpy()
    midpoints = midpoints[:len(q_ids)]

    return embeddings, midpoints


def label_bins_by_reference(
    midpoints: list[float],
    ref_lower: float,
    ref_upper: float,
) -> np.ndarray:
    """Label each bin as 0 (normal) or 1 (abnormal) based on midpoint vs reference range."""
    labels = np.zeros(len(midpoints), dtype=int)
    for i, mp in enumerate(midpoints):
        if mp < ref_lower or mp > ref_upper:
            labels[i] = 1  # abnormal
    return labels


def run_loocv_probe(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> dict:
    """Run leave-one-out cross-validation logistic regression probe.

    Returns dict with accuracy, per-fold predictions, and label counts.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    n = len(labels)
    # Check for degenerate case (all same label)
    if len(np.unique(labels)) < 2:
        return {
            "accuracy": None,
            "n_normal": int((labels == 0).sum()),
            "n_abnormal": int((labels == 1).sum()),
            "note": "All bins have same label; probe not meaningful.",
        }

    predictions = np.zeros(n, dtype=int)
    correct = 0

    for i in range(n):
        # Leave out sample i
        train_mask = np.arange(n) != i
        X_train = embeddings[train_mask]
        y_train = labels[train_mask]
        X_test = embeddings[i:i+1]

        # Skip if training set is degenerate
        if len(np.unique(y_train)) < 2:
            predictions[i] = int(np.bincount(y_train).argmax())
        else:
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            clf = LogisticRegression(max_iter=1000, random_state=42, solver="lbfgs")
            clf.fit(X_train_s, y_train)
            predictions[i] = clf.predict(X_test_s)[0]

        if predictions[i] == labels[i]:
            correct += 1

    return {
        "accuracy": float(correct / n),
        "n_correct": int(correct),
        "n_total": n,
        "n_normal": int((labels == 0).sum()),
        "n_abnormal": int((labels == 1).sum()),
        "predictions": predictions.tolist(),
        "labels": labels.tolist(),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_probe_results(
    results: dict,
    output_path: pathlib.Path,
):
    """Bar chart of probe accuracy per lab test per granularity."""
    import matplotlib.pyplot as plt

    granularities = list(results.keys())
    measurements = set()
    for g_data in results.values():
        measurements.update(g_data.keys())
    measurements = sorted(measurements)

    if not measurements:
        print("  No lab tests to plot.")
        return

    n_groups = len(measurements)
    n_bars = len(granularities)
    x = np.arange(n_groups)
    width = 0.8 / n_bars

    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = plt.cm.Set2(np.linspace(0, 1, n_bars))

    for i, gran in enumerate(granularities):
        accs = []
        for measurement in measurements:
            res = results[gran].get(measurement, {})
            acc = res.get("accuracy")
            accs.append(acc if acc is not None else 0.0)

        bars = ax.bar(x + i * width - (n_bars - 1) * width / 2, accs,
                      width * 0.9, label=gran, color=colors[i], edgecolor="gray",
                      linewidth=0.5)

        # Annotate
        for bar, acc in zip(bars, accs):
            if acc > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{acc:.0%}", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Lab test", fontsize=14)
    ax.set_ylabel("LOO-CV Accuracy", fontsize=14)
    ax.set_title("Clinical Boundary Probe: Embedding Abnormality Classification",
                 fontsize=15, fontweight="bold")
    ax.set_xticks(x)

    # Use display names
    measurement_names = []
    for measurement in measurements:
        info = MEASUREMENT_REFERENCE_RANGES.get(measurement, {})
        measurement_names.append(info.get("name", measurement.split("//")[1] if "//" in measurement else measurement))
    ax.set_xticklabels(measurement_names, rotation=30, ha="right", fontsize=12)
    ax.tick_params(axis="y", labelsize=12)

    ax.set_ylim(0, 1.15)
    ax.axhline(y=0.5, color="red", linestyle="--", linewidth=1, alpha=0.5,
               label="Chance level")
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved probe results plot: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analysis 5: Clinical Boundary Probe"
    )
    parser.add_argument(
        "--models_dir", type=pathlib.Path, default=pathlib.Path("models"),
        help="Directory containing Exp1 model directories",
    )
    parser.add_argument(
        "--output_dir", type=pathlib.Path, default=pathlib.Path("outputs/diagnostics"),
    )
    parser.add_argument("--dry_run", action="store_true")

    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for model_label, model_name, quantizer, n_bins in EXP1_MODELS:
        model_dir = args.models_dir / model_name
        if not model_dir.exists():
            print(f"[SKIP] {model_label}: Model directory not found: {model_dir}")
            continue

        print(f"\n{'='*60}")
        print(f"Processing: {model_label} ({quantizer}, {n_bins} bins)")
        print(f"{'='*60}")

        # Find checkpoint
        try:
            ckpt_dir = find_best_checkpoint(model_dir)
        except FileNotFoundError:
            print(f"  [SKIP] No checkpoint found in {model_dir}")
            continue
        print(f"  Checkpoint: {ckpt_dir}")

        # Find the matching vocab (from the tokenized data)
        # Exp1 unfused models: vocab is in the tokenized data directory or the checkpoint
        # For Exp1, the vocab is in the tokenized data path. We look for it in a known
        # location from the data directory structure.
        #
        # Alternative: use the HF tokenizer in the checkpoint dir. But the fms-ehrs
        # Vocabulary format stores aux data (bin boundaries) which we need.
        #
        # The Exp2 models have vocab.gzip in the model dir. For Exp1, we need to find
        # the tokenized data directory that corresponds to this config.
        vocab_path = None

        # Strategy: look in the cache for the matching tokenized data
        data_version = f"{quantizer}_none_unfused_time_tokens"
        cache_base = args.models_dir.parent / ".cache" / "tokenized" / "mimiciv-3.1_meds_70-10-20"
        tokenized_vocab = cache_base / f"{data_version}-tokenized" / "train" / "vocab.gzip"
        if tokenized_vocab.exists():
            vocab_path = tokenized_vocab
        else:
            # Try the symlinked path
            data_base = args.models_dir.parent / "benchmarks" / "mimic-meds-extraction" / "data" / "meds" / "data"
            tokenized_vocab = data_base / f"{data_version}-tokenized" / "train" / "vocab.gzip"
            if tokenized_vocab.exists():
                vocab_path = tokenized_vocab

        if vocab_path is None:
            print(f"  [SKIP] Could not find vocab.gzip for {data_version}")
            continue
        print(f"  Vocab: {vocab_path}")

        model_results = {}

        for measurement_code, ref_info in MEASUREMENT_REFERENCE_RANGES.items():
            measurement_name = ref_info["name"]

            result = extract_embeddings_for_measurement(
                ckpt_dir, vocab_path, measurement_code, n_bins,
            )
            if result is None:
                print(f"  [{measurement_name}] Not in vocabulary aux, skipping")
                continue

            embeddings, midpoints = result
            actual_bins = len(midpoints)

            # Label bins
            labels = label_bins_by_reference(
                midpoints, ref_info["ref_lower"], ref_info["ref_upper"],
            )

            n_normal = int((labels == 0).sum())
            n_abnormal = int((labels == 1).sum())
            print(f"  [{measurement_name}] {actual_bins} bins: "
                  f"{n_normal} normal, {n_abnormal} abnormal")

            if args.dry_run:
                model_results[measurement_code] = {
                    "name": measurement_name,
                    "n_bins": actual_bins,
                    "n_normal": n_normal,
                    "n_abnormal": n_abnormal,
                    "midpoints": [round(m, 3) for m in midpoints],
                    "labels": labels.tolist(),
                }
                continue

            # Run LOO-CV probe
            probe_result = run_loocv_probe(embeddings, labels)
            probe_result["name"] = measurement_name
            probe_result["midpoints"] = [round(m, 3) for m in midpoints]
            model_results[measurement_code] = probe_result

            acc_str = (f"{probe_result['accuracy']:.1%}"
                       if probe_result['accuracy'] is not None else "N/A")
            print(f"    LOO-CV accuracy: {acc_str}")

        all_results[model_label] = model_results

    # Save results
    results_path = output_dir / "clinical_boundary_probe_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    if not args.dry_run and all_results:
        plot_probe_results(all_results, output_dir / "clinical_boundary_probe.png")


if __name__ == "__main__":
    main()
