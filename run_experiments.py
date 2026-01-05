#!/usr/bin/env python3
"""
Experiment Job Generation for Input Representation Benchmark.

This script generates SLURM job files that call fms-ehrs scripts for all three
experiments. Jobs are generated with valid parameter combinations and proper
dependencies between experiments.

Usage:
    # Generate demo job files (single-seed runs for validation)
    python run_experiments.py --mode demo

    # Generate full job files (5 seeds for statistical robustness)
    python run_experiments.py --mode full

    # Generate for specific experiment
    python run_experiments.py --mode full --exp 1
    python run_experiments.py --mode full --exp 2
    python run_experiments.py --mode full --exp 3

Output:
    slurm/exp1_demo_jobs.sh, slurm/exp2_demo_jobs.sh, slurm/exp3_demo_jobs.sh
    slurm/exp1_full_jobs.sh, slurm/exp2_full_jobs.sh, slurm/exp3_full_jobs.sh

Pipeline per job:
    Exp1: tokenize_w_config.py -> tune_model.py -> extract_outcomes.py -> fine_tune_classification.py
    Exp2: tokenize_w_config.py -> train_representation.py -> extract_outcomes.py -> fine_tune_classification.py
    Exp3: (uses existing tokenized data) -> train_representation.py -> ...

Attribution:
    The MEDS extraction pipeline used in this benchmark is adapted from ETHOS-ARES:
    https://github.com/ipolharvard/ethos-ares (MIT License, Copyright © 2024 Paweł Renc)
    Citation: Renc, P., et al. (2025). GigaScience, 14, giaf107.
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# =============================================================================
# Constants
# =============================================================================

# Training seeds for statistical robustness (5 seeds per configuration)
TRAINING_SEEDS = [42, 123, 456, 789, 1024]

# Data seed is FIXED across all experiments to ensure same data partition
DATA_SEED = 42

# fms-ehrs repository path (relative to benchmark repo)
FMS_EHRS_PATH = "../fms-ehrs"

# Default paths (can be overridden via CLI)
DEFAULT_PATHS = {
    "meds_data_dir": "benchmarks/mimic-meds-extraction/data/meds/data",
    "clif_data_dir": "data/clif",  # CLIF-converted data
    "model_dir": "models",
    "meds_tokenizer_config": f"{FMS_EHRS_PATH}/fms_ehrs/config/mimic-meds-ed.yaml",
    "clif_tokenizer_config": f"{FMS_EHRS_PATH}/fms_ehrs/config/clif-21.yaml",
}

# Quantizer to num_bins mapping
QUANTIZER_BINS = {
    "deciles": 10,
    "ventiles": 20,
    "trentiles": 30,
    "centiles": 100,
}

# =============================================================================
# Experiment Configurations
# =============================================================================


@dataclass
class ExperimentConfig:
    """Configuration for a single experimental condition."""

    quantizer: str
    clinical_anchoring: str
    fused: bool
    representation: str
    temporal: str
    data_format: str  # meds | clif
    # Derived fields
    num_bins: int = field(init=False)
    needs_numeric_values: bool = field(init=False)
    needs_times: bool = field(init=False)

    def __post_init__(self):
        self.num_bins = QUANTIZER_BINS.get(self.quantizer, 20)
        self.needs_numeric_values = self.representation in ("soft", "continuous")
        self.needs_times = self.temporal == "time2vec"

    @property
    def config_id(self) -> str:
        """Unique identifier for this config."""
        return (
            f"{self.data_format}_{self.quantizer}_{self.clinical_anchoring}_"
            f"fused{self.fused}_{self.representation}_{self.temporal}"
        )


# Experiment 1: Granularity (12 configs)
# 6 valid quantizer+anchoring combos × 2 fused options
# All use discrete representation + time_tokens
EXP1_CONFIGS = [
    # Deciles (no clinical anchoring possible)
    ExperimentConfig("deciles", "none", False, "discrete", "time_tokens", "meds"),
    ExperimentConfig("deciles", "none", True, "discrete", "time_tokens", "meds"),
    # Ventiles (population-based)
    ExperimentConfig("ventiles", "none", False, "discrete", "time_tokens", "meds"),
    ExperimentConfig("ventiles", "none", True, "discrete", "time_tokens", "meds"),
    # Ventiles (clinically-anchored 5-10-5)
    ExperimentConfig("ventiles", "5-10-5", False, "discrete", "time_tokens", "meds"),
    ExperimentConfig("ventiles", "5-10-5", True, "discrete", "time_tokens", "meds"),
    # Trentiles (population-based)
    ExperimentConfig("trentiles", "none", False, "discrete", "time_tokens", "meds"),
    ExperimentConfig("trentiles", "none", True, "discrete", "time_tokens", "meds"),
    # Trentiles (clinically-anchored 10-10-10)
    ExperimentConfig("trentiles", "10-10-10", False, "discrete", "time_tokens", "meds"),
    ExperimentConfig("trentiles", "10-10-10", True, "discrete", "time_tokens", "meds"),
    # Centiles (no clinical anchoring possible)
    ExperimentConfig("centiles", "none", False, "discrete", "time_tokens", "meds"),
    ExperimentConfig("centiles", "none", True, "discrete", "time_tokens", "meds"),
]


# Experiment 2: Representation Mechanics (6 configs)
# 3 representation × 2 temporal
# Uses Exp1 winner's quantizer/anchoring (placeholder: ventiles 5-10-5)
# Soft/continuous REQUIRE unfused tokenization
EXP2_CONFIGS = [
    # Discrete + time tokens (baseline, can use fused from Exp1 winner)
    ExperimentConfig("ventiles", "5-10-5", False, "discrete", "time_tokens", "meds"),
    # Discrete + time2vec
    ExperimentConfig("ventiles", "5-10-5", False, "discrete", "time2vec", "meds"),
    # Soft discretization + time tokens (requires unfused)
    ExperimentConfig("ventiles", "5-10-5", False, "soft", "time_tokens", "meds"),
    # Soft discretization + time2vec (requires unfused)
    ExperimentConfig("ventiles", "5-10-5", False, "soft", "time2vec", "meds"),
    # Continuous encoder + time tokens (requires unfused)
    ExperimentConfig("ventiles", "5-10-5", False, "continuous", "time_tokens", "meds"),
    # Continuous encoder + time2vec (requires unfused)
    ExperimentConfig("ventiles", "5-10-5", False, "continuous", "time2vec", "meds"),
]


# Experiment 3: Data Format / Vocabulary Semantics (2 configs)
# Compares MEDS vs CLIF on same ICU cohort
# Uses Exp2 winner (placeholder: soft + time2vec)
EXP3_CONFIGS = [
    # MEDS format (native MIMIC vocabulary)
    ExperimentConfig("ventiles", "5-10-5", False, "soft", "time2vec", "meds"),
    # CLIF format (standardized vocabulary)
    ExperimentConfig("ventiles", "5-10-5", False, "soft", "time2vec", "clif"),
]


# =============================================================================
# Job Generation Functions
# =============================================================================


def get_data_version(config: ExperimentConfig) -> str:
    """Generate data version string for this config."""
    fused_str = "fused" if config.fused else "unfused"
    return f"{config.quantizer}_{config.clinical_anchoring}_{fused_str}"


def get_tokenizer_config(config: ExperimentConfig) -> str:
    """Get tokenizer config path for this config."""
    if config.data_format == "clif":
        base = DEFAULT_PATHS["clif_tokenizer_config"]
        # Use fused config if needed
        if config.fused:
            return base.replace("clif-21.yaml", "clif-21-fused.yaml")
        return base
    else:
        return DEFAULT_PATHS["meds_tokenizer_config"]


def get_data_dir(config: ExperimentConfig) -> str:
    """Get data directory for this config."""
    if config.data_format == "clif":
        return DEFAULT_PATHS["clif_data_dir"]
    return DEFAULT_PATHS["meds_data_dir"]


def generate_exp1_job(
    config: ExperimentConfig, seed: int, job_idx: int
) -> str:
    """Generate Exp1 job script (discrete representation, time_tokens).

    Pipeline: tokenize -> tune_model -> extract_outcomes -> fine_tune_classification
    """
    data_version = get_data_version(config)
    tokenizer_config = get_tokenizer_config(config)
    data_dir = get_data_dir(config)
    model_dir = DEFAULT_PATHS["model_dir"]

    job_name = f"exp1_{config.config_id}_s{seed}"

    script = f"""
# Job {job_idx}: {job_name}
echo "=== Starting {job_name} ==="

# Step 1: Tokenize data
python {FMS_EHRS_PATH}/fms_ehrs/scripts/tokenize_w_config.py \\
    --data_dir {data_dir} \\
    --data_version_in raw \\
    --data_version_out {data_version} \\
    --config_loc {tokenizer_config} \\
    --include_24h_cut

# Step 2: Pretrain model (uses tune_model for Exp1 discrete)
python {FMS_EHRS_PATH}/fms_ehrs/scripts/tune_model.py \\
    --data_dir {data_dir} \\
    --data_version {data_version} \\
    --model_dir {model_dir} \\
    --model_version exp1_{config.config_id} \\
    --collation packed \\
    --n_epochs 5 \\
    --wandb_project input-rep-benchmark-exp1

# Step 3: Extract outcomes
python scripts/extract_outcomes_meds.py \\
    --meds_data_dir {data_dir} \\
    --tokenized_data_dir {data_dir}/{data_version}-tokenized \\
    --output_dir {data_dir}/{data_version}_first_24h-tokenized

# Step 4: Fine-tune for each outcome
for outcome in same_admission_death long_length_of_stay icu_admission imv_event; do
    python {FMS_EHRS_PATH}/fms_ehrs/scripts/fine_tune_classification.py \\
        --model_loc {model_dir}/exp1_{config.config_id}-* \\
        --data_dir {data_dir} \\
        --data_version {data_version}_first_24h \\
        --out_dir {model_dir}/classifiers \\
        --outcome $outcome \\
        --n_epochs 5 \\
        --wandb_project input-rep-benchmark-exp1-classify
done

echo "=== Completed {job_name} ==="
"""
    return script


def generate_exp2_job(
    config: ExperimentConfig, seed: int, job_idx: int
) -> str:
    """Generate Exp2 job script (representation mechanics).

    Pipeline: tokenize -> train_representation -> extract_outcomes -> fine_tune_classification
    """
    data_version = get_data_version(config)
    tokenizer_config = get_tokenizer_config(config)
    data_dir = get_data_dir(config)
    model_dir = DEFAULT_PATHS["model_dir"]

    job_name = f"exp2_{config.config_id}_s{seed}"

    script = f"""
# Job {job_idx}: {job_name}
echo "=== Starting {job_name} ==="

# Step 1: Tokenize data (with numeric_values if needed)
python {FMS_EHRS_PATH}/fms_ehrs/scripts/tokenize_w_config.py \\
    --data_dir {data_dir} \\
    --data_version_in raw \\
    --data_version_out {data_version} \\
    --config_loc {tokenizer_config} \\
    --include_24h_cut

# Step 2: Train with representation mechanics
python {FMS_EHRS_PATH}/fms_ehrs/scripts/train_representation.py \\
    --data_dir {data_dir} \\
    --data_version {data_version} \\
    --model_dir {model_dir} \\
    --model_version exp2_{config.config_id} \\
    --representation {config.representation} \\
    --temporal {config.temporal} \\
    --num_bins {config.num_bins} \\
    --n_epochs 5 \\
    --seed {seed} \\
    --wandb_project input-rep-benchmark-exp2

# Step 3: Extract outcomes
python scripts/extract_outcomes_meds.py \\
    --meds_data_dir {data_dir} \\
    --tokenized_data_dir {data_dir}/{data_version}-tokenized \\
    --output_dir {data_dir}/{data_version}_first_24h-tokenized

# Step 4: Fine-tune for each outcome
for outcome in same_admission_death long_length_of_stay icu_admission imv_event; do
    python {FMS_EHRS_PATH}/fms_ehrs/scripts/fine_tune_classification.py \\
        --model_loc {model_dir}/exp2_{config.config_id}-*/model-{config.representation}-{config.temporal} \\
        --data_dir {data_dir} \\
        --data_version {data_version}_first_24h \\
        --out_dir {model_dir}/classifiers \\
        --outcome $outcome \\
        --n_epochs 5 \\
        --wandb_project input-rep-benchmark-exp2-classify
done

echo "=== Completed {job_name} ==="
"""
    return script


def generate_exp3_job(
    config: ExperimentConfig, seed: int, job_idx: int
) -> str:
    """Generate Exp3 job script (data format comparison).

    Same as Exp2 but compares MEDS vs CLIF formats.
    """
    data_version = get_data_version(config)
    tokenizer_config = get_tokenizer_config(config)
    data_dir = get_data_dir(config)
    model_dir = DEFAULT_PATHS["model_dir"]

    job_name = f"exp3_{config.config_id}_s{seed}"

    script = f"""
# Job {job_idx}: {job_name}
echo "=== Starting {job_name} ==="

# Step 1: Tokenize data
python {FMS_EHRS_PATH}/fms_ehrs/scripts/tokenize_w_config.py \\
    --data_dir {data_dir} \\
    --data_version_in raw \\
    --data_version_out {data_version} \\
    --config_loc {tokenizer_config} \\
    --include_24h_cut

# Step 2: Normalize layout (MEDS -> fms-ehrs compatible)
python scripts/normalize_meds_tokenized_layout.py \\
    --input_dir {data_dir}/{data_version}-tokenized \\
    --output_dir {data_dir}/{data_version}-tokenized-norm \\
    --inplace

# Step 3: Train with representation mechanics
python {FMS_EHRS_PATH}/fms_ehrs/scripts/train_representation.py \\
    --data_dir {data_dir} \\
    --data_version {data_version} \\
    --model_dir {model_dir} \\
    --model_version exp3_{config.config_id} \\
    --representation {config.representation} \\
    --temporal {config.temporal} \\
    --num_bins {config.num_bins} \\
    --n_epochs 5 \\
    --seed {seed} \\
    --wandb_project input-rep-benchmark-exp3

# Step 4: Extract outcomes (format-specific)
if [ "{config.data_format}" = "meds" ]; then
    python scripts/extract_outcomes_meds.py \\
        --meds_data_dir {data_dir} \\
        --tokenized_data_dir {data_dir}/{data_version}-tokenized \\
        --output_dir {data_dir}/{data_version}_first_24h-tokenized
else
    python {FMS_EHRS_PATH}/fms_ehrs/scripts/extract_outcomes.py \\
        --data_dir {data_dir} \\
        --ref_version {data_version} \\
        --data_version {data_version}_first_24h
fi

# Step 5: Fine-tune for each outcome
for outcome in same_admission_death long_length_of_stay icu_admission imv_event; do
    python {FMS_EHRS_PATH}/fms_ehrs/scripts/fine_tune_classification.py \\
        --model_loc {model_dir}/exp3_{config.config_id}-*/model-{config.representation}-{config.temporal} \\
        --data_dir {data_dir} \\
        --data_version {data_version}_first_24h \\
        --out_dir {model_dir}/classifiers \\
        --outcome $outcome \\
        --n_epochs 5 \\
        --wandb_project input-rep-benchmark-exp3-classify
done

echo "=== Completed {job_name} ==="
"""
    return script


def generate_job_file(
    configs: List[ExperimentConfig],
    training_seeds: List[int],
    output_path: Path,
    exp_name: str,
    exp_num: int,
) -> int:
    """Generate a job file with all config+seed combinations.

    Args:
        configs: List of experiment configurations
        training_seeds: List of training seeds to use
        output_path: Path to write job file
        exp_name: Name of experiment for logging
        exp_num: Experiment number (1, 2, or 3)

    Returns:
        Number of jobs generated
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Select job generator
    job_generators = {
        1: generate_exp1_job,
        2: generate_exp2_job,
        3: generate_exp3_job,
    }
    generate_job = job_generators[exp_num]

    with open(output_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write(f"# {exp_name} Job File\n")
        f.write(f"# Generated by run_experiments.py\n")
        f.write(f"# Total jobs: {len(configs) * len(training_seeds)}\n")
        f.write(f"# Configs: {len(configs)}, Seeds: {len(training_seeds)}\n")
        f.write("#\n")
        f.write("# Usage:\n")
        f.write(f"#   bash {output_path.name}\n")
        f.write("#   Or submit individual jobs via SLURM array\n")
        f.write("\n")
        f.write("set -e  # Exit on error\n")
        f.write("\n")

        job_idx = 0
        for config in configs:
            for seed in training_seeds:
                job_script = generate_job(config, seed, job_idx)
                f.write(job_script)
                f.write("\n")
                job_idx += 1

    # Make executable
    output_path.chmod(0o755)

    return job_idx


def main():
    parser = argparse.ArgumentParser(
        description="Generate SLURM job files for experiments"
    )
    parser.add_argument(
        "--mode",
        choices=["demo", "full"],
        default="demo",
        help="demo: single seed, full: 5 seeds for statistical robustness",
    )
    parser.add_argument(
        "--exp",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="Generate jobs for specific experiment only (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("slurm"),
        help="Output directory for job files",
    )
    args = parser.parse_args()

    # Select seeds based on mode
    if args.mode == "demo":
        training_seeds = [42]  # Single seed for validation
    else:
        training_seeds = TRAINING_SEEDS  # 5 seeds for statistical robustness

    print("=" * 60)
    print("INPUT REPRESENTATION BENCHMARK - Job Generation")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Training seeds: {training_seeds}")
    print(f"Data seed: {DATA_SEED} (FIXED)")
    print(f"Output directory: {args.output_dir}")
    print(f"fms-ehrs path: {FMS_EHRS_PATH}")
    print()

    total_jobs = 0
    experiments = [1, 2, 3] if args.exp is None else [args.exp]

    for exp in experiments:
        if exp == 1:
            configs = EXP1_CONFIGS
            exp_name = "Experiment 1: Granularity"
        elif exp == 2:
            configs = EXP2_CONFIGS
            exp_name = "Experiment 2: Representation"
        else:
            configs = EXP3_CONFIGS
            exp_name = "Experiment 3: Data Format"

        output_path = args.output_dir / f"exp{exp}_{args.mode}_jobs.sh"
        n_jobs = generate_job_file(
            configs, training_seeds, output_path, exp_name, exp
        )
        total_jobs += n_jobs

        print(f"{exp_name}:")
        print(f"  Configs: {len(configs)}")
        print(f"  Jobs: {n_jobs}")
        print(f"  Output: {output_path}")
        print()

    print("=" * 60)
    print(f"TOTAL JOBS: {total_jobs}")
    print("=" * 60)
    print()
    print("Next steps:")
    print(f"  1. Review job files in {args.output_dir}/")
    print(f"  2. Run locally: bash {args.output_dir}/exp1_{args.mode}_jobs.sh")
    print("  3. Or submit to SLURM: sbatch --array=0-N slurm/run_from_jobfile.sh")


if __name__ == "__main__":
    main()
