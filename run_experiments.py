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
    Exp1 (two-stage):
      - Stage 0 (tokenize+outcomes; one line per config): slurm/04_exp1_tokenize_jobs.sh
      - Stage 1 (train+classify; one line per config×seed):
          - demo: slurm/07_exp1_train_jobs_demo.sh
          - full: slurm/07_exp1_train_jobs_full.sh

    Exp2 (two-stage):
      - Stage 0 (tokenize+outcomes; dedup by data_version): slurm/08_exp2_tokenize_jobs.sh
      - Stage 1 (train+classify; one line per config×seed):
          - demo: slurm/10_exp2_train_jobs_demo.sh
          - full: slurm/10_exp2_train_jobs_full.sh

    Exp3 (two-stage):
      - Stage 0 (tokenize+outcomes; one line per config): slurm/12_exp3_tokenize_jobs.sh
      - Stage 1 (train+classify; one line per config×seed):
          - demo: slurm/14_exp3_train_jobs_demo.sh
          - full: slurm/14_exp3_train_jobs_full.sh

Pipeline per job:
    Exp1 Stage 0: tokenize_w_config.py -> extract_outcomes_meds.py
    Exp1 Stage 1: tune_model.py -> fine_tune_classification.py
    Exp2 Stage 0: tokenize_w_config.py -> extract_outcomes_meds.py
    Exp2 Stage 1: train_representation.py -> fine_tune_classification.py
    Exp3 Stage 0: tokenize_w_config.py -> (optional normalize) -> extract_outcomes (format-specific)
    Exp3 Stage 1: train_representation.py -> fine_tune_classification.py

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
# Compares MEDS vs CLIF on the same ICU cohort.
#
# NOTE on clinical anchoring:
# The base CLIF 2.1 schema (as implemented by clifpy) does not include reference
# range bounds by default (it includes `reference_unit` but not lower/upper bounds).
# To enable clinically anchored binning symmetrically for MEDS vs CLIF (when Exp1
# winner uses anchoring), we must augment CLIF lab tables with per-row
# `ref_range_lower` / `ref_range_upper` columns derived from MIMIC lab metadata.
# See: scripts/augment_clif_labs_with_ref_ranges.py
#
# Uses Exp2 winner for (representation, temporal) as a placeholder (update after Exp2).
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
    # IMPORTANT: Tokenization differs between `time_tokens` (inserts spacing tokens)
    # and `time2vec` (no spacing tokens; temporal info comes from Time2Vec).
    # If we do not include `temporal` here, time_tokens/time2vec variants will write
    # to the same <data_version>-tokenized directory and race/overwrite each other.
    return f"{config.quantizer}_{config.clinical_anchoring}_{fused_str}_{config.temporal}"


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


def generate_exp1_stage0_job_file(
    configs: List[ExperimentConfig],
    output_path: Path,
) -> int:
    """
    Generate Exp1 Stage 0 job file (tokenize + outcomes), one line per config.

    Stage 0 is deterministic and does NOT depend on the training seed, so it is
    generated once and reused for both demo/full runs.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("# Exp1 Stage 0: tokenization + outcomes (one command per config)\n")
        f.write("# Generated by run_experiments.py\n")
        f.write("# Submit with:\n")
        f.write("#   sbatch --array=0-11%2 slurm/02_run_from_jobfile_cpu.sh slurm/04_exp1_tokenize_jobs.sh\n")
        f.write("\n")

        for config in configs:
            data_version = get_data_version(config)
            include_ref_ranges = "true" if config.clinical_anchoring != "none" else "false"
            include_time_tokens = "true" if config.temporal == "time_tokens" else "false"
            fused_flag = "true" if config.fused else "false"

            f.write(
                "bash slurm/03_stage0_tokenize_and_outcomes_meds.sh "
                f"--data_version_out {data_version} "
                f"--quantizer {config.quantizer} "
                f"--clinical_anchoring {config.clinical_anchoring} "
                f"--include_ref_ranges {include_ref_ranges} "
                f"--fused_category_values {fused_flag} "
                f"--include_time_spacing_tokens {include_time_tokens}\n"
            )

    output_path.chmod(0o755)
    return len(configs)


def generate_exp1_stage1_job_file(
    configs: List[ExperimentConfig],
    training_seeds: List[int],
    output_path: Path,
    *,
    mode: str,
) -> int:
    """
    Generate Exp1 Stage 1 job file (train + classify), one line per config×seed.

    Stage 1 assumes Stage 0 already produced tokenized data + outcomes.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("# Exp1 Stage 1: train + classify (one command per config×seed)\n")
        f.write("# Generated by run_experiments.py\n")
        f.write(f"# Mode: {mode}\n")
        f.write("# Submit with:\n")
        if mode == "demo":
            f.write("#   sbatch --array=0-11%8 slurm/run_from_jobfile.sh slurm/07_exp1_train_jobs_demo.sh\n")
        else:
            f.write("#   sbatch --array=0-59%8 slurm/run_from_jobfile.sh slurm/07_exp1_train_jobs_full.sh\n")
        f.write("\n")

        for config in configs:
            data_version = get_data_version(config)
            for seed in training_seeds:
                f.write(
                    "bash slurm/06_exp1_stage1_train_and_classify.sh "
                    f"--config_id {config.config_id} "
                    f"--data_version {data_version} "
                    f"--seed {seed}\n"
                )

    output_path.chmod(0o755)
    return len(configs) * len(training_seeds)


def generate_exp2_stage0_job_file(
    configs: List[ExperimentConfig],
    output_path: Path,
) -> int:
    """Generate Exp2 Stage 0 jobfile (tokenize + outcomes ONCE per unique data_version).

    In Exp2, multiple representation variants share the same tokenization output
    directory (tokenization does not depend on representation). If we do not
    deduplicate, SLURM arrays can race/overwrite the same <data_version>-tokenized
    directories.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    commands: list[str] = []
    seen: set[tuple[str, str]] = set()

    for config in configs:
        data_version = get_data_version(config)
        data_dir = get_data_dir(config)
        key = (str(data_dir), data_version)
        if key in seen:
            continue
        seen.add(key)

        include_ref_ranges = (
            "true" if config.clinical_anchoring != "none" else "false"
        )
        include_time_tokens = (
            "true" if config.temporal == "time_tokens" else "false"
        )

        # Exp2 uses unfused tokenization to support soft/continuous encoders.
        commands.append(
            "bash slurm/03_stage0_tokenize_and_outcomes_meds.sh "
            f"--data_version_out {data_version} "
            f"--quantizer {config.quantizer} "
            f"--clinical_anchoring {config.clinical_anchoring} "
            f"--include_ref_ranges {include_ref_ranges} "
            "--fused_category_values false "
            f"--include_time_spacing_tokens {include_time_tokens}"
        )

    with open(output_path, "w") as f:
        f.write("# Exp2 Stage 0: tokenization + outcomes (deduplicated by data_version)\n")
        f.write("# Generated by run_experiments.py\n")
        f.write("# Submit with:\n")
        f.write(
            f"#   sbatch --array=0-{len(commands) - 1}%2 slurm/02_run_from_jobfile_cpu.sh {output_path}\n"
        )
        f.write("\n")
        for cmd in commands:
            f.write(cmd + "\n")

    output_path.chmod(0o755)
    return len(commands)


def generate_exp2_stage1_job_file(
    configs: List[ExperimentConfig],
    training_seeds: List[int],
    output_path: Path,
    *,
    mode: str,
) -> int:
    """Generate Exp2 Stage 1 jobfile (train+classify per config×seed)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    commands: list[str] = []
    for config in configs:
        data_version = get_data_version(config)
        for seed in training_seeds:
            commands.append(
                "bash slurm/09_exp2_stage1_train_and_classify.sh "
                f"--config_id {config.config_id} "
                f"--data_version {data_version} "
                f"--representation {config.representation} "
                f"--temporal {config.temporal} "
                f"--num_bins {config.num_bins} "
                f"--seed {seed}"
            )

    with open(output_path, "w") as f:
        f.write(f"# Exp2 Stage 1 ({mode}): train representation + classify (per config×seed)\n")
        f.write("# Generated by run_experiments.py\n")
        f.write("# Submit with:\n")
        f.write(
            f"#   sbatch --array=0-{len(commands) - 1}%8 slurm/run_from_jobfile.sh {output_path}\n"
        )
        f.write("\n")
        for cmd in commands:
            f.write(cmd + "\n")

    output_path.chmod(0o755)
    return len(commands)


def generate_exp3_stage0_job_file(
    configs: List[ExperimentConfig],
    output_path: Path,
) -> int:
    """Generate Exp3 Stage 0 jobfile (tokenize + outcomes once per config)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    commands: list[str] = []
    for config in configs:
        data_version = get_data_version(config)
        tokenizer_config = get_tokenizer_config(config)
        data_dir = get_data_dir(config)

        include_ref_ranges = (
            "true" if config.clinical_anchoring != "none" else "false"
        )
        include_time_tokens = (
            "true" if config.temporal == "time_tokens" else "false"
        )

        commands.append(
            "bash slurm/11_exp3_stage0_tokenize_and_outcomes.sh "
            f"--data_format {config.data_format} "
            f"--data_dir {data_dir} "
            f"--data_version_out {data_version} "
            f"--tokenizer_config {tokenizer_config} "
            f"--quantizer {config.quantizer} "
            f"--clinical_anchoring {config.clinical_anchoring} "
            f"--include_ref_ranges {include_ref_ranges} "
            f"--include_time_spacing_tokens {include_time_tokens}"
        )

    with open(output_path, "w") as f:
        f.write("# Exp3 Stage 0: tokenization + outcomes (one command per config)\n")
        f.write("# Generated by run_experiments.py\n")
        f.write("# Submit with:\n")
        f.write(
            f"#   sbatch --array=0-{len(commands) - 1}%2 slurm/02_run_from_jobfile_cpu.sh {output_path}\n"
        )
        f.write("\n")
        for cmd in commands:
            f.write(cmd + "\n")

    output_path.chmod(0o755)
    return len(commands)


def generate_exp3_stage1_job_file(
    configs: List[ExperimentConfig],
    training_seeds: List[int],
    output_path: Path,
    *,
    mode: str,
) -> int:
    """Generate Exp3 Stage 1 jobfile (train+classify per config×seed)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    commands: list[str] = []
    for config in configs:
        data_version = get_data_version(config)
        data_dir = get_data_dir(config)
        for seed in training_seeds:
            commands.append(
                "bash slurm/13_exp3_stage1_train_and_classify.sh "
                f"--config_id {config.config_id} "
                f"--data_dir {data_dir} "
                f"--data_version {data_version} "
                f"--representation {config.representation} "
                f"--temporal {config.temporal} "
                f"--num_bins {config.num_bins} "
                f"--seed {seed}"
            )

    with open(output_path, "w") as f:
        f.write(f"# Exp3 Stage 1 ({mode}): train representation + classify (per config×seed)\n")
        f.write("# Generated by run_experiments.py\n")
        f.write("# Submit with:\n")
        f.write(
            f"#   sbatch --array=0-{len(commands) - 1}%8 slurm/run_from_jobfile.sh {output_path}\n"
        )
        f.write("\n")
        for cmd in commands:
            f.write(cmd + "\n")

    output_path.chmod(0o755)
    return len(commands)


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

    # Tokenizer overrides to ensure the job actually matches the experiment condition.
    include_ref_ranges = "true" if config.clinical_anchoring != "none" else "false"
    include_time_tokens = "true"  # Exp1 uses time-spacing tokens (baseline)
    fused_flag = "true" if config.fused else "false"

    script = f"""
# Job {job_idx}: {job_name}
echo "=== Starting {job_name} ==="

# Step 1: Tokenize data
python {FMS_EHRS_PATH}/fms_ehrs/scripts/tokenize_w_config.py \\
    --data_dir {data_dir} \\
    --data_version_in raw \\
    --data_version_out {data_version} \\
    --config_loc {tokenizer_config} \\
    --quantizer {config.quantizer} \\
    --clinical_anchoring {config.clinical_anchoring} \\
    --include_ref_ranges {include_ref_ranges} \\
    --include_time_spacing_tokens {include_time_tokens} \\
    --fused_category_values {fused_flag} \\
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
    --meds_events_dir {data_dir} \\
    --tokenized_dir {data_dir}/{data_version}_first_24h-tokenized \\
    --splits train,val

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

    # Tokenizer overrides:
    # - Soft/continuous require unfused tokenization (enforced by configs)
    # - Temporal axis is clean either/or: time_tokens => spacing tokens ON; time2vec => spacing tokens OFF
    include_ref_ranges = "true" if config.clinical_anchoring != "none" else "false"
    include_time_tokens = "true" if config.temporal == "time_tokens" else "false"

    script = f"""
# Job {job_idx}: {job_name}
echo "=== Starting {job_name} ==="

# Step 1: Tokenize data (with numeric_values if needed)
python {FMS_EHRS_PATH}/fms_ehrs/scripts/tokenize_w_config.py \\
    --data_dir {data_dir} \\
    --data_version_in raw \\
    --data_version_out {data_version} \\
    --config_loc {tokenizer_config} \\
    --quantizer {config.quantizer} \\
    --clinical_anchoring {config.clinical_anchoring} \\
    --include_ref_ranges {include_ref_ranges} \\
    --include_time_spacing_tokens {include_time_tokens} \\
    --fused_category_values false \\
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
    --meds_events_dir {data_dir} \\
    --tokenized_dir {data_dir}/{data_version}_first_24h-tokenized \\
    --splits train,val

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

    # Tokenizer overrides:
    # - Clean either/or temporal comparison as in Exp2
    include_ref_ranges = "true" if config.clinical_anchoring != "none" else "false"
    include_time_tokens = "true" if config.temporal == "time_tokens" else "false"

    script = f"""
# Job {job_idx}: {job_name}
echo "=== Starting {job_name} ==="

# Step 1: Tokenize data
python {FMS_EHRS_PATH}/fms_ehrs/scripts/tokenize_w_config.py \\
    --data_dir {data_dir} \\
    --data_version_in raw \\
    --data_version_out {data_version} \\
    --config_loc {tokenizer_config} \\
    --quantizer {config.quantizer} \\
    --clinical_anchoring {config.clinical_anchoring} \\
    --include_ref_ranges {include_ref_ranges} \\
    --include_time_spacing_tokens {include_time_tokens} \\
    --fused_category_values false \\
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
        --meds_events_dir {data_dir} \\
        --tokenized_dir {data_dir}/{data_version}_first_24h-tokenized \\
        --splits train,val
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
    exp1_n0 = exp1_n1 = None
    exp2_n0 = exp2_n1 = None
    exp3_n0 = exp3_n1 = None
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

        if exp == 1:
            # Exp1 uses a two-stage SLURM architecture:
            #   Stage 0 (CPU-heavy): tokenize + outcomes ONCE per config
            #   Stage 1 (GPU): train + classify per config×seed
            stage0_path = args.output_dir / "04_exp1_tokenize_jobs.sh"
            stage1_path = args.output_dir / f"07_exp1_train_jobs_{args.mode}.sh"

            n0 = generate_exp1_stage0_job_file(configs, stage0_path)
            n1 = generate_exp1_stage1_job_file(configs, training_seeds, stage1_path, mode=args.mode)
            total_jobs += (n0 + n1)
            exp1_n0, exp1_n1 = n0, n1

            print(f"{exp_name}:")
            print(f"  Configs: {len(configs)}")
            print("  Stage 0 (tokenize+outcomes):")
            print(f"    Jobs: {n0}")
            print(f"    Output: {stage0_path}")
            print("  Stage 1 (train+classify):")
            print(f"    Jobs: {n1}")
            print(f"    Output: {stage1_path}")
            print()
        elif exp == 2:
            stage0_path = args.output_dir / "08_exp2_tokenize_jobs.sh"
            stage1_path = args.output_dir / f"10_exp2_train_jobs_{args.mode}.sh"

            n0 = generate_exp2_stage0_job_file(configs, stage0_path)
            n1 = generate_exp2_stage1_job_file(configs, training_seeds, stage1_path, mode=args.mode)
            total_jobs += (n0 + n1)
            exp2_n0, exp2_n1 = n0, n1

            print(f"{exp_name}:")
            print(f"  Configs: {len(configs)}")
            print("  Stage 0 (tokenize+outcomes; dedup by data_version):")
            print(f"    Jobs: {n0}")
            print(f"    Output: {stage0_path}")
            print("  Stage 1 (train+classify):")
            print(f"    Jobs: {n1}")
            print(f"    Output: {stage1_path}")
            print()
        else:
            stage0_path = args.output_dir / "12_exp3_tokenize_jobs.sh"
            stage1_path = args.output_dir / f"14_exp3_train_jobs_{args.mode}.sh"

            n0 = generate_exp3_stage0_job_file(configs, stage0_path)
            n1 = generate_exp3_stage1_job_file(configs, training_seeds, stage1_path, mode=args.mode)
            total_jobs += (n0 + n1)
            exp3_n0, exp3_n1 = n0, n1

            print(f"{exp_name}:")
            print(f"  Configs: {len(configs)}")
            print("  Stage 0 (tokenize+outcomes):")
            print(f"    Jobs: {n0}")
            print(f"    Output: {stage0_path}")
            print("  Stage 1 (train+classify):")
            print(f"    Jobs: {n1}")
            print(f"    Output: {stage1_path}")
            print()

    print("=" * 60)
    print(f"TOTAL JOBS: {total_jobs}")
    print("=" * 60)
    print()
    print("Next steps:")
    print(f"  1. Review job files in {args.output_dir}/")
    if exp1_n0 is not None:
        print("  2. Exp1 (two-stage):")
        print(f"     - Stage 0 (CPU): sbatch --array=0-{exp1_n0 - 1}%2 slurm/02_run_from_jobfile_cpu.sh slurm/04_exp1_tokenize_jobs.sh")
        print(f"     - Stage 1 (GPU): sbatch --array=0-{exp1_n1 - 1}%8 slurm/run_from_jobfile.sh slurm/07_exp1_train_jobs_{args.mode}.sh")
    if exp2_n0 is not None:
        print("  3. Exp2 (two-stage):")
        print(f"     - Stage 0 (CPU): sbatch --array=0-{exp2_n0 - 1}%2 slurm/02_run_from_jobfile_cpu.sh slurm/08_exp2_tokenize_jobs.sh")
        print(f"     - Stage 1 (GPU): sbatch --array=0-{exp2_n1 - 1}%8 slurm/run_from_jobfile.sh slurm/10_exp2_train_jobs_{args.mode}.sh")
    if exp3_n0 is not None:
        print("  4. Exp3 (two-stage):")
        print(f"     - Stage 0 (CPU): sbatch --array=0-{exp3_n0 - 1}%2 slurm/02_run_from_jobfile_cpu.sh slurm/12_exp3_tokenize_jobs.sh")
        print(f"     - Stage 1 (GPU): sbatch --array=0-{exp3_n1 - 1}%8 slurm/run_from_jobfile.sh slurm/14_exp3_train_jobs_{args.mode}.sh")


if __name__ == "__main__":
    main()
