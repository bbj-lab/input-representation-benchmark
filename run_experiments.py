#!/usr/bin/env python3
"""
Experiment Job Generation for Input Representation Benchmark.

This script generates SLURM job files that call fms-ehrs scripts for all three
experiments. Jobs are generated with valid parameter combinations and proper
dependencies between experiments.

Usage:
    # Generate demo job files (single-seed runs for validation)
    python run_experiments.py --mode demo

    # Generate for specific experiment
    python run_experiments.py --mode demo --exp 1
    python run_experiments.py --mode demo --exp 2
    python run_experiments.py --mode demo --exp 3

Output:
    This script writes generated jobfiles into:
      slurm/generated/<mode>/

    Each experiment is a 4-stage pipeline:
      - Stage 0 (CPU tier2q): tokenize + outcomes
      - Stage 1 (GPU gpuq): train / tune
      - Stage 2 (GPU gpuq): extract representations (1 GPU)
      - Stage 3 (CPU tier2q): logistic regression on extracted reps

    Example outputs (demo mode):
      - slurm/generated/demo/04_exp1_stage0_tokenize.jobfile
      - slurm/generated/demo/07a_exp1_stage1_train.jobfile
      - slurm/generated/demo/07b_exp1_stage2_extract_reps.jobfile
      - slurm/generated/demo/07c_exp1_stage3_lr.jobfile

Pipeline per experiment (high level):
    - Stage 0: tokenize_w_config.py (via Stage0 scripts) + outcome joining
    - Stage 1: Exp1 uses tune_model.py (packed training); Exp2/3 use train_representation.py
    - Stage 2: extract_hidden_states.py (1-GPU torchrun)
    - Stage 3: transfer_rep_based_preds.py --classifier logistic_regression (CPU-only)

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

# Training seeds:
# This benchmark is standardized on a single training seed per configuration and uses
# bootstrap confidence intervals downstream for uncertainty quantification.
DEFAULT_TRAINING_SEEDS = [42]

# Data seed is FIXED across all experiments to ensure same data partition
DATA_SEED = 42

# fms-ehrs repository path (relative to benchmark repo)
FMS_EHRS_PATH = "../fms-ehrs"
RUN_ARTIFACTS_PATH = "artifacts/runs"
TOKENIZED_DATASET_ID = "mimiciv-3.1_meds_70-10-20"
EXP12_TOKENIZED_DIR = f"{RUN_ARTIFACTS_PATH}/tokenized/{TOKENIZED_DATASET_ID}"

# Default paths (can be overridden via CLI)
DEFAULT_PATHS = {
    "meds_data_dir": "benchmarks/mimic-meds-extraction/data/meds/data",
    "exp12_tokenized_dir": EXP12_TOKENIZED_DIR,
    "model_dir": f"{RUN_ARTIFACTS_PATH}/models",
    "meds_tokenizer_config": f"{FMS_EHRS_PATH}/fms_ehrs/config/mimic-meds-ed.yaml",
    # Exp3 arms (data artifacts produced by scripts/align_cohorts.py,
    # scripts/split_meds_by_hadm_splits.py, scripts/build_exp3_meds_semantics_arms.py)
    "exp3_meds_icu_data_dir": f"{RUN_ARTIFACTS_PATH}/exp3/meds_icu",
    "exp3_meds_mapped_data_dir": f"{RUN_ARTIFACTS_PATH}/exp3/arms/meds_mapped",
    "exp3_meds_randomized_data_dir": f"{RUN_ARTIFACTS_PATH}/exp3/arms/meds_randomized",
    "exp3_meds_freqmatched_data_dir": f"{RUN_ARTIFACTS_PATH}/exp3/arms/meds_freqmatched",
    "exp3_meds_tokenizer_config": f"{FMS_EHRS_PATH}/fms_ehrs/config/mimic-meds-exp3-icu.yaml",
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
    data_format: str  # meds | meds_icu | meds_mapped | meds_randomized | meds_freqmatched
    # Derived fields
    num_bins: int = field(init=False)
    needs_numeric_values: bool = field(init=False)

    def __post_init__(self):
        self.num_bins = QUANTIZER_BINS.get(self.quantizer, 20)
        self.needs_numeric_values = self.representation in ("soft", "xval")

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


# Experiment 2: Representation Mechanics
# -------------------------------------
# NOTE (important for internal validity + scheduling):
# Exp2 is downstream of Exp1 because Exp1 chooses the *tokenization regime*:
#   (quantizer + clinical anchoring + fused/unfused + time spacing tokens).
#
# In this benchmark implementation, BOTH soft discretization and continuous encoding
# depend on the Exp1 binning regime:
# - soft discretization interpolates between adjacent bins and therefore requires
#   the Exp1 bin boundaries (stored in vocab.aux).
# - xVal requires per-code normalization statistics (numeric_stats.json) computed on
#   the training split, and uses the same Exp1 winner regime for event inclusion and
#   time tokenization. (xVal tokenization differs at numeric positions: [NUM].)
#
# As a result, Exp2 has no "immune" subset in this pipeline: Exp2 Stage0/Stage1
# should only be generated after selecting the appropriate Exp1 winner(s).
#
# Soft/xVal REQUIRE unfused tokenization.
def _parse_config_id(config_id: str) -> dict:
    """
    Parse ExperimentConfig.config_id back into its components.

    Expected format:
      <data_format>_<quantizer>_<clinical_anchoring>_fused<True|False>_<representation>_<temporal>

    Example:
      meds_ventiles_5-10-5_fusedFalse_soft_time_rope
    """
    parts = config_id.split("_")
    # NOTE: `temporal` includes an underscore for the default condition (`time_tokens`),
    # so config_id may have more than 6 underscore-separated parts. We parse from the
    # left for the fixed prefix and treat the remainder as `temporal`.
    if len(parts) < 6:
        raise ValueError(
            f"Invalid config_id format (expected at least 6 '_' separated fields): {config_id}"
        )
    data_format, quantizer, clinical_anchoring, fused_part, representation = parts[:5]
    temporal = "_".join(parts[5:])
    if not fused_part.startswith("fused"):
        raise ValueError(f"Invalid fused field in config_id: {config_id}")
    fused_str = fused_part.replace("fused", "", 1)
    if fused_str not in ("True", "False"):
        raise ValueError(f"Invalid fused value in config_id: {config_id}")
    fused = fused_str == "True"
    return {
        "data_format": data_format,
        "quantizer": quantizer,
        "clinical_anchoring": clinical_anchoring,
        "fused": fused,
        "representation": representation,
        "temporal": temporal,
    }


def make_exp2_configs(
    *,
    include_discrete: bool,
    discrete_only: bool,
    include_soft: bool = True,
    include_continuous: bool = True,
    quantizer: str,
    clinical_anchoring: str,
    fused_discrete: bool,
) -> list[ExperimentConfig]:
    configs: list[ExperimentConfig] = []
    if discrete_only and not include_discrete:
        raise ValueError("discrete_only=True requires include_discrete=True")

    if include_discrete:
        configs.extend(
            [
                ExperimentConfig(
                    quantizer,
                    clinical_anchoring,
                    fused_discrete,
                    "discrete",
                    "time_tokens",
                    "meds",
                ),
                ExperimentConfig(
                    quantizer,
                    clinical_anchoring,
                    fused_discrete,
                    "discrete",
                    "time_rope",
                    "meds",
                ),
            ]
        )
    if not discrete_only:
        if include_soft:
            configs.extend(
                [
                    ExperimentConfig(quantizer, clinical_anchoring, False, "soft", "time_tokens", "meds"),
                    ExperimentConfig(quantizer, clinical_anchoring, False, "soft", "time_rope", "meds"),
                ]
            )
        if include_continuous:
            configs.extend(
                [
                    ExperimentConfig(quantizer, clinical_anchoring, False, "xval", "time_tokens", "meds"),
                    ExperimentConfig(quantizer, clinical_anchoring, False, "xval", "time_rope", "meds"),
                ]
            )
    return configs


# Experiment 3: Data Format / Vocabulary Semantics
# ------------------------------------------------
# Exp3 uses a fixed ICU-admission hospitalization cohort H_ICU:
# admissions (hadm_id) with hospital LOS>=24h (hosp/admissions admittime/dischtime)
# AND >=1 linked ICU stay record in icu/icustays for the same hadm_id.
#
# NOTE (optional CLIF-based validation):
# This benchmark's primary Exp3 arms are MEDS-only (native + mapped + null controls) and do not
# require CLIF parquet tables. If you additionally run orthogonal MEDS↔CLIF parity checks, be
# aware that the base CLIF 2.1 schema (as implemented by clifpy) does not include reference
# range bounds by default (it includes `reference_unit` but not lower/upper bounds). For
# clinically anchored quantization parity, CLIF labs may need per-row `ref_range_lower` /
# `ref_range_upper` columns derived from MIMIC metadata; see `scripts/augment_clif_labs_with_ref_ranges.py`.
#
# IMPORTANT:
# Exp3 should be generated only after selecting an Exp2 winner (representation + temporal),
# and after selecting an Exp1 winner tokenization regime (quantizer + clinical anchoring + fused/unfused).
#
# Therefore Exp3 configs are generated by function given the selected settings.
def make_exp3_configs(
    *,
    representation: str,
    temporal: str,
    quantizer: str,
    clinical_anchoring: str,
    fused: bool,
) -> list[ExperimentConfig]:
    if representation in ("soft", "xval", "xval_affine") and fused:
        raise ValueError("Exp3 soft/xval requires unfused tokenization (fused=False).")
    return [
        ExperimentConfig(quantizer, clinical_anchoring, fused, representation, temporal, "meds_icu"),
        ExperimentConfig(quantizer, clinical_anchoring, fused, representation, temporal, "meds_mapped"),
        ExperimentConfig(quantizer, clinical_anchoring, fused, representation, temporal, "meds_randomized"),
        ExperimentConfig(quantizer, clinical_anchoring, fused, representation, temporal, "meds_freqmatched"),
]


# =============================================================================
# Job Generation Functions
# =============================================================================


def get_data_version(config: ExperimentConfig) -> str:
    """Generate data version string for this config."""
    fused_str = "fused" if config.fused else "unfused"
    # IMPORTANT: Tokenization differs between `time_tokens` (inserts spacing tokens)
    # and `time_rope` (no spacing tokens; temporal info comes from relative_times).
    # If we do not include `temporal` here, time_tokens/time_rope variants will write
    # to the same <data_version>-tokenized directory and race/overwrite each other.
    numeric_encoding = "_numencxval" if str(config.representation).startswith("xval") else ""
    return f"{config.quantizer}_{config.clinical_anchoring}_{fused_str}_{config.temporal}{numeric_encoding}"


def get_tokenizer_config(config: ExperimentConfig) -> str:
    """Get tokenizer config path for this config."""
    if config.data_format in ("meds_icu", "meds_mapped", "meds_randomized", "meds_freqmatched"):
        return DEFAULT_PATHS["exp3_meds_tokenizer_config"]
    else:
        return DEFAULT_PATHS["meds_tokenizer_config"]


def get_data_dir(config: ExperimentConfig) -> str:
    """Get data directory for this config."""
    if config.data_format == "meds_icu":
        return DEFAULT_PATHS["exp3_meds_icu_data_dir"]
    if config.data_format == "meds_mapped":
        return DEFAULT_PATHS["exp3_meds_mapped_data_dir"]
    if config.data_format == "meds_randomized":
        return DEFAULT_PATHS["exp3_meds_randomized_data_dir"]
    if config.data_format == "meds_freqmatched":
        return DEFAULT_PATHS["exp3_meds_freqmatched_data_dir"]
    return DEFAULT_PATHS["meds_data_dir"]


def get_tokenized_data_dir(config: ExperimentConfig) -> str:
    """Get canonical tokenized-data root for Stage1/2/3 consumption."""
    if config.data_format in ("meds_icu", "meds_mapped", "meds_randomized", "meds_freqmatched"):
        return get_data_dir(config)
    return DEFAULT_PATHS["exp12_tokenized_dir"]


def generate_exp1_stage0_job_file(
    configs: List[ExperimentConfig],
    output_path: Path,
) -> int:
    """
    Generate Exp1 Stage 0 job file (tokenize + outcomes), one line per config.

    Stage 0 is deterministic and does NOT depend on the training seed, so it is
    generated once and reused across reruns.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("# Exp1 Stage 0: tokenization + outcomes (one command per config)\n")
        f.write("# Generated by run_experiments.py\n")
        f.write("# Submit with:\n")
        f.write(f"#   sbatch --array=0-11 slurm/02_run_stage0_tier2q_tokenize.sh {output_path}\n")
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
                f"--include_time_spacing_tokens {include_time_tokens} "
                "--max_padded_len ${IRB_MAX_PADDED_LEN}\n"
            )

    output_path.chmod(0o755)
    return len(configs)


def _exp1_eval_data_version(config: ExperimentConfig, *, eval_max_padded_len: int) -> str:
    """Evaluation-time tokenization version for Exp1 (fixed vocab; larger max_padded_len)."""
    return f"{get_data_version(config)}_evalL{int(eval_max_padded_len)}"


def generate_exp1_stage0_eval_job_file(
    configs: List[ExperimentConfig],
    output_path: Path,
    *,
    eval_max_padded_len: int,
) -> int:
    """Generate Exp1 Stage 0E: retokenize for evaluation with fixed vocab + larger context.

    This does NOT affect Stage 1 pretraining. It creates a parallel tokenized dataset version
    (<orig>_evalL<maxlen>) used only for Stage 2 extraction and Stage 3 LR evaluation.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("# Exp1 Stage 0E: evaluation retokenization (fixed vocab + larger max_padded_len)\n")
        f.write("# Generated by run_experiments.py\n")
        f.write(f"# eval_max_padded_len: {int(eval_max_padded_len)}\n")
        f.write("# Submit with:\n")
        f.write(f"#   sbatch --array=0-11 slurm/02_run_stage0_tier2q_tokenize.sh {output_path}\n")
        f.write("\n")

        for config in configs:
            dv_train = get_data_version(config)
            dv_eval = _exp1_eval_data_version(config, eval_max_padded_len=eval_max_padded_len)
            include_ref_ranges = "true" if config.clinical_anchoring != "none" else "false"
            include_time_tokens = "true" if config.temporal == "time_tokens" else "false"
            fused_flag = "true" if config.fused else "false"

            # IMPORTANT: Use the *existing* training vocab for this condition so token ids match the pretrained model.
            vocab_path = f"{DEFAULT_PATHS['exp12_tokenized_dir']}/{dv_train}-tokenized/train/vocab.gzip"

            f.write(
                "bash slurm/03_stage0_tokenize_and_outcomes_meds.sh "
                f"--data_version_out {dv_eval} "
                f"--quantizer {config.quantizer} "
                f"--clinical_anchoring {config.clinical_anchoring} "
                f"--include_ref_ranges {include_ref_ranges} "
                f"--fused_category_values {fused_flag} "
                f"--include_time_spacing_tokens {include_time_tokens} "
                f"--vocab_path \"{vocab_path}\" "
                f"--max_padded_len {int(eval_max_padded_len)} "
                "--only_24h_cut true\n"
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
    """Generate Exp1 Stage 1 job file (FM train + small LR sweep), one line per config×seed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("# Exp1 Stage 1: FM training (packed) + small LR sweep\n")
        f.write("# Generated by run_experiments.py\n")
        f.write(f"# Mode: {mode}\n")
        f.write("# Submit with:\n")
        f.write(
            f"#   sbatch --array=0-{(len(configs) * len(training_seeds)) - 1} "
            f"slurm/05_run_stage1_gpu4_train.sh {output_path}\n"
        )
        f.write("\n")

        for config in configs:
            data_version = get_data_version(config)
            for seed in training_seeds:
                f.write(
                    "bash -lc "
                    + "\""
                    + f"export config_id={config.config_id} "
                    + f"data_version={data_version} "
                    + f"seed={seed} "
                    + f"IRB_TOKENIZED_DATASET_DIR={DEFAULT_PATHS['exp12_tokenized_dir']} "
                    # Do not hard-code epochs in the jobfile; allow overrides via env.
                    + "IRB_EXP1_STAGE1_EPOCHS=${IRB_EXP1_STAGE1_EPOCHS:-1} "
                    + f"MODEL_DIR=${{MODEL_DIR:-{DEFAULT_PATHS['model_dir']}}} "
                    + "FMS_EHRS_HOME=${FMS_EHRS_HOME:-../fms-ehrs} "
                    + "&& bash slurm/04_exp1_stage1_tune_packed.sh"
                    + "\"\n"
                )

    output_path.chmod(0o755)
    return len(configs) * len(training_seeds)


def _exp1_best_model_loc(config: ExperimentConfig, seed: int) -> str:
    # Current canonical artifact tree stores the selected Exp1 best-trial checkpoints
    # under legacy run-N/checkpoint-9000 directories rather than the older -hp- alias.
    run_map = {
        "meds_deciles_none_fusedFalse_discrete_time_tokens": "run-0",
        "meds_deciles_none_fusedTrue_discrete_time_tokens": "run-1",
        "meds_ventiles_none_fusedFalse_discrete_time_tokens": "run-1",
        "meds_ventiles_none_fusedTrue_discrete_time_tokens": "run-2",
        "meds_ventiles_5-10-5_fusedFalse_discrete_time_tokens": "run-1",
        "meds_ventiles_5-10-5_fusedTrue_discrete_time_tokens": "run-0",
        "meds_trentiles_none_fusedFalse_discrete_time_tokens": "run-1",
        "meds_trentiles_none_fusedTrue_discrete_time_tokens": "run-2",
        "meds_trentiles_10-10-10_fusedFalse_discrete_time_tokens": "run-1",
        "meds_trentiles_10-10-10_fusedTrue_discrete_time_tokens": "run-0",
        "meds_centiles_none_fusedFalse_discrete_time_tokens": "run-0",
        "meds_centiles_none_fusedTrue_discrete_time_tokens": "run-0",
    }
    run_dir = run_map[config.config_id]
    return f"$MODEL_DIR/exp1_{config.config_id}-s{seed}/{run_dir}/checkpoint-9000"


def _exp2_model_loc(config: ExperimentConfig, seed: int) -> str:
    # Matches slurm/exp2_stage1_train_representation.sh output naming
    return (
        f"$MODEL_DIR/exp2_{config.config_id}-{config.representation}-{config.temporal}-s{seed}"
        f"/model-{config.representation}-{config.temporal}"
    )


def _exp3_model_loc(config: ExperimentConfig, seed: int) -> str:
    # Current canonical artifact tree uses a legacy `meds_icu_native` prefix for
    # the native Exp3 arm, while the remaining arms match their config ids.
    model_prefix = "meds_icu_native" if config.data_format == "meds_icu" else config.config_id
    return (
        f"$MODEL_DIR/exp3_{model_prefix}-{config.representation}-{config.temporal}-s{seed}"
        f"/model-{config.representation}-{config.temporal}"
    )


def generate_exp1_rep_extract_job_file(
    configs: List[ExperimentConfig],
    training_seeds: List[int],
    output_path: Path,
    *,
    mode: str,
    eval_max_padded_len: int | None = None,
) -> int:
    """Exp1 Option-B: extract reps (1 GPU) for each trained FM (config×seed)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("# Exp1 Option-B: extract representations (1-GPU torchrun)\n")
        f.write("# Generated by run_experiments.py\n")
        f.write(f"# Mode: {mode}\n\n")
        for config in configs:
            data_dir = get_tokenized_data_dir(config)
            dv = (
                _exp1_eval_data_version(config, eval_max_padded_len=eval_max_padded_len)
                if eval_max_padded_len is not None
                else get_data_version(config)
            )
            for seed in training_seeds:
                f.write(
                    "bash -lc "
                    + "\""
                    + f"export data_dir={data_dir} "
                    + f"data_version={dv}_first_24h "
                    + f"model_loc={_exp1_best_model_loc(config, seed)} "
                    + "FMS_EHRS_HOME=${FMS_EHRS_HOME:-../fms-ehrs} "
                    + "&& bash slurm/ref_qse/09_extract_reps.sh"
                    + "\"\n"
                )
    output_path.chmod(0o755)
    return len(configs) * len(training_seeds)


def generate_exp1_rep_pred_job_file(
    configs: List[ExperimentConfig],
    training_seeds: List[int],
    output_path: Path,
    *,
    mode: str,
    eval_max_padded_len: int | None = None,
) -> int:
    """Exp1 Option-B: logistic regression on extracted reps (CPU) for each trained FM."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("# Exp1 Option-B: logistic regression on extracted reps (CPU)\n")
        f.write("# Generated by run_experiments.py\n")
        f.write(f"# Mode: {mode}\n\n")
        for config in configs:
            data_dir = get_tokenized_data_dir(config)
            dv = (
                _exp1_eval_data_version(config, eval_max_padded_len=eval_max_padded_len)
                if eval_max_padded_len is not None
                else get_data_version(config)
            )
            for seed in training_seeds:
                f.write(
                    "bash -lc "
                    + "\""
                    + f"export data_dir={data_dir} "
                    + f"data_version={dv}_first_24h "
                    + f"model_loc={_exp1_best_model_loc(config, seed)} "
                    + "FMS_EHRS_HOME=${FMS_EHRS_HOME:-../fms-ehrs} "
                    + "&& bash slurm/ref_qse/10_xfer_rep_based_preds.sh"
                    + "\"\n"
                )
    output_path.chmod(0o755)
    return len(configs) * len(training_seeds)


def generate_exp2_rep_extract_job_file(
    configs: List[ExperimentConfig],
    training_seeds: List[int],
    output_path: Path,
    *,
    mode: str,
) -> int:
    """Exp2 Option-B: extract reps (1 GPU) for each trained representation model."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("# Exp2 Option-B: extract representations (1-GPU torchrun)\n")
        f.write("# Generated by run_experiments.py\n")
        f.write(f"# Mode: {mode}\n\n")
        for config in configs:
            data_dir = get_tokenized_data_dir(config)
            dv = get_data_version(config)
            for seed in training_seeds:
                f.write(
                    "bash -lc "
                    + "\""
                    + f"export data_dir={data_dir} "
                    + f"data_version={dv}_first_24h "
                    + f"model_loc={_exp2_model_loc(config, seed)} "
                    + "FMS_EHRS_HOME=${FMS_EHRS_HOME:-../fms-ehrs} "
                    + "&& bash slurm/ref_qse/09_extract_reps.sh"
                    + "\"\n"
                )
    output_path.chmod(0o755)
    return len(configs) * len(training_seeds)


def generate_exp2_rep_pred_job_file(
    configs: List[ExperimentConfig],
    training_seeds: List[int],
    output_path: Path,
    *,
    mode: str,
) -> int:
    """Exp2 Option-B: logistic regression on extracted reps (CPU)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("# Exp2 Option-B: logistic regression on extracted reps (CPU)\n")
        f.write("# Generated by run_experiments.py\n")
        f.write(f"# Mode: {mode}\n\n")
        for config in configs:
            data_dir = get_tokenized_data_dir(config)
            dv = get_data_version(config)
            for seed in training_seeds:
                f.write(
                    "bash -lc "
                    + "\""
                    + f"export data_dir={data_dir} "
                    + f"data_version={dv}_first_24h "
                    + f"model_loc={_exp2_model_loc(config, seed)} "
                    + "FMS_EHRS_HOME=${FMS_EHRS_HOME:-../fms-ehrs} "
                    + "&& bash slurm/ref_qse/10_xfer_rep_based_preds.sh"
                    + "\"\n"
                )
    output_path.chmod(0o755)
    return len(configs) * len(training_seeds)


def generate_exp3_rep_extract_job_file(
    configs: List[ExperimentConfig],
    training_seeds: List[int],
    output_path: Path,
    *,
    mode: str,
) -> int:
    """Exp3 Option-B: extract reps (1 GPU) for each trained model (MEDS-only arms)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("# Exp3 Option-B: extract representations (1-GPU torchrun)\n")
        f.write("# Generated by run_experiments.py\n")
        f.write(f"# Mode: {mode}\n\n")
        for config in configs:
            data_dir = get_data_dir(config)
            dv = get_data_version(config)
            for seed in training_seeds:
                f.write(
                    "bash -lc "
                    + "\""
                    + f"export data_dir={data_dir} "
                    + f"data_version={dv}_first_24h "
                    + f"model_loc={_exp3_model_loc(config, seed)} "
                    + "FMS_EHRS_HOME=${FMS_EHRS_HOME:-../fms-ehrs} "
                    + "&& bash slurm/ref_qse/09_extract_reps.sh"
                    + "\"\n"
                )
    output_path.chmod(0o755)
    return len(configs) * len(training_seeds)


def generate_exp3_rep_pred_job_file(
    configs: List[ExperimentConfig],
    training_seeds: List[int],
    output_path: Path,
    *,
    mode: str,
) -> int:
    """Exp3: logistic regression on extracted reps (CPU), run for each MEDS-only arm."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("# Exp3 Option-B: logistic regression on extracted reps (CPU)\n")
        f.write("# Generated by run_experiments.py\n")
        f.write(f"# Mode: {mode}\n\n")
        for config in configs:
            data_dir = get_data_dir(config)
            dv = get_data_version(config)
            for seed in training_seeds:
                f.write(
                    "bash -lc "
                    + "\""
                    + f"export data_dir={data_dir} "
                    + f"data_version={dv}_first_24h "
                    + f"model_loc={_exp3_model_loc(config, seed)} "
                    # Exp3 uses a fixed ICU-hospitalization cohort H_ICU:
                    # admissions (hadm_id) with LOS>=24h AND >=1 linked ICU stay record in icu/icustays.
                    # Under this cohort restriction, `icu_admission` is typically degenerate.
                    # We evaluate an ICU-relevant replacement endpoint `prolonged_icu_stay` instead.
                    + "IRB_XFER_PRED_ARGS=\\\"--outcomes same_admission_death long_length_of_stay prolonged_icu_stay imv_event\\\" "
                    + "FMS_EHRS_HOME=${FMS_EHRS_HOME:-../fms-ehrs} "
                    + "&& bash slurm/ref_qse/10_xfer_rep_based_preds.sh"
                    + "\"\n"
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

        fused_flag = "true" if config.fused else "false"
        numeric_encoding = "xval" if str(config.representation).startswith("xval") else "quantile"
        commands.append(
            "bash slurm/03_stage0_tokenize_and_outcomes_meds.sh "
            f"--data_version_out {data_version} "
            f"--quantizer {config.quantizer} "
            f"--clinical_anchoring {config.clinical_anchoring} "
            f"--include_ref_ranges {include_ref_ranges} "
            f"--fused_category_values {fused_flag} "
            f"--include_time_spacing_tokens {include_time_tokens} "
            f"--numeric_encoding {numeric_encoding} "
            "--max_padded_len ${IRB_MAX_PADDED_LEN}"
        )

    with open(output_path, "w") as f:
        f.write("# Exp2 Stage 0: tokenization + outcomes (deduplicated by data_version)\n")
        f.write("# Generated by run_experiments.py\n")
        f.write("# Submit with:\n")
        f.write(
            f"#   sbatch --array=0-{len(commands) - 1} slurm/02_run_stage0_tier2q_tokenize.sh {output_path}\n"
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
    """Generate Exp2 Stage 1 jobfile (train representation mechanics per config×seed)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    commands: list[str] = []
    for config in configs:
        data_version = get_data_version(config)
        for seed in training_seeds:
            commands.append(
                f"IRB_TOKENIZED_DATASET_DIR={DEFAULT_PATHS['exp12_tokenized_dir']} "
                "IRB_EXP23_STAGE1_EPOCHS=${IRB_EXP23_STAGE1_EPOCHS:-1} bash slurm/07_exp2_stage1_train_representation.sh "
                f"--config_id {config.config_id} "
                f"--data_version {data_version} "
                f"--representation {config.representation} "
                f"--temporal {config.temporal} "
                f"--num_bins {config.num_bins} "
                f"--seed {seed}"
            )

    with open(output_path, "w") as f:
        f.write(f"# Exp2 Stage 1 ({mode}): train representation (per config×seed)\n")
        f.write("# Generated by run_experiments.py\n")
        f.write("# Submit with:\n")
        f.write(
            f"#   sbatch --array=0-{len(commands) - 1} slurm/06_run_stage1_gpu1_train.sh {output_path}\n"
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
        fused_flag = "true" if config.fused else "false"
        numeric_encoding = "xval" if str(config.representation).startswith("xval") else "quantile"

        cmd = (
            "bash slurm/04_exp3_stage0_tokenize_and_outcomes.sh "
            f"--data_format {config.data_format} "
            f"--data_dir {data_dir} "
            f"--data_version_out {data_version} "
            f"--tokenizer_config {tokenizer_config} "
            "--max_padded_len ${IRB_MAX_PADDED_LEN} "
            f"--quantizer {config.quantizer} "
            f"--clinical_anchoring {config.clinical_anchoring} "
            f"--include_ref_ranges {include_ref_ranges} "
            f"--include_time_spacing_tokens {include_time_tokens} "
            f"--fused_category_values {fused_flag} "
            f"--numeric_encoding {numeric_encoding}"
        )
        commands.append(cmd
        )

    with open(output_path, "w") as f:
        f.write("# Exp3 Stage 0: tokenization + outcomes (one command per config)\n")
        f.write("# Generated by run_experiments.py\n")
        f.write("# Submit with:\n")
        f.write(
            f"#   sbatch --array=0-{len(commands) - 1} slurm/02_run_stage0_tier2q_tokenize.sh {output_path}\n"
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
    """Generate Exp3 Stage 1 jobfile (train representation mechanics per config×seed)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    commands: list[str] = []
    for config in configs:
        data_version = get_data_version(config)
        data_dir = get_data_dir(config)
        for seed in training_seeds:
            commands.append(
                "IRB_EXP23_STAGE1_EPOCHS=${IRB_EXP23_STAGE1_EPOCHS:-1} bash slurm/08_exp3_stage1_train_representation.sh "
                f"--config_id {config.config_id} "
                f"--data_dir {data_dir} "
                f"--data_version {data_version} "
                f"--representation {config.representation} "
                f"--temporal {config.temporal} "
                f"--num_bins {config.num_bins} "
                f"--seed {seed}"
            )

    with open(output_path, "w") as f:
        f.write(f"# Exp3 Stage 1 ({mode}): train representation (per config×seed)\n")
        f.write("# Generated by run_experiments.py\n")
        f.write("# Submit with:\n")
        f.write(
            f"#   sbatch --array=0-{len(commands) - 1} slurm/06_run_stage1_gpu1_train.sh {output_path}\n"
        )
        f.write("\n")
        for cmd in commands:
            f.write(cmd + "\n")

    output_path.chmod(0o755)
    return len(commands)


## NOTE: This repo is standardized on SLURM jobfiles + stage-specific scripts and reps -> LR evaluation.


def main():
    parser = argparse.ArgumentParser(
        description="Generate SLURM job files for experiments"
    )
    parser.add_argument(
        "--mode",
        choices=["demo"],
        default="demo",
        help="demo: single seed per configuration (standardized in this benchmark)",
    )
    parser.add_argument(
        "--training_seeds",
        type=int,
        nargs="+",
        default=DEFAULT_TRAINING_SEEDS,
        help="Training seeds to generate jobs for (default: 42).",
    )
    parser.add_argument(
        "--exp",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="Generate jobs for specific experiment only (default: all)",
    )
    parser.add_argument(
        "--exp1_eval_max_padded_len",
        type=int,
        default=None,
        help=(
            "If set, generate an Exp1 evaluation retokenization Stage0 jobfile that reuses the original "
            "training vocab.gzip but increases max_padded_len, and point Exp1 Stage2/Stage3 at this eval "
            "tokenized dataset (no Exp1 Stage1 rerun). This is the fairness knob for fused vs unfused "
            "comparisons when the intended comparison is token semantics (not token budget/coverage). "
            "Choose this from the observed token-length distribution (see qc/eval_maxlen_analysis.py); "
            "as a practical starting point, 4096 often substantially reduces truncation while keeping "
            "Stage2 extraction tractable."
        ),
    )
    parser.add_argument(
        "--exp2_include_discrete",
        action="store_true",
        help="Include Exp2 discrete baselines (discrete × {time_tokens,time_rope}).",
    )
    parser.add_argument(
        "--exp2_discrete_only",
        action="store_true",
        help="Generate only the two discrete Exp2 configs (to run after Exp1 winner is known).",
    )
    parser.add_argument(
        "--exp2_include_soft",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include Exp2 soft discretization configs (default: true).",
    )
    parser.add_argument(
        "--exp2_include_continuous",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include Exp2 xVal (continuous value) configs (default: true).",
    )
    parser.add_argument(
        "--exp2_from_exp1_config_id",
        type=str,
        default=None,
        help="Use Exp1 overall winner config_id to parameterize Exp2 *discrete* tokenization regime (quantizer, anchoring, fused).",
    )
    parser.add_argument(
        "--exp2_soft_from_exp1_unfused_config_id",
        type=str,
        default=None,
        help="Use Exp1 unfused winner config_id to parameterize Exp2 soft/xVal tokenization regime (quantizer, anchoring; fused is forced False).",
    )
    parser.add_argument(
        "--exp2_quantizer",
        type=str,
        default="ventiles",
        help="DEPRECATED for main runs. Exp2 should be parameterized from Exp1 winner config_id(s).",
    )
    parser.add_argument(
        "--exp2_clinical_anchoring",
        type=str,
        default="5-10-5",
        help="DEPRECATED for main runs. Exp2 should be parameterized from Exp1 winner config_id(s).",
    )
    parser.add_argument(
        "--exp2_discrete_fused",
        action="store_true",
        help="Use fused tokenization for the discrete Exp2 configs (soft/xVal always use unfused).",
    )
    parser.add_argument(
        "--exp3_representation",
        type=str,
        default=None,
        help="(Required for Exp3) Selected Exp2 winner representation: discrete|soft|xval.",
    )
    parser.add_argument(
        "--exp3_temporal",
        type=str,
        default=None,
        help="(Required for Exp3) Selected Exp2 winner temporal: time_tokens|time_rope.",
    )
    parser.add_argument(
        "--exp3_quantizer",
        type=str,
        default=None,
        help="(Required for Exp3) Selected Exp1 winner quantizer: deciles|ventiles|trentiles|centiles.",
    )
    parser.add_argument(
        "--exp3_clinical_anchoring",
        type=str,
        default=None,
        help="(Required for Exp3) Selected Exp1 winner clinical anchoring: none|5-10-5|10-10-10.",
    )
    parser.add_argument(
        "--exp3_fused",
        action="store_true",
        help="(Optional for Exp3) Use fused tokenization in Exp3 (only valid for discrete representation).",
    )
    parser.add_argument(
        "--exp3_from_exp1_config_id",
        type=str,
        default=None,
        help="Use Exp1 winner config_id to populate Exp3 quantizer/anchoring/fused (no placeholders).",
    )
    parser.add_argument(
        "--exp3_from_exp2_config_id",
        type=str,
        default=None,
        help="Use Exp2 winner config_id to populate Exp3 representation/temporal (no placeholders).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("slurm"),
        help="Output directory for job files",
    )
    parser.add_argument(
        "--meds_data_dir",
        type=str,
        default=DEFAULT_PATHS["meds_data_dir"],
        help="MEDS data directory (contains raw/ and <version>-tokenized/).",
    )
    parser.add_argument(
        "--meds_tokenizer_config",
        type=str,
        default=DEFAULT_PATHS["meds_tokenizer_config"],
        help="Tokenizer config for MEDS (YAML). Default: fms-ehrs/config/mimic-meds-ed.yaml",
    )
    parser.add_argument(
        "--exp3_meds_icu_data_dir",
        type=str,
        default=DEFAULT_PATHS["exp3_meds_icu_data_dir"],
        help=(
            "(Exp3) MEDS events directory already filtered to H_ICU and split into train/val/test. "
            "H_ICU is admissions (hadm_id) with hospital LOS>=24h AND >=1 linked ICU stay record "
            "in MIMIC-IV icu/icustays for the same hadm_id."
        ),
    )
    parser.add_argument(
        "--exp3_meds_mapped_data_dir",
        type=str,
        default=DEFAULT_PATHS["exp3_meds_mapped_data_dir"],
        help="(Exp3) MEDS→CLIF mapped events directory (train/val/test).",
    )
    parser.add_argument(
        "--exp3_meds_randomized_data_dir",
        type=str,
        default=DEFAULT_PATHS["exp3_meds_randomized_data_dir"],
        help="(Exp3) MEDS randomized-mapping control events directory (train/val/test).",
    )
    parser.add_argument(
        "--exp3_meds_freqmatched_data_dir",
        type=str,
        default=DEFAULT_PATHS["exp3_meds_freqmatched_data_dir"],
        help="(Exp3) MEDS frequency-matched collapse control events directory (train/val/test).",
    )
    parser.add_argument(
        "--exp3_meds_tokenizer_config",
        type=str,
        default=DEFAULT_PATHS["exp3_meds_tokenizer_config"],
        help="(Exp3) Tokenizer config for MEDS Exp3 ICU matched-signal subset (YAML).",
    )
    args = parser.parse_args()

    # Apply CLI overrides to default paths used by job generation helpers.
    DEFAULT_PATHS["meds_data_dir"] = args.meds_data_dir
    DEFAULT_PATHS["meds_tokenizer_config"] = args.meds_tokenizer_config
    DEFAULT_PATHS["exp3_meds_icu_data_dir"] = args.exp3_meds_icu_data_dir
    DEFAULT_PATHS["exp3_meds_mapped_data_dir"] = args.exp3_meds_mapped_data_dir
    DEFAULT_PATHS["exp3_meds_randomized_data_dir"] = args.exp3_meds_randomized_data_dir
    DEFAULT_PATHS["exp3_meds_freqmatched_data_dir"] = args.exp3_meds_freqmatched_data_dir
    DEFAULT_PATHS["exp3_meds_tokenizer_config"] = args.exp3_meds_tokenizer_config

    # Write generated jobfiles under a mode-specific subdirectory to avoid
    # ambiguous naming collisions and to keep `slurm/`
    # reserved for hand-authored stage scripts and runners.
    generated_dir = args.output_dir / "generated" / args.mode
    generated_dir.mkdir(parents=True, exist_ok=True)

    training_seeds = [int(s) for s in args.training_seeds]

    print("=" * 60)
    print("INPUT REPRESENTATION BENCHMARK - Job Generation")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Training seeds: {training_seeds}")
    print(f"Data seed: {DATA_SEED} (FIXED)")
    print(f"Output directory: {generated_dir}")
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
            # Exp2 depends on two different Exp1 selections:
            # - Discrete baselines should adopt the *overall* Exp1 winner (including fused/unfused).
            # - Soft/xVal require unfused tokenization; therefore their binning must be chosen
            #   from the Exp1 winner within the unfused group.
            discrete_regime = None
            soft_regime = None

            if args.exp2_include_discrete or args.exp2_discrete_only:
                if args.exp2_from_exp1_config_id is None:
                    # Default to deciles/none/fused if not provided
                    discrete_regime = {
                        "quantizer": "deciles",
                        "clinical_anchoring": "none",
                        "fused": False, # Match the soft default for consistency in demo
                        "representation": "discrete",
                        "temporal": "time_tokens",
                    }
                    print("WARNING: --exp2_from_exp1_config_id not set. Defaulting to deciles/none/unfused.")
                else:
                    discrete_regime = _parse_config_id(args.exp2_from_exp1_config_id)

            if not args.exp2_discrete_only:
                if args.exp2_soft_from_exp1_unfused_config_id is None:
                    # Default to deciles/none/unfused if not provided (reasonable default for demo/testing)
                    soft_regime = {
                        "quantizer": "deciles",
                        "clinical_anchoring": "none",
                        "fused": False,
                        "representation": "discrete", # Placeholder, ignored
                        "temporal": "time_tokens", # Placeholder, ignored
                    }
                    print("WARNING: --exp2_soft_from_exp1_unfused_config_id not set. Defaulting to deciles/none/unfused.")
                else:
                    soft_regime = _parse_config_id(args.exp2_soft_from_exp1_unfused_config_id)
                
                if soft_regime["fused"]:
                    raise ValueError(
                        "--exp2_soft_from_exp1_unfused_config_id must be an unfused Exp1 config_id (fused=False)."
                    )

            configs: list[ExperimentConfig] = []
            if args.exp2_include_discrete:
                assert discrete_regime is not None
                configs.extend(
                    make_exp2_configs(
                        include_discrete=True,
                        discrete_only=True,
                        include_soft=False,
                        include_continuous=False,
                        quantizer=discrete_regime["quantizer"],
                        clinical_anchoring=discrete_regime["clinical_anchoring"],
                        fused_discrete=discrete_regime["fused"],
                    )
                )
            if not args.exp2_discrete_only:
                assert soft_regime is not None
                configs.extend(
                    make_exp2_configs(
                        include_discrete=False,
                        discrete_only=False,
                        include_soft=bool(args.exp2_include_soft),
                        include_continuous=bool(args.exp2_include_continuous),
                        quantizer=soft_regime["quantizer"],
                        clinical_anchoring=soft_regime["clinical_anchoring"],
                        fused_discrete=False,
                    )
                )

            exp_name = "Experiment 2: Representation"
        else:
            # Enforce explicit winner selection; no placeholders.
            for k in [
                "exp3_meds_icu_data_dir",
                "exp3_meds_mapped_data_dir",
                "exp3_meds_randomized_data_dir",
                "exp3_meds_freqmatched_data_dir",
            ]:
                p = Path(DEFAULT_PATHS[k])
                if not p.exists():
                    raise ValueError(
                        f"Exp3 requires {k} to exist: {p}. "
                        "Build these artifacts first:\n"
                        "  - scripts/split_meds_by_hadm_splits.py -> exp3_meds_icu_data_dir\n"
                        "  - scripts/build_exp3_meds_semantics_arms.py -> exp3_meds_{mapped,randomized,freqmatched}_data_dir\n"
                    )
            if args.exp3_from_exp1_config_id is not None:
                exp1 = _parse_config_id(args.exp3_from_exp1_config_id)
                exp3_quantizer = exp1["quantizer"]
                exp3_clinical_anchoring = exp1["clinical_anchoring"]
                exp3_fused = exp1["fused"]
            else:
                exp3_quantizer = args.exp3_quantizer
                exp3_clinical_anchoring = args.exp3_clinical_anchoring
                exp3_fused = args.exp3_fused

            if args.exp3_from_exp2_config_id is not None:
                exp2 = _parse_config_id(args.exp3_from_exp2_config_id)
                exp3_representation = exp2["representation"]
                exp3_temporal = exp2["temporal"]
            else:
                exp3_representation = args.exp3_representation
                exp3_temporal = args.exp3_temporal

            missing = [
                k
                for k, v in {
                    "exp3_representation": exp3_representation,
                    "exp3_temporal": exp3_temporal,
                    "exp3_quantizer": exp3_quantizer,
                    "exp3_clinical_anchoring": exp3_clinical_anchoring,
                }.items()
                if v is None
            ]
            if missing:
                raise SystemExit(
                    "Exp3 requires explicit winner selections (no placeholders). "
                    f"Missing: {', '.join(missing)}. "
                    "Provide either:\n"
                    "  - --exp3_from_exp1_config_id and --exp3_from_exp2_config_id\n"
                    "OR:\n"
                    "  - --exp3_representation --exp3_temporal --exp3_quantizer --exp3_clinical_anchoring"
                )
            configs = make_exp3_configs(
                representation=exp3_representation,
                temporal=exp3_temporal,
                quantizer=exp3_quantizer,
                clinical_anchoring=exp3_clinical_anchoring,
                fused=exp3_fused,
            )
            exp_name = "Experiment 3: Data Format"

        if exp == 1:
            # Exp1 (4-stage; Option B):
            #   Stage 0 (CPU): tokenize + outcomes (once per config)
            #   Stage 1 (GPU; 4 GPUs/job): FM training (packed) per config×seed
            #   Stage 2 (GPU; 1 GPU/job): extract reps (per config×seed)
            #   Stage 3 (CPU tier2q): logistic regression on reps (per config×seed)
            stage0_path = generated_dir / "04_exp1_stage0_tokenize.jobfile"
            stage0_eval_path = generated_dir / "04b_exp1_stage0_eval_retokenize.jobfile"
            stage1_path = generated_dir / "07a_exp1_stage1_train.jobfile"
            repx_path = generated_dir / "07b_exp1_stage2_extract_reps.jobfile"
            pred_path = generated_dir / "07c_exp1_stage3_lr.jobfile"

            n0 = generate_exp1_stage0_job_file(configs, stage0_path)
            n1 = generate_exp1_stage1_job_file(configs, training_seeds, stage1_path, mode=args.mode)
            # Optional: evaluation retokenization to equalize event coverage across fused/unfused
            # (removes token-budget truncation confound in Stage2/Stage3; does not redo Stage1).
            exp1_eval_maxlen = args.exp1_eval_max_padded_len
            n0_eval = 0
            if exp1_eval_maxlen is not None:
                n0_eval = generate_exp1_stage0_eval_job_file(
                    configs,
                    stage0_eval_path,
                    eval_max_padded_len=int(exp1_eval_maxlen),
                )
            nrepx = generate_exp1_rep_extract_job_file(
                configs,
                training_seeds,
                repx_path,
                mode=args.mode,
                eval_max_padded_len=int(exp1_eval_maxlen) if exp1_eval_maxlen is not None else None,
            )
            npred = generate_exp1_rep_pred_job_file(
                configs,
                training_seeds,
                pred_path,
                mode=args.mode,
                eval_max_padded_len=int(exp1_eval_maxlen) if exp1_eval_maxlen is not None else None,
            )
            total_jobs += (n0 + n0_eval + n1 + nrepx + npred)
            exp1_n0, exp1_n1 = n0, n1

            print(f"{exp_name}:")
            print(f"  Configs: {len(configs)}")
            print("  Stage 0 (tokenize+outcomes; CPU):")
            print(f"    Jobs: {n0}")
            print(f"    Output: {stage0_path}")
            if exp1_eval_maxlen is not None:
                print("  Stage 0E (eval retokenize; CPU; fixed vocab + larger max_padded_len):")
                print(f"    Jobs: {n0_eval}")
                print(f"    Output: {stage0_eval_path}")
            print("  Stage 1 (FM train; GPU):")
            print(f"    Jobs: {n1}")
            print(f"    Output: {stage1_path}")
            print("  Stage 2 (extract reps; 1-GPU jobs):")
            print(f"    Jobs: {nrepx}")
            print(f"    Output: {repx_path}")
            print("  Stage 3 (rep-based preds; CPU tier2q jobs):")
            print(f"    Jobs: {npred}")
            print(f"    Output: {pred_path}")
            print()
        elif exp == 2:
            stage0_path = generated_dir / "08_exp2_stage0_tokenize.jobfile"
            stage1_path = generated_dir / "10a_exp2_stage1_train.jobfile"
            repx_path = generated_dir / "10b_exp2_stage2_extract_reps.jobfile"
            pred_path = generated_dir / "10c_exp2_stage3_lr.jobfile"

            n0 = generate_exp2_stage0_job_file(configs, stage0_path)
            n1 = generate_exp2_stage1_job_file(configs, training_seeds, stage1_path, mode=args.mode)
            nrepx = generate_exp2_rep_extract_job_file(configs, training_seeds, repx_path, mode=args.mode)
            npred = generate_exp2_rep_pred_job_file(configs, training_seeds, pred_path, mode=args.mode)
            total_jobs += (n0 + n1 + nrepx + npred)
            exp2_n0, exp2_n1 = n0, n1

            print(f"{exp_name}:")
            print(f"  Configs: {len(configs)}")
            print("  Stage 0 (tokenize+outcomes; CPU; dedup by data_version):")
            print(f"    Jobs: {n0}")
            print(f"    Output: {stage0_path}")
            print("  Stage 1 (train; GPU):")
            print(f"    Jobs: {n1}")
            print(f"    Output: {stage1_path}")
            print("  Stage 2 (extract reps; 1-GPU jobs):")
            print(f"    Jobs: {nrepx}")
            print(f"    Output: {repx_path}")
            print("  Stage 3 (rep-based preds; CPU tier2q jobs):")
            print(f"    Jobs: {npred}")
            print(f"    Output: {pred_path}")
            print()
        else:
            stage0_path = generated_dir / "12_exp3_stage0_tokenize.jobfile"
            stage1_path = generated_dir / "14a_exp3_stage1_train.jobfile"
            repx_path = generated_dir / "14b_exp3_stage2_extract_reps.jobfile"
            pred_path = generated_dir / "14c_exp3_stage3_lr.jobfile"

            n0 = generate_exp3_stage0_job_file(configs, stage0_path)
            n1 = generate_exp3_stage1_job_file(configs, training_seeds, stage1_path, mode=args.mode)
            nrepx = generate_exp3_rep_extract_job_file(configs, training_seeds, repx_path, mode=args.mode)
            npred = generate_exp3_rep_pred_job_file(configs, training_seeds, pred_path, mode=args.mode)
            total_jobs += (n0 + n1 + nrepx + npred)
            exp3_n0, exp3_n1 = n0, n1

            print(f"{exp_name}:")
            print(f"  Configs: {len(configs)}")
            print("  Stage 0 (tokenize+outcomes; CPU):")
            print(f"    Jobs: {n0}")
            print(f"    Output: {stage0_path}")
            print("  Stage 1 (train; GPU):")
            print(f"    Jobs: {n1}")
            print(f"    Output: {stage1_path}")
            print("  Stage 2 (extract reps; 1-GPU jobs):")
            print(f"    Jobs: {nrepx}")
            print(f"    Output: {repx_path}")
            print("  Stage 3 (rep-based preds; CPU tier2q jobs):")
            print(f"    Jobs: {npred}")
            print(f"    Output: {pred_path}")
            print()

    print("=" * 60)
    print(f"TOTAL JOBS: {total_jobs}")
    print("=" * 60)
    print()
    print("Next steps:")
    print(f"  1. Review job files in {generated_dir}/")
    if exp1_n0 is not None:
        print("  2. Exp1:")
        print(f"     - Stage 0 (tier2q CPU): sbatch --array=0-{exp1_n0 - 1} slurm/02_run_stage0_tier2q_tokenize.sh {generated_dir}/04_exp1_stage0_tokenize.jobfile")
        print(f"     - Stage 1 (4 GPUs/job): TRAIN_JID=$(sbatch --parsable --array=0-{exp1_n1 - 1} slurm/05_run_stage1_gpu4_train.sh {generated_dir}/07a_exp1_stage1_train.jobfile)")
        if args.exp1_eval_max_padded_len is not None:
            print(f"     - Stage 0E (tier2q CPU; eval retokenize): EVAL0_JID=$(sbatch --parsable --array=0-{exp1_n0 - 1} slurm/02_run_stage0_tier2q_tokenize.sh {generated_dir}/04b_exp1_stage0_eval_retokenize.jobfile)")
            print(f"     - Stage 2 (1 GPU/job; eval dv): EXTRACT_JID=$(sbatch --parsable --dependency=afterok:\"${{TRAIN_JID}}:${{EVAL0_JID}}\" --array=0-{exp1_n1 - 1} slurm/09_run_stage2_gpu2_extract.sh {generated_dir}/07b_exp1_stage2_extract_reps.jobfile)")
            print("       (If Exp1 Stage1 is already complete and you are only rerunning Stage2/3, skip Stage1 and submit Stage2 dependent only on Stage0E:)")
            print(f"       EXTRACT_JID=$(sbatch --parsable --dependency=afterok:\"${{EVAL0_JID}}\" --array=0-{exp1_n1 - 1} slurm/09_run_stage2_gpu2_extract.sh {generated_dir}/07b_exp1_stage2_extract_reps.jobfile)")
        else:
            print(f"     - Stage 2 (1 GPU/job): EXTRACT_JID=$(sbatch --parsable --dependency=afterok:\"${{TRAIN_JID}}\" --array=0-{exp1_n1 - 1} slurm/09_run_stage2_gpu2_extract.sh {generated_dir}/07b_exp1_stage2_extract_reps.jobfile)")
        print(f"     - Stage 3 (tier2q LR): sbatch --dependency=afterok:\"${{EXTRACT_JID}}\" --array=0-{exp1_n1 - 1} slurm/11_run_stage3_tier2q_lr.sh {generated_dir}/07c_exp1_stage3_lr.jobfile")
    if exp2_n0 is not None:
        print("  3. Exp2 (4-stage):")
        print(f"     - Stage 0 (tier2q CPU): sbatch --array=0-{exp2_n0 - 1} slurm/02_run_stage0_tier2q_tokenize.sh {generated_dir}/08_exp2_stage0_tokenize.jobfile")
        print(f"     - Stage 1 (1 GPU/job): TRAIN2_JID=$(sbatch --parsable --array=0-{exp2_n1 - 1} slurm/06_run_stage1_gpu1_train.sh {generated_dir}/10a_exp2_stage1_train.jobfile)")
        print(f"     - Stage 2 (1 GPU/job): EXTRACT2_JID=$(sbatch --parsable --dependency=afterok:\"${{TRAIN2_JID}}\" --array=0-{exp2_n1 - 1} slurm/09_run_stage2_gpu2_extract.sh {generated_dir}/10b_exp2_stage2_extract_reps.jobfile)")
        print(f"     - Stage 3 (tier2q LR): sbatch --dependency=afterok:\"${{EXTRACT2_JID}}\" --array=0-{exp2_n1 - 1} slurm/11_run_stage3_tier2q_lr.sh {generated_dir}/10c_exp2_stage3_lr.jobfile")
    if exp3_n0 is not None:
        print("  4. Exp3 (4-stage):")
        print(f"     - Stage 0 (tier2q CPU): sbatch --array=0-{exp3_n0 - 1} slurm/02_run_stage0_tier2q_tokenize.sh {generated_dir}/12_exp3_stage0_tokenize.jobfile")
        print(f"     - Stage 1 (1 GPU/job): TRAIN3_JID=$(sbatch --parsable --array=0-{exp3_n1 - 1} slurm/06_run_stage1_gpu1_train.sh {generated_dir}/14a_exp3_stage1_train.jobfile)")
        print(f"     - Stage 2 (1 GPU/job): EXTRACT3_JID=$(sbatch --parsable --dependency=afterok:\"${{TRAIN3_JID}}\" --array=0-{exp3_n1 - 1} slurm/09_run_stage2_gpu2_extract.sh {generated_dir}/14b_exp3_stage2_extract_reps.jobfile)")
        print(f"     - Stage 3 (tier2q LR): sbatch --dependency=afterok:\"${{EXTRACT3_JID}}\" --array=0-{exp3_n1 - 1} slurm/11_run_stage3_tier2q_lr.sh {generated_dir}/14c_exp3_stage3_lr.jobfile")


if __name__ == "__main__":
    main()
