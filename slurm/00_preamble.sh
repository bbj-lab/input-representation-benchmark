#!/bin/bash
# =============================================================================
# Standard preamble for input representation benchmark SLURM jobs
# Adapted from fms-ehrs/slurm/preamble.sh
# =============================================================================

# Log array job info if applicable
if [ -v SLURM_ARRAY_JOB_ID ]; then
    echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID}"
    echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
fi

# Job identification
# Do not call ``scontrol`` here: inside this cluster's job containers, the
# client cannot read /etc/slurm/slurm.conf and retries for up to a minute.
# Slurm already exports the requested job name.
name="${SLURM_JOB_NAME:-unknown}"
jname="${SLURM_JOB_NAME:-unknown}"
parent_dir="$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")"
export name jname parent_dir

# Base paths (bbj-lab cluster structure)
#
# NOTE: On some nodes, name service lookup can fail and USER may be unset or numeric.
# Prefer non-numeric user identifiers to avoid invalid /scratch/<uid> paths.
_user="${USER:-}"
if [[ -z "${_user}" || "${_user}" =~ ^[0-9]+$ ]]; then
  _user="${SLURM_JOB_USER:-${LOGNAME:-}}"
fi
if [[ -z "${_user}" || "${_user}" =~ ^[0-9]+$ ]]; then
  _user="$(id -un 2>/dev/null || true)"
fi
if [[ -z "${_user}" || "${_user}" =~ ^[0-9]+$ ]]; then
  _user="uid_$(id -u 2>/dev/null || echo unknown)"
fi
hm="/gpfs/data/bbj-lab/users/${_user}"

# Project paths (default: infer from this repository location)
IRB_HOME="${IRB_HOME:-${parent_dir}}"
FMS_EHRS_HOME="${FMS_EHRS_HOME:-$(realpath "${IRB_HOME}/../fms-ehrs")}"
RUN_ARTIFACTS_DIR="${RUN_ARTIFACTS_DIR:-${IRB_HOME}/outputs/runs}"
IRB_TOKENIZED_ROOT="${IRB_TOKENIZED_ROOT:-${RUN_ARTIFACTS_DIR}/tokenized}"
IRB_EXP3_ROOT="${IRB_EXP3_ROOT:-${RUN_ARTIFACTS_DIR}/exp3}"
IRB_TOKEN_DATASET_ID="${IRB_TOKEN_DATASET_ID:-mimiciv-3.1_meds_70-10-20}"
IRB_TOKENIZED_DATASET_DIR="${IRB_TOKENIZED_DATASET_DIR:-${IRB_TOKENIZED_ROOT}/${IRB_TOKEN_DATASET_ID}}"
# IMPORTANT: `DATA_DIR` is standardized to point to the MEDS *events directory*
# that contains split subdirectories (train/val/test or train/tuning/test) and
# the raw MEDS parquet shards (or a raw/ shim). Concretely:
#   ${DATA_DIR}/train/
#   ${DATA_DIR}/tuning/ (or val/)
#   ${DATA_DIR}/test/
#   ${DATA_DIR}/*.parquet  (sharded meds table)
#
# This avoids ambiguous "root vs events" semantics like ${DATA_DIR}/data.
DATA_DIR="${DATA_DIR:-${IRB_HOME}/benchmarks/mimic-meds-extraction/data/meds/data}"
MODEL_DIR="${MODEL_DIR:-${RUN_ARTIFACTS_DIR}/models}"
OUTPUT_DIR="${OUTPUT_DIR:-${IRB_HOME}/outputs}"
export IRB_HOME FMS_EHRS_HOME RUN_ARTIFACTS_DIR IRB_TOKENIZED_ROOT IRB_EXP3_ROOT
export IRB_TOKEN_DATASET_ID IRB_TOKENIZED_DATASET_DIR
export DATA_DIR MODEL_DIR OUTPUT_DIR hm

# -----------------------------------------------------------------------------
# Sequence-length defaults (single source of truth for jobs)
#
# - IRB_MAX_PADDED_LEN controls Stage0 tokenization truncation cap (TRUNC beyond cap).
# - IRB_MAX_SEQ_LENGTH controls training-time context/window length.
#
# Both default to 4096 for the current benchmark runs, but are intentionally
# overrideable per job (e.g., if training uses 4096 but evaluation uses 8192).
# -----------------------------------------------------------------------------
export IRB_MAX_PADDED_LEN="${IRB_MAX_PADDED_LEN:-4096}"
export IRB_MAX_SEQ_LENGTH="${IRB_MAX_SEQ_LENGTH:-4096}"

# -----------------------------------------------------------------------------
# Windowed padded training defaults (Exp2/Exp3 Stage1)
#
# In Exp2/Exp3 Stage1, we train on full variable-length admission timelines by
# slicing each admission into fixed-length windows of length IRB_MAX_SEQ_LENGTH.
#
# Design choices for fairness + compute stability:
# - Non-overlapping windows (no overlap between consecutive windows): set stride=L.
#   With TL_CONT enabled, subsequent windows carry (L-1) raw tokens plus a TL_CONT marker,
#   so the implementation advances by (L-1) after the first window to avoid gaps/overlap.
# - Extreme-length guardrail: cap the number of windows emitted per admission to bound
#   compute for pathological outliers (top ~1%). Windows are subsampled deterministically
#   (uniformly across the timeline, always including first+last).
# -----------------------------------------------------------------------------
export IRB_WINDOWED_PADDED="${IRB_WINDOWED_PADDED:-true}"
export IRB_ADD_CONT_TOKEN="${IRB_ADD_CONT_TOKEN:-true}"
export IRB_WINDOW_STRIDE="${IRB_WINDOW_STRIDE:-${IRB_MAX_SEQ_LENGTH}}"
export IRB_MAX_WINDOWS_PER_ADMISSION="${IRB_MAX_WINDOWS_PER_ADMISSION:-128}"

# -----------------------------------------------------------------------------
# Stage1 (epoch budget + shared hypers)
#
# Rationale:
# - All ML4H experiment families use fixed hyperparameters selected by a
#   validation-only preflight and a five-epoch budget per configuration.
# - We centralize defaults here so that job generation, SLURM launchers, and Python
#   entrypoints do not silently diverge.
#
# Override any of these at submit-time by exporting the env var before sourcing
# this preamble (or via `sbatch --export=...`).
# -----------------------------------------------------------------------------
#
# Keep explicit family variables so a deliberate future deviation is visible.
export IRB_EXP1_STAGE1_EPOCHS="${IRB_EXP1_STAGE1_EPOCHS:-5}"
export IRB_EXP23_STAGE1_EPOCHS="${IRB_EXP23_STAGE1_EPOCHS:-5}"
# Backward-compat alias (older scripts referenced IRB_STAGE1_EPOCHS).
export IRB_STAGE1_EPOCHS="${IRB_STAGE1_EPOCHS:-${IRB_EXP23_STAGE1_EPOCHS}}"
export IRB_STAGE1_OPTIMIZER="${IRB_STAGE1_OPTIMIZER:-adamw}"
export IRB_STAGE1_LR="${IRB_STAGE1_LR:-1e-4}"
export IRB_STAGE1_WEIGHT_DECAY="${IRB_STAGE1_WEIGHT_DECAY:-0.01}"
export IRB_ADAM_BETA1="${IRB_ADAM_BETA1:-0.9}"
export IRB_ADAM_BETA2="${IRB_ADAM_BETA2:-0.999}"
export IRB_STAGE1_MAX_GRAD_NORM="${IRB_STAGE1_MAX_GRAD_NORM:-1.0}"
export IRB_STAGE1_LR_SCHEDULER="${IRB_STAGE1_LR_SCHEDULER:-linear}"
export IRB_STAGE1_WARMUP_RATIO="${IRB_STAGE1_WARMUP_RATIO:-0.01}"
export IRB_STAGE1_LOGGING_STEPS="${IRB_STAGE1_LOGGING_STEPS:-100}"
export IRB_PER_DEVICE_TRAIN_BATCH_SIZE="${IRB_PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
export IRB_PER_DEVICE_EVAL_BATCH_SIZE="${IRB_PER_DEVICE_EVAL_BATCH_SIZE:-4}"
export IRB_GRADIENT_ACCUMULATION_STEPS="${IRB_GRADIENT_ACCUMULATION_STEPS:-1}"
export IRB_STAGE1_EVALUATIONS_PER_EPOCH="${IRB_STAGE1_EVALUATIONS_PER_EPOCH:-2}"
export IRB_STAGE1_MAX_STEPS="${IRB_STAGE1_MAX_STEPS:--1}"
# Write a local JSONL/Slurm-log progress heartbeat each minute. This is
# independent of W&B and works on offline compute nodes.
export IRB_PROGRESS_INTERVAL_SECONDS="${IRB_PROGRESS_INTERVAL_SECONDS:-60}"
export IRB_STAGE1_SAVE_STRATEGY="${IRB_STAGE1_SAVE_STRATEGY:-epoch}"
export IRB_STAGE1_SAVE_TOTAL_LIMIT="${IRB_STAGE1_SAVE_TOTAL_LIMIT:-4}"
export IRB_STAGE1_EVAL_STRATEGY="${IRB_STAGE1_EVAL_STRATEGY:-epoch}"
export IRB_LOAD_BEST_MODEL_AT_END="${IRB_LOAD_BEST_MODEL_AT_END:-true}"
# A full Trainer checkpoint is requested independently of epoch-level validation
# and best-model selection. Opportunistic jobs can be preempted after a
# ten-minute grace period, so force their interval to five minutes even when an
# older submission inherited a ten-minute value through --export=ALL. Requeues
# re-read this preamble from disk, so the override applies without resubmitting.
_irb_checkpoint_interval_default=600
if [[ "${SLURM_JOB_QOS:-}" == "opportunistic" ]]; then
  export IRB_STAGE1_CHECKPOINT_INTERVAL_SECONDS="${IRB_OPPORTUNISTIC_CHECKPOINT_INTERVAL_SECONDS:-300}"
else
  export IRB_STAGE1_CHECKPOINT_INTERVAL_SECONDS="${IRB_STAGE1_CHECKPOINT_INTERVAL_SECONDS:-${_irb_checkpoint_interval_default}}"
fi
unset _irb_checkpoint_interval_default

# Method-specific (Exp2/Exp3):
# Position bins are expressed directly in seconds; 60 means one RoPE position
# per minute. The preflight sweeps this alongside the backbone RoPE base.
export IRB_SECONDS_PER_POSITION="${IRB_SECONDS_PER_POSITION:-60.0}"
export IRB_MODEL_MAX_POSITION_EMBEDDINGS="${IRB_MODEL_MAX_POSITION_EMBEDDINGS:-4096}"

# xVal reference implementation uses unit weighting (loss_total = loss_mlm + loss_num).
export IRB_XVAL_NUMERIC_LOSS_WEIGHT="${IRB_XVAL_NUMERIC_LOSS_WEIGHT:-1.0}"
export IRB_XVAL_CLIP_SIGMA="${IRB_XVAL_CLIP_SIGMA:-5.0}"

# -----------------------------------------------------------------------------
# Optional training performance settings (objective-preserving)
#
# These do not change the data, labels, or loss definitions; they only select
# numerics / kernel implementations for speed and memory efficiency.
#
# - IRB_USE_BF16: enable bf16 mixed precision (A100 supports bf16).
# - IRB_ATTN_IMPL: Transformers attention backend, e.g. "sdpa" or "flash_attention_2".
#   (FlashAttention requires optional flash-attn install.)
# -----------------------------------------------------------------------------
export IRB_USE_BF16="${IRB_USE_BF16:-true}"
export IRB_ATTN_IMPL="${IRB_ATTN_IMPL:-}"

# Activate Python environment
source ~/.bashrc 2>/dev/null || true

# Make `conda activate` work across environments (cluster + local).
# - On BBJ cluster, conda is provided via a shared mamba install
# - On local machines, users typically have miniconda/anaconda installed under $HOME
if [[ -f /gpfs/data/bbj-lab/mamba/etc/profile.d/conda.sh ]]; then
    # BBJ cluster
    source /gpfs/data/bbj-lab/mamba/etc/profile.d/conda.sh
else
    # If `conda` is available, source its base shell hook.
    if command -v conda >/dev/null 2>&1; then
        CONDA_BASE="$(conda info --base 2>/dev/null || true)"
        if [[ -n "${CONDA_BASE}" && -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
            # shellcheck disable=SC1090
            source "${CONDA_BASE}/etc/profile.d/conda.sh"
        fi
    fi

    # Common local fallbacks (no-op if they don't exist).
    for _c in \
        "${HOME}/miniconda3/etc/profile.d/conda.sh" \
        "${HOME}/anaconda3/etc/profile.d/conda.sh" \
        "${HOME}/Desktop/miniconda3/etc/profile.d/conda.sh"; do
        if [[ -f "${_c}" ]]; then
            # shellcheck disable=SC1090
            source "${_c}"
            break
        fi
    done
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found in PATH, and no conda.sh could be sourced."
    echo "  - Install Miniconda/Anaconda, or add conda to your PATH."
    echo "  - Or set up your shell so that: source \"$(conda info --base)/etc/profile.d/conda.sh\" works."
    exit 1
fi

IRB_CONDA_ENV="${IRB_CONDA_ENV:-input-rep}"
conda activate "${IRB_CONDA_ENV}"

# Cache directories (following fms-ehrs conventions)
#
# IMPORTANT: Some cluster environments export HF_* cache paths pointing at shared,
# read-only directories. That causes PermissionError during dataset/model loading.
# We therefore prefer a user-writable cache and override non-writable settings.
_hf_home_candidate="${HF_HOME:-}"
# Reject inherited caches on the shared project filesystem. Cache traffic is
# high-volume and ephemeral; it must not consume the /gpfs data allocation.
if [[ "${_hf_home_candidate}" == /gpfs/* ]]; then
    _hf_home_candidate=""
    unset HF_HOME
fi
if [[ -n "${_hf_home_candidate}" ]]; then
    mkdir -p "${_hf_home_candidate}" 2>/dev/null || true
    if [[ -d "${_hf_home_candidate}" && -w "${_hf_home_candidate}" ]]; then
        HF_HOME="${_hf_home_candidate}"
    else
        unset HF_HOME
    fi
fi
if [[ -z "${HF_HOME:-}" ]]; then
    # Prefer per-user scratch. Cache traffic is high-volume and ephemeral, so it
    # belongs on scratch rather than the shared /gpfs project filesystem.
    if [[ -d /scratch && ! "${_user}" =~ ^[0-9]+$ && "${_user}" != uid_* ]]; then
        HF_HOME="/scratch/${_user}/huggingface"
    else
        HF_HOME="${HOME}/.cache/huggingface"
    fi
    if ! mkdir -p "${HF_HOME}" 2>/dev/null; then
        HF_HOME="${HOME}/.cache/huggingface"
        mkdir -p "${HF_HOME}"
    fi
fi

# Keep all HF caches under HF_HOME to avoid accidental writes to shared paths.
HF_DATASETS_CACHE="${HF_HOME}/datasets"
HF_HUB_CACHE="${HF_HOME}/hub"
TRANSFORMERS_CACHE="${HF_HOME}/transformers"
mkdir -p "${HF_DATASETS_CACHE}" "${HF_HUB_CACHE}" "${TRANSFORMERS_CACHE}" 2>/dev/null || true
export IRB_USE_MAPPED_DATASET_CACHE="${IRB_USE_MAPPED_DATASET_CACHE:-true}"
export IRB_MAPPED_DATASET_CACHE_DIR="${IRB_MAPPED_DATASET_CACHE_DIR:-${HF_DATASETS_CACHE}/irb-mapped}"
mkdir -p "${IRB_MAPPED_DATASET_CACHE_DIR}" 2>/dev/null || true
if [[ -d /scratch && ! "${_user}" =~ ^[0-9]+$ && "${_user}" != uid_* ]]; then
    WANDB_CACHE_DIR="/scratch/${_user}/"
    WANDB_DIR="/scratch/${_user}/"
else
    WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${HOME}/.cache/wandb}"
    WANDB_DIR="${WANDB_DIR:-${HOME}/.cache/wandb}"
fi
mkdir -p "${WANDB_CACHE_DIR}" "${WANDB_DIR}" 2>/dev/null || true
PYTHONPATH="${IRB_HOME}:${FMS_EHRS_HOME}:${PYTHONPATH:-}"
export HF_HOME HF_DATASETS_CACHE HF_HUB_CACHE TRANSFORMERS_CACHE WANDB_CACHE_DIR WANDB_DIR PYTHONPATH

# -----------------------------------------------------------------------------
# Attention backend default (FlashAttention-2 when available)
#
# We default to FlashAttention-2 when the optional `flash-attn` package is importable
# in the active environment; otherwise we fall back to PyTorch SDPA.
#
# Users can override explicitly by setting IRB_ATTN_IMPL before sourcing this preamble.
# -----------------------------------------------------------------------------
if [[ -z "${IRB_ATTN_IMPL}" ]]; then
    if python - <<'PY' 2>/dev/null | grep -q '^1$'; then
import importlib.util
print(1 if importlib.util.find_spec("flash_attn") is not None else 0)
PY
        export IRB_ATTN_IMPL="flash_attention_2"
    else
        export IRB_ATTN_IMPL="sdpa"
    fi
fi

# W&B Configuration
# Users should set these in ~/.bashrc or .env:
#   WANDB_API_KEY - Get from https://wandb.ai/authorize
#   WANDB_ENTITY  - Your W&B username or team name
_is_true() {
    case "${1:-}" in
        1|true|TRUE|True|yes|YES|on|ON|y|Y) return 0 ;;
        *) return 1 ;;
    esac
}

IRB_REQUIRE_WANDB="${IRB_REQUIRE_WANDB:-false}"
_has_wandb_auth=0
if [[ -n "${WANDB_API_KEY:-}" ]]; then
    _has_wandb_auth=1
elif python - <<'PY' >/dev/null 2>&1
import sys
try:
    import netrc
    n = netrc.netrc()
    if n.authenticators("api.wandb.ai") or n.authenticators("wandb.ai"):
        sys.exit(0)
except Exception:
    pass
sys.exit(1)
PY
then
    _has_wandb_auth=1
fi

if [[ "${_has_wandb_auth}" -eq 1 ]]; then
    export WANDB_MODE="${WANDB_MODE:-online}"
    export WANDB_LOG_MODEL="${WANDB_LOG_MODEL:-checkpoint}"
    export WANDB_WATCH="${WANDB_WATCH:-false}"
elif _is_true "${IRB_REQUIRE_WANDB}"; then
    echo "ERROR: W&B auth is required but not configured."
    echo "  - Set WANDB_API_KEY in your shell, or run wandb login to populate ~/.netrc."
    echo "  - Key URL: https://wandb.ai/authorize"
    exit 2
else
    echo "NOTE: W&B auth not found. Switching to offline mode."
    echo "      Set WANDB_API_KEY or run wandb login for remote logging."
    export WANDB_MODE=offline
fi

echo "=============================================="
echo "Environment initialized"
echo "=============================================="
echo "  User: ${_user}"
echo "  IRB_HOME: ${IRB_HOME}"
echo "  FMS_EHRS_HOME: ${FMS_EHRS_HOME}"
echo "  DATA_DIR: ${DATA_DIR}"
echo "  HF_HOME: ${HF_HOME}"
echo "  IRB_MAPPED_DATASET_CACHE_DIR: ${IRB_MAPPED_DATASET_CACHE_DIR}"
echo "  Python: $(which python 2>/dev/null || echo 'not found')"
echo "=============================================="

