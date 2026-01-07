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
job_id="${SLURM_JOB_ID:-${SLURM_JOBID:-}}"
name=$(
    if [[ -n "${job_id}" ]]; then
        scontrol show job "${job_id}" \
    | grep -m 1 "Command=" \
    | cut -d "=" -f2 \
            | xargs -I {} basename {} .sh 2>/dev/null
    fi
)
name="${name:-unknown}"
jname=$(
    if [[ -n "${job_id}" ]]; then
        scontrol show job "${job_id}" \
            | grep -oP 'JobName=\K\S+' 2>/dev/null
    fi
)
jname="${jname:-unknown}"
parent_dir="$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")"
export name jname parent_dir

# Base paths (bbj-lab cluster structure)
hm="/gpfs/data/bbj-lab/users/$(whoami)"

# Project paths (default: infer from this repository location)
IRB_HOME="${IRB_HOME:-${parent_dir}}"
FMS_EHRS_HOME="${FMS_EHRS_HOME:-$(realpath "${IRB_HOME}/../fms-ehrs")}"
DATA_DIR="${DATA_DIR:-${IRB_HOME}/benchmarks/mimic-meds-extraction/data/meds}"
MODEL_DIR="${MODEL_DIR:-${IRB_HOME}/models}"
OUTPUT_DIR="${OUTPUT_DIR:-${IRB_HOME}/outputs}"
export IRB_HOME FMS_EHRS_HOME DATA_DIR MODEL_DIR OUTPUT_DIR hm

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
if [[ -d /gpfs/data/bbj-lab/cache/huggingface/ ]]; then
    HF_HOME="/gpfs/data/bbj-lab/cache/huggingface/"
else
    HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
fi
if [[ -d /scratch ]]; then
    WANDB_CACHE_DIR="/scratch/$(whoami)/"
    WANDB_DIR="/scratch/$(whoami)/"
else
    WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${HOME}/.cache/wandb}"
    WANDB_DIR="${WANDB_DIR:-${HOME}/.cache/wandb}"
fi
PYTHONPATH="${IRB_HOME}:${FMS_EHRS_HOME}:${PYTHONPATH:-}"
export HF_HOME WANDB_CACHE_DIR WANDB_DIR PYTHONPATH

# W&B Configuration
# Users should set these in ~/.bashrc or .env:
#   WANDB_API_KEY - Get from https://wandb.ai/authorize
#   WANDB_ENTITY  - Your W&B username or team name
if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "NOTE: WANDB_API_KEY not set. Runs will not be logged to W&B."
    echo "      Get your key at: https://wandb.ai/authorize"
    export WANDB_MODE=offline
fi

echo "=============================================="
echo "Environment initialized"
echo "=============================================="
echo "  User: $(whoami)"
echo "  IRB_HOME: ${IRB_HOME}"
echo "  FMS_EHRS_HOME: ${FMS_EHRS_HOME}"
echo "  DATA_DIR: ${DATA_DIR}"
echo "  HF_HOME: ${HF_HOME}"
echo "  Python: $(which python 2>/dev/null || echo 'not found')"
echo "=============================================="
