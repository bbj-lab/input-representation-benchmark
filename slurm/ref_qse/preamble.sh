#!/bin/bash

# =============================================================================
# Provenance: Quantifying-Surprise-EHRs/slurm/preamble.sh
# Source: /gpfs/data/bbj-lab/users/daniel/Quantifying-Surprise-EHRs/slurm/preamble.sh
# =============================================================================
#
# - Use IRB_HOME + slurm/00_preamble.sh as the single source of truth for env setup.
# - Preserve exported HF_HOME/WANDB dirs as in the reference (cluster-friendly).
#
# =============================================================================

set -euo pipefail

if [ -v SLURM_ARRAY_JOB_ID ]; then
  echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID}"
  echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
fi

# Root of this benchmark repo
IRB_HOME="${IRB_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export IRB_HOME

# Activate our standard benchmark env (contains transformers/torch/etc).
source "${IRB_HOME}/slurm/00_preamble.sh"

# Match reference cache behavior
export HF_HOME="${HF_HOME:-/gpfs/data/bbj-lab/cache/huggingface/}"
_irb_username() {
  # Some compute nodes do not have stable uid->username mapping.
  # Prefer non-numeric identifiers to avoid invalid /scratch/<uid> paths.
  local u="${USER:-}"
  if [[ -n "${u}" && ! "${u}" =~ ^[0-9]+$ ]]; then
    echo "${u}"
    return 0
  fi
  u="${SLURM_JOB_USER:-${LOGNAME:-}}"
  if [[ -n "${u}" && ! "${u}" =~ ^[0-9]+$ ]]; then
    echo "${u}"
    return 0
  fi
  u="$(id -un 2>/dev/null || true)"
  if [[ -n "${u}" && ! "${u}" =~ ^[0-9]+$ ]]; then
    echo "${u}"
    return 0
  fi
  echo "uid_$(id -u)"
}

_IRB_USER="$(_irb_username)"
if [[ -d /scratch && "${_IRB_USER}" != uid_* && ! "${_IRB_USER}" =~ ^[0-9]+$ ]]; then
  export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/scratch/${_IRB_USER}/}"
  export WANDB_DIR="${WANDB_DIR:-/scratch/${_IRB_USER}/}"
else
  export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${HOME}/.cache/wandb}"
  export WANDB_DIR="${WANDB_DIR:-${HOME}/.cache/wandb}"
fi
mkdir -p "${WANDB_CACHE_DIR}" "${WANDB_DIR}" 2>/dev/null || true

