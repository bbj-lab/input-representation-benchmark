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
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/scratch/$(whoami)/}"
export WANDB_DIR="${WANDB_DIR:-/scratch/$(whoami)/}"

