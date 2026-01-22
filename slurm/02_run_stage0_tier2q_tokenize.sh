#!/bin/bash
#SBATCH --job-name=irb-stage0
#SBATCH --output=./slurm/output/%A_%a-%x.stdout
#SBATCH --partition=tier2q
#SBATCH --mem=300GB
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00

# =============================================================================
# Runner (Stage 0; tier2q; CPU-only): executes the Nth line of a jobfile
# =============================================================================

set -euo pipefail

JOBFILE=${1:-""}
if [[ -z "$JOBFILE" ]]; then
  echo "ERROR: Job file not provided." >&2
  echo "Usage: sbatch --array=0-N slurm/02_run_stage0_tier2q_tokenize.sh <jobfile>" >&2
  exit 1
fi
if [[ ! -f "$JOBFILE" ]]; then
  echo "ERROR: Job file not found: $JOBFILE" >&2
  exit 1
fi

find_repo_root() {
  local d="$1"
  while [[ "$d" != "/" ]]; do
    if [[ -f "$d/run_experiments.py" && -d "$d/slurm" ]]; then
      echo "$d"
      return 0
    fi
    d="$(dirname "$d")"
  done
  return 1
}

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
# Robust repo-root resolution:
# - Prefer SLURM_SUBMIT_DIR (normal sbatch behavior)
# - Fall back to the directory of this script (works even if SLURM_SUBMIT_DIR is unset/mis-set)
# NOTE: With `set -e`, we must not allow a failing command substitution to exit the script
# without a diagnostic.
IRB_HOME="$(find_repo_root "${SUBMIT_DIR}" || true)"
if [[ -z "${IRB_HOME}" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  IRB_HOME="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
if [[ ! -f "${IRB_HOME}/run_experiments.py" || ! -d "${IRB_HOME}/slurm" ]]; then
  echo "ERROR: Could not locate IRB repo root." >&2
  echo "  SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>}" >&2
  echo "  pwd=$(pwd)" >&2
  echo "  resolved IRB_HOME=${IRB_HOME}" >&2
  exit 1
fi

export IRB_CONDA_ENV="input-rep"
source "${IRB_HOME}/slurm/00_preamble.sh"
cd "${IRB_HOME}"
mkdir -p slurm/output

echo "=============================================="
echo "SLURM Array Task: ${SLURM_ARRAY_TASK_ID:-0}"
echo "Job File: ${JOBFILE}"
echo "Host: $(hostname)"
echo "=============================================="

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
CMD=$(grep -v '^#' "$JOBFILE" | grep -v '^[[:space:]]*$' | sed -n "$((TASK_ID + 1))p")
if [[ -z "$CMD" ]]; then
  echo "ERROR: No command found for task ${TASK_ID}" >&2
  exit 1
fi

echo ""
echo "Executing command:"
echo "  ${CMD}"
echo ""
echo "=============================================="

eval "$CMD"
