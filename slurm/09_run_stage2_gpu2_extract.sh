#!/bin/bash
#SBATCH --job-name=irb-extract-2gpu
#SBATCH --output=./slurm/output/%A_%a-%x.stdout
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:2
#SBATCH --mem=128GB
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00

# =============================================================================
# Runner (Stage 2; 2 GPUs): executes the Nth line of a jobfile
# =============================================================================

set -euo pipefail

JOBFILE=${1:-""}
if [[ -z "$JOBFILE" ]]; then
  echo "ERROR: Job file not provided." >&2
  echo "Usage: sbatch --array=0-N slurm/09_run_stage2_gpu2_extract.sh <jobfile>" >&2
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
IRB_HOME="$(find_repo_root "${SUBMIT_DIR}")"
source "${IRB_HOME}/slurm/00_preamble.sh"
cd "${IRB_HOME}"
mkdir -p slurm/output

echo "=============================================="
echo "SLURM Array Task: ${SLURM_ARRAY_TASK_ID:-0}"
echo "Job File: ${JOBFILE}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-auto}"
echo "Host: $(hostname)"
echo "=============================================="

START_TS="$(date +%s)"
echo "Start time: $(date -Is)"

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo "GPU info unavailable"

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

END_TS="$(date +%s)"
echo "End time: $(date -Is)"
echo "Walltime (s): $((END_TS - START_TS))"

