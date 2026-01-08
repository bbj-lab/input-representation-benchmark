#!/bin/bash
#SBATCH --job-name=irb-cpu
#SBATCH --output=./slurm/output/%A_%a-%x.stdout
#SBATCH --partition=gpuq
#SBATCH --mem=150GB
#SBATCH --cpus-per-task=8
#SBATCH --time=0-08:00:00

# =============================================================================
# SLURM Array Job Runner (CPU-heavy stages; no GPU request)
# =============================================================================
# Reads commands from a job file and executes the one matching SLURM_ARRAY_TASK_ID.
#
# Intended for CPU-heavy steps (e.g., tokenization + outcome extraction).
#
# Usage:
#   sbatch --array=0-11%2 slurm/02_run_from_jobfile_cpu.sh slurm/04_exp1_tokenize_jobs.sh
#
# Notes:
# - The default partition is `gpuq` (randi target). Override at submit time if needed:
#     sbatch --partition=<your_cpu_partition> --array=0-11%2 slurm/02_run_from_jobfile_cpu.sh slurm/04_exp1_tokenize_jobs.sh
# =============================================================================

set -euo pipefail

# Job file path (passed as argument)
JOBFILE=${1:-""}

if [[ -z "$JOBFILE" ]]; then
    echo "ERROR: Job file not provided."
    echo "Usage: sbatch --array=0-N slurm/02_run_from_jobfile_cpu.sh <jobfile>"
    exit 1
fi

if [[ ! -f "$JOBFILE" ]]; then
    echo "ERROR: Job file not found: $JOBFILE"
    exit 1
fi

# Source environment
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

# Tokenization/outcome extraction uses the input-rep env.
export IRB_CONDA_ENV="input-rep"
source "${IRB_HOME}/slurm/preamble.sh"

# Navigate to project root
cd "$IRB_HOME"

# Create output directory
mkdir -p slurm/output

echo "=============================================="
echo "SLURM Array Task: ${SLURM_ARRAY_TASK_ID:-0}"
echo "Job File: ${JOBFILE}"
echo "Host: $(hostname)"
echo "=============================================="

# Extract the command for this array task
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

# Get the Nth command (0-indexed), skipping comments and empty lines
CMD=$(grep -v '^#' "$JOBFILE" | grep -v '^[[:space:]]*$' | sed -n "$((TASK_ID + 1))p")

if [[ -z "$CMD" ]]; then
    echo "ERROR: No command found for task ${TASK_ID}"
    echo "Job file has $(grep -v '^#' "$JOBFILE" | grep -v '^[[:space:]]*$' | wc -l) commands"
    exit 1
fi

echo ""
echo "Executing command:"
echo "  ${CMD}"
echo ""
echo "=============================================="

eval "$CMD"

EXIT_CODE=$?

echo ""
echo "=============================================="
echo "Task ${TASK_ID} completed with exit code ${EXIT_CODE}"
echo "=============================================="

exit ${EXIT_CODE}

