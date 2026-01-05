#!/bin/bash
#SBATCH --job-name=irb-train
#SBATCH --output=./output/%A_%a-%x.stdout
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00

# =============================================================================
# SLURM Array Job Runner for Input Representation Benchmark
# =============================================================================
# Reads commands from a job file and executes the one matching SLURM_ARRAY_TASK_ID
#
# Usage:
#   # First, generate job file:
#   python run_experiments.py --mode demo --experiment 1
#   
#   # Then submit with array size matching number of jobs:
#   sbatch --array=0-11 slurm/run_from_jobfile.sh slurm/exp1_demo_jobs.txt
#
# Array sizes:
#   Exp1 demo: --array=0-11   (12 jobs)
#   Exp1 full: --array=0-59   (60 jobs)
#   Exp2 demo: --array=0-5    (6 jobs)
#   Exp2 full: --array=0-29   (30 jobs)
#   Exp3 demo: --array=0-1    (2 jobs)
#   Exp3 full: --array=0-9    (10 jobs)
# =============================================================================

set -euo pipefail

# Job file path (passed as argument)
JOBFILE=${1:-"slurm/exp1_demo_jobs.txt"}

if [[ ! -f "$JOBFILE" ]]; then
    echo "ERROR: Job file not found: $JOBFILE"
    echo "Generate it first with: python run_experiments.py --mode demo"
    exit 1
fi

# Source environment
source "$(dirname "$0")/preamble.sh"

# Navigate to project root
cd "$IRB_HOME"

# Create output directory
mkdir -p slurm/output

echo "=============================================="
echo "SLURM Array Task: ${SLURM_ARRAY_TASK_ID:-0}"
echo "Job File: ${JOBFILE}"
echo "=============================================="

# Extract the command for this array task
# Job file format: non-empty, non-comment lines
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

# Execute the command
eval "$CMD"

EXIT_CODE=$?

echo ""
echo "=============================================="
echo "Task ${TASK_ID} completed with exit code ${EXIT_CODE}"
echo "=============================================="

exit ${EXIT_CODE}
