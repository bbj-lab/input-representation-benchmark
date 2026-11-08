#!/bin/bash
#SBATCH --job-name=irb-ml4h-metric-tables
#SBATCH --output=./slurm/output/%A-%x.stdout
#SBATCH --partition=tier1q,tier2q
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

# Build the 24-48h metric tables from the Stage 3 prediction files.
#
# 1. The generator writes the tables to outputs/runs/ml4h_2026/metrics/.
# 2. The validator checks the run, task, and row counts of the tables.
#
# Usage:
#   sbatch slurm/23_build_ml4h_metric_tables.sh

set -euo pipefail

find_repo_root() {
  local d="$1"
  while [[ "$d" != "/" ]]; do
    if [[ -f "$d/pipeline/run_experiments.py" && -d "$d/slurm" ]]; then
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

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_MAX_THREADS="${SLURM_CPUS_PER_TASK}"

EXPERIMENTS=(exp1 exp2 exp3)
METRICS_DIR="${IRB_HOME}/outputs/runs/ml4h_2026/metrics"

python paper/scripts/generate_ml4h_24to48h_provisional.py \
  --output-root "${METRICS_DIR}" \
  --experiments "${EXPERIMENTS[@]}"

python utilities/scripts/validate_ml4h_24to48h_bundle.py \
  --output-root "${METRICS_DIR}" \
  --experiments "${EXPERIMENTS[@]}"
