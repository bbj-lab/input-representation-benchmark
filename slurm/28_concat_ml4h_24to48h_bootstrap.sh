#!/bin/bash
#SBATCH --job-name=irb-ml4h-24to48h-bootc
#SBATCH --output=./outputs/runs/ml4h_2026/bootstrap_24to48h_ci/logs/%A-%x.out
#SBATCH --partition=tier1q
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# Merge unique per-run shards into metrics/metrics_long_with_ci.parquet.
# Does not write metrics_long.parquet.

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

WORK_ROOT="${IRB_BOOTSTRAP_WORK_ROOT:-${IRB_HOME}/outputs/runs/ml4h_2026/bootstrap_24to48h_ci}"
METRICS_LONG="${IRB_METRICS_LONG:-${IRB_HOME}/outputs/runs/ml4h_2026/metrics/metrics_long.parquet}"
OUTPUT_PARQUET="${IRB_CI_PARQUET:-${IRB_HOME}/outputs/runs/ml4h_2026/metrics/metrics_long_with_ci.parquet}"
mkdir -p "${WORK_ROOT}/logs" "${WORK_ROOT}/tmp"

python pipeline/scripts/bootstrap_ml4h_24to48h_metrics.py \
  --mode concat \
  --metrics-long "${METRICS_LONG}" \
  --work-root "${WORK_ROOT}" \
  --output-parquet "${OUTPUT_PARQUET}" \
  --bootstrap-n 1000 \
  --bootstrap-seed 42
