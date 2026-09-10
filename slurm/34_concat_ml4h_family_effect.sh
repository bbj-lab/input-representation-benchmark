#!/bin/bash
#SBATCH --job-name=irb-ml4h-family-bootc
#SBATCH --output=./outputs/runs/ml4h_2026/family_effect_bootstrap/logs/%A-%x.out
#SBATCH --partition=tier1q,tier2q,tier3q
#SBATCH --cpus-per-task=4
#SBATCH --mem=2G
#SBATCH --time=02:00:00

# Merge 17 column shards (eight family tests each) into family_effect_intervals.parquet.

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

WORK_ROOT="${IRB_HOME}/outputs/runs/ml4h_2026/family_effect_bootstrap"
METRICS_LONG="${IRB_HOME}/outputs/runs/ml4h_2026/metrics/metrics_long.parquet"
OUTPUT_PARQUET="${WORK_ROOT}/family_effect_intervals.parquet"
mkdir -p "${WORK_ROOT}/logs" "${WORK_ROOT}/tmp"

python pipeline/scripts/bootstrap_ml4h_family_effects.py \
  --mode concat \
  --metrics "${METRICS_LONG}" \
  --work-root "${WORK_ROOT}" \
  --output "${OUTPUT_PARQUET}" \
  --bootstrap-n 1000 \
  --bootstrap-seed 42
