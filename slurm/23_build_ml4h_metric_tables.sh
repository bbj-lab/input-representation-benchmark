#!/bin/bash
#SBATCH --job-name=irb-ml4h-metric-tables
#SBATCH --output=./slurm/output/%A-%x.stdout
#SBATCH --partition=tier1q,tier2q
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

# Build the 24-48h metric tables from the Stage 3 prediction files.
#
# 1. The generator writes the bundle to outputs/runs/ml4h_2026/metric_bundle/.
# 2. The validator checks the run, task, and row counts of the bundle.
# 3. The script copies the tables in 00_data/ to outputs/runs/ml4h_2026/metrics/.
#
# The script always builds all three experiments, because the copy replaces
# the full tables in metrics/.
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

BUNDLE_ROOT="${IRB_HOME}/outputs/runs/ml4h_2026/metric_bundle"
METRICS_DIR="${IRB_HOME}/outputs/runs/ml4h_2026/metrics"

python paper/scripts/generate_ml4h_24to48h_provisional.py \
  --output-root "${BUNDLE_ROOT}" \
  --experiments "${EXPERIMENTS[@]}"

python utilities/scripts/validate_ml4h_24to48h_bundle.py \
  --output-root "${BUNDLE_ROOT}" \
  --experiments "${EXPERIMENTS[@]}"

mkdir -p "${METRICS_DIR}"
for name in \
  metrics_long.csv metrics_long.parquet \
  run_registry.csv run_registry.parquet \
  label_coverage_long.csv label_coverage_long.parquet \
  outcome_coverage_summary.csv; do
  src="${BUNDLE_ROOT}/00_data/${name}"
  if [[ ! -s "${src}" ]]; then
    echo "ERROR: missing bundle table: ${src}" >&2
    exit 2
  fi
  cp "${src}" "${METRICS_DIR}/${name}"
  echo "Copied ${name} to ${METRICS_DIR}/"
done
