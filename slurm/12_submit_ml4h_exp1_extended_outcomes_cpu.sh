#!/bin/bash
# Refresh extended outcomes for every ML4H Exp1 tokenized dataset in parallel.

set -euo pipefail

IRB_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${IRB_HOME}"

STAGE_DIR="outputs/runs/ml4h_2026/stage3"
LOG_DIR="${STAGE_DIR}/logs"
JOBFILE="${STAGE_DIR}/exp1_refresh_extended_outcomes.jobfile"
CPUS_PER_TASK=8
MEMORY_PER_TASK=256G

python utilities/scripts/generate_ml4h_exp1_stage3_jobfiles.py \
  --output-dir "${STAGE_DIR}" \
  --write-outcome-refresh-jobfile
mkdir -p "${LOG_DIR}"

count="$(rg -v '^(#|[[:space:]]*$)' "${JOBFILE}" | wc -l)"
if [[ "${count}" -ne 12 ]]; then
  echo "ERROR: expected 12 outcome refresh tasks in ${JOBFILE}; found ${count}" >&2
  exit 2
fi

sbatch \
  --parsable \
  --job-name=ml4h-e1-outcomes \
  --partition=tier3q \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task="${CPUS_PER_TASK}" \
  --mem="${MEMORY_PER_TASK}" \
  --time=12:00:00 \
  --open-mode=append \
  --output="${LOG_DIR}/%A_%a-%x.out" \
  --export="ALL,OMP_NUM_THREADS=${CPUS_PER_TASK},POLARS_MAX_THREADS=${CPUS_PER_TASK}" \
  --array="0-$((count - 1))" \
  slurm/15_run_stats_cpu_jobfile.sh "${JOBFILE}"
