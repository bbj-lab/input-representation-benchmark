#!/bin/bash
# Submit the 17-column paired permutation family-mean Δ tests, then concat.
# Split across tier1q / tier2q / tier3q so each partition can take work.
# Work products stay under outputs/runs/ml4h_2026/family_effect_bootstrap/.

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
LOG_DIR="${WORK_ROOT}/logs"
mkdir -p "${LOG_DIR}" "${WORK_ROOT}/shards" "${WORK_ROOT}/tmp"

if [[ -e "${OUTPUT_PARQUET}" ]]; then
  echo "ERROR: refusing to submit while ${OUTPUT_PARQUET} already exists." >&2
  exit 2
fi

count="$(python pipeline/scripts/bootstrap_ml4h_family_effects.py \
  --mode list \
  --metrics "${METRICS_LONG}" \
  --work-root "${WORK_ROOT}" \
  --output "${OUTPUT_PARQUET}" \
  --bootstrap-n 1000 \
  --bootstrap-seed 42 | python -c 'import json,sys; print(json.load(sys.stdin)["n_tasks"])')"

if [[ "${count}" != "17" ]]; then
  echo "ERROR: expected 17 column tasks, found ${count}" >&2
  exit 2
fi

submit_array() {
  local partition="$1"
  local cpus="$2"
  local mem="$3"
  local array="$4"
  sbatch --parsable \
    --partition="${partition}" \
    --cpus-per-task="${cpus}" \
    --mem="${mem}" \
    --time=12:00:00 \
    --job-name=irb-ml4h-family-boot \
    --output="${LOG_DIR}/%A_%a-irb-ml4h-family-boot.out" \
    --array="${array}" \
    slurm/34_run_ml4h_family_effect_task.sh
}

# Pack leftover holes. Mixed nodes already hold small high-mem jobs, so
# idle CPUs come with only a few GB of allocatable RAM. A column task
# peaks around 1.4 GB RSS, so 2G is enough and actually starts.
# tier1 leftover: ~29 CPUs. tier2 leftover: ~26 CPUs. tier3 leftover: ~3 CPUs.
job_tier1="$(submit_array tier1q 29 2G 0-9)"
job_tier2="$(submit_array tier2q 26 2G 10-14)"
job_tier3="$(submit_array tier3q 3 2G 15-16)"

concat_job="$(
  sbatch --parsable \
    --dependency="afterok:${job_tier2}:${job_tier1}:${job_tier3}" \
    --partition=tier1q,tier2q,tier3q \
    --cpus-per-task=4 \
    --mem=2G \
    --time=02:00:00 \
    --job-name=irb-ml4h-family-bootc \
    --output="${LOG_DIR}/%A-irb-ml4h-family-bootc.out" \
    slurm/34_concat_ml4h_family_effect.sh
)"

{
  printf 'family_effect_array_tier1q\t%s\t10\t%s\n' "${job_tier1}" "${WORK_ROOT}"
  printf 'family_effect_array_tier2q\t%s\t5\t%s\n' "${job_tier2}" "${WORK_ROOT}"
  printf 'family_effect_array_tier3q\t%s\t2\t%s\n' "${job_tier3}" "${WORK_ROOT}"
  printf 'family_effect_concat\t%s\t1\t%s\n' "${concat_job}" "${OUTPUT_PARQUET}"
} | tee "${WORK_ROOT}/launch.tsv"
