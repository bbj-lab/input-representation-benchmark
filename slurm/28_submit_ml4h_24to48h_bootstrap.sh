#!/bin/bash
# Submit the 156-run 24–48h bootstrap array, then concat after all tasks succeed.
# Work products stay under outputs/runs/ml4h_2026/bootstrap_24to48h_ci/.
# The live CI table is outputs/runs/ml4h_2026/metrics/metrics_long_with_ci.parquet.

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

WORK_ROOT="${IRB_HOME}/outputs/runs/ml4h_2026/bootstrap_24to48h_ci"
METRICS_LONG="${IRB_HOME}/outputs/runs/ml4h_2026/metrics/metrics_long.parquet"
OUTPUT_PARQUET="${IRB_HOME}/outputs/runs/ml4h_2026/metrics/metrics_long_with_ci.parquet"
LOG_DIR="${WORK_ROOT}/logs"
mkdir -p "${LOG_DIR}" "${WORK_ROOT}/shards" "${WORK_ROOT}/tmp"

if [[ -e "${OUTPUT_PARQUET}" ]]; then
  echo "ERROR: refusing to submit while ${OUTPUT_PARQUET} already exists." >&2
  exit 2
fi
if [[ "$(basename "${METRICS_LONG}")" != "metrics_long.parquet" ]]; then
  echo "ERROR: unexpected metrics_long path: ${METRICS_LONG}" >&2
  exit 2
fi

python pipeline/scripts/bootstrap_ml4h_24to48h_metrics.py \
  --mode list \
  --metrics-long "${METRICS_LONG}" \
  --work-root "${WORK_ROOT}" \
  --output-parquet "${OUTPUT_PARQUET}" \
  --bootstrap-n 1000 \
  --bootstrap-seed 42

count="$(python - "${WORK_ROOT}/runs.csv" <<'PY'
import sys
import pandas as pd
runs = pd.read_csv(sys.argv[1])
if len(runs) != 156:
    raise SystemExit(f"expected 156 runs, found {len(runs)}")
if runs["run_id"].duplicated().any():
    raise SystemExit("run_id values are not unique")
print(len(runs))
PY
)"

array_job="$(
  sbatch --parsable \
    --partition=tier1q \
    --cpus-per-task=4 \
    --mem=32G \
    --time=12:00:00 \
    --job-name=irb-ml4h-24to48h-boot \
    --output="${LOG_DIR}/%A_%a-irb-ml4h-24to48h-boot.out" \
    --array="0-$((count - 1))" \
    slurm/28_run_ml4h_24to48h_bootstrap_task.sh
)"

concat_job="$(
  sbatch --parsable \
    --dependency="afterok:${array_job}" \
    --partition=tier1q \
    --cpus-per-task=4 \
    --mem=32G \
    --time=02:00:00 \
    --job-name=irb-ml4h-24to48h-bootc \
    --output="${LOG_DIR}/%A-irb-ml4h-24to48h-bootc.out" \
    slurm/28_concat_ml4h_24to48h_bootstrap.sh
)"

printf 'bootstrap_array\t%s\t%s\t%s\n' "${array_job}" "${count}" "${WORK_ROOT}" \
  | tee "${WORK_ROOT}/launch.tsv"
printf 'bootstrap_concat\t%s\t1\t%s\n' "${concat_job}" "${OUTPUT_PARQUET}" \
  | tee -a "${WORK_ROOT}/launch.tsv"
