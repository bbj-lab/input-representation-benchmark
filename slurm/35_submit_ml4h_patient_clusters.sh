#!/bin/bash
# Submit patient-cluster bootstrap arrays and a dependent concat job.

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
cd "${IRB_HOME}"

WORK_ROOT="${IRB_HOME}/outputs/runs/ml4h_2026/patient_cluster_bootstrap"
METRICS_LONG="${IRB_HOME}/outputs/runs/ml4h_2026/metrics/metrics_long.parquet"
SUBJECT_MAP="${WORK_ROOT}/hadm_to_subject.parquet"
OUTPUT_PARQUET="${WORK_ROOT}/patient_cluster_intervals.parquet"
LOG_DIR="${WORK_ROOT}/logs"
mkdir -p "${LOG_DIR}" "${WORK_ROOT}/shards" "${WORK_ROOT}/tmp"

if [[ -e "${OUTPUT_PARQUET}" ]]; then
  echo "ERROR: refusing to overwrite ${OUTPUT_PARQUET}." >&2
  exit 2
fi

count="$(python pipeline/scripts/bootstrap_ml4h_patient_clusters.py \
  --mode list \
  --metrics "${METRICS_LONG}" \
  --work-root "${WORK_ROOT}" \
  --output "${OUTPUT_PARQUET}" \
  --bootstrap-n 1000 \
  --bootstrap-seed 42 | python -c 'import json,sys; print(json.load(sys.stdin)["n_tasks"])')"
if [[ "${count}" != "17" ]]; then
  echo "ERROR: expected 17 column tasks, found ${count}." >&2
  exit 2
fi

map_job=""
dependency_args=()
if [[ ! -f "${SUBJECT_MAP}" ]]; then
  map_job="$(
    sbatch --parsable \
      --partition=tier2q \
      --export="ALL,IRB_PATIENT_BOOTSTRAP_WORK_ROOT=${WORK_ROOT},IRB_PATIENT_SUBJECT_MAP=${SUBJECT_MAP}" \
      slurm/35_build_ml4h_patient_map.sh
  )"
  dependency_args=(--dependency="afterok:${map_job}")
fi

submit_array() {
  local partition="$1"
  local cpus="$2"
  local array="$3"
  sbatch --parsable \
    "${dependency_args[@]}" \
    --partition="${partition}" \
    --cpus-per-task="${cpus}" \
    --mem=2G \
    --time=12:00:00 \
    --job-name=irb-ml4h-patient-boot \
    --output="${LOG_DIR}/%A_%a-irb-ml4h-patient-boot.out" \
    --export="ALL,IRB_PATIENT_BOOTSTRAP_WORK_ROOT=${WORK_ROOT},IRB_PATIENT_SUBJECT_MAP=${SUBJECT_MAP},IRB_PATIENT_BOOTSTRAP_N=1000" \
    --array="${array}" \
    slurm/35_run_ml4h_patient_cluster_task.sh
}

job_tier1="$(submit_array tier1q 29 0-9)"
job_tier2="$(submit_array tier2q 26 10-14)"
job_tier3="$(submit_array tier3q 3 15-16)"

concat_job="$(
  sbatch --parsable \
    --dependency="afterok:${job_tier1}:${job_tier2}:${job_tier3}" \
    --partition=tier1q,tier2q,tier3q \
    --cpus-per-task=4 \
    --mem=2G \
    --time=02:00:00 \
    --job-name=irb-ml4h-patient-bootc \
    --output="${LOG_DIR}/%A-irb-ml4h-patient-bootc.out" \
    --export="ALL,IRB_PATIENT_BOOTSTRAP_WORK_ROOT=${WORK_ROOT},IRB_PATIENT_BOOTSTRAP_N=1000" \
    slurm/35_concat_ml4h_patient_clusters.sh
)"

{
  if [[ -n "${map_job}" ]]; then
    printf 'patient_map\t%s\t1\t%s\n' "${map_job}" "${SUBJECT_MAP}"
  fi
  printf 'patient_bootstrap_tier1q\t%s\t10\t%s\n' "${job_tier1}" "${WORK_ROOT}"
  printf 'patient_bootstrap_tier2q\t%s\t5\t%s\n' "${job_tier2}" "${WORK_ROOT}"
  printf 'patient_bootstrap_tier3q\t%s\t2\t%s\n' "${job_tier3}" "${WORK_ROOT}"
  printf 'patient_bootstrap_concat\t%s\t1\t%s\n' "${concat_job}" "${OUTPUT_PARQUET}"
} | tee "${WORK_ROOT}/launch.tsv"
