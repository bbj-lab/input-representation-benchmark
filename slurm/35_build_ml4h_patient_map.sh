#!/bin/bash
#SBATCH --job-name=irb-ml4h-patient-map
#SBATCH --output=./outputs/runs/ml4h_2026/patient_cluster_bootstrap/logs/%j-%x.out
#SBATCH --partition=tier1q,tier2q,tier3q
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00

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

WORK_ROOT="${IRB_PATIENT_BOOTSTRAP_WORK_ROOT:-${IRB_HOME}/outputs/runs/ml4h_2026/patient_cluster_bootstrap}"
SUBJECT_MAP="${IRB_PATIENT_SUBJECT_MAP:-${WORK_ROOT}/hadm_to_subject.parquet}"
MEDS_RAW="${IRB_MEDS_RAW:-${DATA_DIR}/raw}"
mkdir -p "${WORK_ROOT}/logs" "${WORK_ROOT}/shards" "${WORK_ROOT}/tmp"

python pipeline/scripts/bootstrap_ml4h_patient_clusters.py \
  --mode map \
  --meds-raw "${MEDS_RAW}" \
  --subject-map "${SUBJECT_MAP}" \
  --work-root "${WORK_ROOT}"
