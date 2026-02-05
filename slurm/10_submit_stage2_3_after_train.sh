#!/bin/bash
# =============================================================================
# Submit Stage 2 (extract reps) -> Stage 3 (logistic regression) after training
# =============================================================================

set -euo pipefail

TRAIN_JOBID="${1:-}"
EXTRACT_JOBFILE="${2:-}"
PRED_JOBFILE="${3:-}"

if [[ -z "${TRAIN_JOBID}" || -z "${EXTRACT_JOBFILE}" || -z "${PRED_JOBFILE}" ]]; then
  echo "Usage: $0 <train_jobid> <extract_jobfile> <pred_jobfile>" >&2
  exit 2
fi

IRB_HOME="${IRB_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${IRB_HOME}"

for jf in "${EXTRACT_JOBFILE}" "${PRED_JOBFILE}"; do
  if [[ ! -f "${jf}" ]]; then
    echo "ERROR: jobfile not found: ${jf}" >&2
    exit 1
  fi
done

extract_tasks=$(grep -v '^#' "${EXTRACT_JOBFILE}" | grep -v '^[[:space:]]*$' | wc -l)
pred_tasks=$(grep -v '^#' "${PRED_JOBFILE}" | grep -v '^[[:space:]]*$' | wc -l)
if [[ "${extract_tasks}" -le 0 || "${pred_tasks}" -le 0 ]]; then
  echo "ERROR: empty jobfile(s): extract_tasks=${extract_tasks} pred_tasks=${pred_tasks}" >&2
  exit 1
fi

extract_array_max=$((extract_tasks - 1))
pred_array_max=$((pred_tasks - 1))

EXTRACT_JID=$(sbatch --parsable --dependency=afterok:"${TRAIN_JOBID}" --array=0-"${extract_array_max}" slurm/09_run_stage2_gpu2_extract.sh "${EXTRACT_JOBFILE}")
sbatch --dependency=afterok:"${EXTRACT_JID}" --array=0-"${pred_array_max}" slurm/11_run_stage3_tier2q_lr.sh "${PRED_JOBFILE}"
