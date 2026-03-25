#!/bin/bash
# Submit aligned family-level CI/significance regeneration on CPU partitions.

set -euo pipefail

IRB_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARRAY_JOBFILE="${ARRAY_JOBFILE:-${IRB_HOME}/slurm/generated/statistics/aligned_family_stats.jobfile}"
COMBINE_JOBFILE="${COMBINE_JOBFILE:-${IRB_HOME}/slurm/generated/statistics/aligned_family_stats_combine.jobfile}"
RUNNER="${IRB_HOME}/slurm/15_run_stats_cpu_jobfile.sh"

PRIMARY_PARTITION="${PRIMARY_PARTITION:-tier2q}"
FALLBACK_PARTITION="${FALLBACK_PARTITION:-tier1q}"
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-}"

if [[ ! -f "${ARRAY_JOBFILE}" ]]; then
  echo "ERROR: missing jobfile: ${ARRAY_JOBFILE}" >&2
  exit 1
fi
if [[ ! -f "${COMBINE_JOBFILE}" ]]; then
  echo "ERROR: missing combine jobfile: ${COMBINE_JOBFILE}" >&2
  exit 1
fi

count_commands() {
  grep -v '^#' "$1" | grep -v '^[[:space:]]*$' | wc -l | tr -d ' '
}

submit_array() {
  local partition="$1"
  local n_lines="$2"
  if [[ -n "${ARRAY_CONCURRENCY}" && "${ARRAY_CONCURRENCY}" -gt 0 ]]; then
    sbatch --parsable \
      --partition="${partition}" \
      --array="0-$((n_lines - 1))%${ARRAY_CONCURRENCY}" \
      "${RUNNER}" "${ARRAY_JOBFILE}"
  else
    sbatch --parsable \
      --partition="${partition}" \
      --array="0-$((n_lines - 1))" \
      "${RUNNER}" "${ARRAY_JOBFILE}"
  fi
}

submit_combine() {
  local partition="$1"
  local dep="$2"
  sbatch --parsable \
    --partition="${partition}" \
    --dependency="afterok:${dep}" \
    "${RUNNER}" "${COMBINE_JOBFILE}"
}

N_LINES="$(count_commands "${ARRAY_JOBFILE}")"
if [[ "${N_LINES}" -le 0 ]]; then
  echo "ERROR: no commands found in ${ARRAY_JOBFILE}" >&2
  exit 1
fi

echo "[stats] submitting ${N_LINES} family jobs on ${PRIMARY_PARTITION}..."
if ARRAY_JID="$(submit_array "${PRIMARY_PARTITION}" "${N_LINES}")"; then
  PARTITION_USED="${PRIMARY_PARTITION}"
else
  echo "[stats] primary submit failed; retrying on ${FALLBACK_PARTITION}..." >&2
  ARRAY_JID="$(submit_array "${FALLBACK_PARTITION}" "${N_LINES}")"
  PARTITION_USED="${FALLBACK_PARTITION}"
fi

echo "[stats] array job id: ${ARRAY_JID} (partition=${PARTITION_USED})"
COMBINE_JID="$(submit_combine "${PARTITION_USED}" "${ARRAY_JID}")"
echo "[stats] combine job id: ${COMBINE_JID} (partition=${PARTITION_USED})"
