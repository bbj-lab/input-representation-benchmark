#!/bin/bash
# Submit the completed ML4H Exp1 feature sets for CPU-only Stage-3 probes.
#
# Each array task evaluates every task in one family for one checkpoint.  The
# two arrays intentionally have no concurrency cap so Slurm can occupy all
# currently available CPU capacity across the general CPU tiers.

set -euo pipefail

IRB_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${IRB_HOME}"

STAGE_DIR="outputs/runs/ml4h_2026/stage3"
LOG_DIR="${STAGE_DIR}/logs"
LAUNCH_STATE="${STAGE_DIR}/exp1_stage3_launch.tsv"
LAUNCH_LOCK="${STAGE_DIR}/.exp1_stage3_launch.lock"
PARTITIONS="tier1q,tier2q,tier3q"
CPUS_PER_TASK=32
# Feature tensors sit near ~64 GB resident. 64G OOMs; 128G cannot fit the
# common 128 GB nodes after OS reservation, so request 96G.
MEMORY_PER_TASK=96G

python utilities/scripts/generate_ml4h_exp1_stage3_jobfiles.py \
  --output-dir "${STAGE_DIR}"
mkdir -p "${LOG_DIR}"
touch "${LAUNCH_STATE}"
exec 9>"${LAUNCH_LOCK}"
flock 9

submit_family() {
  local family="$1"
  local jobfile="${STAGE_DIR}/exp1_stage3_${family}.jobfile"
  local existing_job_id
  existing_job_id="$(awk -F $'\t' -v family="${family}" '$1 == family {print $2; exit}' "${LAUNCH_STATE}")"
  if [[ -n "${existing_job_id}" ]]; then
    printf '%s\t%s\t%s\t%s\t%s\n' \
      "${family}" "${existing_job_id}" "72" "${jobfile}" "already-submitted"
    return 0
  fi

  local count
  count="$(rg -v '^(#|[[:space:]]*$)' "${jobfile}" | wc -l)"
  if [[ "${count}" -ne 72 ]]; then
    echo "ERROR: expected 72 ${family} tasks in ${jobfile}; found ${count}" >&2
    exit 2
  fi

  local job_id
  job_id="$(
    sbatch \
      --parsable \
      --job-name="ml4h-e1p-${family}" \
      --partition="${PARTITIONS}" \
      --nodes=1 \
      --ntasks=1 \
      --cpus-per-task="${CPUS_PER_TASK}" \
      --mem="${MEMORY_PER_TASK}" \
      --time=2-00:00:00 \
      --open-mode=append \
      --output="${LOG_DIR}/%A_%a-%x.out" \
      --export="ALL,IRB_STAGE3_BOOTSTRAP_N=0,OMP_NUM_THREADS=${CPUS_PER_TASK},OPENBLAS_NUM_THREADS=${CPUS_PER_TASK},MKL_NUM_THREADS=${CPUS_PER_TASK},NUMEXPR_MAX_THREADS=${CPUS_PER_TASK}" \
      --array="0-$((count - 1))" \
      slurm/11_run_stage3_tier2q_lr.sh "${jobfile}"
  )"
  printf '%s\t%s\t%s\n' "${family}" "${job_id}" "$(date -Is)" >> "${LAUNCH_STATE}"
  printf '%s\t%s\t%s\t%s\t%s\n' "${family}" "${job_id}" "${count}" "${jobfile}" "submitted"
}

echo -e "family\tjob_id\ttasks\tjobfile\tstatus"
submit_family binary
submit_family regression
