#!/bin/bash
# Submit provisional Exp2/Exp3 probes immediately from completed feature matrices.

set -euo pipefail

IRB_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${IRB_HOME}"

STAGE_ROOT="outputs/runs/ml4h_2026"
PROVISIONAL_ROOT="${STAGE_ROOT}/provisional_24to48h"
LOG_DIR="${PROVISIONAL_ROOT}/logs"
mkdir -p "${LOG_DIR}"

for experiment in exp2 exp3; do
  python utilities/scripts/generate_ml4h_24to48h_stage3_jobfiles.py \
    --experiment "${experiment}"
done

LAUNCH_STATE="${PROVISIONAL_ROOT}/exp2_exp3_probe_launch.tsv"
printf 'experiment\tfamily\tjob_id\ttasks\tjobfile\n' > "${LAUNCH_STATE}"
PARTITIONS="tier1q,tier2q,tier3q"
CPUS_PER_TASK=32
MEMORY_PER_TASK=96G

submit_family() {
  local experiment="$1"
  local family="$2"
  local expected_count="$3"
  local stage_dir="${STAGE_ROOT}/stage3_24to48h_${experiment}"
  local jobfile="${stage_dir}/${experiment}_24to48h_${family}.jobfile"
  local count
  count="$(rg -v '^(#|[[:space:]]*$)' "${jobfile}" | wc -l)"
  if [[ "${count}" -ne "${expected_count}" ]]; then
    echo "ERROR: ${experiment}/${family} expected ${expected_count} tasks; found ${count}" >&2
    exit 2
  fi
  local job_id
  job_id="$(
    sbatch \
      --parsable \
      --job-name="ml4h-${experiment}-24to48-${family}" \
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
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "${experiment}" "${family}" "${job_id}" "${count}" "${jobfile}" \
    | tee -a "${LAUNCH_STATE}"
}

submit_family exp2 binary 72
submit_family exp2 regression 72
submit_family exp3 binary 12
submit_family exp3 regression 12
