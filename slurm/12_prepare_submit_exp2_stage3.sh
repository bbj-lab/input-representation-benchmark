#!/bin/bash
#SBATCH --job-name=irb-exp2-stage3-prepare
#SBATCH --output=./slurm/output/%A-%x.stdout
#SBATCH --partition=tier2q
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

# After Exp2 Stage-2 finishes: validate features, write Stage-3 jobfiles, submit
# binary + regression arrays at fixed probes (C=0.01, ridge alpha=100).

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

STAGE_DIR="outputs/runs/ml4h_2026/stage3_exp2"
LOG_DIR="${STAGE_DIR}/logs"
mkdir -p "${LOG_DIR}"

python utilities/scripts/generate_ml4h_exp2_stage3_jobfiles.py \
  --stage2-manifest outputs/runs/ml4h_2026/stage2_exp2/exp2_extract_manifest.json \
  --output-dir "${STAGE_DIR}"

PARTITIONS="tier1q,tier2q,tier3q"
CPUS_PER_TASK=32
MEMORY_PER_TASK=96G
LAUNCH_STATE="${STAGE_DIR}/exp2_stage3_launch.tsv"
: > "${LAUNCH_STATE}"

submit_family() {
  local family="$1"
  local jobfile="${STAGE_DIR}/exp2_stage3_${family}.jobfile"
  local count
  count="$(rg -v '^(#|[[:space:]]*$)' "${jobfile}" | wc -l)"
  if [[ "${count}" -ne 72 ]]; then
    echo "ERROR: expected 72 ${family} tasks; found ${count}" >&2
    exit 2
  fi
  local job_id
  job_id="$(
    sbatch \
      --parsable \
      --job-name="ml4h-e2p-${family}" \
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
  printf '%s\t%s\t%s\t%s\n' "${family}" "${job_id}" "${count}" "${jobfile}" | tee -a "${LAUNCH_STATE}"
}

echo -e "family\tjob_id\ttasks\tjobfile"
submit_family binary
submit_family regression
