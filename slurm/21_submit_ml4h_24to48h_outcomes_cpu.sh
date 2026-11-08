#!/bin/bash
# Submit isolated 24-to-48-hour outcome extraction for all ML4H data roots.

set -euo pipefail

IRB_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${IRB_HOME}"

STAGE_DIR="outputs/runs/ml4h_2026/provisional_24to48h"
LOG_DIR="${STAGE_DIR}/logs"
mkdir -p "${LOG_DIR}"

python utilities/scripts/generate_ml4h_24to48h_outcome_jobfiles.py \
  --output-dir "${STAGE_DIR}"

JOBFILE="${STAGE_DIR}/materialize_outcomes_24to48h.jobfile"
count="$(rg -v '^(#|[[:space:]]*$)' "${JOBFILE}" | wc -l)"
if [[ "${count}" -ne 17 ]]; then
  echo "ERROR: expected 17 unique outcome-materialization tasks; found ${count}" >&2
  exit 2
fi

job_id="$(
  sbatch \
    --parsable \
    --job-name=ml4h-24to48-outcomes \
    --partition=tier3q \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=8 \
    --mem=512G \
    --time=12:00:00 \
    --open-mode=append \
    --output="${LOG_DIR}/%A_%a-%x.out" \
    --export=ALL,OMP_NUM_THREADS=8,POLARS_MAX_THREADS=8 \
    --array="0-$((count - 1))" \
    slurm/15_run_stats_cpu_jobfile.sh "${JOBFILE}"
)"

printf 'outcome_materialization\t%s\t%s\t%s\n' "${job_id}" "${count}" "${JOBFILE}" \
  | tee "${STAGE_DIR}/outcome_launch.tsv"

REPAIR_LAUNCH="outputs/runs/ml4h_2026/stage2_exp1_padding_repair/exp1_padding_repair_launch.tsv"
if [[ ! -f "${REPAIR_LAUNCH}" ]]; then
  echo "ERROR: missing active Exp1 padded-feature repair state: ${REPAIR_LAUNCH}" >&2
  exit 2
fi
exp1_stage3_prepare="$(awk -F $'\t' '$1 == "stage3_prepare" {print $2}' "${REPAIR_LAUNCH}")"
if [[ -z "${exp1_stage3_prepare}" ]]; then
  echo "ERROR: missing active Exp1 padded-feature repair Stage-3 preparation job." >&2
  exit 2
fi

probe_prepare_job="$(
  sbatch \
    --parsable \
    --dependency="afterok:${job_id}:${exp1_stage3_prepare}" \
    --partition=tier2q \
    --cpus-per-task=4 \
    --mem=32G \
    --time=01:00:00 \
    --job-name=irb-ml4h-24to48-prepare \
    --output=./slurm/output/%A-%x.stdout \
    slurm/22_prepare_submit_ml4h_24to48h_probes.sh
)"
printf 'probe_prepare\t%s\t1\tslurm/22_prepare_submit_ml4h_24to48h_probes.sh\n' \
  "${probe_prepare_job}" | tee -a "${STAGE_DIR}/outcome_launch.tsv"
