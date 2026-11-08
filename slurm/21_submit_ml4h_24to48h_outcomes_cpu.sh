#!/bin/bash
# Stage 0.5: submit the 24-to-48-hour outcome extraction for all ML4H data roots.
# Then submit Stage 3 (slurm/22_submit_ml4h_24to48h_probes.sh) after it completes.

set -euo pipefail

IRB_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${IRB_HOME}"

STAGE_DIR="outputs/runs/ml4h_2026/outcomes_24to48h"
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

probe_submit_job="$(
  sbatch \
    --parsable \
    --dependency="afterok:${job_id}" \
    slurm/22_submit_ml4h_24to48h_probes.sh
)"
printf 'probe_submit\t%s\t1\tslurm/22_submit_ml4h_24to48h_probes.sh\n' \
  "${probe_submit_job}" | tee -a "${STAGE_DIR}/outcome_launch.tsv"
