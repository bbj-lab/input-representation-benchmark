#!/bin/bash
# Refresh extended outcomes for the ML4H Exp2 tokenized datasets.
#
# The remaining stale artifacts are the test outcome tables for three shared
# Exp2 tokenizations. Train/val already match their current timelines and are
# deliberately left untouched. `event_order` shares the `time_rope`
# tokenization, so three datasets cover all twelve Exp2 arms.

set -euo pipefail

IRB_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${IRB_HOME}"

STAGE_DIR="outputs/runs/ml4h_2026/stage3_exp2"
LOG_DIR="${STAGE_DIR}/logs"
JOBFILE="${STAGE_DIR}/exp2_refresh_test_extended_outcomes.jobfile"
TOKENIZED_ROOT="outputs/runs/tokenized/mimiciv-3.1_meds_70-10-20"
EVENTS_DIR="benchmarks/mimic-meds-extraction/data/meds/data"
CPUS_PER_TASK=8
# Six of the twelve Exp1 datasets exceeded 256G during the July refresh.
MEMORY_PER_TASK=512G

DATA_VERSIONS=(
  deciles_none_unfused_time_rope
  deciles_none_unfused_time_tokens_numencxval
  deciles_none_unfused_time_rope_numencxval
)

mkdir -p "${LOG_DIR}"
{
  echo "# ML4H Exp2 test-only extended-outcome refresh"
  echo "# Replaces only stale 63,750-row test outcomes with 85,057-row outcomes."
  for data_version in "${DATA_VERSIONS[@]}"; do
    tokenized_dir="${TOKENIZED_ROOT}/${data_version}_first_24h-tokenized"
    if [[ ! -d "${tokenized_dir}" ]]; then
      echo "ERROR: missing tokenized dataset ${tokenized_dir}" >&2
      exit 2
    fi
    echo "python pipeline/scripts/extract_extended_outcomes.py" \
      "--meds_events_dir $(realpath "${EVENTS_DIR}")" \
      "--tokenized_dir $(realpath "${tokenized_dir}")" \
      "--splits test --strict"
  done
} > "${JOBFILE}"

count="$(rg -v '^(#|[[:space:]]*$)' "${JOBFILE}" | wc -l)"
if [[ "${count}" -ne "${#DATA_VERSIONS[@]}" ]]; then
  echo "ERROR: expected ${#DATA_VERSIONS[@]} refresh tasks in ${JOBFILE}; found ${count}" >&2
  exit 2
fi

sbatch \
  --parsable \
  --job-name=ml4h-e2-test-outcomes \
  --partition=tier3q \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task="${CPUS_PER_TASK}" \
  --mem="${MEMORY_PER_TASK}" \
  --time=12:00:00 \
  --open-mode=append \
  --output="${LOG_DIR}/%A_%a-%x.out" \
  --export="ALL,OMP_NUM_THREADS=${CPUS_PER_TASK},POLARS_MAX_THREADS=${CPUS_PER_TASK}" \
  --array="0-$((count - 1))" \
  slurm/15_run_stats_cpu_jobfile.sh "${JOBFILE}"
