#!/bin/bash
#SBATCH --job-name=irb-outcome-refresh
#SBATCH --output=./slurm/output/%A-%x.stdout
#SBATCH --partition=tier2q
#SBATCH --mem=128GB
#SBATCH --cpus-per-task=4
#SBATCH --time=0-06:00:00

# =============================================================================
# Refresh all extended outcome parquets needed for current benchmark reruns
# =============================================================================

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
mkdir -p slurm/output

require_dir() {
  local path="$1"
  local what="$2"
  if [[ ! -d "${path}" ]]; then
    echo "ERROR: missing ${what}: ${path}" >&2
    exit 1
  fi
}

require_file() {
  local path="$1"
  local what="$2"
  if [[ ! -f "${path}" ]]; then
    echo "ERROR: missing ${what}: ${path}" >&2
    exit 1
  fi
}

echo "=============================================="
echo "Refreshing extended outcomes for parity reruns"
echo "Host: $(hostname)"
echo "=============================================="

START_TS="$(date +%s)"

EXP12_EVENTS_DIR="benchmarks/mimic-meds-extraction/data/meds/data"
EXP12_TOKEN_ROOT="artifacts/runs/tokenized/mimiciv-3.1_meds_70-10-20"
EXP3_NATIVE_EVENTS_DIR="artifacts/runs/exp3/meds_icu"

declare -a REFRESH_SPECS=(
  "${EXP12_EVENTS_DIR}|${EXP12_TOKEN_ROOT}/deciles_none_unfused_time_tokens_first_24h-tokenized"
  "${EXP12_EVENTS_DIR}|${EXP12_TOKEN_ROOT}/deciles_none_fused_time_tokens_first_24h-tokenized"
  "${EXP12_EVENTS_DIR}|${EXP12_TOKEN_ROOT}/ventiles_none_unfused_time_tokens_first_24h-tokenized"
  "${EXP12_EVENTS_DIR}|${EXP12_TOKEN_ROOT}/ventiles_none_fused_time_tokens_first_24h-tokenized"
  "${EXP12_EVENTS_DIR}|${EXP12_TOKEN_ROOT}/ventiles_5-10-5_unfused_time_tokens_first_24h-tokenized"
  "${EXP12_EVENTS_DIR}|${EXP12_TOKEN_ROOT}/ventiles_5-10-5_fused_time_tokens_first_24h-tokenized"
  "${EXP12_EVENTS_DIR}|${EXP12_TOKEN_ROOT}/trentiles_none_unfused_time_tokens_first_24h-tokenized"
  "${EXP12_EVENTS_DIR}|${EXP12_TOKEN_ROOT}/trentiles_none_fused_time_tokens_first_24h-tokenized"
  "${EXP12_EVENTS_DIR}|${EXP12_TOKEN_ROOT}/trentiles_10-10-10_unfused_time_tokens_first_24h-tokenized"
  "${EXP12_EVENTS_DIR}|${EXP12_TOKEN_ROOT}/trentiles_10-10-10_fused_time_tokens_first_24h-tokenized"
  "${EXP12_EVENTS_DIR}|${EXP12_TOKEN_ROOT}/centiles_none_unfused_time_tokens_first_24h-tokenized"
  "${EXP12_EVENTS_DIR}|${EXP12_TOKEN_ROOT}/centiles_none_fused_time_tokens_first_24h-tokenized"
  "${EXP12_EVENTS_DIR}|${EXP12_TOKEN_ROOT}/deciles_none_unfused_time_rope_first_24h-tokenized"
  "${EXP12_EVENTS_DIR}|${EXP12_TOKEN_ROOT}/deciles_none_unfused_time_tokens_numencxval_first_24h-tokenized"
  "${EXP12_EVENTS_DIR}|${EXP12_TOKEN_ROOT}/deciles_none_unfused_time_rope_numencxval_first_24h-tokenized"
  "${EXP3_NATIVE_EVENTS_DIR}|artifacts/runs/exp3/meds_icu/deciles_none_unfused_time_rope_first_24h-tokenized"
  "${EXP3_NATIVE_EVENTS_DIR}|artifacts/runs/exp3/arms/meds_mapped/deciles_none_unfused_time_rope_first_24h-tokenized"
  "${EXP3_NATIVE_EVENTS_DIR}|artifacts/runs/exp3/arms/meds_randomized/deciles_none_unfused_time_rope_first_24h-tokenized"
  "${EXP3_NATIVE_EVENTS_DIR}|artifacts/runs/exp3/arms/meds_freqmatched/deciles_none_unfused_time_rope_first_24h-tokenized"
)

done_n=0

for spec in "${REFRESH_SPECS[@]}"; do
  IFS="|" read -r events_dir tokdir <<< "${spec}"
  require_dir "${events_dir}" "MEDS events dir"
  require_dir "${tokdir}" "tokenized dir"

  echo ""
  echo "--- Refreshing extended outcomes for: $(basename "${tokdir}") ---"
  echo "    events: ${events_dir}"
  python3 pipeline/scripts/extract_extended_outcomes.py \
    --meds_events_dir "${events_dir}" \
    --tokenized_dir "${tokdir}" \
    --splits train,val,test \
    --strict
  for s in train val test; do
    require_file "${tokdir}/${s}/tokens_timelines_extended_outcomes.parquet" "extended outcomes parquet"
  done
  ((done_n+=1))
done

END_TS="$(date +%s)"
echo ""
echo "Completed roots: ${done_n}"
echo "Walltime (s):    $((END_TS - START_TS))"
