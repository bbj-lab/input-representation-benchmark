#!/bin/bash
#SBATCH --job-name=irb-exp3-outcome-repair
#SBATCH --output=./slurm/output/%A-%x.stdout
#SBATCH --partition=tier2q
#SBATCH --mem=128GB
#SBATCH --cpus-per-task=4
#SBATCH --time=0-02:00:00

# =============================================================================
# Refresh extended outcomes for Exp3 arms using the native ICU clinical events
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
echo "Refreshing Exp3 extended outcomes from native ICU events"
echo "Host: $(hostname)"
echo "=============================================="

START_TS="$(date +%s)"

EXP3_NATIVE_EVENTS_DIR="artifacts/runs/exp3/meds_icu"
require_dir "${EXP3_NATIVE_EVENTS_DIR}" "native Exp3 ICU MEDS events dir"
TOKENIZED_DIRS=(
  "artifacts/runs/exp3/meds_icu/deciles_none_unfused_time_rope_first_24h-tokenized"
  "artifacts/runs/exp3/arms/meds_mapped/deciles_none_unfused_time_rope_first_24h-tokenized"
  "artifacts/runs/exp3/arms/meds_randomized/deciles_none_unfused_time_rope_first_24h-tokenized"
  "artifacts/runs/exp3/arms/meds_freqmatched/deciles_none_unfused_time_rope_first_24h-tokenized"
)

done_n=0
for tokdir in "${TOKENIZED_DIRS[@]}"; do
  require_dir "${tokdir}" "Exp3 tokenized dir"
  echo ""
  echo "--- Repairing $(basename "${tokdir}") with native ICU events ---"
  python3 pipeline/scripts/extract_extended_outcomes.py \
    --meds_events_dir "${EXP3_NATIVE_EVENTS_DIR}" \
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
