#!/bin/bash
#SBATCH --job-name=irb-diag-extract
#SBATCH --output=./slurm/output/%A_%a-%x.stdout
#SBATCH --partition=tier2q
#SBATCH --mem=128GB
#SBATCH --cpus-per-task=4
#SBATCH --time=0-04:00:00

# =============================================================================
# Extract extended outcomes (Tier 2 + Tier 3) from MEDS events
# =============================================================================
#
# This is a one-time extraction step that must run before the Tier 2/3
# diagnostic evaluations. It walks through each data_version directory,
# scans the corresponding MEDS events, and produces
# `tokens_timelines_extended_outcomes.parquet` per split.
#
# Usage:
#   sbatch slurm/13_extract_extended_outcomes.sh
#
# Or run interactively for testing:
#   bash slurm/13_extract_extended_outcomes.sh
# =============================================================================

set -euo pipefail

find_repo_root() {
  local d="$1"
  while [[ "$d" != "/" ]]; do
    if [[ -f "$d/run_experiments.py" && -d "$d/slurm" ]]; then
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

echo "=============================================="
echo "Extracting extended outcomes"
echo "Host: $(hostname)"
echo "=============================================="

START_TS="$(date +%s)"

MEDS_EVENTS_DIR="benchmarks/mimic-meds-extraction/data/meds/data"

# All Exp2 tokenized directories that need extended outcome extraction.
# These are the -tokenized directories under ${MEDS_EVENTS_DIR}/.
# Note: discrete and soft share the same tokenized dirs (time_tokens / time_rope).
# xVal uses its own dirs (time_tokens_numencxval / time_rope_numencxval).
TOKENIZED_DIRS=(
  "${MEDS_EVENTS_DIR}/deciles_none_unfused_time_tokens_first_24h-tokenized"
  "${MEDS_EVENTS_DIR}/deciles_none_unfused_time_rope_first_24h-tokenized"
  "${MEDS_EVENTS_DIR}/deciles_none_unfused_time_tokens_numencxval_first_24h-tokenized"
  "${MEDS_EVENTS_DIR}/deciles_none_unfused_time_rope_numencxval_first_24h-tokenized"
)

for tokdir in "${TOKENIZED_DIRS[@]}"; do
  if [[ ! -d "${tokdir}" ]]; then
    echo "SKIP: ${tokdir} does not exist"
    continue
  fi

  echo ""
  echo "--- Extracting extended outcomes for: $(basename ${tokdir}) ---"
  python3 scripts/extract_extended_outcomes.py \
    --meds_events_dir "${MEDS_EVENTS_DIR}" \
    --tokenized_dir "${tokdir}" \
    --splits train,val,test
done

END_TS="$(date +%s)"
echo ""
echo "Done. Walltime (s): $((END_TS - START_TS))"
