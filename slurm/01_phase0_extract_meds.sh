#!/bin/bash
#SBATCH --job-name=irb-extract-meds
#SBATCH --output=./slurm/output/%j-%x.stdout
#SBATCH --partition=tier2q
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=450GB

# =============================================================================
# Phase 0: MEDS extraction wrapper
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
export IRB_CONDA_ENV="${IRB_CONDA_ENV:-meds-extract}"
source "${IRB_HOME}/slurm/00_preamble.sh"
cd "${IRB_HOME}"

echo "=============================================="
echo "MEDS Extraction (Phase 0)"
echo "=============================================="

(
  cd "${IRB_HOME}/benchmarks/mimic-meds-extraction"
  bash scripts/01_extract_meds_full.sh "${IRB_MEDS_WORKERS:-3}"
)

MEDS_DATA_BASE="${IRB_HOME}/benchmarks/mimic-meds-extraction/data/meds/data"
RAW_BASE="${MEDS_DATA_BASE}/raw"
VAL_SRC=""
if [[ -d "${MEDS_DATA_BASE}/val" ]]; then
  VAL_SRC="val"
elif [[ -d "${MEDS_DATA_BASE}/tuning" ]]; then
  VAL_SRC="tuning"
elif [[ -d "${MEDS_DATA_BASE}/validation" ]]; then
  VAL_SRC="validation"
else
  echo "ERROR: Could not locate a validation split under ${MEDS_DATA_BASE}." >&2
  echo "  Expected one of: val/, tuning/, or validation/." >&2
  exit 1
fi

mkdir -p "${RAW_BASE}"
ensure_raw_link() {
  local link_name="$1"
  local target="$2"
  local link_path="${RAW_BASE}/${link_name}"
  if [[ -L "${link_path}" ]]; then
    local cur
    cur="$(readlink "${link_path}")"
    if [[ "${cur}" != "${target}" ]]; then
      rm -f "${link_path}"
      ln -s "${target}" "${link_path}"
    fi
  elif [[ -e "${link_path}" ]]; then
    echo "ERROR: ${link_path} exists and is not a symlink." >&2
    exit 1
  else
    ln -s "${target}" "${link_path}"
  fi
}
(
  cd "${RAW_BASE}"
  ensure_raw_link train ../train
  ensure_raw_link test ../test
  ensure_raw_link val "../${VAL_SRC}"
)

echo ""
echo "=============================================="
echo "MEDS Extraction Complete!"
echo "=============================================="
echo "Output: ${IRB_HOME}/benchmarks/mimic-meds-extraction/data/meds"
echo ""
echo "Next steps:"
echo "  1. Generate job files: python pipeline/run_experiments.py --mode demo"
echo "  2. Exp1 Stage 0 (tier2q CPU): sbatch --array=0-11 slurm/02_run_stage0_tier2q_tokenize.sh slurm/generated/demo/04_exp1_stage0_tokenize.jobfile"
echo "  3. Exp1 Stage 1 (4 GPUs/job): TRAIN_JID=\$(sbatch --parsable --array=0-11 slurm/05_run_stage1_gpu4_train.sh slurm/generated/demo/07a_exp1_stage1_train.jobfile)"
echo "  4. Exp1 Stage 2 (1 GPU/job): EXTRACT_JID=\$(sbatch --parsable --dependency=afterok:\"\${TRAIN_JID}\" --array=0-11 slurm/09_run_stage2_gpu2_extract.sh slurm/generated/demo/07b_exp1_stage2_extract_reps.jobfile)"
echo "  5. Exp1 Stage 3 (tier2q LR): sbatch --dependency=afterok:\"\${EXTRACT_JID}\" --array=0-11 slurm/11_run_stage3_tier2q_lr.sh slurm/generated/demo/07c_exp1_stage3_lr.jobfile"

