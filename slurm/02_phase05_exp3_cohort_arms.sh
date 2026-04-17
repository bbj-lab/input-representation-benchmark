#!/bin/bash
#SBATCH --job-name=irb-exp3-cohort
#SBATCH --output=./slurm/output/%j-%x.stdout
#SBATCH --partition=tier2q
#SBATCH --time=0-04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=128GB

# =============================================================================
# Stage -0.5: Build Exp3 ICU cohort + semantic-control arms
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

MIMIC_DIR="${IRB_HOME}/physionet.org/files/mimiciv/3.1"
MEDS_EVENTS_DIR="${IRB_HOME}/benchmarks/mimic-meds-extraction/data/meds/data"
EXP3_ROOT="${IRB_HOME}/artifacts/runs/exp3"
COHORT_DIR="${EXP3_ROOT}/cohort"
MEDS_ICU_DIR="${EXP3_ROOT}/meds_icu"
ARMS_DIR="${EXP3_ROOT}/arms"

echo "=============================================="
echo "Stage -0.5: Exp3 cohort + semantic arms"
echo "Host: $(hostname)"
echo "=============================================="

# Step 1: align cohorts (patient-level splits + hadm projection)
echo ""
echo "=== Step 1: align_cohorts ==="
python pipeline/scripts/align_cohorts.py \
  --mimic_dir "${MIMIC_DIR}" \
  --output_dir "${COHORT_DIR}" \
  --data_seed 42

# Step 2: filter MEDS events to ICU cohort (native arm)
echo ""
echo "=== Step 2: split_meds_by_hadm_splits (native ICU) ==="
python pipeline/scripts/split_meds_by_hadm_splits.py \
  --meds_in_dir "${MEDS_EVENTS_DIR}" \
  --splits_dir "${COHORT_DIR}" \
  --meds_out_dir "${MEDS_ICU_DIR}"

# Step 3: build semantic-control arms (mapped, randomized, freqmatched)
echo ""
echo "=== Step 3: build_exp3_meds_semantics_arms ==="
python pipeline/scripts/build_exp3_meds_semantics_arms.py \
  --meds_in_dir "${MEDS_ICU_DIR}" \
  --out_root "${ARMS_DIR}" \
  --clif_mimic_repo "${IRB_HOME}/../CLIF-MIMIC" \
  --seed 42

echo ""
echo "=============================================="
echo "Stage -0.5 complete"
echo "=============================================="
echo "  Cohort: ${COHORT_DIR}"
echo "  Native: ${MEDS_ICU_DIR}"
echo "  Arms:   ${ARMS_DIR}"
