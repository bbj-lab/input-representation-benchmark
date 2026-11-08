#!/bin/bash
#SBATCH --job-name=irb-exp3-cohort
#SBATCH --output=./slurm/output/%j-%x.stdout
#SBATCH --partition=tier2q
#SBATCH --time=0-04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=128GB

# =============================================================================
# Stage -0.5: Build matched raw-MIMIC and full-CLIF Exp3 arms
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
EXP3_ROOT="${IRB_HOME}/outputs/runs/exp3"
COHORT_DIR="${EXP3_ROOT}/cohort"
MEDS_ICU_DIR="${EXP3_ROOT}/meds_icu"
CLIF_SRC_DIR="${CLIF_SRC_DIR:-${IRB_HOME}/../CLIF-SRC/mimic-2.1.0}"
MEDS_CLIF_DIR="${EXP3_ROOT}/meds_clif"
FMS_EHRS_HOME="${FMS_EHRS_HOME:-$(realpath "${IRB_HOME}/../fms-ehrs")}"
COCOA_CONFIG="${FMS_EHRS_HOME}/fms_ehrs/config/cocoa-exp3-clif-full.yaml"
FROZEN_SPLITS_MANIFEST="${COHORT_DIR}/frozen_split_manifest.json"
COCOA_WORK_DIR="${EXP3_ROOT}/cocoa_work"

echo "=============================================="
echo "Stage -0.5: Exp3 raw-MIMIC + CLIF arms"
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

# Step 3: freeze the verified native split, then collate full CLIF through the
# FMS-owned Cocoa adapter. Cocoa's own chronological splitting is never used.
echo ""
echo "=== Step 3a: freeze native H_ICU split ==="
python "${FMS_EHRS_HOME}/fms_ehrs/scripts/collate_clif_with_cocoa.py" \
  freeze-splits \
  --cohort-dir "${COHORT_DIR}" \
  --raw-meds-dir "${MEDS_ICU_DIR}" \
  --output "${FROZEN_SPLITS_MANIFEST}"

echo ""
echo "=== Step 3b: Cocoa collation + FMS MEDS adaptation ==="
python "${FMS_EHRS_HOME}/fms_ehrs/scripts/collate_clif_with_cocoa.py" \
  materialize \
  --raw-data-home "${CLIF_SRC_DIR}" \
  --collation-config "${COCOA_CONFIG}" \
  --frozen-splits-manifest "${FROZEN_SPLITS_MANIFEST}" \
  --output-meds-dir "${MEDS_CLIF_DIR}" \
  --work-dir "${COCOA_WORK_DIR}"

echo ""
echo "=============================================="
echo "Stage -0.5 complete"
echo "=============================================="
echo "  Cohort: ${COHORT_DIR}"
echo "  Raw MIMIC: ${MEDS_ICU_DIR}"
echo "  Full CLIF: ${MEDS_CLIF_DIR}"
