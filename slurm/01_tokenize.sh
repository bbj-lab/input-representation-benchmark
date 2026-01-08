#!/bin/bash
#SBATCH --job-name=tokenize
#SBATCH --output=./slurm/output/%j-%x.stdout
#SBATCH --partition=gpuq
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=150GB

# =============================================================================
# Phase 1: Tokenization
# =============================================================================
# Tokenizes MEDS data using fms-ehrs tokenizer.
# Run this AFTER MEDS extraction (00_extract_meds.sh).
#
# Usage:
#   sbatch slurm/01_tokenize.sh
#
# This creates tokenized data at:
#   ${DATA_DIR}/${DATA_VERSION}-tokenized/
# =============================================================================

set -euo pipefail

# Source environment (robust to SLURM script spooling into /var/spool/...)
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
export IRB_CONDA_ENV="input-rep"
source "${IRB_HOME}/slurm/preamble.sh"

# Default data version (can be overridden)
DATA_VERSION="${DATA_VERSION:-deciles_none_unfused}"
CONFIG_LOC="${FMS_EHRS_HOME}/fms_ehrs/config/mimic-meds-ed.yaml"

echo "=============================================="
echo "Tokenization Job"
echo "=============================================="
echo "  Job ID: ${SLURM_JOB_ID:-local}"
echo "  Data version: ${DATA_VERSION}"
echo "  Config: ${CONFIG_LOC}"
echo "=============================================="

cd "${FMS_EHRS_HOME}"

# =============================================================================
# Ensure MEDS input layout shim exists (raw/{train,val,test})
# =============================================================================
# `tokenize_w_config.py` always reads from:
#   --data_dir/<data_version_in>/{train,val,test}
# and this job sets:
#   --data_version_in raw
# Our MEDS extraction produces:
#   ${DATA_DIR}/data/{train,tuning,test}
# so we create symlinks (no data duplication):
#   ${DATA_DIR}/data/raw/train -> ../train
#   ${DATA_DIR}/data/raw/val   -> ../tuning   (or ../val if it exists)
#   ${DATA_DIR}/data/raw/test  -> ../test
MEDS_DATA_BASE="${DATA_DIR}/data"
RAW_BASE="${MEDS_DATA_BASE}/raw"

if [[ ! -d "${MEDS_DATA_BASE}" ]]; then
    echo "ERROR: Expected MEDS data directory not found: ${MEDS_DATA_BASE}"
    exit 1
fi
if [[ ! -d "${MEDS_DATA_BASE}/train" ]]; then
    echo "ERROR: Missing expected split dir: ${MEDS_DATA_BASE}/train"
    exit 1
fi
if [[ ! -d "${MEDS_DATA_BASE}/test" ]]; then
    echo "ERROR: Missing expected split dir: ${MEDS_DATA_BASE}/test"
    exit 1
fi

VAL_SRC=""
if [[ -d "${MEDS_DATA_BASE}/val" ]]; then
    VAL_SRC="val"
elif [[ -d "${MEDS_DATA_BASE}/tuning" ]]; then
    VAL_SRC="tuning"
else
    echo "ERROR: Missing validation split dir. Expected one of:"
    echo "  - ${MEDS_DATA_BASE}/val"
    echo "  - ${MEDS_DATA_BASE}/tuning"
    exit 1
fi

mkdir -p "${RAW_BASE}"
(
    cd "${RAW_BASE}"
    [[ -e train ]] || ln -s ../train train
    [[ -e test ]] || ln -s ../test test
    [[ -e val ]] || ln -s "../${VAL_SRC}" val
)

echo "Using input split layout:"
echo "  ${RAW_BASE}/train -> ../train"
echo "  ${RAW_BASE}/val   -> ../${VAL_SRC}"
echo "  ${RAW_BASE}/test  -> ../test"

python fms_ehrs/scripts/tokenize_w_config.py \
    --data_dir "${DATA_DIR}/data" \
    --data_version_in raw \
    --data_version_out "${DATA_VERSION}" \
    --config_loc "${CONFIG_LOC}" \
    --include_24h_cut

echo ""
echo "=============================================="
echo "Tokenization Complete!"
echo "=============================================="
echo "Output: ${DATA_DIR}/data/${DATA_VERSION}-tokenized"
