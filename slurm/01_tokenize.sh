#!/bin/bash
#SBATCH --job-name=tokenize
#SBATCH --output=./slurm/output/%j-%x.stdout
#SBATCH --partition=tier2q
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
echo "  Job ID: ${SLURM_JOB_ID}"
echo "  Data version: ${DATA_VERSION}"
echo "  Config: ${CONFIG_LOC}"
echo "=============================================="

cd "${FMS_EHRS_HOME}"

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
