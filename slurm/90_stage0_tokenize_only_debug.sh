#!/bin/bash
#SBATCH --job-name=tokenize
#SBATCH --output=./slurm/output/%j-%x.stdout
#SBATCH --partition=tier2q
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=150GB

# =============================================================================
# Stage 0 (manual): tokenization only (debug helper)
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
export IRB_CONDA_ENV="input-rep"
source "${IRB_HOME}/slurm/00_preamble.sh"

DATA_VERSION="${DATA_VERSION:-deciles_none_unfused_time_tokens}"
CONFIG_LOC="${FMS_EHRS_HOME}/fms_ehrs/config/mimic-meds-ed.yaml"

echo "=============================================="
echo "Tokenization Job"
echo "=============================================="
echo "  Job ID: ${SLURM_JOB_ID:-local}"
echo "  Data version: ${DATA_VERSION}"
echo "  Config: ${CONFIG_LOC}"
echo "=============================================="

cd "${FMS_EHRS_HOME}"

python fms_ehrs/scripts/tokenize_w_config.py \
  --data_dir "${DATA_DIR}" \
  --data_version_in raw \
  --data_version_out "${DATA_VERSION}" \
  --config_loc "${CONFIG_LOC}" \
  --quantizer deciles \
  --clinical_anchoring none \
  --include_ref_ranges false \
  --include_time_spacing_tokens true \
  --fused_category_values false \
  --include_24h_cut

