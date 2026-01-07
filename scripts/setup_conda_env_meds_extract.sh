#!/usr/bin/env bash
#
# Create the `meds-extract` conda environment for MEDS extraction (Job 1 / Phase 0).
#
# Why this exists
# ---------------
# The benchmark uses two separate Python envs due to `polars` version constraints:
# - `meds-extract`: MEDS extraction via `MEDS_transforms`
# - `input-rep`: tokenization + model training/evaluation (fms-ehrs + benchmark)
#
# Usage
# -----
#   bash scripts/setup_conda_env_meds_extract.sh
#
set -euo pipefail

ENV_NAME="${ENV_NAME:-meds-extract}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

# If you want strict reproducibility, pin a known-good version here.
# Set to empty to install latest.
MEDS_TRANSFORMS_VERSION="${MEDS_TRANSFORMS_VERSION:-0.1.1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IRB_HOME="$(cd "${SCRIPT_DIR}/.." && pwd)"

# -----------------------------------------------------------------------------
# Load conda in a way that works across machines.
# -----------------------------------------------------------------------------
if [[ -f /gpfs/data/bbj-lab/mamba/etc/profile.d/conda.sh ]]; then
  source /gpfs/data/bbj-lab/mamba/etc/profile.d/conda.sh
elif command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base 2>/dev/null || true)"
  if [[ -n "${CONDA_BASE}" && -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1090
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
  fi
elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1090
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1090
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/Desktop/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1090
  source "${HOME}/Desktop/miniconda3/etc/profile.d/conda.sh"
else
  echo "ERROR: conda not found. Install Miniconda/Anaconda and retry."
  exit 1
fi

# -----------------------------------------------------------------------------
# Create env if it doesn't exist.
# -----------------------------------------------------------------------------
if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Creating conda env: ${ENV_NAME} (python=${PYTHON_VERSION})"
  conda create -n "${ENV_NAME}" -y "python=${PYTHON_VERSION}" pip
else
  echo "Conda env already exists: ${ENV_NAME}"
fi

conda activate "${ENV_NAME}"

# -----------------------------------------------------------------------------
# Install MEDS_transforms
# -----------------------------------------------------------------------------
python -m pip install --upgrade pip || true

if [[ -n "${MEDS_TRANSFORMS_VERSION}" ]]; then
  python -m pip install "MEDS_transforms[local_parallelism]==${MEDS_TRANSFORMS_VERSION}"
else
  python -m pip install "MEDS_transforms[local_parallelism]"
fi

echo ""
echo "=============================================="
echo "Done: meds-extract environment is ready"
echo "=============================================="
echo "  ENV_NAME: ${ENV_NAME}"
echo "  IRB_HOME: ${IRB_HOME}"
echo ""
echo "Next step (MEDS extraction):"
echo "  conda activate ${ENV_NAME}"
echo "  cd ${IRB_HOME}"
echo "  sbatch slurm/00_extract_meds.sh   # or: bash benchmarks/mimic-meds-extraction/scripts/01_extract_meds_full.sh <N_WORKERS>"

