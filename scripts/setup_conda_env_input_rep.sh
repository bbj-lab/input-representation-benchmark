#!/usr/bin/env bash
#
# Create the `input-rep` conda environment for tokenization + downstream benchmark code.
#
# Why this exists
# ---------------
# - The benchmark uses two separate Python envs due to `polars` version conflicts:
#   - `meds-extract`: MEDS extraction (MEDS_transforms)
#   - `input-rep`: tokenization + model training/evaluation (fms-ehrs + benchmark)
# - We intentionally avoid `pip install -e ../fms-ehrs` *with dependencies* because the
#   upstream `fms-ehrs` pyproject includes heavyweight / platform-specific deps (e.g. jax[cuda12]).
#   Instead, we:
#     1) install `fms-ehrs` editable with `--no-deps`
#     2) install the minimal deps needed for tokenization (and optionally training)
#
# Usage
# -----
#   bash scripts/setup_conda_env_input_rep.sh
#   bash scripts/setup_conda_env_input_rep.sh --with-training-deps
#
set -euo pipefail

WITH_TRAINING_DEPS="0"
if [[ "${1:-}" == "--with-training-deps" ]]; then
  WITH_TRAINING_DEPS="1"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IRB_HOME="$(cd "${SCRIPT_DIR}/.." && pwd)"

# fms-ehrs is expected as a sibling repo by default.
FMS_EHRS_HOME="${FMS_EHRS_HOME:-$(realpath "${IRB_HOME}/../fms-ehrs" 2>/dev/null || true)}"
if [[ -z "${FMS_EHRS_HOME}" || ! -d "${FMS_EHRS_HOME}" ]]; then
  echo "ERROR: Could not find sister repo 'fms-ehrs'."
  echo "Expected at: ${IRB_HOME}/../fms-ehrs"
  echo "Fix by either:"
  echo "  - cloning fms-ehrs as a sibling directory, or"
  echo "  - exporting FMS_EHRS_HOME=/path/to/fms-ehrs"
  exit 1
fi

ENV_NAME="${ENV_NAME:-input-rep}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

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
else
  echo "ERROR: conda not found. Install Miniconda/Anaconda and retry."
  exit 1
fi

# -----------------------------------------------------------------------------
# Create env if it doesn't exist.
# -----------------------------------------------------------------------------
if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Creating conda env: ${ENV_NAME} (python=${PYTHON_VERSION})"
  # Use conda for compiled deps (fast, reliable wheels).
  conda create -n "${ENV_NAME}" -y \
    "python=${PYTHON_VERSION}" \
    pip \
    numpy \
    scipy \
    scikit-learn
else
  echo "Conda env already exists: ${ENV_NAME}"
fi

conda activate "${ENV_NAME}"

# -----------------------------------------------------------------------------
# Install runtime deps (tokenization).
# -----------------------------------------------------------------------------
# NOTE: `polars>=1.33.1` is required for compatibility with clifpy (used in fms-ehrs).
python -m pip install --upgrade pip || true
python -m pip install \
  "polars>=1.33.1" \
  ruamel.yaml \
  loguru

# Editable installs
python -m pip install -e "${IRB_HOME}"
python -m pip install -e "${FMS_EHRS_HOME}" --no-deps

if [[ "${WITH_TRAINING_DEPS}" == "1" ]]; then
  echo "Installing training/eval deps (torch is NOT installed here; install it per your hardware)."
  python -m pip install \
    datasets>=3.2 \
    fire>=0.7 \
    transformers>=4.48 \
    trl==0.13.0 \
    tqdm>=4.67 \
    wandb
  echo ""
  echo "NOTE: You still need PyTorch for training:"
  echo "  - GPU: install the correct CUDA build from pytorch.org"
  echo "  - CPU: pip install torch --index-url https://download.pytorch.org/whl/cpu"
fi

echo ""
echo "=============================================="
echo "Done: input-rep environment is ready"
echo "=============================================="
echo "  ENV_NAME: ${ENV_NAME}"
echo "  IRB_HOME: ${IRB_HOME}"
echo "  FMS_EHRS_HOME: ${FMS_EHRS_HOME}"
echo ""
echo "Next step (tokenization):"
echo "  conda activate ${ENV_NAME}"
echo "  cd ${IRB_HOME}"
echo "  bash slurm/01_tokenize.sh"

