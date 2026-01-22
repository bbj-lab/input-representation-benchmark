#!/bin/bash
#SBATCH --job-name=irb-exp3-clif
#SBATCH --output=./slurm/output/%A_%a-%x.stdout
#SBATCH --partition=tier2q
#SBATCH --mem=300GB
#SBATCH --cpus-per-task=16
#SBATCH --time=1-00:00:00
#
# =============================================================================
# Exp3 prerequisite (tier2q; CPU-only): Build CLIF from PhysioNet MIMIC-IV v3.1
# =============================================================================
#
# Internal validity:
#   - Exp3 requires a CLIF arm derived from the same raw PhysioNet source as MEDS.
#   - Running conversion under SLURM avoids partial/failed artifacts from login-node kills.
#
# Usage:
#   sbatch slurm/01_exp3_build_clif_tier2q.sh
#
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
IRB_HOME="$(find_repo_root "${SUBMIT_DIR}" || true)"
if [[ -z "${IRB_HOME}" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  IRB_HOME="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
if [[ ! -f "${IRB_HOME}/run_experiments.py" || ! -d "${IRB_HOME}/slurm" ]]; then
  echo "ERROR: Could not locate IRB repo root." >&2
  echo "  SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>}" >&2
  echo "  pwd=$(pwd)" >&2
  echo "  resolved IRB_HOME=${IRB_HOME}" >&2
  exit 1
fi

export IRB_CONDA_ENV="input-rep"
source "${IRB_HOME}/slurm/00_preamble.sh"
cd "${IRB_HOME}"
mkdir -p slurm/output

PY="${IRB_PYTHON_BIN:-/gpfs/data/bbj-lab/.envs/input-rep/bin/python}"

echo "=============================================="
echo "Exp3 CLIF build"
echo "Host: $(hostname)"
echo "IRB_HOME: ${IRB_HOME}"
echo "Python: ${PY}"
echo "=============================================="

"${PY}" scripts/build_clif_from_physionet.py \
  --mimic_dir physionet.org/files/mimiciv/3.1 \
  --clif_mimic_repo ../CLIF-MIMIC \
  --clif_out_root data/clif \
  --exp3_root data/exp3 \
  --tables patient hospitalization labs vitals

