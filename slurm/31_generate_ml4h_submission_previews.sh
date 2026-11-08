#!/bin/bash
#SBATCH --job-name=irb-ml4h-preview-figures
#SBATCH --output=/gpfs/data/bbj-lab/users/daniel/input-representation-benchmark/outputs/runs/ml4h_2026/affine_repair/logs/%A-%x.out
#SBATCH --partition=tier1q
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

# Generate approval PDFs under ML4H-DUMP. Do not change ML4H2026.

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

module load gcc/12.1.0
module load texlive/2025

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_MAX_THREADS="${SLURM_CPUS_PER_TASK}"

OUTPUT_DIR="${IRB_HOME}/../ML4H-DUMP/submission-preview"
RASTER_DIR="${IRB_HOME}/outputs/runs/ml4h_2026/submission_preview_rasters"

python paper/scripts/generate_ml4h_submission_previews.py \
  --output-dir "${OUTPUT_DIR}"

mkdir -p "${RASTER_DIR}"
python - "${RASTER_DIR}" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
for path in root.glob("*.png"):
    path.unlink()
PY

for pdf in "${OUTPUT_DIR}"/*.pdf; do
  stem="$(basename "${pdf}" .pdf)"
  pdftoppm -f 1 -singlefile -r 160 -png "${pdf}" "${RASTER_DIR}/${stem}"
done

echo "submission_preview_complete"
