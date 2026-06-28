#!/bin/bash
#SBATCH --job-name=xval_zero_audit
#SBATCH --partition=tier1q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=slurm/logs/xval_zero_audit_%j.out
#SBATCH --error=slurm/logs/xval_zero_audit_%j.err

set -euo pipefail

PROJECT_ROOT="/gpfs/data/bbj-lab/users/daniel/input-representation-benchmark"
RUN_ROOT="${PROJECT_ROOT}/outputs/runs"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/diagnostics_xval_zero_audit"

mkdir -p "${PROJECT_ROOT}/slurm/logs" "${OUTPUT_DIR}"
cd "${RUN_ROOT}"

source /gpfs/data/bbj-lab/mamba/etc/profile.d/conda.sh
conda activate input-rep

echo "=== xVal zero-out audit (tier1q CPU) ==="
echo "Start: $(date)"

python "${PROJECT_ROOT}/pipeline/scripts/diagnostics/diag_xval_zero_out.py" \
  --data_dir "${RUN_ROOT}/tokenized/mimiciv-3.1_meds_70-10-20" \
  --n_samples 500 \
  --batch_size 1 \
  --split test \
  --device cpu \
  --output_dir "${OUTPUT_DIR}"

echo "Done: $(date)"
