#!/bin/bash
#SBATCH --job-name=xgb_baseline
#SBATCH --partition=tier2q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=deprecated/slurm/logs/xgb_baseline_%j.out
#SBATCH --error=deprecated/slurm/logs/xgb_baseline_%j.err

set -euo pipefail

PROJECT_ROOT="/gpfs/data/bbj-lab/users/daniel/input-representation-benchmark"
cd "$PROJECT_ROOT"

mkdir -p deprecated/slurm/logs deprecated/outputs/xgboost_baseline

eval "$(conda shell.bash hook)"
conda activate input-rep

python -c "import xgboost; print(f'xgboost {xgboost.__version__}')" 2>/dev/null || \
    echo "WARNING: xgboost not installed, will try lightgbm"

echo "=== Archived XGBoost/LightGBM Baseline ==="
echo "Start: $(date)"

python deprecated/scripts/xgboost_baseline.py \
    --meds_events_dir data/clif/raw \
    --output_dir deprecated/outputs/xgboost_baseline \
    --model auto \
    --n_bootstrap 1000

echo "Done: $(date)"
