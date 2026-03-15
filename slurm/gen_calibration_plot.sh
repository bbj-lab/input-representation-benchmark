#!/bin/bash
#SBATCH --job-name=gen_calib
#SBATCH --output=slurm/logs/gen_calib_%j.out
#SBATCH --error=slurm/logs/gen_calib_%j.err
#SBATCH --partition=tier2q
#SBATCH --account=bbj-lab
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

PROJECT_ROOT="/gpfs/data/bbj-lab/users/daniel/input-representation-benchmark"
cd "$PROJECT_ROOT"
source "${PROJECT_ROOT}/slurm/00_preamble.sh"

# Activate conda env
eval "$(conda shell.bash hook)"
conda activate input-rep

echo "=== Generating Calibration Plot ==="
echo "Start: $(date)"

PREDS_PKL="${IRB_TOKENIZED_ROOT}/mimiciv-3.1_meds_70-10-20/deciles_none_fused_time_tokens_first_24h-tokenized/test/logistic_regression-preds-checkpoint-9000.pkl"

# First, inspect pkl structure
python -c "
import pickle, sys
with open('${PREDS_PKL}', 'rb') as f:
    data = pickle.load(f)
print('Type:', type(data))
if isinstance(data, dict):
    print('Keys:', list(data.keys()))
    for k, v in data.items():
        if isinstance(v, dict):
            print(f'  {k}: dict with keys={list(v.keys())}')
        else:
            print(f'  {k}: {type(v).__name__}, shape={getattr(v, \"shape\", \"N/A\")}')
"

# Generate the plot
python scripts/generate_calibration_plot.py \
    --preds_pkl "${PREDS_PKL}" \
    --output figures/calibration_curves.pdf

echo "Done: $(date)"
ls -la figures/calibration*
