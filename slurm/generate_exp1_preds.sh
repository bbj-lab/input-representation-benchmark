#!/bin/bash
#SBATCH --job-name=gen_exp1_preds
#SBATCH --partition=tier2q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/gen_exp1_preds_%j.out
#SBATCH --error=slurm/logs/gen_exp1_preds_%j.err

set -euo pipefail

# --- Configuration ---
PROJECT_ROOT="/gpfs/data/bbj-lab/users/daniel/input-representation-benchmark"
cd "$PROJECT_ROOT"

# This should point to your sibling fms-ehrs repo
export FMS_EHRS_HOME="../fms-ehrs"
source "${PROJECT_ROOT}/slurm/00_preamble.sh"

# 1. DATA: The tokenized data for Exp1 Deciles Fused (Phase 1 Winner)
#    We use the main tokenized run tree.
DATA_DIR="${IRB_TOKENIZED_ROOT}/mimiciv-3.1_meds_70-10-20"
DATA_VERSION="deciles_none_fused_time_tokens_first_24h"

# 2. MODEL: The trained model checkpoint
#    Using run-0, checkpoint-9000 (final checkpoint)
MODEL_LOC="${MODEL_DIR}/exp1_meds_deciles_none_fusedTrue_discrete_time_tokens-s42/run-0/checkpoint-9000"

mkdir -p slurm/logs

# Activate conda env
eval "$(conda shell.bash hook)"
conda activate input-rep

echo "=== Generating Predictions for Figure 5 (Calibration) ==="
echo "Data Dir: $DATA_DIR"
echo "Model:    $MODEL_LOC"
echo "Start:    $(date)"

# Run the predictions CLI from fms-ehrs
# Note: we point PYTHONPATH to fms-ehrs so relative imports work if needed, 
# but the script is called directly.
export PYTHONPATH="${FMS_EHRS_HOME}:$PYTHONPATH"

python3 "${FMS_EHRS_HOME}/fms_ehrs/scripts/transfer_rep_based_preds.py" \
  --data_dir_orig "${DATA_DIR}" \
  --data_dir_new "${DATA_DIR}" \
  --data_version "${DATA_VERSION}" \
  --model_loc "${MODEL_LOC}" \
  --classifier logistic_regression \
  --preds_tag calibration-primary_binary \
  --save_preds \
  --drop_icu_adm \
  --outcomes same_admission_death long_length_of_stay imv_event icu_admission

echo "Done: $(date)"
