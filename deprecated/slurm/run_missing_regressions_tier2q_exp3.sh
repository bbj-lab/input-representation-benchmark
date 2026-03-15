#!/bin/bash
#SBATCH --job-name=eval_exp3_regs
#SBATCH --output=slurm/logs/exp3_regs_%A_%a.out
#SBATCH --error=slurm/logs/exp3_regs_%A_%a.err
#SBATCH --partition=tier2q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-3

set -euo pipefail
cd /gpfs/data/bbj-lab/users/daniel/input-representation-benchmark

export FMS_EHRS_HOME=/gpfs/data/bbj-lab/users/daniel
export DIAG_CLASSIFIER=ridge_regression
export DIAG_TASK_TYPE=regression
export DIAG_OUTCOMES="peak_creatinine peak_troponin min_hemoglobin peak_potassium min_glucose peak_bnp time_to_icu_hours"
export DIAG_OUTCOMES_PARQUET="tokens_timelines_extended_outcomes.parquet"

# Exp 3 Models (MEDS native, CLIF mapped, Freq matched, Randomized)
MODELS=(
    "models/exp3_meds_icu_native-discrete-time_rope-s42/run-1/checkpoint-23500"
    "data/exp3/meds_icu/deciles_none_unfused_time_rope_first_24h-tokenized"
    "deciles_none_unfused_time_rope_first_24h"
    
    "models/exp3_meds_mapped-discrete-time_rope-s42/run-1/checkpoint-23500"
    "data/exp3/arms/meds_mapped/deciles_none_unfused_time_rope_first_24h-tokenized"
    "deciles_none_unfused_time_rope_first_24h"
    
    "models/exp3_meds_randomized-discrete-time_rope-s42/run-1/checkpoint-23500"
    "data/exp3/arms/meds_randomized/deciles_none_unfused_time_rope_first_24h-tokenized"
    "deciles_none_unfused_time_rope_first_24h"
    
    "models/exp3_meds_freqmatched-discrete-time_rope-s42/run-1/checkpoint-23500"
    "data/exp3/arms/meds_freqmatched/deciles_none_unfused_time_rope_first_24h-tokenized"
    "deciles_none_unfused_time_rope_first_24h"
)

IDX=$((SLURM_ARRAY_TASK_ID * 3))
export model_loc="${MODELS[$IDX]}"
export data_dir="${MODELS[$((IDX+1))]}"
export data_version="${MODELS[$((IDX+2))]}"

if [ -d "$model_loc" ]; then
    echo "Running ridge regression on: $model_loc"
    bash slurm/ref_qse/11_diag_eval.sh
    
    export DIAG_OUTCOMES="length_of_stay"
    export DIAG_OUTCOMES_PARQUET="tokens_timelines_outcomes.parquet"
    bash slurm/ref_qse/11_diag_eval.sh
else
    echo "Model dir not found: $model_loc"
fi
