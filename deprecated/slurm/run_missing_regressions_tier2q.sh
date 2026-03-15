#!/bin/bash
#SBATCH --job-name=eval_missing_regs
#SBATCH --output=slurm/logs/missing_regs_%A_%a.out
#SBATCH --error=slurm/logs/missing_regs_%A_%a.err
#SBATCH --partition=tier2q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-2  # Adjust based on number of models to probe

set -euo pipefail
cd /gpfs/data/bbj-lab/users/daniel/input-representation-benchmark

export FMS_EHRS_HOME=/gpfs/data/bbj-lab/users/daniel
export DIAG_CLASSIFIER=ridge_regression
export DIAG_TASK_TYPE=regression
export DIAG_OUTCOMES="peak_creatinine peak_troponin min_hemoglobin peak_potassium min_glucose peak_bnp time_to_icu_hours"
export DIAG_OUTCOMES_PARQUET="tokens_timelines_extended_outcomes.parquet"

# Define the models we need to run: Exp 1 mean representatives + Exp 3 Meds/CLIF/Controls
MODELS=(
    # Exp 1 Unfused (deciles)
    "models_archive/ctx1024_or_stale_exp1_20260131_115533/exp1_meds_deciles_none_fusedFalse_discrete_time_tokens-s42/run-1/checkpoint-18500"
    "benchmarks/mimic-meds-extraction/data/meds/data/deciles_none_unfused_time_tokens_evalL4096_first_24h-tokenized"
    "deciles_none_unfused_time_tokens_evalL4096_first_24h"
    
    # Exp 1 Fused (deciles)
    "models_archive/ctx1024_or_stale_exp1_20260131_115533/exp1_meds_deciles_none_fusedTrue_discrete_time_tokens-s42/run-1/checkpoint-18500"
    "benchmarks/mimic-meds-extraction/data/meds/data/deciles_none_fused_time_tokens_evalL4096_first_24h-tokenized"
    "deciles_none_fused_time_tokens_evalL4096_first_24h"
    
    # Exp 3 MEDS
    "models_archive/ctx4096_exp3_20260205_xxxxxx/exp3_meds_deciles_none_fusedFalse_discrete_rope-s42/run-1/checkpoint-xxxxx"
    "benchmarks/mimic-meds-extraction/data/meds/data/deciles_none_unfused_time_tokens_evalL4096_first_24h-tokenized"
    "deciles_none_unfused_time_tokens_evalL4096_first_24h"
)

IDX=$((SLURM_ARRAY_TASK_ID * 3))
export model_loc="${MODELS[$IDX]}"
export data_dir="${MODELS[$((IDX+1))]}"
export data_version="${MODELS[$((IDX+2))]}"

if [ -d "$model_loc" ]; then
    echo "Running ridge regression on: $model_loc"
    bash slurm/ref_qse/11_diag_eval.sh
else
    echo "Model dir not found: $model_loc"
fi

# Also run the LOS one (base outcomes parquet)
export DIAG_OUTCOMES="length_of_stay"
export DIAG_OUTCOMES_PARQUET="tokens_timelines_outcomes.parquet"
if [ -d "$model_loc" ]; then
    bash slurm/ref_qse/11_diag_eval.sh
fi
