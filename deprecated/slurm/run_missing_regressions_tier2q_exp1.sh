#!/bin/bash
#SBATCH --job-name=eval_exp1_regs
#SBATCH --output=slurm/logs/exp1_regs_%A_%a.out
#SBATCH --error=slurm/logs/exp1_regs_%A_%a.err
#SBATCH --partition=tier2q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-11

set -euo pipefail
cd /gpfs/data/bbj-lab/users/daniel/input-representation-benchmark

export FMS_EHRS_HOME=/gpfs/data/bbj-lab/users/daniel/fms-ehrs
export DIAG_CLASSIFIER=ridge_regression
export DIAG_TASK_TYPE=regression
export DIAG_OUTCOMES="peak_creatinine peak_troponin min_hemoglobin peak_potassium min_glucose peak_bnp time_to_icu_hours"
export DIAG_OUTCOMES_PARQUET="tokens_timelines_extended_outcomes.parquet"

MODELS=(
    "models/exp1_meds_deciles_none_fusedFalse_discrete_time_tokens-s42/run-0/checkpoint-18500"
    "deciles_none_unfused_time_tokens_evalL4096_first_24h"
    "models/exp1_meds_deciles_none_fusedTrue_discrete_time_tokens-s42/run-0/checkpoint-18000"
    "deciles_none_fused_time_tokens_evalL4096_first_24h"

    "models/exp1_meds_ventiles_none_fusedFalse_discrete_time_tokens-s42/run-0/checkpoint-18500"
    "ventiles_none_unfused_time_tokens_evalL4096_first_24h"
    "models/exp1_meds_ventiles_none_fusedTrue_discrete_time_tokens-s42/run-0/checkpoint-18000"
    "ventiles_none_fused_time_tokens_evalL4096_first_24h"

    "models/exp1_meds_trentiles_none_fusedFalse_discrete_time_tokens-s42/run-0/checkpoint-18500"
    "trentiles_none_unfused_time_tokens_evalL4096_first_24h"
    "models/exp1_meds_trentiles_none_fusedTrue_discrete_time_tokens-s42/run-0/checkpoint-18000"
    "trentiles_none_fused_time_tokens_evalL4096_first_24h"

    "models/exp1_meds_trentiles_10-10-10_fusedFalse_discrete_time_tokens-s42/run-0/checkpoint-18000"
    "trentiles_10-10-10_unfused_time_tokens_evalL4096_first_24h"
    "models/exp1_meds_trentiles_10-10-10_fusedTrue_discrete_time_tokens-s42/run-0/checkpoint-18500"
    "trentiles_10-10-10_fused_time_tokens_evalL4096_first_24h"

    "models/exp1_meds_ventiles_5-10-5_fusedFalse_discrete_time_tokens-s42/run-0/checkpoint-18500"
    "ventiles_5-10-5_unfused_time_tokens_evalL4096_first_24h"
    "models/exp1_meds_ventiles_5-10-5_fusedTrue_discrete_time_tokens-s42/run-0/checkpoint-18500"
    "ventiles_5-10-5_fused_time_tokens_evalL4096_first_24h"

    "models/exp1_meds_centiles_none_fusedFalse_discrete_time_tokens-s42/run-0/checkpoint-18500"
    "centiles_none_unfused_time_tokens_evalL4096_first_24h"
    "models/exp1_meds_centiles_none_fusedTrue_discrete_time_tokens-s42/run-0/checkpoint-18000"
    "centiles_none_fused_time_tokens_evalL4096_first_24h"
)

IDX=$((SLURM_ARRAY_TASK_ID * 2))
export model_loc="${MODELS[$IDX]}"
export data_version="${MODELS[$((IDX+1))]}"
export data_dir="benchmarks/mimic-meds-extraction/data/meds/data"

echo "Running ridge regression on: $model_loc"
bash slurm/ref_qse/11_diag_eval.sh
