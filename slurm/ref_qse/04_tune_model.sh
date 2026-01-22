#!/bin/bash

# =============================================================================
# Provenance: Quantifying-Surprise-EHRs/slurm/04_tune_model.sh
# Source: /gpfs/data/bbj-lab/users/daniel/Quantifying-Surprise-EHRs/slurm/04_tune_model.sh
# =============================================================================
#
# This script is vendored to preserve the exact training/HPO launch pattern:
# - 8×GPU DDP via torchrun
# - n_epochs=10, n_trials=3
# - scaled-down Llama config overrides (≈67M params)
#
# - Use benchmark paths (IRB_HOME/FMS_EHRS_HOME/MEDS data/model dirs)
# - Use config_id/data_version passed via environment (jobfile sets these)
#
# =============================================================================

#SBATCH --job-name=tune-mdl
#SBATCH --output=./slurm/output/%j-%x.stdout
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:8
#SBATCH --time=1-00:00:00

set -euo pipefail

source slurm/ref_qse/preamble.sh

: "${FMS_EHRS_HOME:?Set FMS_EHRS_HOME (sibling fms-ehrs repo)}"
: "${MODEL_DIR:?Set MODEL_DIR (e.g., ${IRB_HOME}/models)}"
: "${DATA_DIR:?Set DATA_DIR (set by slurm/preamble.sh)}"

# Expect these from the jobfile command
: "${config_id:?Set config_id (experiment config_id)}"
: "${data_version:?Set data_version (tokenization data_version)}"
: "${seed:?Set seed (training seed; used for naming only)}"

echo "Training an FM on MEDS data (Exp1)..."

# Reference uses SLURM job id; we use deterministic seed-based id for reproducible naming.
jid="s${seed}"

NPROC_PER_NODE="${IRB_NPROC_PER_NODE:-8}"
NNODES="${SLURM_JOB_NUM_NODES:-1}"
NODE_RANK="${SLURM_NODEID:-0}"

# Optional overrides for fast dev/debug runs (defaults preserve paper runs).
N_EPOCHS="${IRB_STAGE1_EPOCHS:-4}"
N_TRIALS="${IRB_EXP1_OPTUNA_TRIALS:-3}"

torchrun_args=( "${FMS_EHRS_HOME}/fms_ehrs/scripts/tune_model.py" )
if [[ "${NNODES}" -gt 1 ]]; then
  : "${MASTER_ADDR:?MASTER_ADDR must be set for multi-node torchrun (set by runner)}"
  : "${MASTER_PORT:?MASTER_PORT must be set for multi-node torchrun (set by runner)}"
  torchrun_args=( --nnodes="${NNODES}" --nproc_per_node="${NPROC_PER_NODE}" --node_rank="${NODE_RANK}" --rdzv_backend=c10d --rdzv_id="${SLURM_JOB_ID}" --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" "${torchrun_args[@]}" )
else
  torchrun_args=( --standalone --nproc_per_node="${NPROC_PER_NODE}" "${torchrun_args[@]}" )
fi

torchrun "${torchrun_args[@]}" \
  --n_epochs "${N_EPOCHS}" \
  --n_trials "${N_TRIALS}" \
  --data_dir "${DATA_DIR}" \
  --data_version "${data_version}" \
  --model_dir "${MODEL_DIR}" \
  --model_version "exp1_${config_id}" \
  --model_name "meta-llama/Llama-3.2-1B" \
  --wandb_project "${data_version}" \
  --jid "${jid}" \
  --hidden_size 1024 \
  --intermediate_size 2048 \
  --num_hidden_layers 8 \
  --num_attention_heads 8

# Leaves tuned models at ${MODEL_DIR}/exp1_${config_id}-${jid}-hp-${data_version}

