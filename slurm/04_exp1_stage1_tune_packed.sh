#!/bin/bash
# =============================================================================
# Exp1 Stage 1 (GPU): packed-collation FM training + small LR sweep
#
# The current paper rerun path standardizes on 1 GPU per model for Stage1.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IRB_HOME="$(cd "${SCRIPT_DIR}/.." && pwd)"
export IRB_REQUIRE_WANDB="${IRB_REQUIRE_WANDB:-true}"
export WANDB_LOG_MODEL="${WANDB_LOG_MODEL:-checkpoint}"
source "${IRB_HOME}/slurm/00_preamble.sh"
cd "${IRB_HOME}"

# Expect these from jobfile command (exported in the generated command string).
: "${config_id:?Set config_id (experiment config_id)}"
: "${data_version:?Set data_version (tokenization data_version)}"
: "${seed:?Set seed (training seed; used for naming only)}"

TOKENIZED_DATA_DIR="${IRB_TOKENIZED_DATASET_DIR}"
MODEL_DIR="${MODEL_DIR:-${RUN_ARTIFACTS_DIR}/models}"
FMS_EHRS_HOME="${FMS_EHRS_HOME:-$(realpath "${IRB_HOME}/../fms-ehrs")}"

# Deterministic seed-based id for reproducible naming.
jid="s${seed}"

NPROC_PER_NODE="${IRB_NPROC_PER_NODE:-1}"
NNODES="${SLURM_JOB_NUM_NODES:-1}"
NODE_RANK="${SLURM_NODEID:-0}"

# Benchmark contract: 1 GPU per configuration (single node).
if [[ "${NPROC_PER_NODE}" -ne 1 ]]; then
  echo "ERROR: IRB_NPROC_PER_NODE must be 1 for paper reruns (got ${NPROC_PER_NODE})." >&2
  exit 2
fi
if [[ "${NNODES}" -ne 1 ]]; then
  echo "ERROR: This benchmark assumes single-node training (got SLURM_JOB_NUM_NODES=${NNODES})." >&2
  exit 2
fi

# Optional overrides (defaults preserve paper runs).
N_EPOCHS="${IRB_EXP1_STAGE1_EPOCHS:-1}"
N_TRIALS="${IRB_EXP1_LR_TRIALS:-3}"
DO_HPO="${IRB_EXP1_DO_HPO:-true}"
RESUME_CKPT="${IRB_EXP1_RESUME_CHECKPOINT:-}"

extra_resume_args=()
if [[ -n "${RESUME_CKPT}" ]]; then
  extra_resume_args=( --resume_from_checkpoint "${RESUME_CKPT}" )
fi

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
  --do_hpo "${DO_HPO}" \
  --n_trials "${N_TRIALS}" \
  "${extra_resume_args[@]}" \
  --data_dir "${TOKENIZED_DATA_DIR}" \
  --data_version "${data_version}" \
  --model_dir "${MODEL_DIR}" \
  --model_version "exp1_${config_id}" \
  --model_name "meta-llama/Llama-3.2-1B" \
  --wandb_project "${data_version}" \
  --jid "${jid}" \
  --max_seq_length "${IRB_MAX_SEQ_LENGTH}" \
  --use_bf16 "${IRB_USE_BF16}" \
  --attn_implementation "${IRB_ATTN_IMPL}" \
  --hidden_size 1024 \
  --intermediate_size 2048 \
  --num_hidden_layers 8 \
  --num_attention_heads 8

