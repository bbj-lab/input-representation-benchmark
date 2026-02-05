#!/bin/bash
# =============================================================================
# Exp1 Stage 1 (GPU): packed-collation FM training + small LR sweep
#
# This replaces the legacy 8-GPU vendored reference launcher. The benchmark paper
# uses a fixed 4×A100 allocation for Stage1 across configurations.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IRB_HOME="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${IRB_HOME}/slurm/00_preamble.sh"
cd "${IRB_HOME}"

# Expect these from jobfile command (exported in the generated command string).
: "${config_id:?Set config_id (experiment config_id)}"
: "${data_version:?Set data_version (tokenization data_version)}"
: "${seed:?Set seed (training seed; used for naming only)}"

MODEL_DIR="${MODEL_DIR:-${IRB_HOME}/models}"
FMS_EHRS_HOME="${FMS_EHRS_HOME:-$(realpath "${IRB_HOME}/../fms-ehrs")}"

# Deterministic seed-based id for reproducible naming.
jid="s${seed}"

NPROC_PER_NODE="${IRB_NPROC_PER_NODE:-4}"
NNODES="${SLURM_JOB_NUM_NODES:-1}"
NODE_RANK="${SLURM_NODEID:-0}"

# Benchmark contract: 4 GPUs per configuration (single node).
if [[ "${NPROC_PER_NODE}" -ne 4 ]]; then
  echo "ERROR: IRB_NPROC_PER_NODE must be 4 for benchmark runs (got ${NPROC_PER_NODE})." >&2
  exit 2
fi
if [[ "${NNODES}" -ne 1 ]]; then
  echo "ERROR: This benchmark assumes single-node training (got SLURM_JOB_NUM_NODES=${NNODES})." >&2
  exit 2
fi

# Optional overrides (defaults preserve paper runs).
N_EPOCHS="${IRB_EXP1_STAGE1_EPOCHS:-1}"
N_TRIALS="${IRB_EXP1_LR_TRIALS:-3}"

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
  --max_seq_length "${IRB_MAX_SEQ_LENGTH}" \
  --use_bf16 "${IRB_USE_BF16}" \
  --attn_implementation "${IRB_ATTN_IMPL}" \
  --hidden_size 1024 \
  --intermediate_size 2048 \
  --num_hidden_layers 8 \
  --num_attention_heads 8

