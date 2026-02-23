#!/bin/bash
# =============================================================================
# Exp2 Stage 1 (GPU): train representation mechanics (per config×seed)
# =============================================================================

set -euo pipefail

CONFIG_ID=""
DATA_VERSION=""
REPRESENTATION=""
TEMPORAL=""
NUM_BINS=""
SEED=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config_id) CONFIG_ID="$2"; shift 2 ;;
    --data_version) DATA_VERSION="$2"; shift 2 ;;
    --representation) REPRESENTATION="$2"; shift 2 ;;
    --temporal) TEMPORAL="$2"; shift 2 ;;
    --num_bins) NUM_BINS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${CONFIG_ID}" || -z "${DATA_VERSION}" || -z "${REPRESENTATION}" || -z "${TEMPORAL}" || -z "${NUM_BINS}" || -z "${SEED}" ]]; then
  echo "Missing required args." >&2
  echo "Required: --config_id --data_version --representation --temporal --num_bins --seed" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IRB_HOME="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${IRB_HOME}/slurm/00_preamble.sh"

# `DATA_DIR` is standardized (via slurm/00_preamble.sh) to point to the MEDS events directory.
MEDS_DATA_DIR="${DATA_DIR}"
MODEL_DIR="${MODEL_DIR:-${IRB_HOME}/models}"
FMS_EHRS_HOME="${FMS_EHRS_HOME:-$(realpath "${IRB_HOME}/../fms-ehrs")}"

JOB_NAME="exp2_${CONFIG_ID}_s${SEED}"
JID="s${SEED}"

NNODES="${SLURM_JOB_NUM_NODES:-1}"
NODE_RANK="${SLURM_NODEID:-0}"

# Benchmark contract: 2 GPUs per configuration (single node).
# Paper: "2 × A100 for Experiments 2–3, trading longer wall-clock time for faster queue throughput."
NPROC_PER_NODE="${IRB_NPROC_PER_NODE:-2}"
if [[ "${NPROC_PER_NODE}" -ne 4 && "${NPROC_PER_NODE}" -ne 2 ]]; then
  echo "ERROR: IRB_NPROC_PER_NODE must be 2 or 4 for benchmark runs (got ${NPROC_PER_NODE})." >&2
  exit 2
fi
if [[ "${NNODES}" -ne 1 ]]; then
  echo "ERROR: This benchmark assumes single-node training (got SLURM_JOB_NUM_NODES=${NNODES})." >&2
  exit 2
fi

# Optional override for fast dev/debug runs (defaults preserve paper runs).
N_EPOCHS="${IRB_EXP23_STAGE1_EPOCHS:-${IRB_STAGE1_EPOCHS:-1}}"

torchrun_args=( "${FMS_EHRS_HOME}/fms_ehrs/scripts/train_representation.py" )
if [[ "${NNODES}" -gt 1 ]]; then
  : "${MASTER_ADDR:?MASTER_ADDR must be set for multi-node torchrun (set by runner)}"
  : "${MASTER_PORT:?MASTER_PORT must be set for multi-node torchrun (set by runner)}"
  torchrun_args=( --nnodes="${NNODES}" --nproc_per_node="${NPROC_PER_NODE}" --node_rank="${NODE_RANK}" --rdzv_backend=c10d --rdzv_id="${SLURM_JOB_ID}" --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" "${torchrun_args[@]}" )
else
  torchrun_args=( --standalone --nproc_per_node="${NPROC_PER_NODE}" "${torchrun_args[@]}" )
fi

torchrun "${torchrun_args[@]}" \
  --data_dir "${MEDS_DATA_DIR}" \
  --data_version "${DATA_VERSION}" \
  --model_dir "${MODEL_DIR}" \
  --model_version "exp2_${CONFIG_ID}" \
  --model_name "meta-llama/Llama-3.2-1B" \
  --use_bf16 "${IRB_USE_BF16}" \
  --attn_implementation "${IRB_ATTN_IMPL}" \
  --hidden_size "${IRB_MODEL_HIDDEN_SIZE:-1024}" \
  --intermediate_size "${IRB_MODEL_INTERMEDIATE_SIZE:-2048}" \
  --num_hidden_layers "${IRB_MODEL_NUM_HIDDEN_LAYERS:-8}" \
  --num_attention_heads "${IRB_MODEL_NUM_ATTENTION_HEADS:-8}" \
  --max_seq_length "${IRB_MAX_SEQ_LENGTH}" \
  --windowed_padded "${IRB_WINDOWED_PADDED:-true}" \
  --window_stride "${IRB_WINDOW_STRIDE:-${IRB_MAX_SEQ_LENGTH}}" \
  --max_windows_per_admission "${IRB_MAX_WINDOWS_PER_ADMISSION:-128}" \
  --add_cont_token "${IRB_ADD_CONT_TOKEN:-true}" \
  --representation "${REPRESENTATION}" \
  --temporal "${TEMPORAL}" \
  --num_bins "${NUM_BINS}" \
  --time_rope_scaling "${IRB_TIME_ROPE_SCALING:-60.0}" \
  --numeric_loss_weight "${IRB_XVAL_NUMERIC_LOSS_WEIGHT:-1.0}" \
  --clip_sigma "${IRB_XVAL_CLIP_SIGMA:-5.0}" \
  --optimizer "${IRB_STAGE1_OPTIMIZER:-muon}" \
  --learning_rate "${IRB_STAGE1_LR:-1e-4}" \
  --muon_learning_rate "${IRB_MUON_LR:-${IRB_STAGE1_LR:-1e-4}}" \
  --aux_adamw_learning_rate "${IRB_AUX_ADAMW_LR:-${IRB_STAGE1_LR:-1e-4}}" \
  --weight_decay "${IRB_STAGE1_WEIGHT_DECAY:-0.01}" \
  --n_epochs "${N_EPOCHS}" \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 2 \
  --seed "${SEED}" \
  --jid "${JID}" \
  --wandb_project input-rep-benchmark-exp2

# NOTE: Exp2/Exp3 Stage1 no longer runs Optuna HPO; all hypers are fixed.

echo "=== Completed ${JOB_NAME} ==="

