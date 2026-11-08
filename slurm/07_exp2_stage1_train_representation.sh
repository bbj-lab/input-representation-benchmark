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
export IRB_REQUIRE_WANDB="${IRB_REQUIRE_WANDB:-true}"
export WANDB_LOG_MODEL="${WANDB_LOG_MODEL:-checkpoint}"
source "${IRB_HOME}/slurm/00_preamble.sh"

# Exp2 Stage1 consumes the main tokenized timelines, not the raw MEDS events root.
TOKENIZED_DATA_DIR="${IRB_TOKENIZED_DATASET_DIR}"
MODEL_DIR="${MODEL_DIR:-${RUN_ARTIFACTS_DIR}/models}"
FMS_EHRS_HOME="${FMS_EHRS_HOME:-$(realpath "${IRB_HOME}/../fms-ehrs")}"

JOB_NAME="exp2_${CONFIG_ID}_s${SEED}"
JID="${IRB_RUN_JID:-s${SEED}}"

# Benchmark contract: one process and one GPU per configuration × seed.
NPROC_PER_NODE="${IRB_NPROC_PER_NODE:-1}"
if [[ "${NPROC_PER_NODE}" -ne 1 ]]; then
  echo "ERROR: IRB_NPROC_PER_NODE must be 1 for ML4H runs (got ${NPROC_PER_NODE})." >&2
  exit 2
fi
if [[ "${SLURM_JOB_NUM_NODES:-1}" -ne 1 ]]; then
  echo "ERROR: This benchmark assumes one node (got SLURM_JOB_NUM_NODES=${SLURM_JOB_NUM_NODES})." >&2
  exit 2
fi

# Optional override for fast dev/debug runs (defaults preserve paper runs).
N_EPOCHS="${IRB_EXP23_STAGE1_EPOCHS:-${IRB_STAGE1_EPOCHS:-1}}"
MODEL_NAME="${IRB_MODEL_NAME:-meta-llama/Llama-3.2-1B}"
MODEL_VERSION_PREFIX="${IRB_MODEL_VERSION_PREFIX:-exp2}"
WANDB_PROJECT="${IRB_WANDB_PROJECT:-input-rep-benchmark-exp2}"
USE_MODEL_CONFIG_OVERRIDES="${IRB_USE_MODEL_CONFIG_OVERRIDES:-true}"
SAVE_STRATEGY="${IRB_STAGE1_SAVE_STRATEGY:-epoch}"
EVAL_STRATEGY="${IRB_STAGE1_EVAL_STRATEGY:-epoch}"
LOAD_BEST_MODEL_AT_END="${IRB_LOAD_BEST_MODEL_AT_END:-true}"
ADAM_BETA1="${IRB_ADAM_BETA1:-0.9}"
ADAM_BETA2="${IRB_ADAM_BETA2:-0.999}"

if [[ "${MODEL_VERSION_PREFIX}" == llama10ep_* && "${IRB_LLAMA10EP_WANDB_EPOCH_UPLOADS:-true}" == "true" ]]; then
  export IRB_STAGE1_SAVE_STRATEGY="epoch"
  SAVE_STRATEGY="epoch"
  export WANDB_LOG_MODEL="checkpoint"
fi

if [[ "${SAVE_STRATEGY}" == "steps" && "${SLURM_JOB_PARTITION:-}" == "sxmq" && "${N_EPOCHS}" =~ ^[0-9]+$ && "${N_EPOCHS}" -gt 1 ]]; then
  SXMQ_MAX_SAVE_STEPS="${IRB_SXMQ_MAX_SAVE_STEPS:-500}"
  CURRENT_SAVE_STEPS="${IRB_STAGE1_SAVE_STEPS:-2000}"
  if [[ "${SXMQ_MAX_SAVE_STEPS}" =~ ^[0-9]+$ && "${CURRENT_SAVE_STEPS}" =~ ^[0-9]+$ && "${CURRENT_SAVE_STEPS}" -gt "${SXMQ_MAX_SAVE_STEPS}" ]]; then
    export IRB_STAGE1_SAVE_STEPS="${SXMQ_MAX_SAVE_STEPS}"
  fi
fi

echo "Exp2 Stage 1 configuration:"
echo "  config_id:              ${CONFIG_ID}"
echo "  model_name:             ${MODEL_NAME}"
echo "  model_version:          ${MODEL_VERSION_PREFIX}_${CONFIG_ID}"
echo "  wandb_project:          ${WANDB_PROJECT}"
echo "  n_epochs:               ${N_EPOCHS}"
echo "  execution:              single process, one GPU (no DDP)"
echo "  config_overrides:       ${USE_MODEL_CONFIG_OVERRIDES}"
echo "  resume_from_checkpoint: ${IRB_RESUME_FROM_CHECKPOINT:-auto}"
echo "  save_strategy:          ${SAVE_STRATEGY}"
echo "  save_steps:             ${IRB_STAGE1_SAVE_STEPS:-2000}"
echo "  save_total_limit:       ${IRB_STAGE1_SAVE_TOTAL_LIMIT:-6}"
echo "  eval_strategy:          ${EVAL_STRATEGY}"
echo "  wandb_log_model:        ${WANDB_LOG_MODEL:-unset}"
echo "  load_best_at_end:       ${LOAD_BEST_MODEL_AT_END}"
echo "  learning_rate:          ${IRB_STAGE1_LR:-1e-4}"
echo "  adam_betas:             ${ADAM_BETA1}/${ADAM_BETA2}"
echo "  max_grad_norm:          ${IRB_STAGE1_MAX_GRAD_NORM:-1.0}"
echo "  scheduler/warmup:       ${IRB_STAGE1_LR_SCHEDULER:-linear}/${IRB_STAGE1_WARMUP_RATIO:-0.01}"
echo "  mapped_dataset_cache:   ${IRB_USE_MAPPED_DATASET_CACHE:-true}"
echo "  mapped_cache_dir:       ${IRB_MAPPED_DATASET_CACHE_DIR:-${HF_DATASETS_CACHE}/irb-mapped}"

model_config_args=()
if [[ "${USE_MODEL_CONFIG_OVERRIDES}" == "true" || "${USE_MODEL_CONFIG_OVERRIDES}" == "1" ]]; then
  model_config_args=(
    --hidden_size "${IRB_MODEL_HIDDEN_SIZE:-1024}"
    --intermediate_size "${IRB_MODEL_INTERMEDIATE_SIZE:-2048}"
    --num_hidden_layers "${IRB_MODEL_NUM_HIDDEN_LAYERS:-8}"
    --num_attention_heads "${IRB_MODEL_NUM_ATTENTION_HEADS:-8}"
    --max_position_embeddings "${IRB_MODEL_MAX_POSITION_EMBEDDINGS:-4096}"
  )
fi

python "${FMS_EHRS_HOME}/fms_ehrs/scripts/train_representation.py" \
  --data_dir "${TOKENIZED_DATA_DIR}" \
  --data_version "${DATA_VERSION}" \
  --model_dir "${MODEL_DIR}" \
  --model_version "${MODEL_VERSION_PREFIX}_${CONFIG_ID}" \
  --model_name "${MODEL_NAME}" \
  --rope_theta "${IRB_ROPE_THETA:-10000.0}" \
  --use_bf16 "${IRB_USE_BF16}" \
  --attn_implementation "${IRB_ATTN_IMPL}" \
  "${model_config_args[@]}" \
  --max_seq_length "${IRB_MAX_SEQ_LENGTH}" \
  --windowed_padded "${IRB_WINDOWED_PADDED:-true}" \
  --window_stride "${IRB_WINDOW_STRIDE:-${IRB_MAX_SEQ_LENGTH}}" \
  --max_windows_per_admission "${IRB_MAX_WINDOWS_PER_ADMISSION:-128}" \
  --add_cont_token "${IRB_ADD_CONT_TOKEN:-true}" \
  --representation "${REPRESENTATION}" \
  --temporal "${TEMPORAL}" \
  --num_bins "${NUM_BINS}" \
  --seconds_per_position "${IRB_SECONDS_PER_POSITION:-60.0}" \
  --numeric_loss_weight "${IRB_XVAL_NUMERIC_LOSS_WEIGHT:-1.0}" \
  --clip_sigma "${IRB_XVAL_CLIP_SIGMA:-5.0}" \
  --learning_rate "${IRB_STAGE1_LR:-1e-4}" \
  --adam_beta1 "${ADAM_BETA1}" \
  --adam_beta2 "${ADAM_BETA2}" \
  --weight_decay "${IRB_STAGE1_WEIGHT_DECAY:-0.01}" \
  --max_grad_norm "${IRB_STAGE1_MAX_GRAD_NORM:-1.0}" \
  --lr_scheduler_type "${IRB_STAGE1_LR_SCHEDULER:-linear}" \
  --warmup_ratio "${IRB_STAGE1_WARMUP_RATIO:-0.01}" \
  --logging_steps "${IRB_STAGE1_LOGGING_STEPS:-100}" \
  --evaluations_per_epoch "${IRB_STAGE1_EVALUATIONS_PER_EPOCH:-1}" \
  --max_steps "${IRB_STAGE1_MAX_STEPS:--1}" \
  --n_epochs "${N_EPOCHS}" \
  --per_device_train_batch_size "${IRB_PER_DEVICE_TRAIN_BATCH_SIZE:-4}" \
  --per_device_eval_batch_size "${IRB_PER_DEVICE_EVAL_BATCH_SIZE:-4}" \
  --gradient_accumulation_steps "${IRB_GRADIENT_ACCUMULATION_STEPS:-1}" \
  --resume_from_checkpoint "${IRB_RESUME_FROM_CHECKPOINT:-auto}" \
  --save_strategy "${SAVE_STRATEGY}" \
  --save_steps "${IRB_STAGE1_SAVE_STEPS:-2000}" \
  --save_total_limit "${IRB_STAGE1_SAVE_TOTAL_LIMIT:-6}" \
  --eval_strategy "${EVAL_STRATEGY}" \
  --load_best_model_at_end "${LOAD_BEST_MODEL_AT_END}" \
  --seed "${SEED}" \
  --jid "${JID}" \
  --wandb_project "${WANDB_PROJECT}"

# NOTE: Exp2/Exp3 Stage1 no longer runs Optuna HPO; all hypers are fixed.

echo "=== Completed ${JOB_NAME} ==="

