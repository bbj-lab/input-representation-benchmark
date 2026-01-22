#!/bin/bash
# =============================================================================
# Exp3 Stage 1 (GPU): train representation mechanics (per config×seed)
# =============================================================================

set -euo pipefail

CONFIG_ID=""
DATA_DIR_IN=""
DATA_VERSION=""
REPRESENTATION=""
TEMPORAL=""
NUM_BINS=""
SEED=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config_id) CONFIG_ID="$2"; shift 2 ;;
    --data_dir) DATA_DIR_IN="$2"; shift 2 ;;
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

if [[ -z "${CONFIG_ID}" || -z "${DATA_DIR_IN}" || -z "${DATA_VERSION}" || -z "${REPRESENTATION}" || -z "${TEMPORAL}" || -z "${NUM_BINS}" || -z "${SEED}" ]]; then
  echo "Missing required args." >&2
  echo "Required: --config_id --data_dir --data_version --representation --temporal --num_bins --seed" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IRB_HOME="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${IRB_HOME}/slurm/00_preamble.sh"

FMS_EHRS_HOME="${FMS_EHRS_HOME:-$(realpath "${IRB_HOME}/../fms-ehrs")}"

DATA_DIR="${DATA_DIR_IN}"
if [[ "${DATA_DIR}" != /* ]]; then
  DATA_DIR="${IRB_HOME}/${DATA_DIR}"
fi

MODEL_DIR="${MODEL_DIR:-${IRB_HOME}/models}"
JOB_NAME="exp3_${CONFIG_ID}_s${SEED}"
JID="s${SEED}"

OUT_DIR="${MODEL_DIR}/exp3_${CONFIG_ID}-${REPRESENTATION}-${TEMPORAL}-s${SEED}"
mkdir -p "${OUT_DIR}"

NPROC_PER_NODE="${IRB_NPROC_PER_NODE:-8}"
NNODES="${SLURM_JOB_NUM_NODES:-1}"
NODE_RANK="${SLURM_NODEID:-0}"

# Optional override for fast dev/debug runs (defaults preserve paper runs).
N_EPOCHS="${IRB_STAGE1_EPOCHS:-4}"

extra_args=()
if [[ "${IRB_TUNE_REP_HPARAMS:-}" == "true" ]]; then
  extra_args+=( --tune_representation_hparams true )
fi
NUM_BINS_CHOICES_DEFAULT="${IRB_NUM_BINS_CHOICES:-[${NUM_BINS}]}"
TIME2VEC_DIM_CHOICES_DEFAULT="${IRB_TIME2VEC_DIM_CHOICES:-[32,64,128]}"
CONTINUOUS_NUM_SCALES_CHOICES_DEFAULT="${IRB_CONTINUOUS_NUM_SCALES_CHOICES:-[1,3]}"
if [[ "${IRB_TUNE_REP_HPARAMS:-}" == "true" ]]; then
  extra_args+=( --num_bins_choices "${NUM_BINS_CHOICES_DEFAULT}" )
  extra_args+=( --time2vec_dim_choices "${TIME2VEC_DIM_CHOICES_DEFAULT}" )
  extra_args+=( --continuous_num_scales_choices "${CONTINUOUS_NUM_SCALES_CHOICES_DEFAULT}" )
fi

torchrun_args=( "${FMS_EHRS_HOME}/fms_ehrs/scripts/train_representation.py" )
if [[ "${NNODES}" -gt 1 ]]; then
  : "${MASTER_ADDR:?MASTER_ADDR must be set for multi-node torchrun (set by runner)}"
  : "${MASTER_PORT:?MASTER_PORT must be set for multi-node torchrun (set by runner)}"
  torchrun_args=( --nnodes="${NNODES}" --nproc_per_node="${NPROC_PER_NODE}" --node_rank="${NODE_RANK}" --rdzv_backend=c10d --rdzv_id="${SLURM_JOB_ID}" --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" "${torchrun_args[@]}" )
else
  torchrun_args=( --standalone --nproc_per_node="${NPROC_PER_NODE}" "${torchrun_args[@]}" )
fi

torchrun "${torchrun_args[@]}" \
  --data_dir "${DATA_DIR}" \
  --data_version "${DATA_VERSION}" \
  --model_dir "${OUT_DIR}" \
  --model_version "exp3_${CONFIG_ID}" \
  --model_name "meta-llama/Llama-3.2-1B" \
  --hidden_size "${IRB_MODEL_HIDDEN_SIZE:-1024}" \
  --intermediate_size "${IRB_MODEL_INTERMEDIATE_SIZE:-2048}" \
  --num_hidden_layers "${IRB_MODEL_NUM_HIDDEN_LAYERS:-8}" \
  --num_attention_heads "${IRB_MODEL_NUM_ATTENTION_HEADS:-8}" \
  --representation "${REPRESENTATION}" \
  --temporal "${TEMPORAL}" \
  --num_bins "${NUM_BINS}" \
  --do_hpo true \
  --n_trials "${IRB_EXP3_OPTUNA_TRIALS:-3}" \
  --n_epochs "${N_EPOCHS}" \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gr_acc_min 4 \
  --gr_acc_max 12 \
  --seed "${SEED}" \
  --jid "${JID}" \
  --wandb_project input-rep-benchmark-exp3 \
  "${extra_args[@]}"

echo "=== Completed ${JOB_NAME} ==="

