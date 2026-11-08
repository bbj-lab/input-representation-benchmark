#!/bin/bash
# =============================================================================
# Exp1 Stage 1 (GPU): packed-collation FM training.
#
# ML4H uses fixed, pre-registered hyperparameters selected during the shared
# validation-only preflight; it does not run a per-arm HPO sweep.
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

# Benchmark contract: 1 GPU per configuration (single node).
if [[ "${NPROC_PER_NODE}" -ne 1 ]]; then
  echo "ERROR: IRB_NPROC_PER_NODE must be 1 for paper reruns (got ${NPROC_PER_NODE})." >&2
  exit 2
fi
if [[ "${NNODES}" -ne 1 ]]; then
  echo "ERROR: This benchmark assumes single-node training (got SLURM_JOB_NUM_NODES=${NNODES})." >&2
  exit 2
fi

# Optional overrides.
N_EPOCHS="${IRB_EXP1_STAGE1_EPOCHS:-5}"
N_TRIALS="${IRB_EXP1_LR_TRIALS:-0}"
DO_HPO="${IRB_EXP1_DO_HPO:-false}"
RESUME_CKPT="${IRB_EXP1_RESUME_CHECKPOINT:-${IRB_RESUME_FROM_CHECKPOINT:-auto}}"
ADAM_BETA1="${IRB_ADAM_BETA1:-0.9}"
ADAM_BETA2="${IRB_ADAM_BETA2:-0.999}"
WANDB_PROJECT="${IRB_WANDB_PROJECT:-input-rep-benchmark-exp1}"
MODEL_NAME="${IRB_MODEL_NAME:-meta-llama/Llama-3.2-1B}"
MODEL_VERSION_PREFIX="${IRB_MODEL_VERSION_PREFIX:-exp1}"

extra_resume_args=()
if [[ -n "${RESUME_CKPT}" ]]; then
  extra_resume_args=( --resume_from_checkpoint "${RESUME_CKPT}" )
fi

python "${FMS_EHRS_HOME}/fms_ehrs/scripts/tune_model.py" \
  --n_epochs "${N_EPOCHS}" \
  --do_hpo "${DO_HPO}" \
  --n_trials "${N_TRIALS}" \
  "${extra_resume_args[@]}" \
  --data_dir "${TOKENIZED_DATA_DIR}" \
  --data_version "${data_version}" \
  --model_dir "${MODEL_DIR}" \
  --model_version "${MODEL_VERSION_PREFIX}_${config_id}" \
  --model_name "${MODEL_NAME}" \
  --rope_theta "${IRB_ROPE_THETA:-10000.0}" \
  --wandb_project "${WANDB_PROJECT}" \
  --jid "${jid}" \
  --seed "${seed}" \
  --max_seq_length "${IRB_MAX_SEQ_LENGTH}" \
  --use_bf16 "${IRB_USE_BF16}" \
  --attn_implementation "${IRB_ATTN_IMPL}" \
  --learning_rate "${IRB_STAGE1_LR:-1e-4}" \
  --adam_beta1 "${ADAM_BETA1}" \
  --adam_beta2 "${ADAM_BETA2}" \
  --weight_decay "${IRB_STAGE1_WEIGHT_DECAY:-0.01}" \
  --max_grad_norm "${IRB_STAGE1_MAX_GRAD_NORM:-1.0}" \
  --lr_scheduler_type "${IRB_STAGE1_LR_SCHEDULER:-linear}" \
  --warmup_ratio "${IRB_STAGE1_WARMUP_RATIO:-0.01}" \
  --logging_steps "${IRB_STAGE1_LOGGING_STEPS:-100}" \
  --evaluations_per_epoch "${IRB_STAGE1_EVALUATIONS_PER_EPOCH:-1}" \
  --checkpoint_interval_seconds "${IRB_STAGE1_CHECKPOINT_INTERVAL_SECONDS:-600}" \
  --gradient_accumulation_steps "${IRB_GRADIENT_ACCUMULATION_STEPS:-1}" \
  --per_device_train_batch_size "${IRB_PER_DEVICE_TRAIN_BATCH_SIZE:-1}" \
  --per_device_eval_batch_size "${IRB_PER_DEVICE_EVAL_BATCH_SIZE:-1}" \
  --save_total_limit "${IRB_STAGE1_SAVE_TOTAL_LIMIT:-6}" \
  --hidden_size "${IRB_MODEL_HIDDEN_SIZE:-1024}" \
  --intermediate_size "${IRB_MODEL_INTERMEDIATE_SIZE:-2048}" \
  --num_hidden_layers "${IRB_MODEL_NUM_HIDDEN_LAYERS:-8}" \
  --num_attention_heads "${IRB_MODEL_NUM_ATTENTION_HEADS:-8}" \
  --max_position_embeddings "${IRB_MODEL_MAX_POSITION_EMBEDDINGS:-4096}"

