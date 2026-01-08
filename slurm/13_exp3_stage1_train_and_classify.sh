#!/bin/bash
# =============================================================================
# Exp3 Stage 1 (GPU): train representation mechanics + classify (per config×seed)
# =============================================================================
#
# Inputs (produced by Stage 0)
# ----------------------------
# Tokenized:
#   <data_dir>/<data_version>-tokenized/{train,val}/tokens_timelines.parquet
# Outcomes-joined (24h-cut):
#   <data_dir>/<data_version>_first_24h-tokenized/{train,val}/tokens_timelines_outcomes.parquet
#
# Usage (called by jobfile lines)
# -------------------------------
#   bash slurm/13_exp3_stage1_train_and_classify.sh \
#     --config_id clif_soft_time2vec \
#     --data_dir data/clif \
#     --data_version ventiles_5-10-5_unfused_time2vec \
#     --representation soft \
#     --temporal time2vec \
#     --num_bins 20 \
#     --seed 42
# =============================================================================

set -euo pipefail

# -----------------------------
# Parse args
# -----------------------------
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

# -----------------------------
# Locate repo root + source preamble
# -----------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IRB_HOME="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck disable=SC1091
source "${IRB_HOME}/slurm/preamble.sh"

FMS_EHRS_HOME="${FMS_EHRS_HOME:-$(realpath "${IRB_HOME}/../fms-ehrs")}"

DATA_DIR="${DATA_DIR_IN}"
if [[ "${DATA_DIR}" != /* ]]; then
  DATA_DIR="${IRB_HOME}/${DATA_DIR}"
fi

MODEL_DIR="${MODEL_DIR:-${IRB_HOME}/models}"

JOB_NAME="exp3_${CONFIG_ID}_s${SEED}"
JID="s${SEED}"

echo "=== Starting ${JOB_NAME} ==="
echo "DATA_DIR: ${DATA_DIR}"
echo "DATA_VERSION: ${DATA_VERSION}"
echo "REPRESENTATION: ${REPRESENTATION}"
echo "TEMPORAL: ${TEMPORAL}"
echo "NUM_BINS: ${NUM_BINS}"
echo "SEED: ${SEED}"
echo "MODEL_DIR: ${MODEL_DIR}"
echo "FMS_EHRS_HOME: ${FMS_EHRS_HOME}"

# -----------------------------
# Guardrails: require Stage 0 artifacts
# -----------------------------
TOK_TRAIN="${DATA_DIR}/${DATA_VERSION}-tokenized/train/tokens_timelines.parquet"
TOK_VAL="${DATA_DIR}/${DATA_VERSION}-tokenized/val/tokens_timelines.parquet"
OUT_TRAIN="${DATA_DIR}/${DATA_VERSION}_first_24h-tokenized/train/tokens_timelines_outcomes.parquet"
OUT_VAL="${DATA_DIR}/${DATA_VERSION}_first_24h-tokenized/val/tokens_timelines_outcomes.parquet"

for fp in "${TOK_TRAIN}" "${TOK_VAL}" "${OUT_TRAIN}" "${OUT_VAL}"; do
  if [[ ! -f "${fp}" ]]; then
    echo "Missing required Stage 0 artifact: ${fp}" >&2
    echo "Did you run Exp3 Stage 0 tokenization+outcomes first?" >&2
    exit 1
  fi
done

# -----------------------------
# Step 1: Train representation mechanics (GPU)
# -----------------------------
python "${FMS_EHRS_HOME}/fms_ehrs/scripts/train_representation.py" \
  --data_dir "${DATA_DIR}" \
  --data_version "${DATA_VERSION}" \
  --model_dir "${MODEL_DIR}" \
  --model_version "exp3_${CONFIG_ID}" \
  --representation "${REPRESENTATION}" \
  --temporal "${TEMPORAL}" \
  --num_bins "${NUM_BINS}" \
  --n_epochs 5 \
  --seed "${SEED}" \
  --jid "${JID}" \
  --wandb_project input-rep-benchmark-exp3

MODEL_LOC="${MODEL_DIR}/exp3_${CONFIG_ID}-${REPRESENTATION}-${TEMPORAL}-${JID}/model-${REPRESENTATION}-${TEMPORAL}"
if [[ ! -d "${MODEL_LOC}" ]]; then
  echo "Expected trained model directory not found: ${MODEL_LOC}" >&2
  exit 1
fi

# -----------------------------
# Step 2: Fine-tune classifiers (GPU)
# -----------------------------
for outcome in same_admission_death long_length_of_stay icu_admission imv_event; do
  python "${FMS_EHRS_HOME}/fms_ehrs/scripts/fine_tune_classification.py" \
    --model_loc "${MODEL_LOC}" \
    --data_dir "${DATA_DIR}" \
    --data_version "${DATA_VERSION}_first_24h" \
    --out_dir "${MODEL_DIR}/classifiers" \
    --outcome "${outcome}" \
    --n_epochs 5 \
    --jid "${JID}" \
    --wandb_project input-rep-benchmark-exp3-classify
done

echo "=== Completed ${JOB_NAME} ==="

