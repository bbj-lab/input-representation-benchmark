#!/bin/bash
# =============================================================================
# Exp3 Stage 0 (CPU-heavy): tokenize + outcomes ONCE per format/config
# =============================================================================
#
# Exp3 compares MEDS vs CLIF data formats. Tokenization is deterministic for a
# given (data_dir, data_version_out, tokenizer_config, quantizer, anchoring, temporal),
# so we run it once per config and reuse for all training seeds.
#
# This script writes:
#   <data_dir>/<data_version_out>-tokenized/{train,val,test}/tokens_timelines.parquet
#   <data_dir>/<data_version_out>_first_24h-tokenized/{train,val,test}/tokens_timelines.parquet
# and the outcomes-joined artifacts required for classification:
#   - MEDS: via scripts/extract_outcomes_meds.py (timestamp-based)
#   - CLIF: via fms_ehrs/scripts/extract_outcomes.py (token-presence-based)
#
# Usage (called by generated jobfiles):
#   bash slurm/11_exp3_stage0_tokenize_and_outcomes.sh \
#     --data_format meds \
#     --data_dir benchmarks/mimic-meds-extraction/data/meds/data \
#     --data_version_out ventiles_5-10-5_unfused_time2vec \
#     --tokenizer_config ../fms-ehrs/fms_ehrs/config/mimic-meds-ed.yaml \
#     --quantizer ventiles \
#     --clinical_anchoring 5-10-5 \
#     --include_ref_ranges true \
#     --include_time_spacing_tokens false
# =============================================================================

set -euo pipefail

# -----------------------------
# Parse args
# -----------------------------
DATA_FORMAT=""
DATA_DIR_IN=""
DATA_VERSION_OUT=""
TOKENIZER_CONFIG=""
QUANTIZER=""
CLINICAL_ANCHORING=""
INCLUDE_REF_RANGES=""
INCLUDE_TIME_SPACING_TOKENS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data_format) DATA_FORMAT="$2"; shift 2 ;;
    --data_dir) DATA_DIR_IN="$2"; shift 2 ;;
    --data_version_out) DATA_VERSION_OUT="$2"; shift 2 ;;
    --tokenizer_config) TOKENIZER_CONFIG="$2"; shift 2 ;;
    --quantizer) QUANTIZER="$2"; shift 2 ;;
    --clinical_anchoring) CLINICAL_ANCHORING="$2"; shift 2 ;;
    --include_ref_ranges) INCLUDE_REF_RANGES="$2"; shift 2 ;;
    --include_time_spacing_tokens) INCLUDE_TIME_SPACING_TOKENS="$2"; shift 2 ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${DATA_FORMAT}" || -z "${DATA_DIR_IN}" || -z "${DATA_VERSION_OUT}" || -z "${TOKENIZER_CONFIG}" || -z "${QUANTIZER}" || -z "${CLINICAL_ANCHORING}" || -z "${INCLUDE_REF_RANGES}" || -z "${INCLUDE_TIME_SPACING_TOKENS}" ]]; then
  echo "Missing required args." >&2
  echo "Required: --data_format --data_dir --data_version_out --tokenizer_config --quantizer --clinical_anchoring --include_ref_ranges --include_time_spacing_tokens" >&2
  exit 2
fi

if [[ "${DATA_FORMAT}" != "meds" && "${DATA_FORMAT}" != "clif" ]]; then
  echo "Invalid --data_format: ${DATA_FORMAT} (expected: meds|clif)" >&2
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

# Resolve paths relative to repo root for reproducibility
DATA_DIR="${DATA_DIR_IN}"
if [[ "${DATA_DIR}" != /* ]]; then
  DATA_DIR="${IRB_HOME}/${DATA_DIR}"
fi
TOKENIZER_CONFIG_ABS="${TOKENIZER_CONFIG}"
if [[ "${TOKENIZER_CONFIG_ABS}" != /* ]]; then
  TOKENIZER_CONFIG_ABS="${IRB_HOME}/${TOKENIZER_CONFIG_ABS}"
fi

JOB_NAME="exp3_stage0_${DATA_FORMAT}_${DATA_VERSION_OUT}"
echo "=== Starting ${JOB_NAME} ==="
echo "DATA_FORMAT: ${DATA_FORMAT}"
echo "DATA_DIR: ${DATA_DIR}"
echo "DATA_VERSION_OUT: ${DATA_VERSION_OUT}"
echo "TOKENIZER_CONFIG: ${TOKENIZER_CONFIG_ABS}"

TOKENIZED_DIR="${DATA_DIR}/${DATA_VERSION_OUT}-tokenized"
TOKENIZED_24H_DIR="${DATA_DIR}/${DATA_VERSION_OUT}_first_24h-tokenized"

# -----------------------------
# Step 1: Tokenize (skip if already present)
# -----------------------------
TOK_OK="false"
TOK_OK="true"
for s in train val test; do
  if [[ ! -f "${TOKENIZED_DIR}/${s}/tokens_timelines.parquet" ]]; then
    TOK_OK="false"
    break
  fi
done
if [[ ! -f "${TOKENIZED_DIR}/train/vocab.gzip" ]]; then
  TOK_OK="false"
fi
for s in train val test; do
  if [[ ! -f "${TOKENIZED_24H_DIR}/${s}/tokens_timelines.parquet" ]]; then
    TOK_OK="false"
    break
  fi
done
if [[ ! -f "${TOKENIZED_24H_DIR}/train/vocab.gzip" ]]; then
  TOK_OK="false"
fi

if [[ "${TOK_OK}" == "true" ]]; then
  echo "[stage0] Tokenization exists; skipping tokenize_w_config.py"
else
  # NOTE: fms-ehrs uses argparse.BooleanOptionalAction for boolean overrides:
  #   --flag / --no-flag (NOT `--flag true|false`).
  tokenize_args=(
    --data_dir "${DATA_DIR}"
    --data_version_in raw
    --data_version_out "${DATA_VERSION_OUT}"
    --config_loc "${TOKENIZER_CONFIG_ABS}"
    --quantizer "${QUANTIZER}"
    --clinical_anchoring "${CLINICAL_ANCHORING}"
    --include_24h_cut
  )

  case "${INCLUDE_REF_RANGES}" in
    true) tokenize_args+=( --include_ref_ranges ) ;;
    false) tokenize_args+=( --no-include_ref_ranges ) ;;
    *) echo "ERROR: --include_ref_ranges must be true|false (got: ${INCLUDE_REF_RANGES})" >&2; exit 1 ;;
  esac

  case "${INCLUDE_TIME_SPACING_TOKENS}" in
    true) tokenize_args+=( --include_time_spacing_tokens ) ;;
    false) tokenize_args+=( --no-include_time_spacing_tokens ) ;;
    *) echo "ERROR: --include_time_spacing_tokens must be true|false (got: ${INCLUDE_TIME_SPACING_TOKENS})" >&2; exit 1 ;;
  esac

  # Exp3 Stage0 always uses unfused tokenization (required for soft/continuous + parity).
  tokenize_args+=( --no-fused_category_values )

  python "${FMS_EHRS_HOME}/fms_ehrs/scripts/tokenize_w_config.py" "${tokenize_args[@]}"
fi

# -----------------------------
# Step 2: Extract outcomes (skip if already present)
# -----------------------------
OUT_OK="false"
if [[ -f "${TOKENIZED_24H_DIR}/train/tokens_timelines_outcomes.parquet" && -f "${TOKENIZED_24H_DIR}/val/tokens_timelines_outcomes.parquet" ]]; then
  OUT_OK="true"
fi

if [[ "${OUT_OK}" == "true" ]]; then
  echo "[stage0] Outcomes exist; skipping outcome extraction"
else
  if [[ "${DATA_FORMAT}" == "meds" ]]; then
    python "${IRB_HOME}/scripts/extract_outcomes_meds.py" \
      --meds_events_dir "${DATA_DIR}" \
      --tokenized_dir "${TOKENIZED_24H_DIR}" \
      --splits train,val,test
  else
    python "${FMS_EHRS_HOME}/fms_ehrs/scripts/extract_outcomes.py" \
      --data_dir "${DATA_DIR}" \
      --ref_version "${DATA_VERSION_OUT}" \
      --data_version "${DATA_VERSION_OUT}_first_24h"
  fi
fi

echo "=== Completed ${JOB_NAME} ==="

