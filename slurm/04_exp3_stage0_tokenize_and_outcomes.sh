#!/bin/bash
# =============================================================================
# Exp3 Stage 0 (CPU-heavy): tokenize + outcomes ONCE per format/config
# =============================================================================

set -euo pipefail

DATA_FORMAT=""
DATA_DIR_IN=""
DATA_VERSION_OUT=""
TOKENIZER_CONFIG=""
QUANTIZER=""
CLINICAL_ANCHORING=""
INCLUDE_REF_RANGES=""
INCLUDE_TIME_SPACING_TOKENS=""
FUSED_CATEGORY_VALUES=""

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
    --fused_category_values) FUSED_CATEGORY_VALUES="$2"; shift 2 ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${DATA_FORMAT}" || -z "${DATA_DIR_IN}" || -z "${DATA_VERSION_OUT}" || -z "${TOKENIZER_CONFIG}" || -z "${QUANTIZER}" || -z "${CLINICAL_ANCHORING}" || -z "${INCLUDE_REF_RANGES}" || -z "${INCLUDE_TIME_SPACING_TOKENS}" || -z "${FUSED_CATEGORY_VALUES}" ]]; then
  echo "Missing required args." >&2
  exit 2
fi
if [[ "${DATA_FORMAT}" != "meds_icu" && "${DATA_FORMAT}" != "clif" && "${DATA_FORMAT}" != "meds_randomized" && "${DATA_FORMAT}" != "meds_freqmatched" ]]; then
  echo "Invalid --data_format: ${DATA_FORMAT} (expected: meds_icu|clif|meds_randomized|meds_freqmatched)" >&2
  exit 2
fi
if [[ "${FUSED_CATEGORY_VALUES}" != "true" && "${FUSED_CATEGORY_VALUES}" != "false" ]]; then
  echo "Invalid --fused_category_values: ${FUSED_CATEGORY_VALUES} (expected: true|false)" >&2
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

RAW_BASE="${DATA_DIR}/raw"
VAL_SRC="validation"
if [[ -d "${DATA_DIR}/val" ]]; then
  VAL_SRC="val"
elif [[ -d "${DATA_DIR}/tuning" ]]; then
  # MEDS extraction uses train/tuning/test; tokenization expects raw/{train,val,test}.
  # We link raw/val -> ../tuning to keep the tokenization interface consistent.
  VAL_SRC="tuning"
fi
mkdir -p "${RAW_BASE}"
(
  cd "${RAW_BASE}"
  [[ -e train ]] || ln -s ../train train
  [[ -e test ]] || ln -s ../test test
  [[ -e val ]] || ln -s "../${VAL_SRC}" val
)

python "${FMS_EHRS_HOME}/fms_ehrs/scripts/tokenize_w_config.py" \
  --data_dir "${DATA_DIR}" \
  --data_version_in raw \
  --data_version_out "${DATA_VERSION_OUT}" \
  --config_loc "${TOKENIZER_CONFIG}" \
  --quantizer "${QUANTIZER}" \
  --clinical_anchoring "${CLINICAL_ANCHORING}" \
  --include_ref_ranges "${INCLUDE_REF_RANGES}" \
  --include_time_spacing_tokens "${INCLUDE_TIME_SPACING_TOKENS}" \
  --fused_category_values "${FUSED_CATEGORY_VALUES}" \
  --include_24h_cut

if [[ "${DATA_FORMAT}" = "meds_icu" || "${DATA_FORMAT}" = "meds_randomized" || "${DATA_FORMAT}" = "meds_freqmatched" ]]; then
  python "${IRB_HOME}/scripts/extract_outcomes_meds.py" \
    --meds_events_dir "${DATA_DIR}" \
    --tokenized_dir "${DATA_DIR}/${DATA_VERSION_OUT}_first_24h-tokenized" \
    --splits train,val,test
else
  python "${FMS_EHRS_HOME}/fms_ehrs/scripts/extract_outcomes.py" \
    --data_dir "${DATA_DIR}" \
    --ref_version "${DATA_VERSION_OUT}" \
    --data_version "${DATA_VERSION_OUT}_first_24h"
fi

