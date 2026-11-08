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
MAX_PADDED_LEN=""
NUMERIC_ENCODING=""
OUTCOMES_MEDS_EVENTS_DIR=""
ONLY_24H_CUT="false"
INCLUDE_24H_CUT="true"
SKIP_SUMMARY="false"
DETERMINISTIC_VOCAB="false"
SPLITS_CSV="train,val,test"

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
    --max_padded_len) MAX_PADDED_LEN="$2"; shift 2 ;;
    --numeric_encoding) NUMERIC_ENCODING="$2"; shift 2 ;;
    --outcomes_meds_events_dir) OUTCOMES_MEDS_EVENTS_DIR="$2"; shift 2 ;;
    --only_24h_cut) ONLY_24H_CUT="$2"; shift 2 ;;
    --include_24h_cut) INCLUDE_24H_CUT="$2"; shift 2 ;;
    --skip_summary) SKIP_SUMMARY="$2"; shift 2 ;;
    --deterministic_vocab) DETERMINISTIC_VOCAB="$2"; shift 2 ;;
    --splits) SPLITS_CSV="$2"; shift 2 ;;
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
if [[ "${DATA_FORMAT}" != "meds_icu" && "${DATA_FORMAT}" != "meds_clif" ]]; then
  echo "Invalid --data_format: ${DATA_FORMAT} (expected: meds_icu|meds_clif)" >&2
  exit 2
fi
if [[ "${FUSED_CATEGORY_VALUES}" != "true" && "${FUSED_CATEGORY_VALUES}" != "false" ]]; then
  echo "Invalid --fused_category_values: ${FUSED_CATEGORY_VALUES} (expected: true|false)" >&2
  exit 2
fi
if [[ "${ONLY_24H_CUT}" != "true" && "${ONLY_24H_CUT}" != "false" ]]; then
  echo "Invalid --only_24h_cut: ${ONLY_24H_CUT} (expected: true|false)" >&2
  exit 2
fi
if [[ "${INCLUDE_24H_CUT}" != "true" && "${INCLUDE_24H_CUT}" != "false" ]]; then
  echo "Invalid --include_24h_cut: ${INCLUDE_24H_CUT} (expected: true|false)" >&2
  exit 2
fi
if [[ "${SKIP_SUMMARY}" != "true" && "${SKIP_SUMMARY}" != "false" ]]; then
  echo "Invalid --skip_summary: ${SKIP_SUMMARY} (expected: true|false)" >&2
  exit 2
fi
if [[ "${DETERMINISTIC_VOCAB}" != "true" && "${DETERMINISTIC_VOCAB}" != "false" ]]; then
  echo "Invalid --deterministic_vocab: ${DETERMINISTIC_VOCAB} (expected: true|false)" >&2
  exit 2
fi
if [[ "${ONLY_24H_CUT}" = "true" && "${INCLUDE_24H_CUT}" = "false" ]]; then
  echo "--only_24h_cut requires --include_24h_cut true." >&2
  exit 2
fi
IFS=',' read -r -a SPLITS <<< "${SPLITS_CSV}"
if [[ "${#SPLITS[@]}" -eq 0 ]]; then
  echo "--splits must include at least one of train,val,test." >&2
  exit 2
fi
for split in "${SPLITS[@]}"; do
  if [[ "${split}" != "train" && "${split}" != "val" && "${split}" != "test" ]]; then
    echo "Invalid --splits member: ${split} (expected train,val,test)" >&2
    exit 2
  fi
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IRB_HOME="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${IRB_HOME}/slurm/00_preamble.sh"
FMS_EHRS_HOME="${FMS_EHRS_HOME:-$(realpath "${IRB_HOME}/../fms-ehrs")}"
export POLARS_MAX_THREADS="${POLARS_MAX_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
echo "POLARS_MAX_THREADS=${POLARS_MAX_THREADS}"

require_dir() {
  local path="$1"
  local what="$2"
  if [[ ! -d "${path}" ]]; then
    echo "ERROR: missing ${what}: ${path}" >&2
    exit 1
  fi
}

require_file() {
  local path="$1"
  local what="$2"
  if [[ ! -f "${path}" ]]; then
    echo "ERROR: missing ${what}: ${path}" >&2
    exit 1
  fi
}

if [[ -z "${MAX_PADDED_LEN}" ]]; then
  MAX_PADDED_LEN="${IRB_MAX_PADDED_LEN}"
fi

DATA_DIR="${DATA_DIR_IN}"
if [[ "${DATA_DIR}" != /* ]]; then
  DATA_DIR="${IRB_HOME}/${DATA_DIR}"
fi
OUTCOMES_EVENTS_DIR="${OUTCOMES_MEDS_EVENTS_DIR:-${DATA_DIR}}"
if [[ "${OUTCOMES_EVENTS_DIR}" != /* ]]; then
  OUTCOMES_EVENTS_DIR="${IRB_HOME}/${OUTCOMES_EVENTS_DIR}"
fi

if [[ "${DATA_FORMAT}" = "meds_clif" ]]; then
  ADAPTER_MANIFEST="${DATA_DIR}/cocoa_adapter_manifest.json"
  FROZEN_SPLITS_MANIFEST="${IRB_HOME}/outputs/runs/exp3/cohort/frozen_split_manifest.json"
  require_file "${ADAPTER_MANIFEST}" "Cocoa adapter manifest"
  require_file "${FROZEN_SPLITS_MANIFEST}" "frozen Exp3 split manifest"
  python "${FMS_EHRS_HOME}/fms_ehrs/scripts/collate_clif_with_cocoa.py" \
    validate \
    --meds-dir "${DATA_DIR}" \
    --frozen-splits-manifest "${FROZEN_SPLITS_MANIFEST}" \
    --collation-config "${FMS_EHRS_HOME}/fms_ehrs/config/cocoa-exp3-clif-full.yaml"
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
if [[ ! -d "${DATA_DIR}/train" || ! -d "${DATA_DIR}/test" ]]; then
  echo "ERROR: expected train/ and test/ splits under ${DATA_DIR}" >&2
  exit 1
fi
if [[ ! -d "${DATA_DIR}/${VAL_SRC}" ]]; then
  echo "ERROR: could not locate a validation split under ${DATA_DIR}" >&2
  exit 1
fi
mkdir -p "${RAW_BASE}"
ensure_raw_link() {
  local link_name="$1"
  local target="$2"
  local link_path="${RAW_BASE}/${link_name}"
  if [[ -L "${link_path}" ]]; then
    local cur
    cur="$(readlink "${link_path}")"
    if [[ "${cur}" != "${target}" ]]; then
      rm -f "${link_path}"
      ln -s "${target}" "${link_path}"
    fi
  elif [[ -e "${link_path}" ]]; then
    echo "ERROR: ${link_path} exists and is not a symlink." >&2
    exit 1
  else
    ln -s "${target}" "${link_path}"
  fi
}
(
  cd "${RAW_BASE}"
  ensure_raw_link train ../train
  ensure_raw_link test ../test
  ensure_raw_link val "../${VAL_SRC}"
)
for s in "${SPLITS[@]}"; do
  require_dir "${RAW_BASE}/${s}" "raw MEDS split"
done

# Build tokenize args, handling BooleanOptionalAction flags properly.
tokenize_args=(
  --data_dir "${DATA_DIR}"
  --data_version_in raw
  --data_version_out "${DATA_VERSION_OUT}"
  --config_loc "${TOKENIZER_CONFIG}"
  --max_padded_len "${MAX_PADDED_LEN}"
  --quantizer "${QUANTIZER}"
  --clinical_anchoring "${CLINICAL_ANCHORING}"
  --splits "${SPLITS[@]}"
)

# BooleanOptionalAction: --flag / --no-flag
case "${INCLUDE_REF_RANGES}" in
  true)  tokenize_args+=( --include_ref_ranges ) ;;
  false) tokenize_args+=( --no-include_ref_ranges ) ;;
  *) echo "ERROR: --include_ref_ranges must be true|false (got: ${INCLUDE_REF_RANGES})" >&2; exit 2 ;;
esac
case "${INCLUDE_TIME_SPACING_TOKENS}" in
  true)  tokenize_args+=( --include_time_spacing_tokens ) ;;
  false) tokenize_args+=( --no-include_time_spacing_tokens ) ;;
  *) echo "ERROR: --include_time_spacing_tokens must be true|false (got: ${INCLUDE_TIME_SPACING_TOKENS})" >&2; exit 2 ;;
esac
case "${FUSED_CATEGORY_VALUES}" in
  true)  tokenize_args+=( --fused_category_values ) ;;
  false) tokenize_args+=( --no-fused_category_values ) ;;
  *) echo "ERROR: --fused_category_values must be true|false (got: ${FUSED_CATEGORY_VALUES})" >&2; exit 2 ;;
esac
if [[ -n "${NUMERIC_ENCODING}" ]]; then
  tokenize_args+=( --numeric_encoding "${NUMERIC_ENCODING}" )
fi
if [[ "${INCLUDE_24H_CUT}" = "true" ]]; then
  tokenize_args+=( --include_24h_cut )
else
  tokenize_args+=( --no-include_24h_cut )
fi
if [[ "${ONLY_24H_CUT}" = "true" ]]; then
  tokenize_args+=( --only_24h_cut )
fi
if [[ "${SKIP_SUMMARY}" = "true" ]]; then
  tokenize_args+=( --skip_summary )
fi
if [[ "${DETERMINISTIC_VOCAB}" = "true" ]]; then
  tokenize_args+=( --deterministic_vocab )
else
  tokenize_args+=( --no-deterministic_vocab )
fi

python "${FMS_EHRS_HOME}/fms_ehrs/scripts/tokenize_w_config.py" "${tokenize_args[@]}"

if [[ "${INCLUDE_24H_CUT}" = "true" && ( "${DATA_FORMAT}" = "meds_icu" || "${DATA_FORMAT}" = "meds_clif" ) ]]; then
  python "${IRB_HOME}/pipeline/scripts/extract_outcomes_meds.py" \
    --meds_events_dir "${OUTCOMES_EVENTS_DIR}" \
    --tokenized_dir "${DATA_DIR}/${DATA_VERSION_OUT}_first_24h-tokenized" \
    --splits "${SPLITS_CSV}" \
    --strict
fi

if [[ "${ONLY_24H_CUT}" = "false" ]]; then
  TOKENIZED_FULL_DIR="${DATA_DIR}/${DATA_VERSION_OUT}-tokenized"
  for s in "${SPLITS[@]}"; do
    require_file "${TOKENIZED_FULL_DIR}/${s}/tokens_timelines.parquet" "full tokenized timeline parquet"
  done
  if [[ " ${SPLITS[*]} " == *" train "* ]]; then
    require_file "${TOKENIZED_FULL_DIR}/train/vocab.gzip" "full tokenizer vocabulary"
    require_file "${TOKENIZED_FULL_DIR}/train/numeric_stats.json" "full numeric stats"
  fi
fi

if [[ "${INCLUDE_24H_CUT}" = "true" ]]; then
  TOKENIZED_24H_DIR="${DATA_DIR}/${DATA_VERSION_OUT}_first_24h-tokenized"
  for s in "${SPLITS[@]}"; do
    require_file "${TOKENIZED_24H_DIR}/${s}/tokens_timelines.parquet" "tokenized timeline parquet"
    require_file "${TOKENIZED_24H_DIR}/${s}/tokens_timelines_outcomes.parquet" "base outcomes parquet"
  done
  if [[ " ${SPLITS[*]} " == *" train "* ]]; then
    require_file "${TOKENIZED_24H_DIR}/train/vocab.gzip" "tokenizer vocabulary"
    require_file "${TOKENIZED_24H_DIR}/train/numeric_stats.json" "numeric stats"
  fi
fi

