#!/bin/bash
# =============================================================================
# Stage 0 (CPU-heavy): Tokenize MEDS once per config + write outcomes parquet
# =============================================================================
#
# Usage (recommended: via SLURM array on tier2q):
#   python run_experiments.py --mode demo --exp 1
#   sbatch --array=0-11%2 slurm/02_run_stage0_tier2q_tokenize.sh slurm/generated/demo/04_exp1_stage0_tokenize.jobfile
#
# Validity note:
#   - Internal validity: stage0 enforces required tokenization artifacts (including numeric_stats.json)
#     so downstream representation comparisons do not silently mix incompatible preprocessing.
#
# =============================================================================

set -euo pipefail

usage() {
  echo "Usage: $0 --data_version_out <name> --quantizer <deciles|ventiles|trentiles|centiles> --clinical_anchoring <none|5-10-5|10-10-10> --include_ref_ranges <true|false> --fused_category_values <true|false> [--vocab_path <path>] [--max_padded_len <int>] [--only_24h_cut <true|false>]"
}

DATA_VERSION_OUT=""
QUANTIZER=""
CLINICAL_ANCHORING=""
INCLUDE_REF_RANGES=""
FUSED_CATEGORY_VALUES=""
INCLUDE_TIME_SPACING_TOKENS="true"
VOCAB_PATH=""
MAX_PADDED_LEN=""
ONLY_24H_CUT="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data_version_out) DATA_VERSION_OUT="$2"; shift 2 ;;
    --quantizer) QUANTIZER="$2"; shift 2 ;;
    --clinical_anchoring) CLINICAL_ANCHORING="$2"; shift 2 ;;
    --include_ref_ranges) INCLUDE_REF_RANGES="$2"; shift 2 ;;
    --fused_category_values) FUSED_CATEGORY_VALUES="$2"; shift 2 ;;
    --include_time_spacing_tokens) INCLUDE_TIME_SPACING_TOKENS="$2"; shift 2 ;;
    --vocab_path) VOCAB_PATH="$2"; shift 2 ;;
    --max_padded_len) MAX_PADDED_LEN="$2"; shift 2 ;;
    --only_24h_cut) ONLY_24H_CUT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: Unknown arg: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "${DATA_VERSION_OUT}" || -z "${QUANTIZER}" || -z "${CLINICAL_ANCHORING}" || -z "${INCLUDE_REF_RANGES}" || -z "${FUSED_CATEGORY_VALUES}" ]]; then
  echo "ERROR: Missing required args."
  usage
  exit 1
fi

find_repo_root() {
  local d="$1"
  while [[ "$d" != "/" ]]; do
    if [[ -f "$d/run_experiments.py" && -d "$d/slurm" ]]; then
      echo "$d"
      return 0
    fi
    d="$(dirname "$d")"
  done
  return 1
}

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
IRB_HOME="${IRB_HOME:-$(find_repo_root "${SUBMIT_DIR}")}"
export IRB_CONDA_ENV="${IRB_CONDA_ENV:-input-rep}"
source "${IRB_HOME}/slurm/00_preamble.sh"
cd "${IRB_HOME}"

# `DATA_DIR` is standardized (via slurm/00_preamble.sh) to point to the MEDS
# events directory (the one containing train/val/test or train/tuning/test and
# raw parquet shards). Do NOT append `/data` here.
MEDS_DATA_DIR="${DATA_DIR}"
TOKENIZER_CONFIG="${FMS_EHRS_HOME}/fms_ehrs/config/mimic-meds-ed.yaml"

DATASET_ID="${IRB_TOKEN_CACHE_DATASET_ID:-mimiciv-3.1_meds_70-10-20}"
DEFAULT_CACHE_ROOT=""
if [[ -d "${hm}" ]]; then
  DEFAULT_CACHE_ROOT="${hm}/irb_scratch/tokenized/${DATASET_ID}"
else
  DEFAULT_CACHE_ROOT="${IRB_HOME}/.cache/tokenized/${DATASET_ID}"
fi
IRB_TOKEN_CACHE_ROOT="${IRB_TOKEN_CACHE_ROOT:-${DEFAULT_CACHE_ROOT}}"

mkdir -p "${IRB_TOKEN_CACHE_ROOT}"

TARGET_MAIN="${IRB_TOKEN_CACHE_ROOT}/${DATA_VERSION_OUT}-tokenized"
TARGET_24H="${IRB_TOKEN_CACHE_ROOT}/${DATA_VERSION_OUT}_first_24h-tokenized"

LINK_MAIN="${MEDS_DATA_DIR}/${DATA_VERSION_OUT}-tokenized"
LINK_24H="${MEDS_DATA_DIR}/${DATA_VERSION_OUT}_first_24h-tokenized"

mkdir -p "${TARGET_24H}"
if [[ "${ONLY_24H_CUT}" != "true" ]]; then
  mkdir -p "${TARGET_MAIN}"
fi

ensure_symlink() {
  local target="$1"
  local link="$2"
  if [[ -L "${link}" ]]; then
    local cur
    cur="$(readlink "${link}")"
    if [[ "${cur}" == "${target}" ]]; then
      return 0
    fi
    echo "ERROR: ${link} is a symlink to ${cur}, expected ${target}."
    echo "  Resolve by removing ${link} and re-running."
    exit 1
  fi
  if [[ -e "${link}" ]]; then
    echo "ERROR: ${link} exists and is not a symlink. Refusing to overwrite."
    echo "  Move or delete it, or set IRB_TOKEN_CACHE_ROOT to a new location."
    exit 1
  fi
  ln -s "${target}" "${link}"
}

ensure_symlink "${TARGET_24H}" "${LINK_24H}"

MAIN_OK=1
if [[ "${ONLY_24H_CUT}" == "true" ]]; then
  MAIN_OK=1
else
  ensure_symlink "${TARGET_MAIN}" "${LINK_MAIN}"
  for s in train val test; do
    if [[ ! -f "${LINK_MAIN}/${s}/tokens_timelines.parquet" ]]; then
      MAIN_OK=0
      break
    fi
  done
  if [[ ! -f "${LINK_MAIN}/train/vocab.gzip" ]]; then
    MAIN_OK=0
  fi
  # Continuous encoders require quantizer/anchoring-independent numeric stats
  # computed from raw training values. This file is produced by
  # `fms_ehrs/scripts/tokenize_w_config.py` (train split) and must exist for
  # Experiment 2 continuous configurations to be method-faithful and fair across
  # quantization/anchoring settings.
  if [[ ! -f "${LINK_MAIN}/train/numeric_stats.json" ]]; then
    MAIN_OK=0
  fi
fi

H24_OK=1
for s in train val test; do
  if [[ ! -f "${LINK_24H}/${s}/tokens_timelines.parquet" ]]; then
    H24_OK=0
    break
  fi
done
if [[ ! -f "${LINK_24H}/train/vocab.gzip" ]]; then
  H24_OK=0
fi
if [[ ! -f "${LINK_24H}/train/numeric_stats.json" ]]; then
  H24_OK=0
fi

if [[ "${MAIN_OK}" -eq 1 && "${H24_OK}" -eq 1 ]]; then
  echo "[stage0] Tokenization outputs already exist for ${DATA_VERSION_OUT}; skipping tokenization."
else
  tokenize_args=(
    --data_dir "${MEDS_DATA_DIR}"
    --data_version_in raw
    --data_version_out "${DATA_VERSION_OUT}"
    --config_loc "${TOKENIZER_CONFIG}"
    --quantizer "${QUANTIZER}"
    --clinical_anchoring "${CLINICAL_ANCHORING}"
  )
  if [[ "${ONLY_24H_CUT}" == "true" ]]; then
    tokenize_args+=( --only_24h_cut )
  else
    tokenize_args+=( --include_24h_cut )
  fi
  if [[ -n "${VOCAB_PATH}" ]]; then
    tokenize_args+=( --vocab_path "${VOCAB_PATH}" )
  fi
  if [[ -n "${MAX_PADDED_LEN}" ]]; then
    tokenize_args+=( --max_padded_len "${MAX_PADDED_LEN}" )
  fi

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

  case "${FUSED_CATEGORY_VALUES}" in
    true) tokenize_args+=( --fused_category_values ) ;;
    false) tokenize_args+=( --no-fused_category_values ) ;;
    *) echo "ERROR: --fused_category_values must be true|false (got: ${FUSED_CATEGORY_VALUES})" >&2; exit 1 ;;
  esac

  python "${FMS_EHRS_HOME}/fms_ehrs/scripts/tokenize_w_config.py" "${tokenize_args[@]}"

  # ---------------------------------------------------------------------------
  # Post-tokenization validity checks (fail loudly)
  #
  # Rationale: Stage0 is the source-of-truth for downstream training inputs.
  # We must not allow Stage0 to "succeed" while silently missing artifacts
  # required for method-faithful continuous encoders (numeric_stats.json).
  # ---------------------------------------------------------------------------
  for s in train val test; do
    if [[ "${ONLY_24H_CUT}" != "true" ]]; then
      if [[ ! -f "${LINK_MAIN}/${s}/tokens_timelines.parquet" ]]; then
        echo "ERROR: missing tokenization output: ${LINK_MAIN}/${s}/tokens_timelines.parquet" >&2
        exit 1
      fi
    fi
    if [[ ! -f "${LINK_24H}/${s}/tokens_timelines.parquet" ]]; then
      echo "ERROR: missing tokenization output: ${LINK_24H}/${s}/tokens_timelines.parquet" >&2
      exit 1
    fi
  done
  if [[ "${ONLY_24H_CUT}" != "true" ]]; then
    if [[ ! -f "${LINK_MAIN}/train/vocab.gzip" ]]; then
      echo "ERROR: missing vocabulary: ${LINK_MAIN}/train/vocab.gzip" >&2
      exit 1
    fi
  fi
  if [[ ! -f "${LINK_24H}/train/vocab.gzip" ]]; then
    echo "ERROR: missing vocabulary: ${LINK_24H}/train/vocab.gzip" >&2
    exit 1
  fi
  if [[ "${ONLY_24H_CUT}" != "true" ]]; then
    if [[ ! -f "${LINK_MAIN}/train/numeric_stats.json" ]]; then
      echo "ERROR: missing required numeric stats: ${LINK_MAIN}/train/numeric_stats.json" >&2
      echo "  Continuous encoders require these quantizer/anchoring-independent per-code statistics." >&2
      exit 1
    fi
  fi
  if [[ ! -f "${LINK_24H}/train/numeric_stats.json" ]]; then
    echo "ERROR: missing required numeric stats: ${LINK_24H}/train/numeric_stats.json" >&2
    echo "  Continuous encoders require these quantizer/anchoring-independent per-code statistics." >&2
    exit 1
  fi
fi

OUT_OK=0
if [[ -f "${LINK_24H}/train/tokens_timelines_outcomes.parquet" && -f "${LINK_24H}/val/tokens_timelines_outcomes.parquet" && -f "${LINK_24H}/test/tokens_timelines_outcomes.parquet" ]]; then
  OUT_OK=1
fi

if [[ "${OUT_OK}" -eq 1 ]]; then
  echo "[stage0] Outcomes already exist for ${DATA_VERSION_OUT}; skipping outcome extraction."
else
  python scripts/extract_outcomes_meds.py \
    --meds_events_dir "${MEDS_DATA_DIR}" \
    --tokenized_dir "${LINK_24H}" \
    --splits train,val,test
fi

echo "[stage0] Done: ${DATA_VERSION_OUT}"
