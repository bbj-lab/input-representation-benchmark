#!/bin/bash
# =============================================================================
# Stage 0 (CPU-heavy): Tokenize MEDS once per config + write outcomes parquet
# =============================================================================
#
# Design goal
# -----------
# Tokenization depends on (quantizer, clinical_anchoring, fused/unfused, time tokens),
# but it does NOT depend on:
#   - random seed
#   - Optuna trial
# Therefore we tokenize once per config and reuse for all downstream training jobs.
#
# This script also computes MEDS-derived outcomes and writes:
#   <data_dir>/<data_version_out>_first_24h-tokenized/{train,val,test}/tokens_timelines_outcomes.parquet
#
# Cache layout (shared filesystem)
# -------------------------------
# Tokenized directories are placed under a shared cache root on /gpfs and symlinked
# back into the repo’s expected location under:
#   ${DATA_DIR}/data/<data_version_out>-tokenized
#   ${DATA_DIR}/data/<data_version_out>_first_24h-tokenized
#
# Users may override the cache location by setting:
#   IRB_TOKEN_CACHE_ROOT=/gpfs/data/bbj-lab/users/$USER/irb_scratch/tokenized/<dataset_id>
#
# Usage (direct):
#   bash slurm/03_stage0_tokenize_and_outcomes_meds.sh \
#     --data_version_out deciles_none_unfused_time_tokens \
#     --quantizer deciles \
#     --clinical_anchoring none \
#     --include_ref_ranges false \
#     --fused_category_values false
#
# Usage (recommended: via SLURM array):
#   # First generate jobfile with 12 lines:
#   python run_experiments.py --mode demo --exp 1
#   # Then:
#   sbatch --array=0-11%2 slurm/02_run_from_jobfile_cpu.sh slurm/04_exp1_tokenize_jobs.sh
# =============================================================================

set -euo pipefail

usage() {
  echo "Usage: $0 --data_version_out <name> --quantizer <deciles|ventiles|trentiles|centiles> --clinical_anchoring <none|5-10-5|10-10-10> --include_ref_ranges <true|false> --fused_category_values <true|false>"
}

DATA_VERSION_OUT=""
QUANTIZER=""
CLINICAL_ANCHORING=""
INCLUDE_REF_RANGES=""
FUSED_CATEGORY_VALUES=""
INCLUDE_TIME_SPACING_TOKENS="true"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data_version_out) DATA_VERSION_OUT="$2"; shift 2 ;;
    --quantizer) QUANTIZER="$2"; shift 2 ;;
    --clinical_anchoring) CLINICAL_ANCHORING="$2"; shift 2 ;;
    --include_ref_ranges) INCLUDE_REF_RANGES="$2"; shift 2 ;;
    --fused_category_values) FUSED_CATEGORY_VALUES="$2"; shift 2 ;;
    --include_time_spacing_tokens) INCLUDE_TIME_SPACING_TOKENS="$2"; shift 2 ;;
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
source "${IRB_HOME}/slurm/preamble.sh"
cd "${IRB_HOME}"

MEDS_DATA_DIR="${DATA_DIR}/data"
TOKENIZER_CONFIG="${FMS_EHRS_HOME}/fms_ehrs/config/mimic-meds-ed.yaml"

# Shared cache root for tokenized artifacts (must be on a shared filesystem).
# Default is a user-owned directory on /gpfs (BBJ cluster). For non-cluster/local
# runs, fall back to a repo-local cache directory.
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

# Create targets
mkdir -p "${TARGET_MAIN}" "${TARGET_24H}"

# Create or validate symlinks
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

ensure_symlink "${TARGET_MAIN}" "${LINK_MAIN}"
ensure_symlink "${TARGET_24H}" "${LINK_24H}"

# Skip tokenization if the key outputs already exist
MAIN_OK=0
H24_OK=0
MAIN_OK=1
for s in train val test; do
  if [[ ! -f "${LINK_MAIN}/${s}/tokens_timelines.parquet" ]]; then
    MAIN_OK=0
    break
  fi
done
if [[ ! -f "${LINK_MAIN}/train/vocab.gzip" ]]; then
  MAIN_OK=0
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

if [[ "${MAIN_OK}" -eq 1 && "${H24_OK}" -eq 1 ]]; then
  echo "[stage0] Tokenization outputs already exist for ${DATA_VERSION_OUT}; skipping tokenization."
else
  # NOTE: fms-ehrs uses argparse.BooleanOptionalAction for boolean overrides.
  # That means the correct CLI is:
  #   --flag      (sets True)
  #   --no-flag   (sets False)
  # Passing `--flag false` *does not* set False; it sets True and treats "false" as unknown.
  tokenize_args=(
    --data_dir "${MEDS_DATA_DIR}"
    --data_version_in raw
    --data_version_out "${DATA_VERSION_OUT}"
    --config_loc "${TOKENIZER_CONFIG}"
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

  case "${FUSED_CATEGORY_VALUES}" in
    true) tokenize_args+=( --fused_category_values ) ;;
    false) tokenize_args+=( --no-fused_category_values ) ;;
    *) echo "ERROR: --fused_category_values must be true|false (got: ${FUSED_CATEGORY_VALUES})" >&2; exit 1 ;;
  esac

  python "${FMS_EHRS_HOME}/fms_ehrs/scripts/tokenize_w_config.py" "${tokenize_args[@]}"
fi

# Outcomes are deterministic given MEDS + tokenized timelines; compute once per config.
# Write into the 24h-cut tokenized directory (used for classification).
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

