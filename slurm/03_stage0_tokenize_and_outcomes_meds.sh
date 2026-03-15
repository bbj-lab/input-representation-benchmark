#!/bin/bash
# =============================================================================
# Stage 0 (CPU-heavy): Tokenize MEDS once per config + write outcomes parquet
# =============================================================================
#
# Usage (recommended: via SLURM array on tier2q):
#   python run_experiments.py --mode demo --exp 1
#   sbatch --array=0-11 slurm/02_run_stage0_tier2q_tokenize.sh slurm/generated/demo/04_exp1_stage0_tokenize.jobfile
#
# Validity note:
#   - Internal validity: stage0 enforces required tokenization artifacts (including numeric_stats.json)
#     so downstream representation comparisons do not silently mix incompatible preprocessing.
#
# =============================================================================

set -euo pipefail

usage() {
  echo "Usage: $0 --data_version_out <name> --quantizer <deciles|ventiles|trentiles|centiles> --clinical_anchoring <none|5-10-5|10-10-10> --include_ref_ranges <true|false> --fused_category_values <true|false> [--numeric_encoding <quantile|xval>] [--vocab_path <path>] [--max_padded_len <int>] [--only_24h_cut <true|false>]"
}

DATA_VERSION_OUT=""
QUANTIZER=""
CLINICAL_ANCHORING=""
INCLUDE_REF_RANGES=""
FUSED_CATEGORY_VALUES=""
INCLUDE_TIME_SPACING_TOKENS="true"
NUMERIC_ENCODING=""
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
    --numeric_encoding) NUMERIC_ENCODING="$2"; shift 2 ;;
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

# Default tokenization truncation cap: keep deterministic and centralized.
# This controls where we append a TRUNC token when a timeline exceeds the cap.
if [[ -z "${MAX_PADDED_LEN}" ]]; then
  MAX_PADDED_LEN="${IRB_MAX_PADDED_LEN}"
fi

# `DATA_DIR` is standardized (via slurm/00_preamble.sh) to point to the MEDS
# events directory (the one containing train/val/test or train/tuning/test and
# raw parquet shards). Do NOT append `/data` here.
MEDS_DATA_DIR="${DATA_DIR}"
TOKENIZER_CONFIG="${FMS_EHRS_HOME}/fms_ehrs/config/mimic-meds-ed.yaml"

for s in train val test; do
  require_dir "${MEDS_DATA_DIR}/raw/${s}" "raw MEDS split"
done

DATASET_ID="${IRB_TOKEN_CACHE_DATASET_ID:-mimiciv-3.1_meds_70-10-20}"
DEFAULT_CACHE_ROOT="${IRB_TOKENIZED_ROOT}/${DATASET_ID}"
IRB_TOKEN_CACHE_ROOT="${IRB_TOKEN_CACHE_ROOT:-${DEFAULT_CACHE_ROOT}}"
IRB_STAGE0_CREATE_TOKENIZED_LINKS="${IRB_STAGE0_CREATE_TOKENIZED_LINKS:-false}"

mkdir -p "${IRB_TOKEN_CACHE_ROOT}"

TARGET_MAIN="${IRB_TOKEN_CACHE_ROOT}/${DATA_VERSION_OUT}-tokenized"
TARGET_24H="${IRB_TOKEN_CACHE_ROOT}/${DATA_VERSION_OUT}_first_24h-tokenized"
LEGACY_LINK_MAIN="${MEDS_DATA_DIR}/${DATA_VERSION_OUT}-tokenized"
LEGACY_LINK_24H="${MEDS_DATA_DIR}/${DATA_VERSION_OUT}_first_24h-tokenized"
LINK_MAIN="${TARGET_MAIN}"
LINK_24H="${TARGET_24H}"

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

if [[ "${IRB_STAGE0_CREATE_TOKENIZED_LINKS}" == "true" ]]; then
  ensure_symlink "${TARGET_24H}" "${LEGACY_LINK_24H}"
  LINK_24H="${LEGACY_LINK_24H}"
fi

MAIN_OK=1
if [[ "${ONLY_24H_CUT}" == "true" ]]; then
  MAIN_OK=1
else
  if [[ "${IRB_STAGE0_CREATE_TOKENIZED_LINKS}" == "true" ]]; then
    ensure_symlink "${TARGET_MAIN}" "${LEGACY_LINK_MAIN}"
    LINK_MAIN="${LEGACY_LINK_MAIN}"
  fi
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

# -----------------------------------------------------------------------------
# Stage0E fast path (avoid re-tokenizing raw MEDS):
#
# If this Stage0 invocation is an evaluation-time retokenization (i.e., we are
# given a *training vocab* via --vocab_path and we are producing ONLY the 24h-cut
# dataset), we can cheaply derive the eval dataset from an existing 24h-cut
# tokenized dataset that already contains full variable-length `tokens`:
#   - Drop fully-padded fixed-length columns (e.g., `padded`, `padded_*`)
#   - Optionally truncate variable-length sequences to max_padded_len and append TRUNC
#   - Reuse vocab.gzip + numeric_stats.json from the training condition
#   - Reuse outcomes parquet (same cohort/labels) while dropping padded columns
#
# This is dramatically faster than re-parsing raw MEDS events and is sufficient
# because Stage2 extraction dynamically pads per-batch when `padded` is absent.
# -----------------------------------------------------------------------------
if [[ "${ONLY_24H_CUT}" == "true" && -n "${VOCAB_PATH}" && -n "${MAX_PADDED_LEN}" ]]; then
  VOCAB_DIR="$(dirname "${VOCAB_PATH}")"                # .../<dv>-tokenized/train
  TOKENIZED_DIR="$(dirname "${VOCAB_DIR}")"             # .../<dv>-tokenized
  DV_TRAIN="$(basename "${TOKENIZED_DIR}")"
  DV_TRAIN="${DV_TRAIN%-tokenized}"
  SRC_24H="$(dirname "${TOKENIZED_DIR}")/${DV_TRAIN}_first_24h-tokenized"

  # Only proceed if the source 24h-cut dataset exists and looks complete.
  SRC_OK=1
  for s in train val test; do
    if [[ ! -f "${SRC_24H}/${s}/tokens_timelines.parquet" ]]; then
      SRC_OK=0
      break
    fi
  done
  if [[ ! -f "${SRC_24H}/train/vocab.gzip" || ! -f "${SRC_24H}/train/numeric_stats.json" ]]; then
    SRC_OK=0
  fi

  if [[ "${SRC_OK}" -eq 1 && "${H24_OK}" -ne 1 ]]; then
    echo "[stage0] Stage0E fast-path: deriving eval tokenization from existing 24h-cut tokens."
    echo "  train dv: ${DV_TRAIN}"
    echo "  src_24h:  ${SRC_24H}"
    echo "  dst_24h:  ${LINK_24H}"
    echo "  max_len:  ${MAX_PADDED_LEN}"

    # Ensure split directories exist in the destination.
    mkdir -p "${LINK_24H}/train" "${LINK_24H}/val" "${LINK_24H}/test"

    # Reuse vocab + numeric stats exactly (token IDs must match the pretrained model).
    ln -sf "${SRC_24H}/train/vocab.gzip" "${LINK_24H}/train/vocab.gzip"
    ln -sf "${SRC_24H}/train/numeric_stats.json" "${LINK_24H}/train/numeric_stats.json"

    export SRC_24H
    export DST_24H="${LINK_24H}"
    export MAX_PADDED_LEN

    python - <<'PY'
import os
import polars as pl

IRB_HOME = os.environ["IRB_HOME"]
FMS_EHRS_HOME = os.environ["FMS_EHRS_HOME"]
SRC_24H = os.environ["SRC_24H"]
DST_24H = os.environ["DST_24H"]
MAX_LEN = int(os.environ["MAX_PADDED_LEN"])

import sys
sys.path.insert(0, FMS_EHRS_HOME)
from fms_ehrs.framework.vocabulary import Vocabulary

vocab = Vocabulary().load(os.path.join(SRC_24H, "train", "vocab.gzip"))
TRUNC = int(vocab("TRUNC"))

def _transform(in_fp: str, out_fp: str):
    lf = pl.scan_parquet(in_fp)
    cols = set(lf.collect_schema().names())
    drop_cols = [c for c in cols if c.startswith("padded")]
    if drop_cols:
        lf = lf.drop(drop_cols)

    # Truncate variable-length lists and append TRUNC to tokens (and null to aligned arrays).
    lf = lf.with_columns(seq_len=pl.col("tokens").list.len())
    lf = lf.with_columns(
        tokens=pl.when(pl.col("seq_len") > MAX_LEN).then(
            pl.concat_list(pl.col("tokens").list.slice(0, MAX_LEN - 1), pl.lit(TRUNC))
        ).otherwise(pl.col("tokens"))
    )
    if "times" in cols:
        lf = lf.with_columns(
            times=pl.when(pl.col("seq_len") > MAX_LEN).then(
                pl.concat_list(
                    pl.col("times").list.slice(0, MAX_LEN - 1),
                    pl.lit(None).cast(pl.Datetime(time_unit="ms")),
                )
            ).otherwise(pl.col("times"))
        )
    if "numeric_values" in cols:
        lf = lf.with_columns(
            numeric_values=pl.when(pl.col("seq_len") > MAX_LEN).then(
                pl.concat_list(
                    pl.col("numeric_values").list.slice(0, MAX_LEN - 1),
                    pl.lit(None).cast(pl.Float32),
                )
            ).otherwise(pl.col("numeric_values"))
        )
    lf = lf.drop("seq_len")
    lf.collect().write_parquet(out_fp)

for split in ("train", "val", "test"):
    os.makedirs(os.path.join(DST_24H, split), exist_ok=True)
    _transform(
        os.path.join(SRC_24H, split, "tokens_timelines.parquet"),
        os.path.join(DST_24H, split, "tokens_timelines.parquet"),
    )

    src_out = os.path.join(SRC_24H, split, "tokens_timelines_outcomes.parquet")
    if os.path.exists(src_out):
        _transform(
            src_out,
            os.path.join(DST_24H, split, "tokens_timelines_outcomes.parquet"),
        )
PY

    # Mark H24_OK so downstream branches skip the expensive raw tokenization.
    H24_OK=1
  fi
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
  if [[ -n "${NUMERIC_ENCODING}" ]]; then
    tokenize_args+=( --numeric_encoding "${NUMERIC_ENCODING}" )
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
  # required for method-faithful xVal scaling (numeric_stats.json).
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
    --splits train,val,test \
    --strict
fi

for s in train val test; do
  require_file "${LINK_24H}/${s}/tokens_timelines_outcomes.parquet" "base outcomes parquet"
done

echo "[stage0] Done: ${DATA_VERSION_OUT}"
