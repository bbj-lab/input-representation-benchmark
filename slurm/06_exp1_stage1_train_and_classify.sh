#!/bin/bash
# =============================================================================
# Stage 1 (GPU): Exp1 training + classification (assumes Stage 0 completed)
# =============================================================================
#
# This job assumes that, for the given config:
#   - <data_version>-tokenized/{train,val}/tokens_timelines.parquet exists
#   - <data_version>_first_24h-tokenized/{train,val}/tokens_timelines_outcomes.parquet exists
#
# It then runs:
#   - Exp1 pretraining (tune_model.py; Optuna-based by default)
#   - 4 downstream classifiers (fine_tune_classification.py)
#
# Important: tokenization/outcome extraction is NOT performed here to avoid
# write races across seeds and to eliminate redundant CPU-heavy work.
# =============================================================================

set -euo pipefail

usage() {
  echo "Usage: $0 --config_id <id> --data_version <data_version_out> --seed <int> [--n_trials <int>]"
}

CONFIG_ID=""
DATA_VERSION=""
SEED=""
N_TRIALS="${IRB_EXP1_OPTUNA_TRIALS:-5}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config_id) CONFIG_ID="$2"; shift 2 ;;
    --data_version) DATA_VERSION="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --n_trials) N_TRIALS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: Unknown arg: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "${CONFIG_ID}" || -z "${DATA_VERSION}" || -z "${SEED}" ]]; then
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

# Require stage0 outputs
test -f "${MEDS_DATA_DIR}/${DATA_VERSION}-tokenized/train/tokens_timelines.parquet"
test -f "${MEDS_DATA_DIR}/${DATA_VERSION}-tokenized/train/vocab.gzip"
test -f "${MEDS_DATA_DIR}/${DATA_VERSION}-tokenized/val/tokens_timelines.parquet"
test -f "${MEDS_DATA_DIR}/${DATA_VERSION}_first_24h-tokenized/train/tokens_timelines_outcomes.parquet"
test -f "${MEDS_DATA_DIR}/${DATA_VERSION}_first_24h-tokenized/val/tokens_timelines_outcomes.parquet"

# Deterministic naming to avoid glob/ambiguity across reruns.
# NOTE: tune_model.py does not currently accept an explicit RNG seed; this `jid`
# is for naming/logging only. (Exp2/Exp3 training scripts do accept --seed.)
MODEL_VERSION="exp1_${CONFIG_ID}"
JID="s${SEED}"

python "${FMS_EHRS_HOME}/fms_ehrs/scripts/tune_model.py" \
  --data_dir "${MEDS_DATA_DIR}" \
  --data_version "${DATA_VERSION}" \
  --model_dir "${MODEL_DIR}" \
  --model_version "${MODEL_VERSION}" \
  --jid "${JID}" \
  --collation packed \
  --n_epochs 5 \
  --n_trials "${N_TRIALS}" \
  --wandb_project input-rep-benchmark-exp1

BEST_MODEL_DIR="${MODEL_DIR}/${MODEL_VERSION}-${JID}-hp-${DATA_VERSION}"
test -d "${BEST_MODEL_DIR}"

for outcome in same_admission_death long_length_of_stay icu_admission imv_event; do
  python "${FMS_EHRS_HOME}/fms_ehrs/scripts/fine_tune_classification.py" \
    --model_loc "${BEST_MODEL_DIR}" \
    --data_dir "${MEDS_DATA_DIR}" \
    --data_version "${DATA_VERSION}_first_24h" \
    --out_dir "${MODEL_DIR}/classifiers" \
    --outcome "${outcome}" \
    --n_epochs 5 \
    --wandb_project input-rep-benchmark-exp1-classify
done

echo "[stage1] Done: ${CONFIG_ID} seed=${SEED}"

