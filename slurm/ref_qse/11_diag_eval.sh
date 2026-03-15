#!/bin/bash

# =============================================================================
# Diagnostic evaluation wrapper: MLP probe / Ridge regression on extracted reps
# =============================================================================
#
# This is the diagnostic counterpart of ref_qse/10_xfer_rep_based_preds.sh.
# It supports additional classifiers (mlp, ridge_regression) and task types
# (classification, regression) for the Exp2 diagnostic analyses.
#
# Required env vars (set by the jobfile):
#   data_dir       - tokenized data root (e.g., benchmarks/.../data/meds/data)
#   data_version   - data version string (should end with _first_24h)
#   model_loc      - HF model directory
#
# Optional env vars:
#   DIAG_CLASSIFIER     - classifier name (default: logistic_regression)
#   DIAG_TASK_TYPE       - classification or regression (default: classification)
#   DIAG_OUTCOMES        - space-separated outcome list (default: script default)
#   DIAG_OUTCOMES_PARQUET - custom parquet filename (default: tokens_timelines_outcomes.parquet)
#   DIAG_MLP_HIDDEN      - MLP hidden sizes (default: 256)
#   DIAG_EXTRA_ARGS      - any extra arguments
#
# Usage:
#   # MLP probe (classification)
#   export DIAG_CLASSIFIER=mlp data_dir=... data_version=... model_loc=... && bash slurm/ref_qse/11_diag_eval.sh
#
#   # Ridge regression on LOS
#   export DIAG_CLASSIFIER=ridge_regression DIAG_TASK_TYPE=regression \
#          DIAG_OUTCOMES="length_of_stay icu_length_of_stay" \
#          data_dir=... data_version=... model_loc=... && bash slurm/ref_qse/11_diag_eval.sh
#
#   # Ridge regression on extended lab outcomes
#   export DIAG_CLASSIFIER=ridge_regression DIAG_TASK_TYPE=regression \
#          DIAG_OUTCOMES="peak_creatinine peak_troponin min_hemoglobin peak_potassium min_glucose peak_bnp time_to_icu_hours" \
#          DIAG_OUTCOMES_PARQUET=tokens_timelines_extended_outcomes.parquet \
#          data_dir=... data_version=... model_loc=... && bash slurm/ref_qse/11_diag_eval.sh
# =============================================================================

set -euo pipefail

source slurm/ref_qse/preamble.sh

: "${FMS_EHRS_HOME:?Set FMS_EHRS_HOME (sibling fms-ehrs repo)}"

# Expect these from the jobfile command
: "${data_dir:?Set data_dir (tokenized data root)}"
: "${data_version:?Set data_version (should end with _first_24h)}"
: "${model_loc:?Set model_loc (HF model directory)}"

DIAG_CLASSIFIER="${DIAG_CLASSIFIER:-logistic_regression}"
DIAG_TASK_TYPE="${DIAG_TASK_TYPE:-classification}"
DIAG_OUTCOMES_PARQUET="${DIAG_OUTCOMES_PARQUET:-tokens_timelines_outcomes.parquet}"
DIAG_MLP_HIDDEN="${DIAG_MLP_HIDDEN:-256}"
PARQUET_STEM="$(basename "${DIAG_OUTCOMES_PARQUET}" .parquet)"
if [[ -n "${DIAG_OUTCOMES:-}" ]]; then
  OUTCOME_SIG="$(printf '%s' "${DIAG_OUTCOMES}" | sha1sum | cut -c1-12)"
else
  OUTCOME_SIG="default"
fi
DIAG_PREDS_TAG="${DIAG_PREDS_TAG:-diag-${DIAG_TASK_TYPE}-${DIAG_CLASSIFIER}-${PARQUET_STEM}-${OUTCOME_SIG}}"

# Build the command
CMD=(
  python3 "${FMS_EHRS_HOME}/fms_ehrs/scripts/transfer_rep_based_preds.py"
  --data_dir_orig "${data_dir}"
  --data_dir_new "${data_dir}"
  --data_version "${data_version}"
  --model_loc "${model_loc}"
  --classifier "${DIAG_CLASSIFIER}"
  --task_type "${DIAG_TASK_TYPE}"
  --outcomes_parquet "${DIAG_OUTCOMES_PARQUET}"
  --preds_tag "${DIAG_PREDS_TAG}"
  --save_preds
)

if [[ -n "${DIAG_OUTCOMES:-}" ]]; then
  # shellcheck disable=SC2206  # intentional word-splitting
  CMD+=(--outcomes ${DIAG_OUTCOMES})
fi

if [[ "${DIAG_CLASSIFIER}" == "mlp" ]]; then
  CMD+=(--mlp_hidden_sizes "${DIAG_MLP_HIDDEN}")
fi

# Append any extra args
if [[ -n "${DIAG_EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  CMD+=(${DIAG_EXTRA_ARGS})
fi

echo "Diagnostic evaluation:"
echo "  classifier:  ${DIAG_CLASSIFIER}"
echo "  task_type:   ${DIAG_TASK_TYPE}"
echo "  outcomes:    ${DIAG_OUTCOMES:-<default>}"
echo "  parquet:     ${DIAG_OUTCOMES_PARQUET}"
echo "  preds_tag:   ${DIAG_PREDS_TAG}"
echo "  model:       ${model_loc}"
echo ""
echo "  ${CMD[*]}"
echo ""

"${CMD[@]}"
