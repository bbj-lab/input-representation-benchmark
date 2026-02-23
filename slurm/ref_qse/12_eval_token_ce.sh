#!/bin/bash

# =============================================================================
# Token-level CE evaluation wrapper (GPU required)
# =============================================================================
#
# Runs eval_token_ce.py on a single model. Called by the per-model jobfile.
#
# Required env vars:
#   data_dir       - tokenized data root
#   data_version   - data version string
#   model_loc      - HF model directory
#
# Optional env vars:
#   CE_SPLITS      - comma-separated splits (default: test)
#   CE_BATCH_SZ    - batch size (default: 16)
#   CE_EXTRA_ARGS  - additional arguments
# =============================================================================

set -euo pipefail

source slurm/ref_qse/preamble.sh

: "${FMS_EHRS_HOME:?Set FMS_EHRS_HOME (sibling fms-ehrs repo)}"
: "${data_dir:?Set data_dir}"
: "${data_version:?Set data_version}"
: "${model_loc:?Set model_loc}"

CE_SPLITS="${CE_SPLITS:-test}"
CE_BATCH_SZ="${CE_BATCH_SZ:-16}"

CMD=(
  python3 "${FMS_EHRS_HOME}/fms_ehrs/scripts/eval_token_ce.py"
  --data_dir "${data_dir}"
  --data_version "${data_version}"
  --model_loc "${model_loc}"
  --splits "${CE_SPLITS}"
  --batch_sz "${CE_BATCH_SZ}"
)

if [[ -n "${CE_EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  CMD+=(${CE_EXTRA_ARGS})
fi

echo "Token-level CE evaluation:"
echo "  model: ${model_loc}"
echo "  data:  ${data_dir}/${data_version}"
echo "  splits: ${CE_SPLITS}"
echo ""

"${CMD[@]}"
