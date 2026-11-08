#!/bin/bash
# =============================================================================
# ML4H batch-geometry calibration (CPU diagnostics + GPU probes).
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IRB_HOME="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${IRB_HOME}/slurm/00_preamble.sh"
cd "${IRB_HOME}"

CALIB_DIR="${IRB_HOME}/outputs/runs/ml4h_2026/calibration"
mkdir -p "${CALIB_DIR}"
PYTHON="${IRB_PYTHON:-/gpfs/data/bbj-lab/.envs/input-rep/bin/python}"

# Drop stale mapped-cache locks left by cancelled jobs (no matching _SUCCESS).
if [[ -n "${IRB_MAPPED_DATASET_CACHE_DIR:-}" && -d "${IRB_MAPPED_DATASET_CACHE_DIR}" ]]; then
  for lock in "${IRB_MAPPED_DATASET_CACHE_DIR}"/*.lock; do
    [[ -e "${lock}" ]] || continue
    base="${lock%.lock}"
    if [[ ! -f "${base}/_SUCCESS" ]]; then
      echo "[calib] Removing stale mapped-cache lock: ${lock}"
      rm -f "${lock}"
      rm -rf "${base}"
    fi
  done
fi

if [[ "${IRB_CALIB_SKIP_DIAGNOSTICS:-0}" == "1" ]]; then
  echo "=== Reusing existing CPU diagnostics ==="
else
  echo "=== Window exposure diagnostics ==="
  "${PYTHON}" utilities/scripts/diagnose_ml4h_window_exposure.py \
    --output "${CALIB_DIR}/window_exposure.json"

  echo "=== Representation one-batch checks ==="
  "${PYTHON}" utilities/scripts/check_ml4h_representation_batches.py \
    --batch-size "${IRB_CALIB_CHECK_BATCH:-2}" \
    --output "${CALIB_DIR}/representation_batch_checks.json"
fi

echo "=== GPU batch-geometry probes ==="
# shellcheck disable=SC2206
CALIB_BATCH_SIZES=(${IRB_CALIB_BATCH_SIZES:-1 2 4 8})
CALIB_MODE_ARGS=()
if [[ "${IRB_CALIB_CLIP_ONLY:-0}" == "1" ]]; then
  CALIB_MODE_ARGS+=(--clip-only)
  # shellcheck disable=SC2206
  CALIB_CLIP_VALUES=(${IRB_CALIB_CLIP_VALUES:-1.0 32.0})
  CALIB_MODE_ARGS+=(--clip-values "${CALIB_CLIP_VALUES[@]}")
  [[ -n "${IRB_CALIB_CLIP_BATCH:-}" ]] && CALIB_MODE_ARGS+=(--clip-batch-size "${IRB_CALIB_CLIP_BATCH}")
fi
"${PYTHON}" utilities/scripts/calibrate_ml4h_batch_geometry.py \
  --output-dir "${CALIB_DIR}" \
  --probe-steps "${IRB_CALIB_PROBE_STEPS:-20}" \
  --clip-probe-steps "${IRB_CALIB_CLIP_STEPS:-40}" \
  --batch-sizes "${CALIB_BATCH_SIZES[@]}" \
  "${CALIB_MODE_ARGS[@]+"${CALIB_MODE_ARGS[@]}"}" \
  --window-diagnostics "${CALIB_DIR}/window_exposure.json" \
  --batch-checks "${CALIB_DIR}/representation_batch_checks.json"

echo "=== Calibration complete ==="
"${PYTHON}" - <<'PY'
import json
from pathlib import Path
report = json.loads(Path("outputs/runs/ml4h_2026/calibration/batch_geometry_report.json").read_text())
print(json.dumps({"gate_passed": report["gate_passed"], "selected": report["selected"], "gate_errors": report["gate_errors"]}, indent=2))
raise SystemExit(0 if report["gate_passed"] else 2)
PY
