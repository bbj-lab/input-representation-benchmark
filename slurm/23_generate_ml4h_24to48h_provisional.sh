#!/bin/bash
#SBATCH --job-name=irb-ml4h-24to48-figures
#SBATCH --output=./slurm/output/%A-%x.stdout
#SBATCH --partition=tier1q
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

# Render and validate only ML4H-DUMP/08_ML4H_provisional.

set -euo pipefail

find_repo_root() {
  local d="$1"
  while [[ "$d" != "/" ]]; do
    if [[ -f "$d/pipeline/run_experiments.py" && -d "$d/slurm" ]]; then
      echo "$d"
      return 0
    fi
    d="$(dirname "$d")"
  done
  return 1
}

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
IRB_HOME="$(find_repo_root "${SUBMIT_DIR}")"
source "${IRB_HOME}/slurm/00_preamble.sh"
cd "${IRB_HOME}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_MAX_THREADS="${SLURM_CPUS_PER_TASK}"

OUTPUT_ROOT="${IRB_HOME}/../Archive/ml4h-dump/08_ML4H_provisional"
SNAPSHOT_PATH="${1:-outputs/runs/ml4h_2026/provisional_24to48h/07_ml4h_ready_before.json}"
PRIMARY_ROOT="${IRB_HOME}/../Archive/ml4h-dump/07_ML4H_ready"

python - "${PRIMARY_ROOT}" "${SNAPSHOT_PATH}" <<'PY'
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
snapshot_path = Path(sys.argv[2])
if not root.is_dir():
    raise SystemExit(f"Primary ML4H figure directory does not exist: {root}")
snapshot_path.parent.mkdir(parents=True, exist_ok=True)
hashes = {}
for path in sorted((item for item in root.rglob("*") if item.is_file()), key=str):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    hashes[str(path.relative_to(root))] = digest.hexdigest()
snapshot_path.write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n")
print(f"Snapshotted {len(hashes)} primary figure files.")
PY

python paper/scripts/generate_ml4h_24to48h_provisional.py \
  --output-root "${OUTPUT_ROOT}"

python utilities/scripts/validate_ml4h_24to48h_bundle.py \
  --output-root "${OUTPUT_ROOT}" \
  --primary-snapshot "${SNAPSHOT_PATH}"
