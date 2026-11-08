#!/bin/bash
#SBATCH --job-name=irb-ml4h-figures
#SBATCH --output=./slurm/output/%A-%x.stdout
#SBATCH --partition=tier1q
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

# Rebuild the complete ML4H figure bundle from the current, completed Stage-3
# artifacts. This is CPU-bound: no GPU is requested.

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

OUTPUT_ROOT="${IRB_HOME}/../Archive/ml4h-dump"

python paper/scripts/generate_ml4h_figures.py \
  --experiments all \
  --output-root "${OUTPUT_ROOT}"

python - "${OUTPUT_ROOT}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

output_root = Path(sys.argv[1])
manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
expected = {
    "exp1": (72, 11_232, 85_057),
    "exp2": (72, 11_232, 85_057),
    "exp3": (12, 1_824, 16_529),
}

if manifest["figure_count"] != 57 or len(manifest["figures"]) != 57:
    raise SystemExit(
        f"Expected 57 figures; manifest reports {manifest['figure_count']} / "
        f"{len(manifest['figures'])}."
    )
for figure in manifest["figures"]:
    for key in ("pdf", "png", "source", "metadata"):
        path = output_root / figure[key]
        if not path.is_file() or path.stat().st_size == 0:
            raise SystemExit(f"Missing or empty {key}: {path}")

coverage = pd.read_csv(output_root / "00_data" / "coverage.csv").set_index("experiment")
metrics = pd.read_parquet(output_root / "00_data" / "metrics_long.parquet")
for experiment, (runs, rows, test_rows) in expected.items():
    record = coverage.loc[experiment]
    if int(record["observed_runs"]) != runs:
        raise SystemExit(f"{experiment}: expected {runs} runs, found {record['observed_runs']}")
    if int(record["observed_metric_rows"]) != rows:
        raise SystemExit(
            f"{experiment}: expected {rows} metrics, found {record['observed_metric_rows']}"
        )
    values = metrics.loc[metrics["experiment"] == experiment, "test_cohort_rows"]
    if set(values.astype(int)) != {test_rows}:
        raise SystemExit(
            f"{experiment}: expected test cohort {test_rows}, found {sorted(values.unique())}"
        )
    if metrics.loc[metrics["experiment"] == experiment, "source_sha256"].isna().any():
        raise SystemExit(f"{experiment}: missing source artifact hash")

print("Verified 57 figure pairs and all three experiment coverage contracts.")
PY
