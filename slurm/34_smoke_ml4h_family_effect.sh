#!/bin/bash
#SBATCH --job-name=irb-ml4h-family-smoke
#SBATCH --output=./outputs/runs/ml4h_2026/family_effect_bootstrap/logs/%j-smoke.out
#SBATCH --partition=tier2q
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00

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

python - <<'PY'
import sys

sys.path.insert(0, "paper/scripts")
import pandas as pd
import ml4h_main_table_contrasts as contrasts

metrics = pd.read_parquet("outputs/runs/ml4h_2026/metrics/metrics_long.parquet")
columns = contrasts.column_specs()
if len(columns) != contrasts.COLUMN_COUNT:
    raise SystemExit(f"expected {contrasts.COLUMN_COUNT} columns, found {len(columns)}")
leaves = contrasts.leaf_specs()
if len(leaves) != contrasts.LEAF_COUNT:
    raise SystemExit(f"expected {contrasts.LEAF_COUNT} leaves, found {len(leaves)}")
leaf0 = contrasts.unrounded_table_delta(
    metrics,
    experiment=str(leaves[0]["experiment"]),
    condition_key=str(leaves[0]["condition"]),
    level_label=str(leaves[0]["level"]),
    task_type=str(leaves[0]["task_type"]),
    family=str(leaves[0]["family"]),
)
print("n_columns", len(columns))
print("n_leaves", len(leaves))
print("leaf0", leaf0)
PY

python pipeline/scripts/bootstrap_ml4h_family_effects.py \
  --mode run \
  --task-index 0 \
  --metrics outputs/runs/ml4h_2026/metrics/metrics_long.parquet \
  --work-root outputs/runs/ml4h_2026/family_effect_bootstrap_smoke \
  --output outputs/runs/ml4h_2026/family_effect_bootstrap_smoke/family_effect_intervals.parquet \
  --bootstrap-n 2 \
  --bootstrap-seed 42 \
  --workers "${SLURM_CPUS_PER_TASK:-4}"
