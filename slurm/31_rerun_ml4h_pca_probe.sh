#!/bin/bash
#SBATCH --job-name=irb-ml4h-pca-probe
#SBATCH --output=/gpfs/data/bbj-lab/users/daniel/input-representation-benchmark/outputs/runs/ml4h_2026/affine_repair/logs/%A-%x.out
#SBATCH --partition=tier2q
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

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

METRICS="${IRB_HOME}/outputs/runs/ml4h_2026/metrics"
PAPER_FIG="${IRB_HOME}/../ML4H2026/ML4H/figures"
LIVE_PROBE="${IRB_HOME}/outputs/diagnostics_ml4h_clinical_boundary_probe"
STALE_PROBE="${IRB_HOME}/outputs/diagnostics_clinical_boundary_probe"

rm -f \
  "${METRICS}/pca_centile_geometry_grid_source.csv" \
  "${METRICS}/clinical_boundary_probe_results.json" \
  "${METRICS}/pca_probe_best_checkpoint_summary.json" \
  "${PAPER_FIG}/pca_centile_geometry_grid.pdf" \
  "${PAPER_FIG}/pca_centile_geometry_grid.png" \
  "${PAPER_FIG}/clinical_boundary_probe.png" \
  "${PAPER_FIG}/clinical_boundary_probe.pdf" \
  "${PAPER_FIG}/sources/pca_centile_geometry_grid_source.csv"
rm -rf "${LIVE_PROBE}" "${STALE_PROBE}"

python pipeline/scripts/diagnostics/rebuild_ml4h_pca_and_probe.py

python - <<'PY'
import json
from pathlib import Path
p = Path("/gpfs/data/bbj-lab/users/daniel/input-representation-benchmark/outputs/runs/ml4h_2026/metrics/pca_probe_best_checkpoint_summary.json")
print(p.read_text())
PY
