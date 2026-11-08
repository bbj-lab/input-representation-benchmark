#!/bin/bash
#SBATCH --job-name=irb-ml4h-patient-test
#SBATCH --output=./outputs/runs/ml4h_2026/patient_cluster_bootstrap/logs/%j-%x.out
#SBATCH --partition=tier1q,tier2q,tier3q
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
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

python -m pytest -q \
  pipeline/tests/unit/test_bootstrap_ml4h_patient_clusters.py \
  pipeline/tests/unit/test_bootstrap_ml4h_family_effects.py
