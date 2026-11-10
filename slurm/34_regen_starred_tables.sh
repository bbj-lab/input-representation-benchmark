#!/bin/bash
#SBATCH --job-name=irb-ml4h-star-tables
#SBATCH --output=./outputs/runs/ml4h_2026/model_level_tests/logs/%j-star-tables.out
#SBATCH --partition=tier2q
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00

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

PAPER_DIR="${IRB_HOME}/../ML4H2026/ML4H"
python paper/scripts/generate_ml4h_appendix_result_tables.py \
  --metrics-long "${IRB_HOME}/outputs/runs/ml4h_2026/metrics/metrics_long.csv" \
  --intervals "${IRB_HOME}/outputs/runs/ml4h_2026/model_level_tests/family_tests.csv" \
  --output-dir "${PAPER_DIR}/generated"

python - <<'PY'
from pathlib import Path
root = Path("/gpfs/data/bbj-lab/users/daniel/ML4H2026/ML4H/generated")
for name in (
    "main_exp1_quantization_table.tex",
    "main_exp1_fusion_table.tex",
    "main_exp2_numeric_table.tex",
    "main_exp2_temporal_table.tex",
    "main_exp3_family_table.tex",
):
    text = (root / name).read_text(encoding="utf-8")
    print(name, "stars", text.count(r"\textsuperscript{\textbf{\large *}}"))
PY
