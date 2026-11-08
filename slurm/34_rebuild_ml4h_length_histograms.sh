#!/bin/bash
#SBATCH --job-name=irb-ml4h-length-hists
#SBATCH --output=/gpfs/data/bbj-lab/users/daniel/input-representation-benchmark/outputs/runs/ml4h_2026/affine_repair/logs/%A-%x.out
#SBATCH --partition=tier2q
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

# Rebuild the two ML4H token-length histogram figures with a title that
# does not mention the parquet column name.

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
TOKEN_BASE="${IRB_HOME}/outputs/runs/tokenized/mimiciv-3.1_meds_70-10-20"
PLOT_SCRIPT="${IRB_HOME}/utilities/qc/plot_length_histograms.py"
TITLE="Token-length distributions"

python "${PLOT_SCRIPT}" \
  --base "${TOKEN_BASE}" \
  --versions \
    deciles_none_unfused_time_tokens \
    deciles_none_fused_time_tokens \
    deciles_none_unfused_time_tokens_first_24h \
    deciles_none_fused_time_tokens_first_24h \
  --labels \
    'unfused: full timeline' \
    'fused: full timeline' \
    'unfused: first 24h' \
    'fused: first 24h' \
  --split train \
  --title "${TITLE}" \
  --out_pdf "${PAPER_DIR}/figures/length_histograms_train.pdf"

python "${PLOT_SCRIPT}" \
  --base "${TOKEN_BASE}" \
  --versions \
    deciles_none_unfused_time_tokens \
    deciles_none_unfused_time_rope \
    deciles_none_unfused_time_tokens_first_24h \
    deciles_none_unfused_time_rope_first_24h \
  --labels \
    'explicit time tokens: full timeline' \
    'no time tokens: full timeline\n(event order / admission-relative RoPE)' \
    'explicit time tokens: first 24h' \
    'no time tokens: first 24h\n(event order / admission-relative RoPE)' \
  --split train \
  --title "${TITLE}" \
  --out_pdf "${PAPER_DIR}/figures/length_histograms_temporal_train.pdf"

python - "${PAPER_DIR}/figures" "${TITLE}" <<'PY'
from pathlib import Path
import subprocess
import sys

figures_dir = Path(sys.argv[1])
expected_title = sys.argv[2]
for name in (
    "length_histograms_train.pdf",
    "length_histograms_temporal_train.pdf",
):
    path = figures_dir / name
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"ERROR: missing rebuilt figure: {path}")
    extracted = subprocess.check_output(
        ["pdftotext", "-layout", str(path), "-"],
        text=True,
        errors="replace",
    )
    if "(tokens column)" in extracted:
        raise SystemExit(f"ERROR: leftover title phrase in {name}")
    if "\u2014" in extracted:
        raise SystemExit(f"ERROR: em-dash found in {name}")
    if expected_title not in extracted:
        raise SystemExit(f"ERROR: expected title missing from {name}")
    print(f"verified title in {name}")
PY

echo "length_histograms_rebuild_complete"
