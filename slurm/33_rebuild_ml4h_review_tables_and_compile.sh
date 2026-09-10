#!/bin/bash
#SBATCH --job-name=irb-ml4h-review-compile
#SBATCH --output=/gpfs/data/bbj-lab/users/daniel/input-representation-benchmark/outputs/runs/ml4h_2026/affine_repair/logs/%A-%x.out
#SBATCH --partition=tier2q
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

# Rebuild result tables and the boundary-probe figure, then compile the
# ML4H manuscript.

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
METRICS="${IRB_HOME}/outputs/runs/ml4h_2026/metrics/metrics_long.csv"
INTERVALS="${IRB_HOME}/outputs/runs/ml4h_2026/family_effect_bootstrap/family_effect_intervals.parquet"
PROBE_JSON="${IRB_HOME}/outputs/runs/ml4h_2026/metrics/clinical_boundary_probe_results.json"

if [[ ! -s "${INTERVALS}" ]]; then
  echo "ERROR: missing family-effect intervals: ${INTERVALS}" >&2
  exit 2
fi

python paper/scripts/generate_ml4h_appendix_result_tables.py \
  --metrics-long "${METRICS}" \
  --intervals "${INTERVALS}" \
  --output-dir "${PAPER_DIR}/generated"

python paper/scripts/generate_ml4h_boundary_probe_plot.py \
  --results-json "${PROBE_JSON}" \
  --output "${PAPER_DIR}/figures/clinical_boundary_probe.png"

for path in \
    "${PAPER_DIR}/generated/main_exp1_quantization_table.tex" \
    "${PAPER_DIR}/generated/main_exp1_fusion_table.tex" \
    "${PAPER_DIR}/generated/main_exp2_numeric_table.tex" \
    "${PAPER_DIR}/generated/main_exp2_temporal_table.tex" \
    "${PAPER_DIR}/generated/main_exp3_family_table.tex" \
    "${PAPER_DIR}/figures/clinical_boundary_probe.png"; do
  if [[ ! -s "${path}" ]]; then
    echo "ERROR: missing rebuilt artifact: ${path}" >&2
    exit 2
  fi
done

module load gcc/12.1.0
module load texlive/2025

cd "${PAPER_DIR}"
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error main.tex

if [[ ! -s "${PAPER_DIR}/main.pdf" ]]; then
  echo "ERROR: expected nonempty manuscript PDF: ${PAPER_DIR}/main.pdf" >&2
  exit 2
fi

python - <<'PY'
from pathlib import Path
import subprocess

text = Path("main.log").read_text(encoding="utf-8", errors="replace")
for line in text.splitlines():
    if "Output written on" in line:
        print(line)
    if "Float too large" in line:
        print(line)

figures_dir = Path("figures")
for name in (
    "length_histograms_train.pdf",
    "length_histograms_temporal_train.pdf",
):
    extracted = subprocess.check_output(
        ["pdftotext", "-layout", str(figures_dir / name), "-"],
        text=True,
        errors="replace",
    )
    if "(tokens column)" in extracted:
        raise SystemExit(f"ERROR: leftover title phrase in {name}")
    if "Token-length distributions" not in extracted:
        raise SystemExit(f"ERROR: expected title missing from {name}")

manuscript = subprocess.check_output(
    ["pdftotext", "-layout", "main.pdf", "-"],
    text=True,
    errors="replace",
)
for phrase in (
    "(tokens column)",
    "before optimization begins",
    "figure-1",
    "Adm.-rel.",
    r"Mult. xVal",
):
    if phrase in manuscript:
        raise SystemExit(f"ERROR: leftover phrase in main.pdf: {phrase}")
if "Code-normalized xVal" not in manuscript:
    raise SystemExit("ERROR: expected Code-normalized xVal in main.pdf")
if "Benjamini" not in manuscript:
    raise SystemExit("ERROR: expected Benjamini-Hochberg caption in main.pdf")
if "permutation" not in manuscript:
    raise SystemExit("ERROR: expected permutation testing language in main.pdf")
if "hierarchical" in manuscript:
    raise SystemExit("ERROR: leftover hierarchical testing language in main.pdf")
if "unadjusted 95" in manuscript:
    raise SystemExit("ERROR: leftover unadjusted-interval starring language in main.pdf")
table_text = "".join(
    (Path("generated") / name).read_text(encoding="utf-8")
    for name in (
        "main_exp1_quantization_table.tex",
        "main_exp1_fusion_table.tex",
        "main_exp2_numeric_table.tex",
        "main_exp2_temporal_table.tex",
        "main_exp3_family_table.tex",
    )
)
if r"$^{*}$}" not in table_text:
    raise SystemExit("ERROR: expected at least one starred Δ in the main-text tables")

for line in text.splitlines():
    if "Overfull" in line and (
        "openreview.net" in line
        or "FlashAttention" in line
        or "MOTOR" in line
        or "proceedings.iclr.cc" in line
    ):
        raise SystemExit(f"ERROR: bibliography overflow: {line}")
print("verified histogram titles, table labels, and leftover-phrase checks")
PY

echo "review_tables_and_compile_complete"
