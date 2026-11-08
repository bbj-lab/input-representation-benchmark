#!/bin/bash
#SBATCH --job-name=irb-ml4h-spiders
#SBATCH --output=./slurm/output/%A-%x.stdout
#SBATCH --partition=tier1q
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

# Render three baseline-delta spider figures (24–48h; Exp1/2 split by
# condition, Exp3 by architecture) and their summary tables into
# ML4H-DUMP/08_ML4H_provisional/04_spider_charts/.

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

python paper/scripts/generate_ml4h_spider_charts.py \
  --output-root "${OUTPUT_ROOT}"

python - "${OUTPUT_ROOT}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

output_root = Path(sys.argv[1])
section = output_root / "04_spider_charts"
manifest = json.loads((section / "manifest.json").read_text(encoding="utf-8"))
expected = ("exp1_factor_spiders", "exp2_factor_spiders", "exp3_factor_spiders")
expected_tables = (
    "exp1_factor_summary_table",
    "exp2_factor_summary_table",
    "exp3_factor_summary_table",
)
stems = tuple(figure["stem"] for figure in manifest["figures"])
if stems != expected:
    raise SystemExit(f"Unexpected stems: {stems}")
if manifest["figure_count"] != 3:
    raise SystemExit(f"Expected 3 figures, found {manifest['figure_count']}")
for figure in manifest["figures"]:
    for key in ("pdf", "png", "source", "metadata"):
        path = output_root / figure[key]
        if not path.is_file() or path.stat().st_size == 0:
            raise SystemExit(f"Missing or empty {key}: {path}")
    for key in (
        "summary_table_pdf",
        "summary_table_png",
        "summary_table_source",
    ):
        path = output_root / figure[key]
        if not path.is_file() or path.stat().st_size == 0:
            raise SystemExit(f"Missing or empty {key}: {path}")
allowed_prefixes = expected + expected_tables
extras = sorted(
    path.name
    for path in section.iterdir()
    if path.is_file()
    and path.name not in {"figure_index.csv", "manifest.json"}
    and not any(path.name.startswith(stem) for stem in allowed_prefixes)
)
if extras:
    raise SystemExit(f"Unexpected files in spider section: {extras}")
print("Verified 3 baseline-delta spider figures and 3 summary tables.")
PY
