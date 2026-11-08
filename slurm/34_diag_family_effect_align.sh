#!/bin/bash
#SBATCH --job-name=irb-ml4h-family-diag
#SBATCH --output=./outputs/runs/ml4h_2026/family_effect_bootstrap/logs/%j-diag.out
#SBATCH --partition=tier2q
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00

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
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "paper/scripts")
import ml4h_main_table_contrasts as contrasts

metrics = pd.read_parquet("outputs/runs/ml4h_2026/metrics/metrics_long.parquet")
task = contrasts.column_specs()[0]
condition = contrasts.find_condition(str(task["experiment"]), str(task["condition"]))
level = contrasts.find_level(condition, str(task["level"]))
print("task", task)

def artifacts(filters):
    rows = metrics.loc[
        (metrics["experiment"] == task["experiment"])
        & (metrics["task_type"] == "binary")
        & (metrics["metric"] == "roc_auc")
    ].copy()
    rows = contrasts.filter_rows(rows, filters)
    return rows[["config_id", "backbone", "seed", "source_artifact"]].drop_duplicates()

base = artifacts(contrasts.level_baseline_filters(condition, level))
comp = artifacts(level.filters)
print("n_base", len(base), "n_comp", len(comp))
print("base_configs", sorted(base["config_id"].unique()))
print("comp_configs", sorted(comp["config_id"].unique()))

outcome = "same_admission_death"
payloads = []
for name, table in (("baseline", base), ("level", comp)):
    row = table.loc[(table["backbone"] == "llama") & (table["seed"] == 42)].iloc[0]
    with Path(row["source_artifact"]).open("rb") as handle:
        payload = pickle.load(handle)
    print(name, "keys", sorted(payload.keys()))
    meta = payload.get("metadata", {})
    if isinstance(meta, dict):
        print(name, "meta_keys", sorted(meta.keys())[:40])
    labels = np.asarray(payload["labels"][outcome])
    preds = np.asarray(payload["predictions"][outcome])
    quals = np.asarray(payload["qualifiers"][outcome])
    print(
        name,
        "label_shape",
        labels.shape,
        "pred_shape",
        preds.shape,
        "qual_shape",
        quals.shape,
        "qual_sum",
        int(quals.sum()),
        "label_nan",
        int(np.isnan(labels.astype(float)).sum()) if np.issubdtype(labels.dtype, np.number) else "nonnumeric",
        "label_dtype",
        labels.dtype,
    )
    payloads.append((name, labels, preds, quals, payload))

(_, a, _, _, pa), (_, b, _, _, pb) = payloads
print("shape_equal", a.shape == b.shape)
if a.shape == b.shape:
    print("allclose", np.allclose(a.astype(float), b.astype(float), equal_nan=True))
    diff = np.abs(a.astype(float) - b.astype(float))
    print("max_abs_diff", np.nanmax(diff) if diff.size else None)
    print("n_diff", int(np.sum(~np.isclose(a.astype(float), b.astype(float), equal_nan=True))))
    print("n_nan_a", int(np.isnan(a.astype(float)).sum()), "n_nan_b", int(np.isnan(b.astype(float)).sum()))
id_keys = [k for k in pa.keys() if any(s in k.lower() for s in ("hadm", "subject", "id", "index", "row"))]
print("id_like_keys", id_keys)
PY
