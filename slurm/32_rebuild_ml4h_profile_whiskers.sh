#!/bin/bash
#SBATCH --job-name=irb-ml4h-profile-whiskers
#SBATCH --output=/gpfs/data/bbj-lab/users/daniel/input-representation-benchmark/outputs/runs/ml4h_2026/affine_repair/logs/%A-%x.out
#SBATCH --partition=tier2q
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

# Rebuild family-mean profile plots with six-model whiskers and appendix tables.

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
LIVE_METRICS="${IRB_HOME}/outputs/runs/ml4h_2026/metrics/metrics_long.parquet"
LIVE_METRICS_CSV="${IRB_HOME}/outputs/runs/ml4h_2026/metrics/metrics_long.csv"
PAPER_FIGURES="${IRB_HOME}/../ML4H2026/ML4H/figures"
PAPER_GENERATED="${IRB_HOME}/../ML4H2026/ML4H/generated"
PRIOR_DIR="${SLURM_TMPDIR:-/tmp}/ml4h_profile_prior_$$"

if [[ ! -s "${LIVE_METRICS}" ]]; then
  echo "ERROR: missing metric table: ${LIVE_METRICS}" >&2
  exit 2
fi
if [[ ! -s "${LIVE_METRICS_CSV}" ]]; then
  echo "ERROR: missing metric CSV: ${LIVE_METRICS_CSV}" >&2
  exit 2
fi
mkdir -p "${PRIOR_DIR}"
for experiment in exp1 exp2 exp3; do
  src="${PAPER_FIGURES}/sources/${experiment}-trace-plot.csv"
  if [[ ! -s "${src}" ]]; then
    echo "ERROR: missing prior figure source: ${src}" >&2
    exit 2
  fi
  cp "${src}" "${PRIOR_DIR}/${experiment}-trace-plot.csv"
done

python -m py_compile \
  paper/scripts/generate_ml4h_outcome_family_traces.py \
  paper/scripts/generate_ml4h_outcome_factor_traces.py \
  paper/scripts/generate_ml4h_appendix_result_tables.py \
  paper/scripts/generate_ml4h_provisional_visual_alternatives.py

python paper/scripts/generate_ml4h_outcome_family_traces.py \
  --output-root "${OUTPUT_ROOT}" \
  --metrics-long "${LIVE_METRICS}" \
  --submission-figures-dir "${PAPER_FIGURES}"

python paper/scripts/generate_ml4h_appendix_result_tables.py \
  --metrics-long "${LIVE_METRICS_CSV}" \
  --intervals "${IRB_HOME}/outputs/runs/ml4h_2026/family_effect_bootstrap/family_effect_intervals.parquet" \
  --output-dir "${PAPER_GENERATED}"

python - "${PRIOR_DIR}" \
  "${LIVE_METRICS_CSV}" \
  "${IRB_HOME}/paper/scripts" \
  "${PAPER_FIGURES}/sources/exp1-trace-plot.csv" \
  "${PAPER_FIGURES}/sources/exp2-trace-plot.csv" \
  "${PAPER_FIGURES}/sources/exp3-trace-plot.csv" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

prior_dir = Path(sys.argv[1])
metrics_path = Path(sys.argv[2])
sys.path.insert(0, sys.argv[3])
from generate_ml4h_appendix_result_tables import performance_rows, six_model_family

keys = ["experiment", "condition", "level_id", "task_type", "outcome_family"]
metrics = performance_rows(pd.read_csv(metrics_path))
for path in [Path(item) for item in sys.argv[4:]]:
    table = pd.read_csv(path)
    prior = pd.read_csv(prior_dir / path.name)
    if table["point"].isna().any():
        raise SystemExit(f"{path}: missing family-mean points")
    if table["lo"].isna().any() or table["hi"].isna().any():
        raise SystemExit(f"{path}: missing six-model whiskers")
    if not (table["lo"] <= table["point"] + 1e-12).all():
        raise SystemExit(f"{path}: lo exceeds point")
    if not (table["hi"] + 1e-12 >= table["point"]).all():
        raise SystemExit(f"{path}: hi below point")
    if not (table["n_models"] == 6).all():
        raise SystemExit(f"{path}: n_models is not 6")
    merged = prior.merge(
        table[keys + ["point", "lo", "hi"]],
        on=keys,
        suffixes=("_old", "_new"),
    )
    if merged.empty:
        raise SystemExit(f"{path.name}: point-check merge is empty")
    if (merged["point_old"] - merged["point_new"]).abs().max() > 1e-12:
        raise SystemExit(f"{path.name}: family-mean points changed")
    span_old = merged["hi_old"] - merged["lo_old"]
    span_new = merged["hi_new"] - merged["lo_new"]
    prior_already_six_model = (
        "n_models" in prior.columns and (prior["n_models"] == 6).all()
    )
    if not prior_already_six_model:
        if not (span_new <= span_old + 1e-12).all():
            raise SystemExit(
                f"{path.name}: six-model range is wider than member-outcome range"
            )
        if not (span_new < span_old - 1e-12).any():
            raise SystemExit(
                f"{path.name}: lo/hi still look like member-outcome ranges"
            )

    experiment = str(table["experiment"].iloc[0])
    family = six_model_family(metrics.loc[metrics["experiment"] == experiment])
    meta = (
        metrics.loc[metrics["experiment"] == experiment]
        .drop_duplicates("config_id")[
            [
                "config_id",
                "quantizer",
                "anchoring",
                "fusion",
                "representation",
                "temporal",
                "config_label",
            ]
        ]
    )
    family = family.merge(meta, on="config_id", how="left")
    trained = table.loc[table["condition"] != "temporal"]
    for row in trained.itertuples(index=False):
        if experiment == "exp1":
            quantizer, anchoring, fusion = str(row.level_id).split("|")
            match = family.loc[
                (family["quantizer"] == quantizer)
                & (family["anchoring"] == anchoring)
                & (family["fusion"] == fusion)
                & (family["task_type"] == row.task_type)
                & (family["outcome_family"] == row.outcome_family)
            ]
        elif experiment == "exp2":
            representation, temporal = str(row.level_id).split("|")
            match = family.loc[
                (family["representation"] == representation)
                & (family["temporal"] == temporal)
                & (family["task_type"] == row.task_type)
                & (family["outcome_family"] == row.outcome_family)
            ]
        else:
            match = family.loc[
                (family["config_label"] == row.level_id)
                & (family["task_type"] == row.task_type)
                & (family["outcome_family"] == row.outcome_family)
            ]
        if len(match) != 1:
            raise SystemExit(
                f"{path.name} {row.condition} {row.level_id} "
                f"{row.outcome_family}: expected one table cell, found {len(match)}"
            )
        cell = match.iloc[0]
        if abs(float(cell["mean"]) - float(row.point)) > 1e-12:
            raise SystemExit(
                f"{path.name} {row.level_id} {row.outcome_family}: "
                "table family mean differs from figure point"
            )
        if abs(float(cell["lo"]) - float(row.lo)) > 1e-12:
            raise SystemExit(
                f"{path.name} {row.level_id} {row.outcome_family}: "
                "table min differs from figure lo"
            )
        if abs(float(cell["hi"]) - float(row.hi)) > 1e-12:
            raise SystemExit(
                f"{path.name} {row.level_id} {row.outcome_family}: "
                "table max differs from figure hi"
            )
print("profile_whisker_point_check_ok")
print("appendix_table_figure_match_ok")
PY

for experiment in exp1 exp2 exp3; do
  for suffix in pdf png; do
    path="${PAPER_FIGURES}/${experiment}-trace-plot.${suffix}"
    if [[ ! -s "${path}" ]]; then
      echo "ERROR: missing ${path}" >&2
      exit 2
    fi
  done
  for kind in family outcome; do
    path="${PAPER_GENERATED}/appendix_${kind}_means_${experiment}.tex"
    if [[ ! -s "${path}" ]]; then
      echo "ERROR: missing ${path}" >&2
      exit 2
    fi
  done
done

echo "profile_whisker_rebuild_complete"
