#!/bin/bash
# =============================================================================
# Prepare + optionally submit full paper-scope rerun (Exp1/2/3) with
# point-estimate / CI / paired-test statistics regeneration.
#
# Scope:
# - Includes Exp1/2/3 primary + additional binary + regression families
# - Excludes MLP and xgBoost surfaces
# - Uses partition fallbacks:
#   - GPU: gpudev <-> gpuq (as needed)
#   - CPU: tier2q -> tier1q
# - Uses full arrays (no concurrency threshold)
# =============================================================================

set -euo pipefail

SUBMIT_MODE="submit"
if [[ "${1:-}" == "--prepare-only" ]]; then
  SUBMIT_MODE="prepare-only"
fi

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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IRB_HOME="$(find_repo_root "${SCRIPT_DIR}")"
cd "${IRB_HOME}"

DEMO_DIR="${IRB_HOME}/slurm/generated/demo"
STATS_DIR="${IRB_HOME}/slurm/generated/statistics"
STATS_ROOT="${IRB_HOME}/outputs/runs/statistics/paper_stats_run_outputs"
mkdir -p "${DEMO_DIR}" "${STATS_DIR}" "${STATS_ROOT}" "${IRB_HOME}/slurm/output"

echo "[prepare] Generating Exp1/2/3 jobfiles..."
python "pipeline/run_experiments.py" --mode demo --exp 1
python "pipeline/run_experiments.py" \
  --mode demo \
  --exp 2 \
  --exp2_include_discrete \
  --exp2_include_xval_affine \
  --exp2_from_exp1_config_id "meds_deciles_none_fusedFalse_discrete_time_tokens" \
  --exp2_soft_from_exp1_unfused_config_id "meds_deciles_none_fusedFalse_discrete_time_tokens"
python "pipeline/run_experiments.py" \
  --mode demo \
  --exp 3 \
  --exp3_from_exp1_config_id "meds_deciles_none_fusedFalse_discrete_time_tokens" \
  --exp3_from_exp2_config_id "meds_deciles_none_fusedFalse_discrete_time_rope"

echo "[prepare] Building paper-scope Stage3 extra-outcome jobfiles and stats inventories..."
python - <<'PY'
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path.cwd()
DEMO = ROOT / "slurm" / "generated" / "demo"
STATS_DIR = ROOT / "slurm" / "generated" / "statistics"
STATS_ROOT = ROOT / "outputs" / "runs" / "statistics" / "paper_stats_run_outputs"
STATS_DIR.mkdir(parents=True, exist_ok=True)
STATS_ROOT.mkdir(parents=True, exist_ok=True)

ADDITIONAL_BINARY = (
    "hyperkalemia severe_hypokalemia severe_anemia hypoglycemia "
    "profound_hyponatremia severe_hypernatremia tachycardia_hr130 "
    "severe_hypertension vasopressor_initiation hypotension "
    "crrt_initiation hemodialysis_initiation"
)
EXTENDED_REGRESSION = (
    "peak_creatinine peak_troponin min_hemoglobin peak_potassium "
    "min_potassium min_glucose min_sodium max_sodium peak_bnp "
    "max_heart_rate max_sbp max_dbp"
)


def _active_cmds(path: Path) -> list[str]:
    out: list[str] = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.append(s)
    return out


def _write(path: Path, title: str, lines: list[str]) -> None:
    payload = [f"# {title}", ""] + lines
    path.write_text("\n".join(payload) + "\n")


def _diagify(
    *,
    src_jobfile: Path,
    dst_jobfile: Path,
    classifier: str,
    task_type: str,
    outcomes_parquet: str,
    outcomes: str,
    preds_tag: str | None,
) -> None:
    new_lines: list[str] = []
    for line in _active_cmds(src_jobfile):
        # Remove Exp3 primary override if present; diagnostic wrapper sets outcomes explicitly.
        line = line.replace(
            'IRB_XFER_PRED_ARGS=\\"--outcomes same_admission_death long_length_of_stay prolonged_icu_stay imv_event\\" ',
            "",
        )
        if "&& bash slurm/ref_qse/10_xfer_rep_based_preds.sh" not in line:
            raise ValueError(f"Unexpected Stage3 command format in {src_jobfile}: {line}")
        env_bits = [
            f"DIAG_CLASSIFIER={classifier}",
            f"DIAG_TASK_TYPE={task_type}",
            f"DIAG_OUTCOMES_PARQUET={outcomes_parquet}",
            f"DIAG_OUTCOMES='{outcomes}'",
        ]
        if preds_tag:
            env_bits.append(f"DIAG_PREDS_TAG={preds_tag}")
        env_join = " ".join(env_bits)
        line = line.replace(
            " && bash slurm/ref_qse/10_xfer_rep_based_preds.sh",
            f" {env_join} && bash slurm/ref_qse/11_diag_eval.sh",
        )
        new_lines.append(line)
    _write(dst_jobfile, f"{dst_jobfile.stem.replace('_', ' ')}", new_lines)


_diagify(
    src_jobfile=DEMO / "07c_exp1_stage3_lr.jobfile",
    dst_jobfile=DEMO / "07d_exp1_stage3_additional_binary_lr.jobfile",
    classifier="logistic_regression",
    task_type="classification",
    outcomes_parquet="tokens_timelines_extended_outcomes.parquet",
    outcomes=ADDITIONAL_BINARY,
    preds_tag=None,
)
_diagify(
    src_jobfile=DEMO / "07c_exp1_stage3_lr.jobfile",
    dst_jobfile=DEMO / "07e_exp1_stage3_length_of_stay_ridge.jobfile",
    classifier="ridge_regression",
    task_type="regression",
    outcomes_parquet="tokens_timelines_outcomes.parquet",
    outcomes="length_of_stay",
    preds_tag="paper-length_of_stay",
)
_diagify(
    src_jobfile=DEMO / "07c_exp1_stage3_lr.jobfile",
    dst_jobfile=DEMO / "07f_exp1_stage3_extended_regression_ridge.jobfile",
    classifier="ridge_regression",
    task_type="regression",
    outcomes_parquet="tokens_timelines_extended_outcomes.parquet",
    outcomes=EXTENDED_REGRESSION,
    preds_tag=None,
)
_diagify(
    src_jobfile=DEMO / "10c_exp2_stage3_lr.jobfile",
    dst_jobfile=DEMO / "10d_exp2_stage3_additional_binary_lr.jobfile",
    classifier="logistic_regression",
    task_type="classification",
    outcomes_parquet="tokens_timelines_extended_outcomes.parquet",
    outcomes=ADDITIONAL_BINARY,
    preds_tag=None,
)
_diagify(
    src_jobfile=DEMO / "10c_exp2_stage3_lr.jobfile",
    dst_jobfile=DEMO / "10e_exp2_stage3_length_of_stay_ridge.jobfile",
    classifier="ridge_regression",
    task_type="regression",
    outcomes_parquet="tokens_timelines_outcomes.parquet",
    outcomes="length_of_stay",
    preds_tag="paper-length_of_stay",
)
_diagify(
    src_jobfile=DEMO / "10c_exp2_stage3_lr.jobfile",
    dst_jobfile=DEMO / "10f_exp2_stage3_extended_regression_ridge.jobfile",
    classifier="ridge_regression",
    task_type="regression",
    outcomes_parquet="tokens_timelines_extended_outcomes.parquet",
    outcomes=EXTENDED_REGRESSION,
    preds_tag=None,
)
_diagify(
    src_jobfile=DEMO / "14c_exp3_stage3_lr.jobfile",
    dst_jobfile=DEMO / "14d_exp3_stage3_additional_binary_lr.jobfile",
    classifier="logistic_regression",
    task_type="classification",
    outcomes_parquet="tokens_timelines_extended_outcomes.parquet",
    outcomes=ADDITIONAL_BINARY,
    preds_tag=None,
)
_diagify(
    src_jobfile=DEMO / "14c_exp3_stage3_lr.jobfile",
    dst_jobfile=DEMO / "14e_exp3_stage3_length_of_stay_ridge.jobfile",
    classifier="ridge_regression",
    task_type="regression",
    outcomes_parquet="tokens_timelines_outcomes.parquet",
    outcomes="length_of_stay",
    preds_tag="paper-length_of_stay",
)
_diagify(
    src_jobfile=DEMO / "14c_exp3_stage3_lr.jobfile",
    dst_jobfile=DEMO / "14f_exp3_stage3_extended_regression_ridge.jobfile",
    classifier="ridge_regression",
    task_type="regression",
    outcomes_parquet="tokens_timelines_extended_outcomes.parquet",
    outcomes=EXTENDED_REGRESSION,
    preds_tag=None,
)


def _pred_name(classifier: str, *, regression: bool, model_stem: str, tag: str | None) -> str:
    prefix = f"reg_{classifier}" if regression else classifier
    if tag:
        return f"{prefix}-preds-{tag}-{model_stem}.pkl"
    return f"{prefix}-preds-{model_stem}.pkl"


rows: list[dict[str, str]] = []


def _append_family_rows(
    *,
    family_name: str,
    task_type: str,
    handle: str,
    pred_path: Path,
    outcomes_path: Path,
) -> None:
    rows.append(
        {
            "family_name": family_name,
            "task_type": task_type,
            "handle": handle,
            "pred_path": str(pred_path.resolve()),
            "outcomes_parquet": str(outcomes_path.resolve()),
        }
    )


token_root = ROOT / "outputs" / "runs" / "tokenized" / "mimiciv-3.1_meds_70-10-20"
exp3_root = ROOT / "outputs" / "runs" / "exp3"

# ---------------------------
# Exp1 handles (12)
# ---------------------------
exp1_specs = [
    ("deciles_unfused", "meds_deciles_none_fusedFalse_discrete_time_tokens", "deciles_none_unfused_time_tokens"),
    ("deciles_fused", "meds_deciles_none_fusedTrue_discrete_time_tokens", "deciles_none_fused_time_tokens"),
    ("ventiles_unfused", "meds_ventiles_none_fusedFalse_discrete_time_tokens", "ventiles_none_unfused_time_tokens"),
    ("ventiles_fused", "meds_ventiles_none_fusedTrue_discrete_time_tokens", "ventiles_none_fused_time_tokens"),
    ("ventiles_5_10_5_unfused", "meds_ventiles_5-10-5_fusedFalse_discrete_time_tokens", "ventiles_5-10-5_unfused_time_tokens"),
    ("ventiles_5_10_5_fused", "meds_ventiles_5-10-5_fusedTrue_discrete_time_tokens", "ventiles_5-10-5_fused_time_tokens"),
    ("trentiles_unfused", "meds_trentiles_none_fusedFalse_discrete_time_tokens", "trentiles_none_unfused_time_tokens"),
    ("trentiles_fused", "meds_trentiles_none_fusedTrue_discrete_time_tokens", "trentiles_none_fused_time_tokens"),
    ("trentiles_10_10_10_unfused", "meds_trentiles_10-10-10_fusedFalse_discrete_time_tokens", "trentiles_10-10-10_unfused_time_tokens"),
    ("trentiles_10_10_10_fused", "meds_trentiles_10-10-10_fusedTrue_discrete_time_tokens", "trentiles_10-10-10_fused_time_tokens"),
    ("centiles_unfused", "meds_centiles_none_fusedFalse_discrete_time_tokens", "centiles_none_unfused_time_tokens"),
    ("centiles_fused", "meds_centiles_none_fusedTrue_discrete_time_tokens", "centiles_none_fused_time_tokens"),
]

for handle, config_id, dv in exp1_specs:
    test_dir = token_root / f"{dv}_first_24h-tokenized" / "test"
    outcomes_base = test_dir / "tokens_timelines_outcomes.parquet"
    outcomes_ext = test_dir / "tokens_timelines_extended_outcomes.parquet"
    model_stem = f"exp1_{config_id}-s42-hp-{dv}"
    _append_family_rows(
        family_name="exp1_primary_binary",
        task_type="classification",
        handle=handle,
        pred_path=test_dir / _pred_name("logistic_regression", regression=False, model_stem=model_stem, tag="base-primary_binary"),
        outcomes_path=outcomes_base,
    )
    _append_family_rows(
        family_name="exp1_additional_binary",
        task_type="classification",
        handle=handle,
        pred_path=test_dir / _pred_name("logistic_regression", regression=False, model_stem=model_stem, tag=None),
        outcomes_path=outcomes_ext,
    )
    _append_family_rows(
        family_name="exp1_length_of_stay",
        task_type="regression",
        handle=handle,
        pred_path=test_dir / _pred_name("ridge_regression", regression=True, model_stem=model_stem, tag="paper-length_of_stay"),
        outcomes_path=outcomes_base,
    )
    _append_family_rows(
        family_name="exp1_extended_regression",
        task_type="regression",
        handle=handle,
        pred_path=test_dir / _pred_name("ridge_regression", regression=True, model_stem=model_stem, tag=None),
        outcomes_path=outcomes_ext,
    )

# ---------------------------
# Exp2 handles (8)
# ---------------------------
exp2_specs = [
    ("discrete_tt", "deciles_none_unfused_time_tokens", "model-discrete-time_tokens"),
    ("soft_tt", "deciles_none_unfused_time_tokens", "model-soft-time_tokens"),
    ("xval_tt", "deciles_none_unfused_time_tokens_numencxval", "model-xval-time_tokens"),
    ("xval_affine_tt", "deciles_none_unfused_time_tokens_numencxval", "model-xval_affine-time_tokens"),
    ("discrete_rope", "deciles_none_unfused_time_rope", "model-discrete-time_rope"),
    ("soft_rope", "deciles_none_unfused_time_rope", "model-soft-time_rope"),
    ("xval_rope", "deciles_none_unfused_time_rope_numencxval", "model-xval-time_rope"),
    ("xval_affine_rope", "deciles_none_unfused_time_rope_numencxval", "model-xval_affine-time_rope"),
]

for handle, dv, model_stem in exp2_specs:
    test_dir = token_root / f"{dv}_first_24h-tokenized" / "test"
    outcomes_base = test_dir / "tokens_timelines_outcomes.parquet"
    outcomes_ext = test_dir / "tokens_timelines_extended_outcomes.parquet"
    _append_family_rows(
        family_name="exp2_primary_binary",
        task_type="classification",
        handle=handle,
        pred_path=test_dir / _pred_name("logistic_regression", regression=False, model_stem=model_stem, tag="base-primary_binary"),
        outcomes_path=outcomes_base,
    )
    _append_family_rows(
        family_name="exp2_additional_binary",
        task_type="classification",
        handle=handle,
        pred_path=test_dir / _pred_name("logistic_regression", regression=False, model_stem=model_stem, tag=None),
        outcomes_path=outcomes_ext,
    )
    _append_family_rows(
        family_name="exp2_length_of_stay",
        task_type="regression",
        handle=handle,
        pred_path=test_dir / _pred_name("ridge_regression", regression=True, model_stem=model_stem, tag="paper-length_of_stay"),
        outcomes_path=outcomes_base,
    )
    _append_family_rows(
        family_name="exp2_extended_regression",
        task_type="regression",
        handle=handle,
        pred_path=test_dir / _pred_name("ridge_regression", regression=True, model_stem=model_stem, tag=None),
        outcomes_path=outcomes_ext,
    )

# ---------------------------
# Exp3 handles (4)
# ---------------------------
exp3_specs = [
    ("meds", exp3_root / "meds_icu"),
    ("mapped", exp3_root / "arms" / "meds_mapped"),
    ("randomized", exp3_root / "arms" / "meds_randomized"),
    ("freqmatched", exp3_root / "arms" / "meds_freqmatched"),
]

for handle, data_dir in exp3_specs:
    test_dir = data_dir / "deciles_none_unfused_time_rope_first_24h-tokenized" / "test"
    outcomes_base = test_dir / "tokens_timelines_outcomes.parquet"
    outcomes_ext = test_dir / "tokens_timelines_extended_outcomes.parquet"
    model_stem = "model-discrete-time_rope"
    _append_family_rows(
        family_name="exp3_primary_binary",
        task_type="classification",
        handle=handle,
        pred_path=test_dir / _pred_name("logistic_regression", regression=False, model_stem=model_stem, tag="base-primary_binary"),
        outcomes_path=outcomes_base,
    )
    _append_family_rows(
        family_name="exp3_additional_binary",
        task_type="classification",
        handle=handle,
        pred_path=test_dir / _pred_name("logistic_regression", regression=False, model_stem=model_stem, tag=None),
        outcomes_path=outcomes_ext,
    )
    _append_family_rows(
        family_name="exp3_length_of_stay",
        task_type="regression",
        handle=handle,
        pred_path=test_dir / _pred_name("ridge_regression", regression=True, model_stem=model_stem, tag="paper-length_of_stay"),
        outcomes_path=outcomes_base,
    )
    _append_family_rows(
        family_name="exp3_extended_regression",
        task_type="regression",
        handle=handle,
        pred_path=test_dir / _pred_name("ridge_regression", regression=True, model_stem=model_stem, tag=None),
        outcomes_path=outcomes_ext,
    )

source_csv = STATS_DIR / "source_metrics_inventory.csv"
with source_csv.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["family_name", "task_type", "handle", "pred_path", "outcomes_parquet"],
    )
    writer.writeheader()
    writer.writerows(rows)

families = [
    ("exp1_primary_binary", "none"),
    ("exp1_additional_binary", "none"),
    ("exp1_length_of_stay", "none"),
    ("exp1_extended_regression", "none"),
    ("exp2_primary_binary", "discrete_tt"),
    ("exp2_additional_binary", "discrete_tt"),
    ("exp2_length_of_stay", "discrete_tt"),
    ("exp2_extended_regression", "discrete_tt"),
    ("exp3_primary_binary", "meds"),
    ("exp3_additional_binary", "meds"),
    ("exp3_length_of_stay", "meds"),
    ("exp3_extended_regression", "meds"),
]

array_jobfile = STATS_DIR / "aligned_family_stats.jobfile"
combine_jobfile = STATS_DIR / "aligned_family_stats_combine.jobfile"

array_lines: list[str] = []
for family_name, baseline in families:
    cmd = [
        "python",
        "\"${IRB_HOME}/pipeline/scripts/regenerate_aligned_family_stats.py\"",
        "--source_metrics_csv",
        f"\"{source_csv.resolve()}\"",
        "--out_root",
        f"\"{STATS_ROOT.resolve()}\"",
        "--bootstrap_n",
        "2000",
        "--permutation_n",
        "2000",
        "--alpha",
        "0.05",
        "--fdr",
        "bh",
        "--family_name",
        family_name,
        "--skip_combine",
    ]
    if baseline != "none":
        cmd.extend(["--baseline_handle", baseline])
    array_lines.append(" ".join(cmd))

combine_lines = [
    "python "
    "\"${IRB_HOME}/pipeline/scripts/regenerate_aligned_family_stats.py\" "
    f"--source_metrics_csv \"{source_csv.resolve()}\" "
    f"--out_root \"{STATS_ROOT.resolve()}\" "
    "--combine_only "
    "&& python "
    "\"${IRB_HOME}/paper/scripts/recompute_baseline_pairwise_view.py\" "
    f"--pairwise_csv \"{(STATS_ROOT / 'all_family_pairwise.csv').resolve()}\" "
    f"--out_csv \"{(STATS_ROOT / 'all_family_pairwise_baseline.csv').resolve()}\" "
    "--exp1_baseline deciles_unfused"
]

_write(array_jobfile, "Aligned family stats array jobs", array_lines)
_write(combine_jobfile, "Aligned family stats combine jobs", combine_lines)

paper_refresh_jobfile = STATS_DIR / "paper_refresh.jobfile"
paper_refresh_cmd = (
    "python \"${IRB_HOME}/paper/scripts/generate_mlhc_appendix_tables.py\" "
    "&& python \"${IRB_HOME}/paper/scripts/generate_mlhc_appendix_outcome_descriptives.py\" "
    "&& python \"${IRB_HOME}/paper/scripts/generate_mlhc_paper_figures.py\""
)
_write(paper_refresh_jobfile, "Paper refresh after stats", [paper_refresh_cmd])

print(f"Wrote Stage3 extras under {DEMO}")
print(f"Wrote stats inventory: {source_csv}")
print(f"Wrote stats array jobfile: {array_jobfile}")
print(f"Wrote stats combine jobfile: {combine_jobfile}")
print(f"Wrote paper refresh jobfile: {paper_refresh_jobfile}")
PY

if [[ "${SUBMIT_MODE}" == "prepare-only" ]]; then
  echo "[prepare] Completed in prepare-only mode (no sbatch submit)."
  exit 0
fi

require_dir() {
  local p="$1"
  local what="$2"
  if [[ ! -d "${p}" ]]; then
    echo "MISSING ${what}: ${p}"
    return 1
  fi
  return 0
}

echo "[submit] Checking required data/event prerequisites..."
MISSING_PREREQ=0
for s in train val test; do
  if ! require_dir "${IRB_HOME}/benchmarks/mimic-meds-extraction/data/meds/data/raw/${s}" "Exp1/2 raw split"; then
    MISSING_PREREQ=1
  fi
done
for d in \
  "${IRB_HOME}/outputs/runs/exp3/meds_icu/train" \
  "${IRB_HOME}/outputs/runs/exp3/meds_icu/test" \
  "${IRB_HOME}/outputs/runs/exp3/arms/meds_mapped/train" \
  "${IRB_HOME}/outputs/runs/exp3/arms/meds_randomized/train" \
  "${IRB_HOME}/outputs/runs/exp3/arms/meds_freqmatched/train"; do
  if ! require_dir "${d}" "Exp3 events split"; then
    MISSING_PREREQ=1
  fi
done

if [[ "${MISSING_PREREQ}" -ne 0 ]]; then
  echo "[submit] Prerequisites are incomplete. Jobfiles are prepared, but submission is skipped."
  echo "[submit] Re-run this script after raw MEDS/event splits are materialized."
  exit 0
fi

count_cmds() {
  awk 'NF && $1 !~ /^#/{n+=1} END{print n+0}' "$1"
}

submit_array_fallback() {
  local primary="$1"
  local fallback="$2"
  local dep="${3:-}"
  local runner="$4"
  local jobfile="$5"
  local n
  n="$(count_cmds "${jobfile}")"
  if [[ "${n}" -le 0 ]]; then
    echo "ERROR: no commands in ${jobfile}" >&2
    return 1
  fi
  local array_spec="0-$((n - 1))"
  local jid=""
  if [[ -n "${dep}" ]]; then
    if jid="$(sbatch --parsable --partition="${primary}" --dependency="afterok:${dep}" --array="${array_spec}" "${runner}" "${jobfile}")"; then
      :
    else
      jid="$(sbatch --parsable --partition="${fallback}" --dependency="afterok:${dep}" --array="${array_spec}" "${runner}" "${jobfile}")"
    fi
  else
    if jid="$(sbatch --parsable --partition="${primary}" --array="${array_spec}" "${runner}" "${jobfile}")"; then
      :
    else
      jid="$(sbatch --parsable --partition="${fallback}" --array="${array_spec}" "${runner}" "${jobfile}")"
    fi
  fi
  echo "${jid}"
}

submit_single_fallback() {
  local primary="$1"
  local fallback="$2"
  local dep="${3:-}"
  shift 3
  local jid=""
  if [[ -n "${dep}" ]]; then
    if jid="$(sbatch --parsable --partition="${primary}" --dependency="afterok:${dep}" "$@")"; then
      :
    else
      jid="$(sbatch --parsable --partition="${fallback}" --dependency="afterok:${dep}" "$@")"
    fi
  else
    if jid="$(sbatch --parsable --partition="${primary}" "$@")"; then
      :
    else
      jid="$(sbatch --parsable --partition="${fallback}" "$@")"
    fi
  fi
  echo "${jid}"
}

echo "[submit] Submitting stage chains..."

# Stage 0 (CPU)
JID_E1_S0="$(submit_array_fallback tier2q tier1q "" "slurm/02_run_stage0_tier2q_tokenize.sh" "${DEMO_DIR}/04_exp1_stage0_tokenize.jobfile")"
JID_E2_S0="$(submit_array_fallback tier2q tier1q "" "slurm/02_run_stage0_tier2q_tokenize.sh" "${DEMO_DIR}/08_exp2_stage0_tokenize.jobfile")"
JID_E3_S0="$(submit_array_fallback tier2q tier1q "" "slurm/02_run_stage0_tier2q_tokenize.sh" "${DEMO_DIR}/12_exp3_stage0_tokenize.jobfile")"

# Stage 0.5 extended outcomes refresh (CPU)
JID_EXT="$(submit_single_fallback tier2q tier1q "${JID_E1_S0}:${JID_E2_S0}:${JID_E3_S0}" "slurm/13_refresh_all_extended_outcomes.sh")"

# Stage 1 (GPU)
JID_E1_S1="$(submit_array_fallback gpudev gpuq "${JID_E1_S0}" "slurm/06_run_stage1_gpu1_train.sh" "${DEMO_DIR}/07a_exp1_stage1_train.jobfile")"
JID_E2_S1="$(submit_array_fallback gpudev gpuq "${JID_E2_S0}" "slurm/06_run_stage1_gpu1_train.sh" "${DEMO_DIR}/10a_exp2_stage1_train.jobfile")"
JID_E3_S1="$(submit_array_fallback gpudev gpuq "${JID_E3_S0}" "slurm/06_run_stage1_gpu1_train.sh" "${DEMO_DIR}/14a_exp3_stage1_train.jobfile")"

# Stage 2 (GPU)
JID_E1_S2="$(submit_array_fallback gpudev gpuq "${JID_E1_S1}" "slurm/09_run_stage2_gpu2_extract.sh" "${DEMO_DIR}/07b_exp1_stage2_extract_reps.jobfile")"
JID_E2_S2="$(submit_array_fallback gpudev gpuq "${JID_E2_S1}" "slurm/09_run_stage2_gpu2_extract.sh" "${DEMO_DIR}/10b_exp2_stage2_extract_reps.jobfile")"
JID_E3_S2="$(submit_array_fallback gpudev gpuq "${JID_E3_S1}" "slurm/09_run_stage2_gpu2_extract.sh" "${DEMO_DIR}/14b_exp3_stage2_extract_reps.jobfile")"

# Stage 3 primary (CPU)
JID_E1_S3P="$(submit_array_fallback tier2q tier1q "${JID_E1_S2}" "slurm/11_run_stage3_tier2q_lr.sh" "${DEMO_DIR}/07c_exp1_stage3_lr.jobfile")"
JID_E2_S3P="$(submit_array_fallback tier2q tier1q "${JID_E2_S2}" "slurm/11_run_stage3_tier2q_lr.sh" "${DEMO_DIR}/10c_exp2_stage3_lr.jobfile")"
JID_E3_S3P="$(submit_array_fallback tier2q tier1q "${JID_E3_S2}" "slurm/11_run_stage3_tier2q_lr.sh" "${DEMO_DIR}/14c_exp3_stage3_lr.jobfile")"

# Stage 3 additional binary + regression (CPU; no MLP/xgBoost)
JID_E1_S3D="$(submit_array_fallback tier2q tier1q "${JID_E1_S2}:${JID_EXT}" "slurm/11_run_stage3_tier2q_lr.sh" "${DEMO_DIR}/07d_exp1_stage3_additional_binary_lr.jobfile")"
JID_E1_S3E="$(submit_array_fallback tier2q tier1q "${JID_E1_S2}:${JID_EXT}" "slurm/11_run_stage3_tier2q_lr.sh" "${DEMO_DIR}/07e_exp1_stage3_length_of_stay_ridge.jobfile")"
JID_E1_S3F="$(submit_array_fallback tier2q tier1q "${JID_E1_S2}:${JID_EXT}" "slurm/11_run_stage3_tier2q_lr.sh" "${DEMO_DIR}/07f_exp1_stage3_extended_regression_ridge.jobfile")"

JID_E2_S3D="$(submit_array_fallback tier2q tier1q "${JID_E2_S2}:${JID_EXT}" "slurm/11_run_stage3_tier2q_lr.sh" "${DEMO_DIR}/10d_exp2_stage3_additional_binary_lr.jobfile")"
JID_E2_S3E="$(submit_array_fallback tier2q tier1q "${JID_E2_S2}:${JID_EXT}" "slurm/11_run_stage3_tier2q_lr.sh" "${DEMO_DIR}/10e_exp2_stage3_length_of_stay_ridge.jobfile")"
JID_E2_S3F="$(submit_array_fallback tier2q tier1q "${JID_E2_S2}:${JID_EXT}" "slurm/11_run_stage3_tier2q_lr.sh" "${DEMO_DIR}/10f_exp2_stage3_extended_regression_ridge.jobfile")"

JID_E3_S3D="$(submit_array_fallback tier2q tier1q "${JID_E3_S2}:${JID_EXT}" "slurm/11_run_stage3_tier2q_lr.sh" "${DEMO_DIR}/14d_exp3_stage3_additional_binary_lr.jobfile")"
JID_E3_S3E="$(submit_array_fallback tier2q tier1q "${JID_E3_S2}:${JID_EXT}" "slurm/11_run_stage3_tier2q_lr.sh" "${DEMO_DIR}/14e_exp3_stage3_length_of_stay_ridge.jobfile")"
JID_E3_S3F="$(submit_array_fallback tier2q tier1q "${JID_E3_S2}:${JID_EXT}" "slurm/11_run_stage3_tier2q_lr.sh" "${DEMO_DIR}/14f_exp3_stage3_extended_regression_ridge.jobfile")"

DEP_STATS="${JID_E1_S3P}:${JID_E2_S3P}:${JID_E3_S3P}:${JID_E1_S3D}:${JID_E1_S3E}:${JID_E1_S3F}:${JID_E2_S3D}:${JID_E2_S3E}:${JID_E2_S3F}:${JID_E3_S3D}:${JID_E3_S3E}:${JID_E3_S3F}"
JID_STATS_ARRAY="$(submit_array_fallback tier2q tier1q "${DEP_STATS}" "slurm/15_run_stats_cpu_jobfile.sh" "${STATS_DIR}/aligned_family_stats.jobfile")"
JID_STATS_COMBINE="$(submit_single_fallback tier2q tier1q "${JID_STATS_ARRAY}" "slurm/15_run_stats_cpu_jobfile.sh" "${STATS_DIR}/aligned_family_stats_combine.jobfile")"
JID_PAPER="$(submit_single_fallback tier2q tier1q "${JID_STATS_COMBINE}" "slurm/15_run_stats_cpu_jobfile.sh" "${STATS_DIR}/paper_refresh.jobfile")"

echo "[submit] Done. Job IDs:"
echo "  Exp1 Stage0=${JID_E1_S0} Stage1=${JID_E1_S1} Stage2=${JID_E1_S2}"
echo "  Exp2 Stage0=${JID_E2_S0} Stage1=${JID_E2_S1} Stage2=${JID_E2_S2}"
echo "  Exp3 Stage0=${JID_E3_S0} Stage1=${JID_E3_S1} Stage2=${JID_E3_S2}"
echo "  Extended outcomes refresh=${JID_EXT}"
echo "  Exp1 Stage3 primary/add/reg=${JID_E1_S3P}/${JID_E1_S3D}/${JID_E1_S3E}/${JID_E1_S3F}"
echo "  Exp2 Stage3 primary/add/reg=${JID_E2_S3P}/${JID_E2_S3D}/${JID_E2_S3E}/${JID_E2_S3F}"
echo "  Exp3 Stage3 primary/add/reg=${JID_E3_S3P}/${JID_E3_S3D}/${JID_E3_S3E}/${JID_E3_S3F}"
echo "  Stats array/combine=${JID_STATS_ARRAY}/${JID_STATS_COMBINE}"
echo "  Paper refresh=${JID_PAPER}"
