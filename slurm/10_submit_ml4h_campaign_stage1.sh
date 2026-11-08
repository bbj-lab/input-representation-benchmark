#!/bin/bash
# =============================================================================
# Submit ML4H Stage-1 Exp1 only (Exp2 is gated on Exp1 quantization winners).
#
# Dual route, non-overlapping task ranges (no duplicate arms):
#   1) Exp1 tasks 0-39  on BBJ priority RTX (cri22cn426)
#   2) Exp1 tasks 40-71 on H200 opportunistic
#
# Exp2 jobfile is held under:
#   slurm/generated/ml4h_2026/held_until_exp1/
#
# Usage (from repo root):
#   bash slurm/10_submit_ml4h_campaign_stage1.sh
#   DRY_RUN=1 bash slurm/10_submit_ml4h_campaign_stage1.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IRB_HOME="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${IRB_HOME}"

GENERATED="slurm/generated/ml4h_2026"
EXP1_JOBFILE="${GENERATED}/02_exp1_stage1.jobfile"
EXP2_HELD="${GENERATED}/held_until_exp1/03_exp2_stage1.jobfile"
SUBMISSIONS_DIR="outputs/runs/ml4h_2026/submissions"
RUNNER="slurm/06_run_stage1_gpu1_train.sh"
DRY_RUN="${DRY_RUN:-0}"
TIME_LIMIT="${TIME_LIMIT:-1-12:00:00}"
# Split: first 40 tasks on non-preemptible RTX; remainder on H200.
RTX_LO=0
RTX_HI=39
H200_LO=40
H200_HI=71

mkdir -p \
  "${SUBMISSIONS_DIR}" \
  slurm/output/ml4h_exp1_rtx \
  slurm/output/ml4h_exp1_h200 \
  outputs/runs/ml4h_2026/models \
  "${GENERATED}/held_until_exp1"

if [[ ! -f "${EXP1_JOBFILE}" ]]; then
  echo "ERROR: missing ${EXP1_JOBFILE}" >&2
  exit 2
fi
if [[ ! -f "${EXP2_HELD}" ]]; then
  echo "WARNING: Exp2 jobfile not in held_until_exp1/ (expected after Exp1-first gate)." >&2
fi

N_EXP1=$(grep -cve '^[[:space:]]*$' "${EXP1_JOBFILE}" || true)
if [[ "${N_EXP1}" -ne 72 ]]; then
  echo "ERROR: expected 72 Exp1 tasks, found ${N_EXP1}" >&2
  exit 2
fi
if [[ "${H200_HI}" -ne $((N_EXP1 - 1)) ]]; then
  echo "ERROR: H200_HI=${H200_HI} does not match last Exp1 index $((N_EXP1 - 1))" >&2
  exit 2
fi

echo "Building task index..."
/gpfs/data/bbj-lab/.envs/input-rep/bin/python \
  utilities/scripts/build_ml4h_campaign_index.py \
  --generated-dir "${GENERATED}" \
  --output "${SUBMISSIONS_DIR}/campaign_task_index.json" || true

# Index builder expects Exp2 jobfile in generated/; rebuild Exp1-only index if needed.
/gpfs/data/bbj-lab/.envs/input-rep/bin/python - <<'PY'
import json, re
from pathlib import Path
root = Path("slurm/generated/ml4h_2026")
out = Path("outputs/runs/ml4h_2026/submissions")
lines = [ln.strip() for ln in (root / "02_exp1_stage1.jobfile").read_text().splitlines() if ln.strip()]
tasks = []
for i, line in enumerate(lines):
    prefix = re.search(r"IRB_MODEL_VERSION_PREFIX=(\S+)", line)
    config = re.search(r"config_id=(\S+)", line)
    seed = re.search(r"\bseed=(\d+)", line)
    backbone = "llama" if prefix and "llama" in prefix.group(1) else "qwen"
    tasks.append({
        "task_id": i,
        "experiment": "exp1",
        "backbone": backbone,
        "config_id": config.group(1) if config else None,
        "seed": int(seed.group(1)) if seed else None,
        "route": "rtx" if i <= 39 else "h200",
    })
payload = {"exp1": {"n_tasks": len(tasks), "tasks": tasks}, "exp2": {"status": "held_until_exp1"}}
out.mkdir(parents=True, exist_ok=True)
(out / "campaign_task_index.json").write_text(json.dumps(payload, indent=2) + "\n")
with (out / "campaign_task_index.tsv").open("w") as handle:
    handle.write("experiment\ttask_id\troute\tbackbone\tseed\tconfig_id\n")
    for row in tasks:
        handle.write(
            f"exp1\t{row['task_id']}\t{row['route']}\t{row['backbone']}\t{row['seed']}\t{row['config_id']}\n"
        )
print(f"Wrote Exp1-only index ({len(tasks)} tasks)")
PY

echo
echo "Live capacity snapshot:"
for node in cri22cn426 cri22cn417 cri22cn418 cri22cn423 cri22cn424; do
  scontrol show node "${node}" 2>/dev/null | tr ' ' '\n' | \
    rg '^(NodeName|State|AllocTRES|CfgTRES)=' | paste -sd' ' - || true
done

echo
echo "Planned Exp1-only submissions (Exp2 held until Exp1 quantization winners):"
echo "  Exp1 RTX : array=${RTX_LO}-${RTX_HI}  → cri22cn426"
echo "  Exp1 H200: array=${H200_LO}-${H200_HI} → opportunistic H200"
echo "  time=${TIME_LIMIT}"
echo "  Exp2 jobfile: ${EXP2_HELD}"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN=1 — not submitting."
  exit 0
fi

# Per-task HF datasets cache avoids multi-job parquet race on shared /scratch.
COMMON_EXPORTS=(--export=ALL,IRB_ISOLATE_DATASETS_CACHE=true)
H200_EXPORTS=(--export=ALL,IRB_ISOLATE_DATASETS_CACHE=true,IRB_STAGE1_CHECKPOINT_INTERVAL_SECONDS=300)

EXP1_RTX_JID=$(sbatch --parsable \
  "${COMMON_EXPORTS[@]}" \
  --job-name=ml4h-exp1-rtx \
  --partition=bbj-wanq \
  --qos=bbj-wan_priority \
  --nodelist=cri22cn426 \
  --gres=gpu:nvidia_rtxpro6000-96gb:1 \
  --mem=128GB \
  --cpus-per-task=8 \
  --time="${TIME_LIMIT}" \
  --requeue \
  --signal=B:USR1@600 \
  --array="${RTX_LO}-${RTX_HI}%4" \
  --output=./slurm/output/ml4h_exp1_rtx/%A_%a-%x.stdout \
  "${RUNNER}" "${EXP1_JOBFILE}")

EXP1_H200_JID=$(sbatch --parsable \
  "${H200_EXPORTS[@]}" \
  --job-name=ml4h-exp1-h200 \
  --partition=catherineq,siweiq,zhoulabq,gpuq,porrasq \
  --qos=opportunistic \
  --gres=gpu:nvidia_h200-141gb:1 \
  --mem=128GB \
  --cpus-per-task=8 \
  --time="${TIME_LIMIT}" \
  --requeue \
  --signal=B:USR1@600 \
  --array="${H200_LO}-${H200_HI}" \
  --output=./slurm/output/ml4h_exp1_h200/%A_%a-%x.stdout \
  "${RUNNER}" "${EXP1_JOBFILE}")

STAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
RECORD="${SUBMISSIONS_DIR}/stage1_exp1_only_${STAMP}.json"
LATEST="${SUBMISSIONS_DIR}/stage1_submission_latest.json"

/gpfs/data/bbj-lab/.envs/input-rep/bin/python - <<PY
import json
from pathlib import Path
record = {
    "submitted_at_utc": "${STAMP}",
    "mode": "exp1_only",
    "reason": "Exp2 numeric quantization degree is selected from Exp1 results; Exp2 held.",
    "time_limit": "${TIME_LIMIT}",
    "arrays": {
        "exp1_rtx": {
            "job_id": "${EXP1_RTX_JID}",
            "job_name": "ml4h-exp1-rtx",
            "array": "${RTX_LO}-${RTX_HI}",
            "n_tasks": ${RTX_HI} - ${RTX_LO} + 1,
            "recipe": "bbj_priority_rtx",
            "nodelist": "cri22cn426",
            "jobfile": "${EXP1_JOBFILE}",
            "logs": "slurm/output/ml4h_exp1_rtx/",
        },
        "exp1_h200": {
            "job_id": "${EXP1_H200_JID}",
            "job_name": "ml4h-exp1-h200",
            "array": "${H200_LO}-${H200_HI}",
            "n_tasks": ${H200_HI} - ${H200_LO} + 1,
            "recipe": "h200_opportunistic",
            "jobfile": "${EXP1_JOBFILE}",
            "logs": "slurm/output/ml4h_exp1_h200/",
        },
    },
    "held": {
        "exp2_jobfile": "${EXP2_HELD}",
        "cancelled_arrays": ["13240376", "13240377"],
    },
    "monitor": {
        "command": "python utilities/scripts/monitor_ml4h_runs.py --jobs ${EXP1_RTX_JID},${EXP1_H200_JID} --runs-root outputs/runs/ml4h_2026",
        "task_index": "outputs/runs/ml4h_2026/submissions/campaign_task_index.tsv",
    },
}
path = Path("${RECORD}")
path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\\n")
Path("${LATEST}").write_text(path.read_text())
print(path)
PY

echo
echo "Submitted Exp1-only:"
echo "  Exp1 RTX  array: ${EXP1_RTX_JID}  (${RTX_LO}-${RTX_HI})"
echo "  Exp1 H200 array: ${EXP1_H200_JID} (${H200_LO}-${H200_HI})"
echo "  Exp2: held (not submitted)"
echo "  Record: ${RECORD}"
