#!/bin/bash
#SBATCH --job-name=irb-gpu4-serial
#SBATCH --output=./slurm/output/%j-%x.stdout
#SBATCH --partition=sxmq
#SBATCH --qos=normal
#SBATCH --gres=gpu:8
#SBATCH --mem=256GB
#SBATCH --cpus-per-task=16
#SBATCH --time=10-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@900

# =============================================================================
# Runner (GPU; 4 DDP ranks): executes every command in a jobfile serially within
# one long Slurm allocation.
#
# This is intended for sxmq when we want to hold the node across a whole chain
# rather than yielding between array elements. The allocation reserves all 8 GPUs
# but, by default on sxmq, pins work to GPUs 4-7 because GPUs 0-3 have been
# occupied by non-Slurm processes on cri22cn501.
# =============================================================================

set -euo pipefail

JOBFILE=${1:-""}
STATE_FILE=${2:-""}
if [[ -z "${JOBFILE}" ]]; then
  echo "ERROR: Job file not provided." >&2
  echo "Usage: sbatch slurm/18_run_gpu4_jobfile_serial.sh <jobfile> [state-file]" >&2
  exit 2
fi
if [[ ! -f "${JOBFILE}" ]]; then
  echo "ERROR: Job file not found: ${JOBFILE}" >&2
  exit 1
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

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
IRB_HOME="$(find_repo_root "${SUBMIT_DIR}")"
source "${IRB_HOME}/slurm/00_preamble.sh"
cd "${IRB_HOME}"
mkdir -p slurm/output slurm/state

if [[ -z "${STATE_FILE}" ]]; then
  STATE_FILE="slurm/state/${SLURM_JOB_ID:-manual}-$(basename "${JOBFILE}").last_completed"
fi

TRAIN_PID=""
REQUEUE_IN_PROGRESS=0

stop_training_process() {
  if [[ -z "${TRAIN_PID}" ]] || ! kill -0 "${TRAIN_PID}" 2>/dev/null; then
    return 0
  fi

  echo "[requeue] Forwarding SIGTERM to process group ${TRAIN_PID}."
  kill -TERM "-${TRAIN_PID}" 2>/dev/null || kill -TERM "${TRAIN_PID}" 2>/dev/null || true

  local grace="${IRB_REQUEUE_GRACE_SECONDS:-120}"
  local elapsed=0
  while kill -0 "${TRAIN_PID}" 2>/dev/null && [[ "${elapsed}" -lt "${grace}" ]]; do
    sleep 5
    elapsed=$((elapsed + 5))
  done

  if kill -0 "${TRAIN_PID}" 2>/dev/null; then
    echo "[requeue] Process did not stop after ${grace}s; sending SIGKILL."
    kill -KILL "-${TRAIN_PID}" 2>/dev/null || kill -KILL "${TRAIN_PID}" 2>/dev/null || true
  fi
}

requeue_current_job() {
  local signal_name="$1"
  local job_id="${SLURM_JOB_ID:?SLURM_JOB_ID is required for requeue}"
  echo "[requeue] Received ${signal_name}; requeueing ${job_id}."
  REQUEUE_IN_PROGRESS=1
  if ! scontrol requeue "${job_id}"; then
    echo "[requeue] ERROR: failed to requeue ${job_id}." >&2
    REQUEUE_IN_PROGRESS=0
    stop_training_process
    exit 1
  fi
  trap 'terminate_current_job TERM; exit 0' TERM
  trap 'terminate_current_job INT; exit 0' INT
  stop_training_process
  exit 0
}

terminate_current_job() {
  local signal_name="$1"
  if [[ "${REQUEUE_IN_PROGRESS}" -eq 1 ]]; then
    echo "[requeue] Received ${signal_name} during intentional requeue; exiting cleanly."
    exit 0
  fi
  echo "[requeue] Received ${signal_name}; terminating without requeue."
  stop_training_process
}

trap 'requeue_current_job USR1' USR1
trap 'terminate_current_job TERM; exit 143' TERM
trap 'terminate_current_job INT; exit 130' INT

export IRB_NPROC_PER_NODE="${IRB_NPROC_PER_NODE:-4}"
if [[ "${IRB_NPROC_PER_NODE}" -ne 4 ]]; then
  echo "ERROR: IRB_NPROC_PER_NODE must be 4 for this runner (got ${IRB_NPROC_PER_NODE})." >&2
  exit 2
fi
if [[ "${SLURM_JOB_NUM_NODES:-1}" -ne 1 ]]; then
  echo "ERROR: This benchmark assumes single-node execution (got SLURM_JOB_NUM_NODES=${SLURM_JOB_NUM_NODES})." >&2
  exit 2
fi

if [[ -n "${IRB_STAGE1_CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${IRB_STAGE1_CUDA_VISIBLE_DEVICES}"
elif [[ "${SLURM_JOB_PARTITION:-}" == "sxmq" && "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3,4,5,6,7" ]]; then
  export CUDA_VISIBLE_DEVICES="4,5,6,7"
fi

if [[ -n "${SLURM_JOB_NODELIST:-}" ]]; then
  MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
else
  MASTER_ADDR="$(hostname -s)"
fi
MASTER_PORT="$((10000 + (${SLURM_JOB_ID:-0} % 50000)))"
export MASTER_ADDR MASTER_PORT

mapfile -t COMMANDS < <(grep -v '^#' "${JOBFILE}" | grep -v '^[[:space:]]*$')
if [[ "${#COMMANDS[@]}" -eq 0 ]]; then
  echo "ERROR: no commands found in ${JOBFILE}" >&2
  exit 1
fi

last_completed="-1"
if [[ -f "${STATE_FILE}" ]]; then
  last_completed="$(<"${STATE_FILE}")"
fi
if ! [[ "${last_completed}" =~ ^-?[0-9]+$ ]]; then
  echo "ERROR: invalid state file value in ${STATE_FILE}: ${last_completed}" >&2
  exit 1
fi

echo "=============================================="
echo "Serial GPU jobfile runner"
echo "Job File: ${JOBFILE}"
echo "State File: ${STATE_FILE}"
echo "Commands: ${#COMMANDS[@]}"
echo "Last completed index: ${last_completed}"
echo "IRB_NPROC_PER_NODE: ${IRB_NPROC_PER_NODE}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-auto}"
echo "Host: $(hostname)"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "=============================================="

for i in "${!COMMANDS[@]}"; do
  if [[ "${i}" -le "${last_completed}" ]]; then
    echo "[serial] Skipping completed command index ${i}."
    continue
  fi

  CMD="${COMMANDS[$i]}"
  echo ""
  echo "=============================================="
  echo "[serial] Starting command index ${i}/$(( ${#COMMANDS[@]} - 1 ))"
  echo "${CMD}"
  echo "=============================================="

  if command -v setsid >/dev/null 2>&1; then
    setsid bash -lc "${CMD}" &
  else
    bash -lc "${CMD}" &
  fi
  TRAIN_PID=$!
  set +e
  wait "${TRAIN_PID}"
  status=$?
  set -e
  TRAIN_PID=""

  if [[ "${status}" -ne 0 ]]; then
    echo "[serial] Command index ${i} failed with status ${status}." >&2
    exit "${status}"
  fi

  echo "${i}" > "${STATE_FILE}"
  echo "[serial] Completed command index ${i}."
done

echo "[serial] All commands completed."
