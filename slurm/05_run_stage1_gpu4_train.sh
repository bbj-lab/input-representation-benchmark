#!/bin/bash
#SBATCH --job-name=irb-train-4gpu
#SBATCH --output=./slurm/output/%A_%a-%x.stdout
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:4
#SBATCH --mem=256GB
#SBATCH --cpus-per-task=16
#SBATCH --time=0-12:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@900

# =============================================================================
# Runner (Stage 1; 4 GPUs): executes the Nth line of a jobfile
#
# Rationale:
# - gpuq nodes are 8-GPU; requesting 4-GPU + 16 CPU allows 2 concurrent jobs/node
# - Faster scheduling when the partition is busy
# - Must also set IRB_NPROC_PER_NODE=4 so torchrun world size matches allocated GPUs
# =============================================================================

set -euo pipefail

JOBFILE=${1:-"slurm/generated/demo/07a_exp1_stage1_train.jobfile"}
if [[ ! -f "$JOBFILE" ]]; then
  echo "ERROR: Job file not found: $JOBFILE" >&2
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
mkdir -p slurm/output

TRAIN_PID=""
REQUEUE_IN_PROGRESS=0

current_slurm_task_id() {
  if [[ -n "${SLURM_ARRAY_JOB_ID:-}" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
  else
    echo "${SLURM_JOB_ID:?SLURM_JOB_ID is required for requeue}"
  fi
}

stop_training_process() {
  if [[ -z "${TRAIN_PID}" ]] || ! kill -0 "${TRAIN_PID}" 2>/dev/null; then
    return 0
  fi

  echo "[requeue] Forwarding SIGTERM to training process group ${TRAIN_PID}."
  kill -TERM "-${TRAIN_PID}" 2>/dev/null || kill -TERM "${TRAIN_PID}" 2>/dev/null || true

  local grace="${IRB_REQUEUE_GRACE_SECONDS:-120}"
  local elapsed=0
  while kill -0 "${TRAIN_PID}" 2>/dev/null && [[ "${elapsed}" -lt "${grace}" ]]; do
    sleep 5
    elapsed=$((elapsed + 5))
  done

  if kill -0 "${TRAIN_PID}" 2>/dev/null; then
    echo "[requeue] Training process did not stop after ${grace}s; sending SIGKILL."
    kill -KILL "-${TRAIN_PID}" 2>/dev/null || kill -KILL "${TRAIN_PID}" 2>/dev/null || true
  fi
}

requeue_current_task() {
  local signal_name="$1"
  local task_id
  task_id="$(current_slurm_task_id)"

  echo "[requeue] Received ${signal_name}; requeueing ${task_id}."
  REQUEUE_IN_PROGRESS=1
  if ! scontrol requeue "${task_id}"; then
    echo "[requeue] ERROR: failed to requeue ${task_id}." >&2
    REQUEUE_IN_PROGRESS=0
    stop_training_process
    exit 1
  fi
  trap 'terminate_current_task TERM; exit 0' TERM
  trap 'terminate_current_task INT; exit 0' INT
  stop_training_process
  exit 0
}

terminate_current_task() {
  local signal_name="$1"
  if [[ "${REQUEUE_IN_PROGRESS}" -eq 1 ]]; then
    echo "[requeue] Received ${signal_name} during intentional requeue; exiting cleanly."
    exit 0
  fi
  echo "[requeue] Received ${signal_name}; terminating without requeue."
  stop_training_process
}

trap 'requeue_current_task USR1' USR1
trap 'terminate_current_task TERM; exit 143' TERM
trap 'terminate_current_task INT; exit 130' INT

# Ensure torchrun launches the correct number of processes for the allocated GPUs.
export IRB_NPROC_PER_NODE="${IRB_NPROC_PER_NODE:-4}"
if [[ "${IRB_NPROC_PER_NODE}" -ne 4 ]]; then
  echo "ERROR: IRB_NPROC_PER_NODE must be 4 for benchmark runs (got ${IRB_NPROC_PER_NODE})." >&2
  exit 2
fi
if [[ "${SLURM_JOB_NUM_NODES:-1}" -ne 1 ]]; then
  echo "ERROR: This benchmark assumes single-node training (got SLURM_JOB_NUM_NODES=${SLURM_JOB_NUM_NODES})." >&2
  exit 2
fi

if [[ -n "${IRB_STAGE1_CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${IRB_STAGE1_CUDA_VISIBLE_DEVICES}"
elif [[ "${SLURM_JOB_PARTITION:-}" == "sxmq" && "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3,4,5,6,7" ]]; then
  # cri22cn501 GPUs 0-3 currently carry active/orphan processes despite Slurm
  # reporting the node idle. Reserve the full SXM node, but run 4 DDP ranks on
  # the clean devices until an 8-GPU health check shows all devices are clear.
  export CUDA_VISIBLE_DEVICES="4,5,6,7"
fi

echo "=============================================="
echo "SLURM Array Task: ${SLURM_ARRAY_TASK_ID:-0}"
echo "Job File: ${JOBFILE}"
echo "IRB_NPROC_PER_NODE: ${IRB_NPROC_PER_NODE}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-auto}"
echo "Host: $(hostname)"
echo "=============================================="

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo "GPU info unavailable"

# -----------------------------------------------------------------------------
# Multi-node DDP rendezvous info (safe for single-node too)
# -----------------------------------------------------------------------------
if [[ -n "${SLURM_JOB_NODELIST:-}" ]]; then
  MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
else
  MASTER_ADDR="$(hostname -s)"
fi
MASTER_PORT="$((10000 + (${SLURM_JOB_ID:-0} % 50000)))"
export MASTER_ADDR MASTER_PORT

echo "DDP rendezvous:"
echo "  SLURM_JOB_NUM_NODES=${SLURM_JOB_NUM_NODES:-1}"
echo "  SLURM_NODEID=${SLURM_NODEID:-0}"
echo "  MASTER_ADDR=${MASTER_ADDR}"
echo "  MASTER_PORT=${MASTER_PORT}"

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
CMD=$(grep -v '^#' "$JOBFILE" | grep -v '^[[:space:]]*$' | sed -n "$((TASK_ID + 1))p")
if [[ -z "$CMD" ]]; then
  echo "ERROR: No command found for task ${TASK_ID}" >&2
  exit 1
fi

echo ""
echo "Executing command:"
echo "  ${CMD}"
echo ""
echo "=============================================="

echo "DDP/NCCL env (selected):"
echo "  MASTER_ADDR=${MASTER_ADDR:-}"
echo "  MASTER_PORT=${MASTER_PORT:-}"
echo "  RANK=${RANK:-}"
echo "  LOCAL_RANK=${LOCAL_RANK:-}"
echo "  WORLD_SIZE=${WORLD_SIZE:-}"
echo "  NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-}"
echo "  NCCL_DEBUG=${NCCL_DEBUG:-}"
echo "  TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG:-}"
echo ""

if command -v setsid >/dev/null 2>&1; then
  setsid bash -lc "${CMD}" &
else
  bash -lc "${CMD}" &
fi
TRAIN_PID=$!
set +e
wait "${TRAIN_PID}"
TRAIN_STATUS=$?
set -e
TRAIN_PID=""
exit "${TRAIN_STATUS}"

