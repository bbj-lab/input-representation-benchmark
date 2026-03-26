#!/bin/bash
#SBATCH --job-name=irb-train-2gpu
#SBATCH --output=./slurm/output/%A_%a-%x.stdout
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:2
#SBATCH --mem=128GB
#SBATCH --cpus-per-task=8
#SBATCH --time=0-12:00:00

# =============================================================================
# Runner (Stage 1; 2 GPUs): executes the Nth line of a jobfile
#
# Rationale:
# - gpuq nodes are 8-GPU; requesting 2-GPU allows 4 concurrent jobs/node
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

# Ensure torchrun launches the correct number of processes for the allocated GPUs.
export IRB_NPROC_PER_NODE="${IRB_NPROC_PER_NODE:-2}"
if [[ "${IRB_NPROC_PER_NODE}" -ne 2 ]]; then
  echo "ERROR: IRB_NPROC_PER_NODE must be 2 for benchmark runs (got ${IRB_NPROC_PER_NODE})." >&2
  exit 2
fi
if [[ "${SLURM_JOB_NUM_NODES:-1}" -ne 1 ]]; then
  echo "ERROR: This benchmark assumes single-node training (got SLURM_JOB_NUM_NODES=${SLURM_JOB_NUM_NODES})." >&2
  exit 2
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

eval "$CMD"

