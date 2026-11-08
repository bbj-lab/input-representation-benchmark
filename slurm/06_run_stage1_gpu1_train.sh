#!/bin/bash
#SBATCH --job-name=irb-train-1gpu
#SBATCH --output=./slurm/output/%A_%a-%x.stdout
#SBATCH --partition=gpudev
#SBATCH --gres=gpu:1
#SBATCH --mem=128GB
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@600

# =============================================================================
# Runner (Stage 1; 1 GPU): executes the Nth line of a jobfile
#
# Rationale:
# - Reduce GPU request to improve scheduling latency (1 GPU/job).
# - Set IRB_NPROC_PER_NODE=1 to enforce one process per allocation.
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
# Existing arrays inherited the standard 10-minute value at submission time.
# Enforce the shorter recovery interval at execution time for opportunistic
# allocations, including their later Slurm requeues.
if [[ "${SLURM_JOB_QOS:-}" == "opportunistic" ]]; then
  export IRB_STAGE1_CHECKPOINT_INTERVAL_SECONDS="${IRB_OPPORTUNISTIC_CHECKPOINT_INTERVAL_SECONDS:-300}"
fi
cd "${IRB_HOME}"
mkdir -p slurm/output

# Optional: isolate HuggingFace datasets cache per array task so concurrent
# jobs on the same node do not race while materializing the same parquet set.
if [[ "${IRB_ISOLATE_DATASETS_CACHE:-false}" == "true" || "${IRB_ISOLATE_DATASETS_CACHE:-0}" == "1" ]]; then
  base_cache="${HF_DATASETS_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/datasets}"
  export HF_DATASETS_CACHE="${base_cache}/ml4h-array/${SLURM_ARRAY_JOB_ID:-0}_${SLURM_ARRAY_TASK_ID:-0}"
  mkdir -p "${HF_DATASETS_CACHE}"
  echo "IRB_ISOLATE_DATASETS_CACHE: HF_DATASETS_CACHE=${HF_DATASETS_CACHE}"
fi

TRAIN_PID=""
TRAIN_SIGNAL_FORWARDING=false

forward_checkpoint_signal() {
  local received_signal="$1"
  echo "[checkpoint] Received Slurm ${received_signal}; requesting a safe checkpoint."
  if [[ "${TRAIN_SIGNAL_FORWARDING}" != true ]]; then
    echo "[checkpoint] Process-group forwarding is unavailable; periodic checkpointing remains active."
  elif [[ -n "${TRAIN_PID}" ]] && kill -0 "${TRAIN_PID}" 2>/dev/null; then
    # Forward USR1 rather than TERM: the wrapper shell remains available to wait
    # for Python, while SignalCheckpointCallback requests a save at a safe step.
    kill -USR1 "-${TRAIN_PID}" 2>/dev/null || kill -USR1 "${TRAIN_PID}" 2>/dev/null || true
  fi
}

# Slurm may deliver its time-limit warning or an opportunistic-preemption TERM
# to this batch shell. Both become a checkpoint request at Python's next safe
# optimizer step. Native Slurm requeue remains enabled for preemption; this
# runner deliberately avoids scontrol inside the container.
trap 'forward_checkpoint_signal USR1' USR1
trap 'forward_checkpoint_signal TERM' TERM

# Enforce one Python process for the allocated GPU.
export IRB_NPROC_PER_NODE="${IRB_NPROC_PER_NODE:-1}"
if [[ "${IRB_NPROC_PER_NODE}" -ne 1 ]]; then
  echo "ERROR: IRB_NPROC_PER_NODE must be 1 for 1-GPU runs (got ${IRB_NPROC_PER_NODE})." >&2
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
echo "SLURM_RESTART_COUNT: ${SLURM_RESTART_COUNT:-0}"
echo "Host: $(hostname)"
echo "=============================================="

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo "GPU info unavailable"

# -----------------------------------------------------------------------------
# Training is intentionally a plain, single-process Python invocation. Do not
# set distributed rendezvous variables or use a distributed launcher.
echo "Execution mode: single process on one GPU (no DDP); 24-hour requeueable allocation"

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

if command -v setsid >/dev/null 2>&1; then
  # Keep the shell that owns the process group alive when it forwards USR1.
  # Python replaces the inherited ignored disposition with its checkpoint handler.
  setsid bash -c 'trap "" USR1; exec bash -lc "$1"' -- "${CMD}" &
  TRAIN_SIGNAL_FORWARDING=true
else
  bash -lc "${CMD}" &
fi
TRAIN_PID=$!

set +e
while true; do
  wait "${TRAIN_PID}"
  TRAIN_STATUS=$?
  if ! kill -0 "${TRAIN_PID}" 2>/dev/null; then
    break
  fi
  echo "[checkpoint] Wait interrupted while training continues; resuming wait."
done
set -e

TRAIN_PID=""
exit "${TRAIN_STATUS}"

