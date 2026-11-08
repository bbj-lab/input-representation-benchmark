#!/bin/bash
# Submit the completed ML4H Exp1 models for one-GPU representation extraction.
#
# Current live-capacity shard: 46 H200 tasks, 6 Zhou RTX tasks, 12 BBJ RTX
# tasks, and 8 Yang RTX tasks.
# Every route is intentionally opportunistic and therefore requeueable.

set -euo pipefail

IRB_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${IRB_HOME}"

STAGE_DIR="outputs/runs/ml4h_2026/stage2"
LOG_DIR="${STAGE_DIR}/logs"

python utilities/scripts/generate_ml4h_exp1_stage2_jobfiles.py \
  --output-dir "${STAGE_DIR}" \
  --h200-count 46 \
  --rtx-zhou-count 6 \
  --rtx-bbj-count 12 \
  --rtx-yang-count 8

mkdir -p "${LOG_DIR}"

submit_route() {
  local route="$1"
  local partition="$2"
  local gres="$3"
  local jobfile="$4"
  local nodelist="${5:-}"
  local concurrency="${6:-}"
  local count
  count="$(rg -v '^(#|[[:space:]]*$)' "${jobfile}" | wc -l)"
  if [[ "${count}" -eq 0 ]]; then
    echo "ERROR: ${route} jobfile has no tasks: ${jobfile}" >&2
    exit 2
  fi

  local array_spec="0-$((count - 1))"
  if [[ -n "${concurrency}" ]]; then
    array_spec="${array_spec}%${concurrency}"
  fi

  local args=(
    --parsable
    --job-name="ml4h-e1x-${route}"
    --partition="${partition}"
    --qos=opportunistic
    --gres="${gres}"
    --nodes=1
    --cpus-per-task=4
    --mem=64G
    --time=12:00:00
    --requeue
    --signal=B:TERM@600
    --open-mode=append
    --output="${LOG_DIR}/%A_%a-%x.out"
    --export=ALL,IRB_STAGE2_NPROC_PER_NODE=1,IRB_STAGE2_BATCH_SZ=8,IRB_ISOLATE_DATASETS_CACHE=true
    --array="${array_spec}"
  )
  if [[ -n "${nodelist}" ]]; then
    args+=(--nodelist="${nodelist}")
  fi

  local job_id
  job_id="$(sbatch "${args[@]}" slurm/09_run_stage2_gpu2_extract.sh "${jobfile}")"
  printf '%s\t%s\t%s\t%s\n' "${route}" "${job_id}" "${count}" "${jobfile}"
}

echo -e "route\tjob_id\ttasks\tjobfile"
submit_route \
  h200 \
  catherineq,siweiq,zhoulabq,gpuq,porrasq \
  gpu:nvidia_h200-141gb:1 \
  "${STAGE_DIR}/exp1_extract_h200.jobfile"
submit_route \
  rtx-zhou \
  zhoulabq \
  gpu:nvidia_rtxpro6000-96gb:1 \
  "${STAGE_DIR}/exp1_extract_rtx_zhou.jobfile" \
  cri22cn416
submit_route \
  rtx-bbj \
  bbj-wanq \
  gpu:nvidia_rtxpro6000-96gb:1 \
  "${STAGE_DIR}/exp1_extract_rtx_bbj.jobfile" \
  cri22cn426 \
  4
submit_route \
  rtx-yang \
  yangq \
  gpu:nvidia_rtxpro6000-96gb:1 \
  "${STAGE_DIR}/exp1_extract_rtx_yang.jobfile" \
  cri22cn425
