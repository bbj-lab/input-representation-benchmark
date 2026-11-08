#!/bin/bash
# Submit ML4H Exp2 Stage-2 representation extraction across H200 + RTX 6000 Pro.
#
# Shard: 48 H200 / 8 Zhou RTX / 8 BBJ RTX / 8 Yang RTX.
# H200 omits broken porrasq/cri22cn424. BBJ uses priority QOS with %4 throttle.
# Opportunistic routes are requeueable; completed splits are skipped on resume.

set -euo pipefail

IRB_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${IRB_HOME}"

STAGE_DIR="outputs/runs/ml4h_2026/stage2_exp2"
LOG_DIR="${STAGE_DIR}/logs"
LAUNCH_STATE="${STAGE_DIR}/exp2_stage2_launch.tsv"

python utilities/scripts/generate_ml4h_exp2_stage2_jobfiles.py \
  --output-dir "${STAGE_DIR}" \
  --h200-count 48 \
  --rtx-zhou-count 8 \
  --rtx-bbj-count 8 \
  --rtx-yang-count 8

mkdir -p "${LOG_DIR}" "${STAGE_DIR}/submissions"
touch "${LAUNCH_STATE}"

submit_route() {
  local route="$1"
  local partition="$2"
  local qos="$3"
  local gres="$4"
  local jobfile="$5"
  local nodelist="${6:-}"
  local concurrency="${7:-}"
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
    --job-name="ml4h-e2x-${route}"
    --partition="${partition}"
    --qos="${qos}"
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
  printf '%s\t%s\t%s\t%s\n' "${route}" "${job_id}" "${count}" "${jobfile}" | tee -a "${LAUNCH_STATE}"
}

echo -e "route\tjob_id\ttasks\tjobfile"
submit_route \
  h200 \
  catherineq,siweiq,zhoulabq,gpuq \
  opportunistic \
  gpu:nvidia_h200-141gb:1 \
  "${STAGE_DIR}/exp2_extract_h200.jobfile"
submit_route \
  rtx-zhou \
  zhoulabq \
  opportunistic \
  gpu:nvidia_rtxpro6000-96gb:1 \
  "${STAGE_DIR}/exp2_extract_rtx_zhou.jobfile" \
  cri22cn416
submit_route \
  rtx-bbj \
  bbj-wanq \
  bbj-wan_priority \
  gpu:nvidia_rtxpro6000-96gb:1 \
  "${STAGE_DIR}/exp2_extract_rtx_bbj.jobfile" \
  cri22cn426 \
  4
submit_route \
  rtx-yang \
  yangq \
  opportunistic \
  gpu:nvidia_rtxpro6000-96gb:1 \
  "${STAGE_DIR}/exp2_extract_rtx_yang.jobfile" \
  cri22cn425
