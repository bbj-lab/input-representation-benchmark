#!/bin/bash
# Submit ML4H Exp3 Stage-2 representation extraction across H200 + RTX 6000 Pro.
#
# Shard tuned to live free capacity at submit time (Yang skipped while blocked).
# H200 omits broken porrasq/cri22cn424. BBJ uses priority QOS with %4 throttle.

set -euo pipefail

IRB_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${IRB_HOME}"

STAGE_DIR="outputs/runs/ml4h_2026/stage2_exp3"
LOG_DIR="${STAGE_DIR}/logs"
LAUNCH_STATE="${STAGE_DIR}/exp3_stage2_launch.tsv"

H200_COUNT="${H200_COUNT:-8}"
RTX_ZHOU_COUNT="${RTX_ZHOU_COUNT:-1}"
RTX_BBJ_COUNT="${RTX_BBJ_COUNT:-3}"
RTX_YANG_COUNT="${RTX_YANG_COUNT:-0}"

python utilities/scripts/generate_ml4h_exp3_stage2_jobfiles.py \
  --output-dir "${STAGE_DIR}" \
  --h200-count "${H200_COUNT}" \
  --rtx-zhou-count "${RTX_ZHOU_COUNT}" \
  --rtx-bbj-count "${RTX_BBJ_COUNT}" \
  --rtx-yang-count "${RTX_YANG_COUNT}"

mkdir -p "${LOG_DIR}"
: > "${LAUNCH_STATE}"

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
    echo "SKIP ${route}: empty jobfile"
    return 0
  fi

  local array_spec="0-$((count - 1))"
  if [[ -n "${concurrency}" ]]; then
    array_spec="${array_spec}%${concurrency}"
  fi

  local args=(
    --parsable
    --job-name="ml4h-e3x-${route}"
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
  "${STAGE_DIR}/exp3_extract_h200.jobfile"
submit_route \
  rtx-zhou \
  zhoulabq \
  opportunistic \
  gpu:nvidia_rtxpro6000-96gb:1 \
  "${STAGE_DIR}/exp3_extract_rtx_zhou.jobfile" \
  cri22cn416
submit_route \
  rtx-bbj \
  bbj-wanq \
  bbj-wan_priority \
  gpu:nvidia_rtxpro6000-96gb:1 \
  "${STAGE_DIR}/exp3_extract_rtx_bbj.jobfile" \
  cri22cn426 \
  4
if [[ "${RTX_YANG_COUNT}" -gt 0 ]]; then
  submit_route \
    rtx-yang \
    yangq \
    opportunistic \
    gpu:nvidia_rtxpro6000-96gb:1 \
    "${STAGE_DIR}/exp3_extract_rtx_yang.jobfile" \
    cri22cn425
fi
