#!/bin/bash
#SBATCH --job-name=irb-ml4h-24to48-prepare
#SBATCH --output=./slurm/output/%A-%x.stdout
#SBATCH --partition=tier2q
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

# After the Exp1 feature repair, submit its isolated provisional probe arrays
# and chain the final bundle behind those arrays plus Exp2/Exp3 probes.

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

STAGE_ROOT="outputs/runs/ml4h_2026"
PROVISIONAL_ROOT="${STAGE_ROOT}/provisional_24to48h"
LOG_DIR="${PROVISIONAL_ROOT}/logs"
mkdir -p "${LOG_DIR}"

python utilities/scripts/generate_ml4h_24to48h_stage3_jobfiles.py \
  --experiment exp1

LAUNCH_STATE="${PROVISIONAL_ROOT}/exp1_probe_launch.tsv"
printf 'experiment\tfamily\tjob_id\ttasks\tjobfile\n' > "${LAUNCH_STATE}"
PARTITIONS="tier1q,tier2q,tier3q"
CPUS_PER_TASK=32
MEMORY_PER_TASK=96G

submit_family() {
  local experiment="$1"
  local family="$2"
  local expected_count="$3"
  local stage_dir="${STAGE_ROOT}/stage3_24to48h_${experiment}"
  local jobfile="${stage_dir}/${experiment}_24to48h_${family}.jobfile"
  local count
  count="$(rg -v '^(#|[[:space:]]*$)' "${jobfile}" | wc -l)"
  if [[ "${count}" -ne "${expected_count}" ]]; then
    echo "ERROR: ${experiment}/${family} expected ${expected_count} tasks; found ${count}" >&2
    exit 2
  fi
  local job_id
  job_id="$(
    sbatch \
      --parsable \
      --job-name="ml4h-${experiment}-24to48-${family}" \
      --partition="${PARTITIONS}" \
      --nodes=1 \
      --ntasks=1 \
      --cpus-per-task="${CPUS_PER_TASK}" \
      --mem="${MEMORY_PER_TASK}" \
      --time=2-00:00:00 \
      --open-mode=append \
      --output="${LOG_DIR}/%A_%a-%x.out" \
      --export="ALL,IRB_STAGE3_BOOTSTRAP_N=0,OMP_NUM_THREADS=${CPUS_PER_TASK},OPENBLAS_NUM_THREADS=${CPUS_PER_TASK},MKL_NUM_THREADS=${CPUS_PER_TASK},NUMEXPR_MAX_THREADS=${CPUS_PER_TASK}" \
      --array="0-$((count - 1))" \
      slurm/11_run_stage3_tier2q_lr.sh "${jobfile}"
  )"
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "${experiment}" "${family}" "${job_id}" "${count}" "${jobfile}" \
    | tee -a "${LAUNCH_STATE}"
}

submit_family exp1 binary 72
submit_family exp1 regression 72

early_launch="${PROVISIONAL_ROOT}/exp2_exp3_probe_launch.tsv"
if [[ ! -f "${early_launch}" ]]; then
  echo "ERROR: missing completed/provisional Exp2/Exp3 launch state: ${early_launch}" >&2
  exit 2
fi

primary_launch="${STAGE_ROOT}/stage3_padded_repair/exp1_stage3_launch.tsv"
if [[ ! -f "${primary_launch}" ]]; then
  echo "ERROR: missing repaired Exp1 Stage-3 launch state: ${primary_launch}" >&2
  exit 2
fi
primary_figure_job="$(awk -F $'\t' '$1 == "figure_bundle" {print $2}' "${primary_launch}")"
if [[ -z "${primary_figure_job}" ]]; then
  echo "ERROR: repaired Exp1 figure job is missing from ${primary_launch}" >&2
  exit 2
fi

job_controller_state() {
  local job_id="$1"
  local state
  state="$(
    scontrol show job "${job_id}" 2>/dev/null \
      | tr ' ' '\n' \
      | awk -F= '/^JobState=/ {print $2; exit}'
  )"
  if [[ -n "${state}" ]]; then
    printf '%s\n' "${state}"
    return 0
  fi
  sacct -j "${job_id}" -n -X -P -o State 2>/dev/null | head -1 | sed 's/+.*//'
}

collect_afterok_jobs() {
  local job_id state
  local -a active=()
  for job_id in "$@"; do
    [[ -z "${job_id}" ]] && continue
    state="$(job_controller_state "${job_id}")"
    state="${state%%+*}"
    case "${state}" in
      COMPLETED)
        continue
        ;;
      FAILED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|CANCELLED|BOOT_FAIL|DEADLINE|PREEMPTED)
        echo "ERROR: prerequisite ${job_id} ended in ${state}" >&2
        exit 2
        ;;
      "")
        echo "WARNING: ${job_id} is no longer visible; treating it as already finished." >&2
        continue
        ;;
      *)
        active+=("${job_id}")
        ;;
    esac
  done
  if [[ "${#active[@]}" -eq 0 ]]; then
    return 0
  fi
  local IFS=':'
  printf '%s\n' "${active[*]}"
}

probe_job_ids=(
  $(awk -F $'\t' 'FNR > 1 {print $3}' "${early_launch}" "${LAUNCH_STATE}")
)
afterok_jobs="$(collect_afterok_jobs "${probe_job_ids[@]}" "${primary_figure_job}")"
figure_sbatch=(sbatch --parsable)
if [[ -n "${afterok_jobs}" ]]; then
  figure_sbatch+=(--dependency="afterok:${afterok_jobs}")
fi
figure_job="$("${figure_sbatch[@]}" slurm/23_generate_ml4h_24to48h_provisional.sh)"
printf 'bundle\tfigures\t%s\t9\tslurm/23_generate_ml4h_24to48h_provisional.sh\n' "${figure_job}" \
  | tee -a "${LAUNCH_STATE}"
echo "Submitted provisional bundle build: ${figure_job}"
