#!/bin/bash
#SBATCH --job-name=irb-ml4h-24to48-probes
#SBATCH --output=./slurm/output/%A-%x.stdout
#SBATCH --partition=tier1q,tier2q
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

# Stage 3: submit the 24-48h linear probe arrays, then the metric table build.
#
# 1. Write the binary and regression jobfiles for each experiment.
# 2. Submit one probe array for each experiment and task family.
# 3. Submit slurm/23_build_ml4h_metric_tables.sh after all probe arrays complete.
#    That script rebuilds the tables for all three experiments.
#
# Usage:
#   sbatch slurm/22_submit_ml4h_24to48h_probes.sh               # all experiments
#   sbatch slurm/22_submit_ml4h_24to48h_probes.sh exp2 exp3     # subset

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

if [[ "$#" -gt 0 ]]; then
  EXPERIMENTS=("$@")
else
  EXPERIMENTS=(exp1 exp2 exp3)
fi

STAGE_ROOT="outputs/runs/ml4h_2026"
LOG_DIR="${STAGE_ROOT}/probe_logs"
mkdir -p "${LOG_DIR}"
LAUNCH_STATE="${LOG_DIR}/probe_launch.tsv"
printf 'experiment\tfamily\tjob_id\ttasks\tjobfile\n' > "${LAUNCH_STATE}"

PARTITIONS="tier1q,tier2q,tier3q"
CPUS_PER_TASK=32
MEMORY_PER_TASK=96G

expected_runs() {
  case "$1" in
    exp1|exp2) echo 72 ;;
    exp3) echo 12 ;;
    *) echo "ERROR: unknown experiment $1" >&2; exit 2 ;;
  esac
}

probe_jobs=()
submit_family() {
  local experiment="$1"
  local family="$2"
  local expected_count="$3"
  local jobfile="${STAGE_ROOT}/stage3_24to48h_${experiment}/${experiment}_24to48h_${family}.jobfile"
  local count
  count="$(grep -cvE '^(#|[[:space:]]*$)' "${jobfile}")"
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
  probe_jobs+=("${job_id}")
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "${experiment}" "${family}" "${job_id}" "${count}" "${jobfile}" \
    | tee -a "${LAUNCH_STATE}"
}

for experiment in "${EXPERIMENTS[@]}"; do
  python utilities/scripts/generate_ml4h_24to48h_stage3_jobfiles.py \
    --experiment "${experiment}"
  runs="$(expected_runs "${experiment}")"
  submit_family "${experiment}" binary "${runs}"
  submit_family "${experiment}" regression "${runs}"
done

dependency="$(IFS=':'; echo "${probe_jobs[*]}")"
metric_job="$(
  sbatch --parsable --dependency="afterok:${dependency}" \
    slurm/23_build_ml4h_metric_tables.sh
)"
printf 'all\tmetric_tables\t%s\t1\tslurm/23_build_ml4h_metric_tables.sh\n' "${metric_job}" \
  | tee -a "${LAUNCH_STATE}"
