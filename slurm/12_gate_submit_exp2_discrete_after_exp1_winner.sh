#!/bin/bash
#SBATCH --job-name=irb-gate-exp2disc
#SBATCH --output=./slurm/output/%A_%x.stdout
#SBATCH --partition=gpuq
#SBATCH --mem=8GB
#SBATCH --cpus-per-task=1
#SBATCH --time=1-00:00:00

# =============================================================================
# Gate job: submit Exp2 discrete-only after the benchmark Exp1 winner is available
# =============================================================================
#
# Usage:
#   sbatch --dependency=afterok:<EXP1_STAGE1_ARRAY_JOBID> slurm/12_gate_submit_exp2_discrete_after_exp1_winner.sh
#   If outputs/exp1_winner_config_id.txt is absent, this script materializes the
#   current benchmark-pinned reference winner files before submitting Exp2.
# =============================================================================

set -euo pipefail

find_repo_root() {
  local d="$1"
  while [[ "$d" != "/" ]]; do
    if [[ -f "$d/run_experiments.py" && -d "$d/slurm" ]]; then
      echo "$d"
      return 0
    fi
    d="$(dirname "$d")"
  done
  return 1
}

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
IRB_HOME="${IRB_HOME:-$(find_repo_root "${SUBMIT_DIR}")}"
export IRB_CONDA_ENV="${IRB_CONDA_ENV:-input-rep}"
source "${IRB_HOME}/slurm/00_preamble.sh"
cd "${IRB_HOME}"

WINNER_FILE="${IRB_HOME}/outputs/exp1_winner_config_id.txt"
mkdir -p "${IRB_HOME}/outputs"

if [[ ! -s "${WINNER_FILE}" ]]; then
  echo "[gate] Winner file missing; writing benchmark-pinned reference winners."
  python "${IRB_HOME}/scripts/write_reference_winner_files.py" \
    --output_dir "${IRB_HOME}/outputs"
fi

if [[ ! -s "${WINNER_FILE}" ]]; then
  echo "ERROR: Exp1 winner config_id file not found/non-empty after reference write: ${WINNER_FILE}" >&2
  exit 1
fi

EXP1_WINNER_CONFIG_ID="$(head -n 1 "${WINNER_FILE}" | tr -d '[:space:]')"
if [[ -z "${EXP1_WINNER_CONFIG_ID}" ]]; then
  echo "ERROR: Winner config_id was empty in ${WINNER_FILE}" >&2
  exit 1
fi

echo "[gate] Using Exp1 winner config_id: ${EXP1_WINNER_CONFIG_ID}"
echo "[gate] Generating Exp2 discrete-only jobfiles..."

python run_experiments.py \
  --mode demo \
  --exp 2 \
  --exp2_include_discrete \
  --exp2_discrete_only \
  --exp2_from_exp1_config_id "${EXP1_WINNER_CONFIG_ID}"

EXP2_S0_JOBFILE="slurm/generated/demo/08_exp2_stage0_tokenize.jobfile"
EXP2_S1_JOBFILE="slurm/generated/demo/10a_exp2_stage1_train.jobfile"
EXP2_S2_JOBFILE="slurm/generated/demo/10b_exp2_stage2_extract_reps.jobfile"
EXP2_S3_JOBFILE="slurm/generated/demo/10c_exp2_stage3_lr.jobfile"

N_S0=$(grep -v '^#' "${EXP2_S0_JOBFILE}" | grep -v '^[[:space:]]*$' | wc -l | tr -d ' ')
N_S1=$(grep -v '^#' "${EXP2_S1_JOBFILE}" | grep -v '^[[:space:]]*$' | wc -l | tr -d ' ')
N_S2=$(grep -v '^#' "${EXP2_S2_JOBFILE}" | grep -v '^[[:space:]]*$' | wc -l | tr -d ' ')
N_S3=$(grep -v '^#' "${EXP2_S3_JOBFILE}" | grep -v '^[[:space:]]*$' | wc -l | tr -d ' ')

if [[ "${N_S1}" -ne 2 ]]; then
  echo "ERROR: Expected exactly 2 Exp2 discrete-only Stage1 commands, got ${N_S1}." >&2
  exit 1
fi
if [[ "${N_S2}" -ne "${N_S1}" || "${N_S3}" -ne "${N_S1}" ]]; then
  echo "ERROR: Expected Stage2/Stage3 command counts to match Stage1 (${N_S1}), got Stage2=${N_S2}, Stage3=${N_S3}." >&2
  exit 1
fi

S0_DEP=""
if [[ "${N_S0}" -gt 0 ]]; then
  echo "[gate] Submitting Exp2 Stage0 (tokenize/outcomes) array on tier2q..."
  JID_S0=$(sbatch --parsable --time=1-00:00:00 --array=0-$((N_S0-1)) slurm/02_run_stage0_tier2q_tokenize.sh "${EXP2_S0_JOBFILE}")
  echo "[gate] Submitted Exp2 Stage0 job: ${JID_S0}"
  S0_DEP="--dependency=afterok:${JID_S0}"
else
  echo "[gate] No Stage0 work needed; tokenization/outcomes already present."
fi

echo "[gate] Submitting Exp2 Stage1 (discrete-only) array..."
JID_S1=$(sbatch --parsable ${S0_DEP} --array=0-$((N_S1-1)) slurm/05_run_stage1_gpu4_train.sh "${EXP2_S1_JOBFILE}")
echo "[gate] Submitted Exp2 Stage1 discrete-only job: ${JID_S1}"

echo "[gate] Submitting Exp2 Stage2 (extract reps; 1 GPU/job) array..."
JID_S2=$(sbatch --parsable --dependency=afterok:"${JID_S1}" --array=0-$((N_S2-1)) slurm/09_run_stage2_gpu2_extract.sh "${EXP2_S2_JOBFILE}")
echo "[gate] Submitted Exp2 Stage2 job: ${JID_S2}"

echo "[gate] Submitting Exp2 Stage3 (LR; CPU tier2q) array..."
JID_S3=$(sbatch --parsable --dependency=afterok:"${JID_S2}" --array=0-$((N_S3-1)) slurm/11_run_stage3_tier2q_lr.sh "${EXP2_S3_JOBFILE}")
echo "[gate] Submitted Exp2 Stage3 job: ${JID_S3}"

echo "[gate] Done."

