#!/bin/bash
#SBATCH --job-name=meds-extract
#SBATCH --output=./slurm/output/%j-%x.stdout
#SBATCH --partition=gpuq
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=300GB

# =============================================================================
# Phase 0: MEDS Data Extraction from MIMIC-IV
# =============================================================================
# This job extracts MIMIC-IV data into MEDS format.
# Run this BEFORE any training experiments.
#
# The MEDS extraction pipeline is adapted from ETHOS-ARES:
#   Repository: https://github.com/ipolharvard/ethos-ares
#   Citation: Renc, P., et al. (2025). GigaScience, 14, giaf107.
#   License: MIT (Copyright (c) 2024 Paweł Renc)
#
# Usage:
#   sbatch slurm/00_extract_meds.sh
#
# Prerequisites:
#   1. MIMIC-IV data downloaded to physionet.org/files/mimiciv/3.1/
#   2. Python environment with MEDS_transforms installed
# =============================================================================

set -euo pipefail

# Source environment (robust to SLURM script spooling into /var/spool/...)
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
IRB_HOME="$(find_repo_root "${SUBMIT_DIR}")"
export IRB_CONDA_ENV="meds-extract"
source "${IRB_HOME}/slurm/preamble.sh"

# Create output directory
mkdir -p "${IRB_HOME}/slurm/output"

echo "=============================================="
echo "MEDS Extraction Job"
echo "=============================================="
echo "  Job ID: ${SLURM_JOB_ID}"
echo "  Host: $(hostname)"
echo "  Date: $(date)"
echo "=============================================="

# Run the extraction script
cd "${IRB_HOME}"
# NOTE: MEDS_transforms worker parallelism can be very memory-hungry.
# Default to a conservative worker count unless overridden at submit time:
#   sbatch --export=ALL,N_WORKERS=4 slurm/00_extract_meds.sh
# Empirically, N_WORKERS=3 can exceed a 300GB job memory limit during
# `stage=shard_events` (e.g., while sharding `icu/chartevents.csv.gz`).
N_WORKERS="${N_WORKERS:-2}"
echo "Using N_WORKERS=${N_WORKERS}"
bash benchmarks/mimic-meds-extraction/scripts/01_extract_meds_full.sh "${N_WORKERS}"

# =============================================================================
# Post-step: Create tokenization input layout shim (raw/{train,val,test})
# =============================================================================
# `fms_ehrs/scripts/tokenize_w_config.py` expects split subdirectories under:
#   <data_dir>/<data_version_in>/{train,val,test}
# Our MEDS pipeline produces:
#   ${IRB_HOME}/benchmarks/mimic-meds-extraction/data/meds/data/{train,tuning,test}
# So we create:
#   .../data/raw/train -> ../train
#   .../data/raw/val   -> ../tuning   (or ../val if it already exists)
#   .../data/raw/test  -> ../test
MEDS_DATA_BASE="${IRB_HOME}/benchmarks/mimic-meds-extraction/data/meds/data"
RAW_BASE="${MEDS_DATA_BASE}/raw"

if [[ ! -d "${MEDS_DATA_BASE}" ]]; then
    echo "ERROR: Expected MEDS data directory not found: ${MEDS_DATA_BASE}"
    exit 1
fi
if [[ ! -d "${MEDS_DATA_BASE}/train" ]]; then
    echo "ERROR: Missing expected split dir: ${MEDS_DATA_BASE}/train"
    exit 1
fi
if [[ ! -d "${MEDS_DATA_BASE}/test" ]]; then
    echo "ERROR: Missing expected split dir: ${MEDS_DATA_BASE}/test"
    exit 1
fi

VAL_SRC=""
if [[ -d "${MEDS_DATA_BASE}/val" ]]; then
    VAL_SRC="val"
elif [[ -d "${MEDS_DATA_BASE}/tuning" ]]; then
    VAL_SRC="tuning"
else
    echo "ERROR: Missing validation split dir. Expected one of:"
    echo "  - ${MEDS_DATA_BASE}/val"
    echo "  - ${MEDS_DATA_BASE}/tuning"
    exit 1
fi

mkdir -p "${RAW_BASE}"
(
    cd "${RAW_BASE}"
    [[ -e train ]] || ln -s ../train train
    [[ -e test ]] || ln -s ../test test
    [[ -e val ]] || ln -s "../${VAL_SRC}" val
)

echo ""
echo "=============================================="
echo "Tokenization input layout shim created"
echo "=============================================="
echo "  Base: ${MEDS_DATA_BASE}"
echo "  raw/train -> ../train"
echo "  raw/val   -> ../${VAL_SRC}"
echo "  raw/test  -> ../test"

echo ""
echo "=============================================="
echo "MEDS Extraction Complete!"
echo "=============================================="
echo "Output: ${IRB_HOME}/benchmarks/mimic-meds-extraction/data/meds"
echo ""
echo "Next steps:"
echo "  1. Generate job files: python run_experiments.py --mode demo"
echo "  2. Exp1 Stage 0 (CPU): sbatch --array=0-11%2 slurm/02_run_from_jobfile_cpu.sh slurm/04_exp1_tokenize_jobs.sh"
echo "  3. Exp1 Stage 1 (GPU): sbatch --array=0-11%8 slurm/run_from_jobfile.sh slurm/07_exp1_train_jobs_demo.sh"
echo "  4. Exp2 Stage 0 (CPU): sbatch --array=0-1%2  slurm/02_run_from_jobfile_cpu.sh slurm/08_exp2_tokenize_jobs.sh"
echo "  5. Exp2 Stage 1 (GPU): sbatch --array=0-5%8  slurm/run_from_jobfile.sh slurm/10_exp2_train_jobs_demo.sh"
echo "  6. Exp3 Stage 0 (CPU): sbatch --array=0-1%2  slurm/02_run_from_jobfile_cpu.sh slurm/12_exp3_tokenize_jobs.sh"
echo "  7. Exp3 Stage 1 (GPU): sbatch --array=0-1%8  slurm/run_from_jobfile.sh slurm/14_exp3_train_jobs_demo.sh"