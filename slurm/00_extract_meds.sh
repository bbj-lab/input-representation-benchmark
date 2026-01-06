#!/bin/bash
#SBATCH --job-name=meds-extract
#SBATCH --output=./slurm/output/%j-%x.stdout
#SBATCH --partition=sxmq
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=150GB

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

# Source environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export IRB_CONDA_ENV="meds-extract"
source "${SCRIPT_DIR}/preamble.sh"

# Create output directory
mkdir -p "${SCRIPT_DIR}/output"

echo "=============================================="
echo "MEDS Extraction Job"
echo "=============================================="
echo "  Job ID: ${SLURM_JOB_ID}"
echo "  Host: $(hostname)"
echo "  Date: $(date)"
echo "=============================================="

# Run the extraction script
cd "${IRB_HOME}"
bash benchmarks/mimic-meds-extraction/scripts/01_extract_meds_full.sh 8

echo ""
echo "=============================================="
echo "MEDS Extraction Complete!"
echo "=============================================="
echo "Output: ${IRB_HOME}/benchmarks/mimic-meds-extraction/data/meds"
echo ""
echo "Next steps:"
echo "  1. Generate job files: python run_experiments.py --mode demo"
echo "  2. Submit Exp1: sbatch --array=0-11%8 slurm/run_from_jobfile.sh slurm/exp1_demo_jobs.sh"
