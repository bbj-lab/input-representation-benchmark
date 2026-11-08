#!/bin/bash
#SBATCH --job-name=irb-ml4h-clip-adamw
#SBATCH --output=./slurm/output/%j-%x.stdout
#SBATCH --partition=bbj-wanq
#SBATCH --qos=bbj-wan_priority
#SBATCH --nodelist=cri22cn426
#SBATCH --gres=gpu:nvidia_rtxpro6000-96gb:1
#SBATCH --mem=128GB
#SBATCH --cpus-per-task=8
#SBATCH --time=00:45:00

# =============================================================================
# AdamW clipping comparison at the selected microbatch (batch 4, 1000 steps).
# Replaces the earlier Muon-era clip checks that ran on a frozen model.
# =============================================================================

set -euo pipefail
IRB_HOME="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${IRB_HOME}"
source "${IRB_HOME}/slurm/00_preamble.sh"

IRB_CALIB_SKIP_DIAGNOSTICS=1 \
IRB_CALIB_CLIP_ONLY=1 \
IRB_CALIB_CLIP_VALUES="1.0 32.0" \
IRB_CALIB_CLIP_STEPS=1000 \
IRB_CALIB_CLIP_BATCH=4 \
  bash slurm/09_ml4h_batch_geometry_calibration.sh
