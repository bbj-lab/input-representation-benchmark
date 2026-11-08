#!/bin/bash
#SBATCH --job-name=irb-ml4h-calib2
#SBATCH --output=./slurm/output/%j-%x.stdout
#SBATCH --partition=bbj-wanq
#SBATCH --qos=bbj-wan_priority
#SBATCH --nodelist=cri22cn426
#SBATCH --gres=gpu:nvidia_rtxpro6000-96gb:1
#SBATCH --mem=128GB
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00

# =============================================================================
# Follow-up to the batch-geometry calibration.
#
# Pass 1 extends the microbatch sweep to 16 to confirm where throughput stops
# improving. Pass 2 reruns only the clipping comparison for 1000 steps, since
# the original 40-step comparison is too short to separate loss curves.
#
# Both passes reuse the existing mapped dataset caches and CPU diagnostics.
# =============================================================================

set -euo pipefail

# sbatch runs a spool copy of this file, so resolve the repo from the submit
# directory rather than from BASH_SOURCE.
IRB_HOME="${IRB_HOME:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
if [[ ! -f "${IRB_HOME}/slurm/09_ml4h_batch_geometry_calibration.sh" ]]; then
  echo "ERROR: could not locate repo root (tried ${IRB_HOME})" >&2
  exit 1
fi
cd "${IRB_HOME}"

echo "=== Pass 1: microbatch sweep through 16 ==="
IRB_CALIB_SKIP_DIAGNOSTICS=1 \
IRB_CALIB_BATCH_SIZES="1 2 4 8 16" \
IRB_CALIB_PROBE_STEPS=20 \
IRB_CALIB_CLIP_STEPS=40 \
  bash slurm/09_ml4h_batch_geometry_calibration.sh

echo "=== Pass 2: 1000-step clipping comparison ==="
IRB_CALIB_SKIP_DIAGNOSTICS=1 \
IRB_CALIB_CLIP_ONLY=1 \
IRB_CALIB_CLIP_VALUES="1.0 32.0" \
IRB_CALIB_CLIP_STEPS=1000 \
  bash slurm/09_ml4h_batch_geometry_calibration.sh
