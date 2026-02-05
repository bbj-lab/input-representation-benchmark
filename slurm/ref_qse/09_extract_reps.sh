#!/bin/bash

# =============================================================================
# Provenance: Quantifying-Surprise-EHRs/slurm/09_extract_reps.sh
# Source: /gpfs/data/bbj-lab/users/daniel/Quantifying-Surprise-EHRs/slurm/09_extract_reps.sh
# =============================================================================
#
# Vendored to preserve the exact torchrun extraction pattern.
#
# - jobfile sets: data_dir, data_version, model_loc
# - remove the (mimic/ucmc × method × pct) grid; our grid is config×seed
#
# =============================================================================

#SBATCH --job-name=extract-states
#SBATCH --output=./slurm/output/%A_%a-%x.stdout
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00

set -euo pipefail

source slurm/ref_qse/preamble.sh

: "${FMS_EHRS_HOME:?Set FMS_EHRS_HOME (sibling fms-ehrs repo)}"

# Expect these from the jobfile command
: "${data_dir:?Set data_dir (tokenized data root)}"
: "${data_version:?Set data_version (should end with _first_24h)}"
: "${model_loc:?Set model_loc (HF model directory)}"

echo "Extracting representations..."
torchrun --nproc_per_node=1 \
  --rdzv_backend c10d \
  --rdzv-id "${SLURM_ARRAY_TASK_ID:-0}" \
  --rdzv-endpoint=localhost:0 \
  "${FMS_EHRS_HOME}/fms_ehrs/scripts/extract_hidden_states.py" \
  --data_dir "${data_dir}" \
  --data_version "${data_version}" \
  --model_loc "${model_loc}" \
  --batch_sz "$((2 ** 5))"

