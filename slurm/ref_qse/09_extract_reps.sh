#!/bin/bash

# =============================================================================
# Provenance: Quantifying-Surprise-EHRs/slurm/09_extract_reps.sh
# Source: /gpfs/data/bbj-lab/users/daniel/Quantifying-Surprise-EHRs/slurm/09_extract_reps.sh
# =============================================================================
#
# Vendored to preserve extraction semantics while sharding inference across GPUs.
#
# - jobfile sets: data_dir, data_version, model_loc
# - remove the (mimic/ucmc × method × pct) grid; our grid is config×seed
#
# =============================================================================

#SBATCH --job-name=extract-states
#SBATCH --output=./slurm/output/%A_%a-%x.stdout
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:4
#SBATCH --time=1-00:00:00

set -euo pipefail

source slurm/ref_qse/preamble.sh

: "${FMS_EHRS_HOME:?Set FMS_EHRS_HOME (sibling fms-ehrs repo)}"

# Expect these from the jobfile command
: "${data_dir:?Set data_dir (tokenized data root)}"
: "${data_version:?Set data_version (should end with _first_24h)}"
: "${model_loc:?Set model_loc (HF model directory)}"

echo "Extracting representations..."
IRB_STAGE2_NPROC_PER_NODE="${IRB_STAGE2_NPROC_PER_NODE:-4}"
IRB_STAGE2_BATCH_SZ="${IRB_STAGE2_BATCH_SZ:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
if [[ -n "${IRB_STAGE2_CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${IRB_STAGE2_CUDA_VISIBLE_DEVICES}"
elif [[ "${SLURM_JOB_PARTITION:-}" == "sxmq" && "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3,4,5,6,7" ]]; then
  # GPUs 0-3 on cri22cn501 currently carry orphan memory despite Slurm reporting
  # the node idle. Use the clean SXM A100s while still reserving the full node.
  export CUDA_VISIBLE_DEVICES="4,5,6,7"
fi
echo "torchrun CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-auto}"

torchrun --nproc_per_node="${IRB_STAGE2_NPROC_PER_NODE}" \
  --rdzv_backend c10d \
  --rdzv-id "${SLURM_ARRAY_TASK_ID:-0}" \
  --rdzv-endpoint=localhost:0 \
  "${FMS_EHRS_HOME}/fms_ehrs/scripts/extract_hidden_states.py" \
  --data_dir "${data_dir}" \
  --data_version "${data_version}" \
  --model_loc "${model_loc}" \
  --batch_sz "${IRB_STAGE2_BATCH_SZ}"

