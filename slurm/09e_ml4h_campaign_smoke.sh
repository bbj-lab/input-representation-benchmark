#!/bin/bash
#SBATCH --job-name=irb-ml4h-smoke
#SBATCH --output=./slurm/output/%j-%x.stdout
#SBATCH --partition=bbj-wanq
#SBATCH --qos=bbj-wan_priority
#SBATCH --nodelist=cri22cn426
#SBATCH --gres=gpu:nvidia_rtxpro6000-96gb:1
#SBATCH --mem=128GB
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00

# =============================================================================
# Smoke-test one Exp2 campaign command under the AdamW contract (50 steps).
# =============================================================================

set -euo pipefail
IRB_HOME="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${IRB_HOME}"
source "${IRB_HOME}/slurm/00_preamble.sh"

export IRB_REQUIRE_WANDB=false
export WANDB_MODE=offline
export WANDB_LOG_MODEL=false
export MODEL_DIR=outputs/runs/ml4h_2026/smoke/models
mkdir -p "${MODEL_DIR}"

# Llama discrete time_rope line from the regenerated Exp2 jobfile, rewritten for
# a fast 50-step smoke: point at the smoke model dir, disable eval/save/best so
# the run does not require a full-epoch checkpoint, and cap steps.
rg "ml4h_llama_exp2 .*--representation discrete --temporal time_rope --num_bins 10 --seed 42$" \
  slurm/generated/ml4h_2026/03_exp2_stage1.jobfile | head -n 1 | \
  sed 's|MODEL_DIR=outputs/runs/ml4h_2026/models|MODEL_DIR='"${MODEL_DIR}"' IRB_STAGE1_MAX_STEPS=50 IRB_STAGE1_LOGGING_STEPS=5|' | \
  sed 's|IRB_STAGE1_EVALUATIONS_PER_EPOCH=2|IRB_STAGE1_EVALUATIONS_PER_EPOCH=1|' | \
  sed 's|IRB_STAGE1_SAVE_STRATEGY=epoch|IRB_STAGE1_SAVE_STRATEGY=no|' | \
  sed 's|IRB_STAGE1_EVAL_STRATEGY=epoch|IRB_STAGE1_EVAL_STRATEGY=no|' | \
  sed 's|IRB_LOAD_BEST_MODEL_AT_END=true|IRB_LOAD_BEST_MODEL_AT_END=false|' | \
  bash

echo "=== Smoke complete ==="
/gpfs/data/bbj-lab/.envs/input-rep/bin/python - <<'PY'
import json
from pathlib import Path
runs = sorted(Path("outputs/runs/ml4h_2026/smoke/models").glob("*-discrete-time_rope-*"))
assert runs, "no smoke run directory found"
rec = json.loads((runs[-1] / "run_record.json").read_text())
opt = rec["optimizer"]
train = rec["training"]
print(json.dumps({
    "run_dir": str(runs[-1]),
    "optimizer": opt.get("name"),
    "learning_rate": opt.get("learning_rate"),
    "global_step": train.get("global_step"),
    "per_device_train_batch_size": train.get("per_device_train_batch_size"),
    "finite_clip_fraction": train.get("gradient_norm_clipped_fraction"),
}, indent=2))
assert opt.get("name") == "adamw"
assert abs(float(opt.get("learning_rate")) - 1e-4) < 1e-12
assert int(train.get("global_step") or 0) >= 50
assert int(train.get("per_device_train_batch_size") or 0) == 4
PY
