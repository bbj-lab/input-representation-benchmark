#!/bin/bash
#SBATCH --job-name=prevalence
#SBATCH --partition=tier2q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/prevalence_%j.out
#SBATCH --error=slurm/logs/prevalence_%j.err

set -euo pipefail

PROJECT_ROOT="/gpfs/data/bbj-lab/users/daniel/input-representation-benchmark"
cd "$PROJECT_ROOT"

mkdir -p slurm/logs

# Activate conda env
eval "$(conda shell.bash hook)"
conda activate input-rep

echo "=== Computing test-set prevalence for Table 8 ==="
echo "Start: $(date)"

python scripts/compute_prevalence.py \
    --meds_events_dir data/clif/raw \
    --exp3_tokenized_dir artifacts/runs/exp3/meds_icu/deciles_none_unfused_time_rope_first_24h-tokenized

echo "Done: $(date)"
