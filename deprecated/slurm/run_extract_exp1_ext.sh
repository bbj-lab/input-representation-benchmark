#!/bin/bash
#SBATCH --job-name=extract_exp1_ext
#SBATCH --output=slurm/logs/extract_exp1_ext_%j.out
#SBATCH --error=slurm/logs/extract_exp1_ext_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --partition=tier2q

set -euo pipefail

cd /gpfs/data/bbj-lab/users/daniel/input-representation-benchmark

eval "$(conda shell.bash hook)"
conda activate input-rep

DATA_VERSIONS=(
    "deciles_none_unfused_time_tokens_evalL4096_first_24h"
    "deciles_none_fused_time_tokens_evalL4096_first_24h"
    "ventiles_none_unfused_time_tokens_evalL4096_first_24h"
    "ventiles_none_fused_time_tokens_evalL4096_first_24h"
    "trentiles_none_unfused_time_tokens_evalL4096_first_24h"
    "trentiles_none_fused_time_tokens_evalL4096_first_24h"
    "trentiles_10-10-10_unfused_time_tokens_evalL4096_first_24h"
    "trentiles_10-10-10_fused_time_tokens_evalL4096_first_24h"
    "ventiles_5-10-5_unfused_time_tokens_evalL4096_first_24h"
    "ventiles_5-10-5_fused_time_tokens_evalL4096_first_24h"
    "centiles_none_unfused_time_tokens_evalL4096_first_24h"
    "centiles_none_fused_time_tokens_evalL4096_first_24h"
)

MEDS_DIR="benchmarks/mimic-meds-extraction/data/meds/data"

for dv in "${DATA_VERSIONS[@]}"; do
    tok_dir="${MEDS_DIR}/${dv}-tokenized"
    echo "Extracting extended outcomes for $dv"
    python scripts/extract_extended_outcomes.py \
        --meds_events_dir "$MEDS_DIR" \
        --tokenized_dir "$tok_dir" \
        --splits train,val,test \
        --base_outcomes_filename "tokens_timelines_outcomes.parquet" \
        --output_filename "tokens_timelines_extended_outcomes.parquet" || echo "Failed $dv"
done
