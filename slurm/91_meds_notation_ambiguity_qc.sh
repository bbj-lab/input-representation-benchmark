#!/bin/bash
#SBATCH --job-name=irb-meds-ambiguity
#SBATCH --output=./slurm/output/%j-%x.stdout
#SBATCH --partition=tier2q,tier3q
#SBATCH --mem=300GB
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00

# =============================================================================
# MEDS notation ambiguity QC
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IRB_HOME="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "${IRB_HOME}/slurm/00_preamble.sh"
cd "${IRB_HOME}"

mkdir -p outputs/meds_ambiguity slurm/output

/gpfs/data/bbj-lab/.envs/input-rep/bin/python scripts/analyze_meds_notation_ambiguity.py \
  --meds_data_dir benchmarks/mimic-meds-extraction/data/meds/data \
  --split train \
  --out_dir outputs/meds_ambiguity \
  --top_k_forms 25

