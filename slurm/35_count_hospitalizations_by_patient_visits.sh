#!/bin/bash
#SBATCH --job-name=irb-visit-bins
#SBATCH --output=/gpfs/data/bbj-lab/users/daniel/input-representation-benchmark/outputs/runs/ml4h_2026/metrics/logs/%A-%x.out
#SBATCH --partition=tier2q
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00

set -euo pipefail

IRB_HOME="/gpfs/data/bbj-lab/users/daniel/input-representation-benchmark"
source "${IRB_HOME}/slurm/00_preamble.sh"
cd "${IRB_HOME}"
mkdir -p outputs/runs/ml4h_2026/metrics/logs

python utilities/scripts/count_hospitalizations_by_patient_visits.py
