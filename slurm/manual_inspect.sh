#!/bin/bash
#SBATCH --job-name=inspect_data
#SBATCH --partition=tier2q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=slurm/logs/inspect_%j.out
#SBATCH --error=slurm/logs/inspect_%j.err

set -euo pipefail

# Activate conda env
eval "$(conda shell.bash hook)"
conda activate input-rep

python inspect_data.py
