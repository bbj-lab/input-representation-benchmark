#!/bin/bash
#SBATCH --job-name=gen_figures
#SBATCH --partition=tier2q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/gen_figures_%j.out
#SBATCH --error=slurm/logs/gen_figures_%j.err

set -euo pipefail

PROJECT_ROOT="/gpfs/data/bbj-lab/users/daniel/input-representation-benchmark"
cd "$PROJECT_ROOT"

mkdir -p slurm/logs figures

# Activate conda env
eval "$(conda shell.bash hook)"
conda activate input-rep

echo "=== Generating Figures ==="
echo "Start: $(date)"

# Figure 3: Radar plot (uses hardcoded Table 8 values)
echo "[1/2] Generating radar plot..."
python scripts/generate_radar_plot.py --output methods/MLHC2025-ResearchTrack-Template/figures/radar_exp1.pdf

# Figure 5: Calibration curves (needs prediction data)
# NOTE: This requires Stage 3 evaluation outputs. If they don't exist,
# the calibration plot cannot be generated and must be created separately.
# python scripts/generate_calibration_plot.py --output methods/MLHC2025-ResearchTrack-Template/figures/calibration_mortality.pdf

echo "Done: $(date)"
echo ""
echo "NEXT STEPS:"
echo "1. Uncomment Fig 3 includegraphics in paper.tex (line ~320)"
echo "2. Generate Fig 5 calibration plot from prediction data"
echo "3. Create Fig 1 pipeline diagram (TikZ or external)"
