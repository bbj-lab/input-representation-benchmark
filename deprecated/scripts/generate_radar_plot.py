
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

# Hardcoded data from Table 8 (to be filled/updated by user if needed)
# Current values are placeholders or extracted from logs where available
# For now, we plot the "Exp 1: Granularity" results as an example

def create_radar_plot(output_path):
    # Categories
    categories = ['Mortality', 'Long LoS', 'ICU Adm.', 'IMV', 'Prolonged ICU']
    N = len(categories)

    # Values (AUROC) for different methods
    # Example data: [Method 1, Method 2, Method 3]
    # Llama-3.2 (Base)
    values_base = [0.85, 0.78, 0.92, 0.88, 0.81]
    
    # XGBoost Baseline (Approximate/Placeholder until run completes)
    values_xgb = [0.82, 0.75, 0.89, 0.85, 0.78]
    
    # Best Model (Exp 1 Fused)
    values_best = [0.88, 0.81, 0.94, 0.90, 0.84]

    # Angles
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    values_base += values_base[:1]
    values_xgb += values_xgb[:1]
    values_best += values_best[:1]

    # Plot
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    
    # Draw one axe per variable + labels
    plt.xticks(angles[:-1], categories, color='grey', size=8)
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.6, 0.7, 0.8, 0.9], ["0.6", "0.7", "0.8", "0.9"], color="grey", size=7)
    plt.ylim(0.5, 1.0)
    
    # Plot data
    ax.plot(angles, values_base, linewidth=1, linestyle='solid', label='Llama-3.2 Base')
    ax.fill(angles, values_base, 'b', alpha=0.1)
    
    ax.plot(angles, values_xgb, linewidth=1, linestyle='dashed', label='XGBoost Baseline')
    ax.fill(angles, values_xgb, 'r', alpha=0.1)
    
    ax.plot(angles, values_best, linewidth=2, linestyle='solid', label='Best (Proposed)')
    ax.fill(angles, values_best, 'g', alpha=0.1)
    
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Radar plot saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default="radar_plot.pdf")
    args = parser.parse_args()
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    create_radar_plot(args.output)
