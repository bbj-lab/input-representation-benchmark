#!/usr/bin/env python3
"""Regenerate the attention washout comparison plot from saved JSON results."""
import json, pathlib, shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = pathlib.Path(__file__).resolve().parents[2]

with open(root / 'outputs/diagnostics/attention_washout_results.json') as f:
    results = json.load(f)

model_names = list(results.keys())
fig, axes = plt.subplots(
    1, len(model_names),
    figsize=(7 * len(model_names), 5),
    constrained_layout=True,
)
if len(model_names) == 1:
    axes = [axes]

for ax, model_name in zip(axes, model_names):
    attn_data = results[model_name]['attention_patterns']
    layers = sorted([k for k in attn_data if k.startswith('layer_')],
                   key=lambda x: int(x.split('_')[1]))
    n_layers = len(layers)
    first_layer = attn_data[layers[0]]
    n_heads = len([k for k in first_layer if k.startswith('head_')])

    matrix = np.zeros((n_layers, n_heads))
    for i, layer_name in enumerate(layers):
        layer = attn_data[layer_name]
        for j in range(n_heads):
            matrix[i, j] = layer[f'head_{j}']['numeric_share']

    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=0.5)
    ax.set_xlabel('Head', fontsize=12)
    ax.set_ylabel('Layer', fontsize=12)
    ax.set_xticks(range(n_heads))
    ax.set_yticks(range(n_layers))
    ax.set_title(model_name, fontsize=13, fontweight='bold')

    for i in range(n_layers):
        for j in range(n_heads):
            ax.text(j, i, f'{matrix[i,j]:.2f}', ha='center', va='center',
                   fontsize=7, color='white' if matrix[i,j] > 0.25 else 'black')

cbar = fig.colorbar(
    im, ax=axes, orientation='horizontal',
    shrink=0.6, pad=0.12, aspect=30,
    label='Numeric Attention Share',
)
cbar.ax.tick_params(labelsize=10)
fig.suptitle('Attention Share to Numeric Tokens\n(from Categorical Query Positions)',
             fontsize=14, fontweight='bold')

out = root / 'outputs/diagnostics/attention_washout_comparison.png'
fig.savefig(out, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {out}')

dst = root / 'methods/MLHC2025-ResearchTrack-Template/figures/attention_washout_comparison.png'
shutil.copy(str(out), str(dst))
print(f'Copied to: {dst}')
