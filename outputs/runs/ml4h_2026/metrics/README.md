# ML4H 24–48h metric tables

These files are the live copies used to rebuild paper tables and traces.
They are aggregated scores, not patient timelines or model weights.

## What was measured

Each model saw only the first 24 hours of a stay and was scored on events
in hours 24–48. There are 156 trained runs (Llama and Qwen, three random
seeds, three experiments). Binary tasks report AUROC, Brier, and 15-bin ECE.
Regression tasks report Spearman ρ, R², MAE, and RMSE. Average precision is
not in the bootstrap table.

## Files on disk

Family-mean traces were built from `metrics_long.csv`, not from parquet.

- `metrics_long.csv` / `metrics_long.parquet` — one row per run × outcome ×
  metric (23,040 rows). Use the CSV as the trace source.
- `metrics_long_with_ci.parquet` — the same grid with percentile intervals
  from 1,000 patient-level resamples, seed 42 (20,400 rows; seven metrics;
  no average precision). `metrics_long_with_ci.json` is sidecar metadata
  (row counts, hashes, generator path), not the table.
- `../family_effect_bootstrap/family_effect_intervals.parquet` — 136 family
  Δ tests for the five main-text tables (1,000 paired admission swaps per
  column, seed 42). Each cell is a paired permutation test of that arm versus
  the table baseline. Stars require within-table Benjamini–Hochberg
  significance at q=0.05. Experiment 1 fusion contrasts are fused minus
  unfused at the matched grain. Experiment 2 temporal contrasts use discrete
  values only.
- `run_registry.csv` / `run_registry.parquet` — which config, seed, and
  backbone each run used.
- `label_coverage_long.csv` / `label_coverage_long.parquet` and
  `outcome_coverage_summary.csv` — how many labeled stays each outcome had.
- `exp1-trace-plot.csv`, `exp2-trace-plot.csv`, `exp3-trace-plot.csv` — numbers
  behind the three family-mean traces in the paper figures folder.
- `pca_centile_geometry_grid_source.csv` — coordinates behind the Llama
  seed-42 centile PCA grid. Embeddings come from each run's
  `best_model_checkpoint`.
- `clinical_boundary_probe_results.json` — leave-one-out probe accuracies on
  Llama seed-42 unfused population-quantile embeddings (same checkpoint rule).
- `pca_probe_best_checkpoint_summary.json` — realized bin counts, probe
  ranges, and the checkpoint paths used for those two analyses.

Dated `*.pre_affine_*` and `*.pre_affine_figure_rebuild` files are backups
from the affine-repair rebuild. Do not use them for new figures.

A frozen duplicate of the tables remains under
`Archive/ml4h-dump/08_ML4H_provisional/00_data/`. Do not treat Archive as the
live path for new figures.
