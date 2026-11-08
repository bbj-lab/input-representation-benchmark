# ML4H 2026 24-to-48-Hour Metric Tables

These files store the aggregated evaluation metrics and mechanistic analysis sources that populate the ML4H 2026 manuscript tables and figures. They contain summary statistics only and do not contain patient timelines or model weights.

## Evaluation Protocol

Each pretrained model encodes the first 24 hours of an admission (`[0h, 24h]`). Locked linear probes (`LogisticRegression(C=0.01)` and `Ridge(alpha=100.0)`) predict clinical outcomes in hours 24 to 48 (`(24h, 48h]`, 37 tasks in Experiments 1 and 2, and 36 tasks in Experiment 3) across 156 trained models (`Llama-3.2`-style and `Qwen3`-style decoders, three random seeds, and three experiments). Binary tasks report AUROC, Brier score, and 15-bin expected calibration error (ECE). Continuous regression tasks report Spearman rank correlation (`spearman_rho`), `R2`, mean absolute error (MAE), and root-mean-square error (RMSE).

## Files in This Directory

- `metrics_long.csv` and `metrics_long.parquet`: One row for each run, outcome, and metric (`23,040` rows).
- `metrics_long_with_ci.parquet` and `metrics_long_with_ci.json`: Outcome metrics with bootstrap confidence intervals (`1,000` admission resamples, `seed = 42`, `20,400` rows across seven metrics).
- `run_registry.csv` and `run_registry.parquet`: Metadata that maps each of the 156 runs to its experiment, representation, decoder architecture, and seed.
- `label_coverage_long.csv`, `label_coverage_long.parquet`, and `outcome_coverage_summary.csv`: Evaluable admission counts and positive event rates for each of the 37 outcomes.
- `exp1-trace-plot.csv`, `exp2-trace-plot.csv`, and `exp3-trace-plot.csv`: Family-mean profile coordinates plotted in Figures 3, 4, and 5 of the manuscript.
- `pca_centile_geometry_grid_source.csv`: Principal component coordinates for the Llama seed-42 centile embedding geometry grid (Figure 2A and Appendix F).
- `clinical_boundary_probe_results.json` and `pca_probe_best_checkpoint_summary.json`: Leave-one-out clinical boundary probe accuracies and validation-selected checkpoint paths (Figure 2B and Appendix F).
- `exp3_train_token_totals.json`: Training token and vocabulary counts for Experiment 3 native MIMIC-IV MEDS versus CLIF 2.1.0.
- `hospitalizations_by_patient_visits.md`: Distribution of single-admission versus repeat-admission patients in the held-out test splits.

## Sibling Statistical Directories

- `../family_effect_bootstrap/family_effect_intervals.parquet`: 136 family-mean difference tests for Tables 1 through 4 (`1,000` paired admission-swap permutations and bootstrap draws per contrast, `seed = 42`, with within-table Benjamini-Hochberg adjustment at `q = 0.05`).
- `../patient_cluster_bootstrap/patient_cluster_intervals.parquet`: Patient-cluster bootstrap sensitivity intervals (`1,000` patient-cluster resamples, `seed = 42`) and concordance summary (`agreement_summary.json`).
- `../model_level_tests/family_tests.csv` and `outcome_tests.csv`: Model-level paired t-tests (`df = 5`) across the six architecture-by-seed replicates.
