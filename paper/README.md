# Manuscript tables and figures

Scripts in [`scripts/`](scripts/) build the tables and figures of the paper from `outputs/runs/ml4h_2026/`.

## Paper outputs

| Paper item | Script | Launcher |
| :-- | :-- | :-- |
| Main tables (Experiments 1–3) and appendix family and outcome tables | `generate_ml4h_appendix_result_tables.py` | `slurm/33_rebuild_ml4h_review_tables_and_compile.sh` |
| Clinical boundary probe figure | `generate_ml4h_boundary_probe_plot.py` | `slurm/33_rebuild_ml4h_review_tables_and_compile.sh` |
| Centile embedding PCA grid | `pipeline/scripts/diagnostics/rebuild_ml4h_pca_and_probe.py` | `slurm/31_rerun_ml4h_pca_probe.sh` |
| Sequence length histograms | `utilities/qc/plot_length_histograms.py` | `slurm/34_rebuild_ml4h_length_histograms.sh` |

## Statistics

| Script | Output |
| :-- | :-- |
| `ml4h_main_table_contrasts.py` | Defines the 17 comparison columns and 136 family comparisons in the main tables |
| `ml4h_model_level_tests.py` | Six-pair *t*-tests (df = 5) with BH adjustment, written to `model_level_tests/` |
| `summarize_ml4h_patient_clusters.py` | Compares patient-clustered bootstrap intervals with the model-level tests |

## Supporting scripts

These scripts are not paper outputs. Other scripts import them, or they build the intermediate metric bundle.

| Script | Role |
| :-- | :-- |
| `generate_ml4h_24to48h_provisional.py` | Builds the 24–48h metric bundle from Stage 3 prediction files (run by `slurm/23_build_ml4h_metric_tables.sh`) |
| `generate_ml4h_figures.py` | Shared figure helpers |
| `generate_ml4h_outcome_factor_traces.py`, `generate_ml4h_outcome_family_traces.py` | Outcome and family trace plots for internal review |
| `generate_ml4h_spider_charts.py`, `generate_ml4h_provisional_visual_alternatives.py`, `generate_ml4h_submission_previews.py` | Plot constants and condition definitions that the trace and bootstrap scripts import |
| `generate_ml4h_benchmark_schematic.py` | Earlier schematic of the benchmark design |
