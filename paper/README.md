# Manuscript Figure and Table Generators

`paper/scripts/` contains the Python scripts that generate the ML4H 2026 manuscript tables, trace plots, and mechanistic figures from `outputs/runs/ml4h_2026/`.

| Script | Purpose |
| --- | --- |
| `ml4h_main_table_contrasts.py` | Define the 17 comparison columns and 136 family-level contrasts in Tables 1 through 4 |
| `generate_ml4h_appendix_result_tables.py` | Render main-text Tables 1 through 4 and Appendix Tables 5 through 14 into `../ML4H2026/generated/` |
| `generate_ml4h_outcome_family_traces.py` | Render the main-text family-mean profile traces (`exp1-trace-plot.pdf`, `exp2-trace-plot.pdf`, `exp3-trace-plot.pdf`) |
| `generate_ml4h_outcome_factor_traces.py` | Render the Appendix H outcome-level factor trace plots (`exp1_factor_traces.pdf`, `exp2_factor_traces.pdf`, `exp3_factor_traces.pdf`) |
| `generate_ml4h_boundary_probe_plot.py` | Render the Experiment 1 centile PCA and clinical boundary probe composite (`figure-2.pdf`) |
| `generate_ml4h_benchmark_schematic.py` | Render the benchmark study schematic (`figure-1.pdf`) |
| `ml4h_model_level_tests.py` | Compute model-level paired t-tests (`df = 5`) across the six architecture-by-seed replicates |
| `summarize_ml4h_patient_clusters.py` | Compare admission-level and patient-cluster bootstrap confidence intervals |
| `generate_ml4h_24to48h_provisional.py` | Aggregate Stage 3 prediction files into `metrics_long.parquet` and `label_coverage_long.parquet` |
| `generate_ml4h_figures.py` | Shared plotting primitives and forest-plot renderer |
