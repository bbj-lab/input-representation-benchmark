# ML4H 2026 metric tables

These files hold the aggregated scores behind the paper. They contain no patient timelines and no model weights.

## Evaluation

Each model encodes the first 24 hours of a hospitalization. Fixed linear probes (`LogisticRegression(C=0.01)` and `Ridge(alpha=100.0)`) predict outcomes in hours 24–48. Experiments 1 and 2 have 37 tasks. Experiment 3 has 36 tasks. There are 156 trained models (two decoder architectures, three seeds, three experiments).

| Task type | Metrics |
| :-- | :-- |
| Binary | AUROC, Brier score, 15-bin expected calibration error, average precision |
| Regression | Spearman ρ, R², mean absolute error, root mean squared error |

The paper does not report average precision. `metrics_long` keeps it, and `metrics_long_with_ci` leaves it out.

## Files in this folder

| File | Contents |
| :-- | :-- |
| `metrics_long.csv`, `metrics_long.parquet` | One row for each run, outcome, and metric (23,040 rows) |
| `metrics_long_with_ci.parquet`, `metrics_long_with_ci.json` | The same grid with 1,000-draw hospitalization bootstrap 95% intervals, seed 42 (20,400 rows, seven metrics) |
| `run_registry.csv`, `run_registry.parquet` | Experiment, configuration, architecture, and seed of each run |
| `label_coverage_long.*`, `outcome_coverage_summary.csv` | Labeled hospitalizations and positive rates for each outcome |
| `pca_centile_geometry_grid_source.csv` | Coordinates for the centile embedding PCA figure |
| `clinical_boundary_probe_results.json`, `pca_probe_best_checkpoint_summary.json` | Boundary probe accuracies and the checkpoints used for the PCA and probe analyses |
| `exp3_train_token_totals.json` | Training token totals for the native-MIMIC and CLIF arms of Experiment 3 |
| `hospitalizations_by_patient_visits.md` | Hospitalizations grouped by the number of visits from the same patient, for each split |

## Related folders

| Folder | Contents |
| :-- | :-- |
| `../model_level_tests/` | Six-pair *t*-tests (`df = 5`), BH adjustment, and Llama-versus-Qwen Welch *t*-tests for the 136 family comparisons (`family_tests.csv`) and the 628 individual-outcome comparisons (`outcome_tests.csv`). |
| `../patient_cluster_bootstrap/` | 1,000-draw patient-clustered bootstrap 95% intervals (seed 42) for the 136 family-mean differences, plus agreement summaries against `family_tests.csv`. |

## Column notes for `../model_level_tests/`

| Column | Tables | Meaning |
| :-- | :-- | :-- |
| `base`, `alt` | `outcome_tests.csv` | Six-model mean score for the baseline and alternative configurations on that clinical outcome. |
| `delta`, `delta_min`, `delta_max`, `n_positive` | `family_tests.csv`, `outcome_tests.csv` | Mean, minimum, maximum, and positive count across the six architecture–seed paired differences. |
| `ci_lo`, `ci_hi` | `family_tests.csv`, `outcome_tests.csv` | Unadjusted two-sided 95% *t*-interval (`df = 5`) around `delta`. |
| `t`, `p_t`, `p_t_adj`, `star_t` | `family_tests.csv`, `outcome_tests.csv` | One-sample *t*-statistic, unadjusted *p*-value, BH-adjusted *p*-value (`q = 0.05`), and significance flag. `family_tests.csv` adjusts inside each printed table, and `outcome_tests.csv` adjusts across outcomes inside each contrast column. |
| `delta_llama`, `delta_qwen` | `family_tests.csv`, `outcome_tests.csv` | Three-seed mean difference for the Llama-3.2-style and Qwen3-style decoders. |
| `p_arch`, `p_arch_adj` | `family_tests.csv`, `outcome_tests.csv` (`p_arch` only) | Two-sided Welch *t*-test between the three Llama-3.2-style and three Qwen3-style differences, plus within-table BH adjustment in `family_tests.csv`. |
