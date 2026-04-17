# Run artifact tree

This directory stores active outputs from benchmark runs. The benchmark
pipeline should read from and write to this tree directly.

## Active layout

```text
artifacts/
  runs/
    models/
      exp1_*/
      exp2_*/
      exp3_*/
    tokenized/
      mimiciv-3.1_meds_70-10-20/
    exp3/
      cohort/
      meds_icu/
      arms/
    statistics/
      paper_stats_combined/
        all_family_metrics.csv
        all_family_pairwise_baseline.csv
      paper_stats_exp23_discrete_none_baseline/
      paper_stats_run_artifacts/
        all_family_metrics.csv
        all_family_pairwise.csv
        all_family_pairwise_baseline.csv
      paper_stats_run_artifacts_shards/
```

## Main data files for reporting

- `runs/statistics/paper_stats_combined/all_family_metrics.csv` is the main
  table of metrics and 95% bootstrap confidence intervals.
- `runs/statistics/paper_stats_combined/all_family_pairwise_baseline.csv` is
  the main table of paired baseline comparisons and adjusted p-values.
- `runs/statistics/paper_stats_run_artifacts/` stores per-family tables used
  to build the combined reporting files.
- `runs/models/`, `runs/tokenized/`, and `runs/exp3/` store trained models and
  tokenized timelines used by the statistics pipeline.

## Refresh order for ground-truth statistics updates

1. Refresh aligned family statistics with
   `pipeline/scripts/regenerate_aligned_family_stats.py`.
2. Rebuild baseline-centered pairwise tables when needed with
   `paper/scripts/recompute_baseline_pairwise_view.py`.
3. Regenerate tables and figures with:
   - `paper/scripts/generate_mlhc_appendix_tables.py`
   - `paper/scripts/generate_mlhc_appendix_outcome_descriptives.py`
   - `paper/scripts/generate_mlhc_paper_figures.py`
4. Rebuild your report PDF in the writing workspace.

## Policy

- Keep only active, non-deprecated artifacts in this tree.
- Move stale or retired outputs to `deprecated/`.
- Run `pipeline/scripts/build_run_artifact_tree.py --dry-run` before any
  structural migration.
