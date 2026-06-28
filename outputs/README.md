# Run output tree

Start with [`../README.md`](../README.md) for paper stats and additional-run output
paths. This file describes the on-disk tree under `outputs/runs/`.

## Active layout

```text
outputs/runs/
  models/          exp1_*, exp2_*, exp3_*, qwen3_*, llama10ep_*
  tokenized/       mimiciv-3.1_meds_70-10-20/
  exp3/            cohort/, meds_icu/, arms/
  statistics/
    paper_stats_combined/
    paper_stats_run_outputs/
    paper_stats_exp23_discrete_none_baseline/
    generalizability_tests/qwen3_0p6b_llama10ep/, generalizability_tests/qwen3_0p6b_fused_only/, generalizability_tests/qwen3_scaled/
  figures/         training-loss and stability-check plots
```

## Paper reporting files

| Path | Use |
| --- | --- |
| `statistics/paper_stats_combined/all_family_metrics.csv` | main metric table with 95% CIs |
| `statistics/paper_stats_combined/all_family_pairwise_baseline.csv` | baseline pairwise comparisons |
| `statistics/paper_stats_run_outputs/` | per-family tables that feed the combined dir |
| `statistics/generalizability_tests/qwen3_0p6b_llama10ep/all_family_metrics.csv` | combined Qwen3 + Llama10ep additional-run metric table |
| `statistics/generalizability_tests/qwen3_0p6b_llama10ep/llama10ep_*/` | Llama10ep five-seed per-family metric and pairwise tables |

## Reproducibility

1. `pipeline/scripts/regenerate_aligned_family_stats.py`
2. `paper/scripts/recompute_baseline_pairwise_view.py` (when needed)
3. `paper/scripts/generate_mlhc_*.py`


## Policy

- Keep only active outputs here; move stale material to `deprecated/`.
- Before any tree migration, run `pipeline/scripts/build_run_output_tree.py --dry-run`.
