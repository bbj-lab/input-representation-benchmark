# Input Representation Benchmark

This repository is the orchestration layer for benchmark reruns:

- dataset preparation
- SLURM job generation and stage launch
- statistics refresh (including bootstrap and paired tests)
- manuscript table and figure refresh

Model-side execution is handled by the sibling `../fms-ehrs` repository.

## Repo responsibilities

| Repo | Responsibility |
| --- | --- |
| `input-representation-benchmark` | experiment matrix, stage orchestration, stats refresh, paper refresh |
| `../fms-ehrs` | tokenizer, training, hidden-state extraction, downstream prediction scripts |
| `../697b81f1f269207e5416f18d/MLHC` | manuscript source and compiled paper |

## Active run map

1. **Stage -1 (MEDS extraction)**  
   - `benchmarks/mimic-meds-extraction/scripts/01_extract_meds_full.sh`  
   - `slurm/01_phase0_extract_meds.sh`
2. **Stage 0 (tokenization + base outcomes)**  
   - orchestration: `pipeline/run_experiments.py` + `slurm/03_stage0_tokenize_and_outcomes_meds.sh`  
   - model repo call: `../fms-ehrs/fms_ehrs/scripts/tokenize_w_config.py`  
   - outcome join: `pipeline/scripts/extract_outcomes_meds.py`
3. **Stage 0.5 (optional extended outcomes)**  
   - `pipeline/scripts/extract_extended_outcomes.py`  
   - `slurm/13_refresh_all_extended_outcomes.sh`, `slurm/13_refresh_exp3_extended_outcomes.sh`
4. **Stage 1 (representation training)**  
   - launchers: `slurm/04_exp1_stage1_tune_packed.sh`, `slurm/07_exp2_stage1_train_representation.sh`, `slurm/08_exp3_stage1_train_representation.sh`  
   - model repo calls: `../fms-ehrs/fms_ehrs/scripts/tune_model.py`, `../fms-ehrs/fms_ehrs/scripts/train_representation.py`
5. **Stage 2 (hidden-state extraction)**  
   - launchers: `slurm/09_run_stage2_gpu2_extract.sh`, `slurm/ref_qse/09_extract_reps.sh`  
   - model repo call: `../fms-ehrs/fms_ehrs/scripts/extract_hidden_states.py`
6. **Stage 3 (downstream probes)**  
   - launchers: `slurm/11_run_stage3_tier2q_lr.sh`, `slurm/ref_qse/10_xfer_rep_based_preds.sh`  
   - model repo call: `../fms-ehrs/fms_ehrs/scripts/transfer_rep_based_preds.py`
7. **Statistics refresh**  
   - `pipeline/scripts/regenerate_aligned_family_stats.py` + `slurm/15_submit_aligned_family_stats.sh`  
   - model repo backend: `../fms-ehrs/fms_ehrs/scripts/aggregate_version_preds.py`
8. **Paper refresh**  
   - `paper/scripts/generate_mlhc_appendix_tables.py`  
   - `paper/scripts/generate_mlhc_appendix_outcome_descriptives.py`  
   - `paper/scripts/generate_mlhc_paper_figures.py`

For the full file-by-file stage walkthrough, use `PIPELINE.md`.

Exp3 note:
The arm-construction step rewrites more code domains upstream, but the tokenizer used for the reported Exp3 runs reads only the `LAB` and `VITAL` event blocks. See `pipeline/scripts/build_exp3_meds_semantics_arms.py` together with `../fms-ehrs/fms_ehrs/config/mimic-meds-exp3-icu.yaml`.

## MLHC paper audit

| Surface | What it stores | First use |
| --- | --- | --- |
| `artifacts/runs/statistics/paper_audit_20260316_idaligned_fullstats/all_family_metrics.csv` | Master point estimates and 95% CIs for every reported handle, outcome, and metric. | First stop for nearly every AUROC, Spearman rho, AUPRC, Brier, and ECE value in `paper.tex`. |
| `artifacts/runs/statistics/paper_audit_20260316_idaligned_fullstats/all_family_pairwise.csv` | Paired permutation deltas and BH-adjusted p-values for all within-family handle comparisons. | Use for inferential comparisons and non-baseline pairwise checks. |
| `artifacts/runs/statistics/paper_audit_20260316_idaligned_fullstats/all_family_pairwise_baseline.csv` | Baseline-centered subset of pairwise results. | Main-text delta heatmaps and text that compares each experiment to its shared reference arm. |
| `../697b81f1f269207e5416f18d/MLHC/generated/appendix_binary_outcome_descriptives.tex` | Binary outcome prevalence, eligible `N`, and descriptive coverage. | Outcome sample sizes and appendix descriptive tables. |
| `../697b81f1f269207e5416f18d/MLHC/generated/appendix_regression_outcome_descriptives.tex` | Regression outcome eligible `N`, summary ranges, and descriptive coverage. | Regression sample sizes and appendix descriptive tables. |
| `../697b81f1f269207e5416f18d/MLHC/generated/appendix_binary_sweep.tex` | Manuscript binary outcome sweep table with point estimates and CIs. | Appendix binary metrics included by LaTeX. |
| `../697b81f1f269207e5416f18d/MLHC/generated/appendix_regression_sweep.tex` | Manuscript regression sweep table with point estimates and CIs. | Appendix regression metrics included by LaTeX. |
| `../697b81f1f269207e5416f18d/MLHC/generated/appendix_stats_coverage.tex` | Coverage summary for which stats surfaces are included in the paper appendix. | Appendix stats-coverage table. |
| `../697b81f1f269207e5416f18d/MLHC/figures/` | Final paper figures consumed by LaTeX. | What the manuscript actually renders. |
| `artifacts/runs/models/exp1_*`, `artifacts/runs/models/exp2_*`, `artifacts/runs/models/exp3_*` | Model configs, checkpoints, vocab sizes, representation metadata, and training logs. | Model inventory, vocab sizes, parameter counts, and pretraining-loss source. |
| `artifacts/runs/tokenized/mimiciv-3.1_meds_70-10-20/` and `artifacts/runs/exp3/` tokenized directories | Tokenized Parquet timelines and vocabularies. | Sequence lengths, overflow fractions, realized bin counts, and zero-frequency token audits. |
| `artifacts/runs/exp3/arms/meds_mapped/mappings/meta.json` | Train-split event remapping counts for Experiment 3. | LAB/VITAL remapping percentages reported in `paper.tex`. |
| `outputs/diagnostics_xval_zero_audit/xval_zero_out_results.json` | xVal near-zero suppression ratios and counts. | Mechanistic xVal trench claims. Regenerate with `slurm/xval_zero_out_audit_tier1q.sh`. |

### Builder and recompute entrypoints

| Surface | Builder / implementation |
| --- | --- |
| Aligned family metrics and CIs | `pipeline/scripts/regenerate_aligned_family_stats.py` |
| Baseline-centered pairwise stats | `paper/scripts/recompute_baseline_pairwise_view.py` |
| Appendix sweep and coverage tables | `paper/scripts/generate_mlhc_appendix_tables.py` |
| Appendix descriptive tables | `paper/scripts/generate_mlhc_appendix_outcome_descriptives.py` |
| Main-text and appendix figures | `paper/scripts/generate_mlhc_paper_figures.py` |
| Outcome extraction and clinical definitions | `pipeline/scripts/extract_outcomes_meds.py`, `pipeline/scripts/extract_extended_outcomes.py` |
| xVal zero-out diagnostic | `pipeline/scripts/diagnostics/diag_xval_zero_out.py`, `slurm/xval_zero_out_audit_tier1q.sh` |
| Soft-vs-discrete contextual CE diagnostic | `pipeline/scripts/diagnostics/diag_attention_washout.py` |
| Clinical boundary probe | `pipeline/scripts/diagnostics/diag_clinical_boundary_probe.py` |
| Embedding geometry / sparsity diagnostics | `pipeline/scripts/diagnostics/diag_embedding_geometry.py`, `pipeline/scripts/diagnostics/diag_potassium_geometry_grid.py`, `pipeline/scripts/diagnostics/diag_sparsity_distance.py` |
| Model-side tokenization and representation mechanics | `../fms-ehrs/fms_ehrs/scripts/tokenize_w_config.py`, `../fms-ehrs/fms_ehrs/framework/tokenizer.py`, `../fms-ehrs/fms_ehrs/framework/soft_discretization.py`, `../fms-ehrs/fms_ehrs/framework/xval.py`, `../fms-ehrs/fms_ehrs/framework/model_wrapper.py`, `../fms-ehrs/fms_ehrs/scripts/train_representation.py` |

### Paper section to artifact groups

| Paper area | Primary artifact files | Supporting code / metadata |
| --- | --- | --- |
| Abstract, intro takeaways, conclusion | `artifacts/runs/statistics/paper_audit_20260316_idaligned_fullstats/all_family_metrics.csv`, `artifacts/runs/statistics/paper_audit_20260316_idaligned_fullstats/all_family_pairwise_baseline.csv`, `artifacts/runs/models/exp1_*`, `artifacts/runs/models/exp2_*`, `artifacts/runs/models/exp3_*` | `paper/scripts/generate_mlhc_paper_figures.py` |
| Outcome definitions and eligible sample sizes | `../697b81f1f269207e5416f18d/MLHC/generated/appendix_binary_outcome_descriptives.tex`, `../697b81f1f269207e5416f18d/MLHC/generated/appendix_regression_outcome_descriptives.tex`, `../697b81f1f269207e5416f18d/MLHC/paper.tex` Table `tab:outcomes_sources` | `pipeline/scripts/extract_outcomes_meds.py`, `pipeline/scripts/extract_extended_outcomes.py` |
| Cohort and extraction details | `artifacts/runs/tokenized/mimiciv-3.1_meds_70-10-20/`, `artifacts/runs/exp3/`, `../697b81f1f269207e5416f18d/MLHC/paper.tex` Table `tab:cohort_summary` | `pipeline/scripts/align_cohorts.py`, `benchmarks/mimic-meds-extraction/` configs and scripts |
| Experiment 1 claims | `artifacts/runs/statistics/paper_audit_20260316_idaligned_fullstats/all_family_metrics.csv`, `artifacts/runs/statistics/paper_audit_20260316_idaligned_fullstats/all_family_pairwise_baseline.csv`, `../697b81f1f269207e5416f18d/MLHC/generated/appendix_binary_sweep.tex`, `../697b81f1f269207e5416f18d/MLHC/generated/appendix_regression_sweep.tex` | `paper/scripts/generate_mlhc_paper_figures.py` |
| Experiment 2 claims | `artifacts/runs/statistics/paper_audit_20260316_idaligned_fullstats/all_family_metrics.csv`, `artifacts/runs/statistics/paper_audit_20260316_idaligned_fullstats/all_family_pairwise_baseline.csv`, `artifacts/runs/models/exp2_*`, `outputs/diagnostics_xval_zero_audit/xval_zero_out_results.json` | `../fms-ehrs/fms_ehrs/framework/xval.py`, `../fms-ehrs/fms_ehrs/framework/soft_discretization.py`, `pipeline/scripts/diagnostics/diag_xval_zero_out.py`, `pipeline/scripts/diagnostics/diag_attention_washout.py` |
| Experiment 3 claims | `artifacts/runs/statistics/paper_audit_20260316_idaligned_fullstats/all_family_metrics.csv`, `artifacts/runs/statistics/paper_audit_20260316_idaligned_fullstats/all_family_pairwise_baseline.csv`, `artifacts/runs/exp3/arms/meds_mapped/mappings/meta.json`, `artifacts/runs/models/exp3_*` | `paper/scripts/recompute_baseline_pairwise_view.py` |
| Main-text and appendix figures | `../697b81f1f269207e5416f18d/MLHC/figures/` | `paper/scripts/generate_mlhc_paper_figures.py` |
| Appendix tables | `../697b81f1f269207e5416f18d/MLHC/generated/` | `paper/scripts/generate_mlhc_appendix_tables.py`, `paper/scripts/generate_mlhc_appendix_outcome_descriptives.py` |

## Archived benchmark side paths

- `deprecated/scripts/xgboost_baseline.py`
- `deprecated/slurm/xgboost_baseline.sh`
- `deprecated/outputs/xgboost_baseline/xgboost_baseline_results.json`

## Statistics settings for release reruns

For Exp2/Exp3 stats regeneration:

- bootstrap samples: `2000`
- paired permutation tests: enabled with `permutation_n=2000` when requested
- baseline-only pairwise mode: pass `--baseline_handle` (`discrete_tt` for Exp2, `meds` for Exp3)

Use `slurm/15_submit_aligned_family_stats.sh` to generate the local stats rerun jobfiles under `slurm/generated/statistics/`.
The completed strict-parity rerun sheets are archived under `deprecated/slurm/strict_parity_exp23/generated/demo/`.

## Experiment orchestration tests and dry-runs

Use the `input-rep` environment:

```bash
conda activate input-rep
```

Unit + contract tests:

```bash
pytest pipeline/tests/unit
```

If `pytest` is not available, you can still run core contract checks directly:

```bash
python - <<'PY'
import sys
sys.path.insert(0, ".")
from pipeline.tests.unit import test_pipeline_script_contracts as t
t.test_every_pipeline_script_has_main_and_entry_guard()
t.test_every_pipeline_script_has_dryrun_wrapper()
t.test_dryrun_manifest_matches_pipeline_scripts()
print("pipeline contract checks passed")
PY
```

Dry-run wrappers (one wrapper per pipeline script):

```bash
bash pipeline/tests/dryrun/run_all.sh
```

Optional execute-mode dry-runs for scripts configured with safe CLI checks:

```bash
IRB_DRYRUN_EXECUTE_MODE=1 bash pipeline/tests/dryrun/run_all.sh
```

## Directory map

| Path | Role |
| --- | --- |
| `pipeline/` | stage orchestration scripts and control logic |
| `pipeline/tests/unit/` | unit and contract tests for orchestration |
| `pipeline/tests/dryrun/` | dry-run wrappers for each pipeline script |
| `paper/` | manuscript table/figure builders |
| `utilities/` | non-stage helper scripts and QC |
| `slurm/` | stage launchers and local generated jobfiles |
| `artifacts/` | run outputs |
| `deprecated/` | archived CLIF/UCMC/legacy material |

## Docs

- `PIPELINE.md`: stage-by-stage run notes
- `pipeline/tests/README.md`: orchestration test and dry-run details
- `slurm/README.md`: launcher layout
- `docs/layout.md`: repository layout notes
- `docs/surface_inventory.md`: active vs utility vs archived surfaces

When docs and scripts disagree, follow the scripts and then update the docs.
