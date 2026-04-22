# Input Representation Benchmark

This repository runs the benchmark:

- dataset preparation
- SLURM job generation and stage launch
- statistics calculation, including bootstrap intervals and paired tests
- manuscript table and figure generation

The sibling `../fms-ehrs` repository handles tokenization, model training, and
representation extraction.

## Repo responsibilities

| Repo | Responsibility |
| --- | --- |
| `input-representation-benchmark` | experiment matrix, stage orchestration, statistics calculation, and table/figure generation |
| `../fms-ehrs` | tokenizer, training, hidden-state extraction, and downstream prediction scripts |

## Reproduction environment

To match the paper environment, clone `fms-ehrs` next to this repository and
run:

```bash
conda env create -f environment.yml
conda activate input-rep
```

`environment.yml` mirrors the `input-rep` conda environment used for the
reported runs, including editable installs for this repository and the sibling
`../fms-ehrs` checkout. The file keeps the CUDA, JAX, and FlashAttention
packages used for the paper runs, so CPU-only users may need to adapt it.

## Active run map

1. **Stage -1 (MEDS extraction)**  
   - `benchmarks/mimic-meds-extraction/scripts/01_extract_meds_full.sh`  
   - `slurm/01_phase0_extract_meds.sh`
2. **Stage 0 (tokenization + base outcomes)**  
   - orchestration: `pipeline/run_experiments.py` + `slurm/03_stage0_tokenize_and_outcomes_meds.sh`  
   - model repo call: `../fms-ehrs/fms_ehrs/scripts/tokenize_w_config.py`  
   - outcome join: `pipeline/scripts/extract_outcomes_meds.py`
3. **Stage 0.5 (extended outcomes)**  
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
7. **Statistics calculation**  
   - `pipeline/scripts/regenerate_aligned_family_stats.py` + `slurm/15_submit_aligned_family_stats.sh`  
   - model repo backend: `../fms-ehrs/fms_ehrs/scripts/aggregate_version_preds.py`
8. **Table/Figure generation**  
   - `paper/scripts/generate_mlhc_appendix_tables.py`  
   - `paper/scripts/generate_mlhc_appendix_outcome_descriptives.py`  
   - `paper/scripts/generate_mlhc_paper_figures.py`

For the full file-by-file stage walkthrough, use `PIPELINE.md`.

Experiment 3 note:
The arm-construction step rewrites CLIF-covered `LAB`, `VITAL`, `MEDICATION`,
and `INFUSION` codes upstream, but the tokenizer used for the reported
Experiment 3 runs reads only the `LAB` and `VITAL` event blocks. See
`pipeline/scripts/build_exp3_meds_semantics_arms.py` together with
`../fms-ehrs/fms_ehrs/config/mimic-meds-exp3-icu.yaml`.

## Model input coverage

The 28 reported transformers do not consume every extracted MEDS field. The
table below lists all the MIMIC columns that become model inputs after MEDS
transformation.

| Source MIMIC table | Exact source columns used | Exp1-2 | Exp3 | Model input created |
| --- | --- | --- | --- | --- |
| `hosp/admissions` | `admittime`, `admission_type`, `dischtime`, `discharge_location`, `insurance`, `language`, `marital_status`, `race` | yes | yes | admission and discharge time anchors plus `ADMN_*`, `DSCG_*`, `INSR_*`, `LANG_*`, `MRRD_*`, and `RACE_*` tokens |
| `hosp/patients` | `gender`, `anchor_year`, `anchor_age` | yes | yes | `SEX_*` plus age-at-admission token after deriving `year_of_birth = anchor_year - anchor_age` |
| `hosp/labevents` | `itemid`, `valueuom`, `storetime`, `valuenum`, `ref_range_lower`, `ref_range_upper` | yes | yes | `LAB//itemid//unit` codes, numeric values, and clinically anchored laboratory bins when enabled |
| `hosp/emar` | `medication`, `event_txt`, `storetime` | yes | no | `MEDICATION//drug//action` codes |
| `hosp/transfers` | `eventtype`, `careunit`, `intime` | yes | no | `TRANSFER_TO//eventtype//careunit` codes |
| `icu/icustays` | `first_careunit`, `intime`, `last_careunit`, `outtime` | yes | cohort only | `ICU_ADMISSION//*` and `ICU_DISCHARGE//*` codes in Exp1-2; ICU-stay linkage only in Exp3 |
| `icu/chartevents` | `itemid`, `valueuom`, `storetime`, `valuenum` | no | yes | `VITAL//itemid//unit` codes and numeric values |
| `icu/procedureevents` | `itemid`, `storetime` | yes | no | aggregated `PROC_*` suffix tokens |

Important notes:

- Reported Exp1-2 models use shared prefix tokens (`RACE`, `LANG`, `SEX`,
  age, `INSR`, `MRRD`, `ADMN`), event blocks (`LAB`, `MEDICATION`,
  `TRANSFER_TO`, `ICU_ADMISSION`, `ICU_DISCHARGE`), and suffix tokens
  (`DSCG`, `PROC`).
- Reported Exp3 models use the same static prefix and suffix scaffold, but only
  the `LAB` and `VITAL` event blocks. The upstream arm builder rewrites
  `MEDICATION` and `INFUSION` codes too, but the reported Exp3 tokenizer does
  not feed those families to the model.
- Event order follows information availability: `storetime` is used for
  `labevents`, `emar`, `chartevents`, and `procedureevents`; native admission,
  transfer, and ICU-stay timestamps are used elsewhere.
- Post-discharge billing tables are excluded from the reported models to minimize
  leakage risk: `hosp/diagnoses_icd`, `hosp/procedures_icd`,
  `hosp/hcpcsevents`, and `hosp/drgcodes`.
- The transformers are trained on full tokenized timelines in Stage 1.
  `_first_24h-tokenized` directories are the extraction and evaluation data for
  Stage 2 and Stage 3.

## Reported MLHC compute environment

The paper appendix keeps a short compute summary. Here is the fuller
stage-by-stage record for the reported MLHC runs:

- `Stage0` (`tokenization + outcomes`, CPU-only): 8 CPUs, 300 GB RAM.
- `Stage0E` (`evaluation retokenization`, CPU-only): 8 CPUs, 80 GB RAM.
- `Stage1` (`generative medical event model training`, GPU): 1 x `A100`, 4 CPUs, 128 GB RAM.
- `Stage2` (`representation extraction`, GPU): 1 x `A100`, 4 CPUs, 32 GB RAM.
- `Stage3` (`logistic regression probes`, CPU-only): 8 CPUs, 256 GB RAM.

All reported jobs were single-node runs. All reported Stage1 runs used
`FlashAttention-2`; those precision and kernel choices affected runtime and
memory, but not the model objective or evaluation definitions.

## Statistics files for reproducibility

| File name | What it stores | First use |
| --- | --- | --- |
| `artifacts/runs/statistics/paper_stats_run_artifacts/all_family_metrics.csv` | Master point estimates and 95% CIs for every reported handle, outcome, and metric. | First stop for nearly every AUROC, Spearman rho, AUPRC, Brier, and ECE value in `paper.tex`. |
| `artifacts/runs/statistics/paper_stats_run_artifacts/all_family_pairwise.csv` | Paired permutation deltas and BH-adjusted p-values for all within-family handle comparisons. | Use for inferential comparisons and non-baseline pairwise checks. |
| `artifacts/runs/statistics/paper_stats_run_artifacts/all_family_pairwise_baseline.csv` | Baseline-centered subset of pairwise results. | Main-text delta heatmaps and text that compares each experiment to its shared reference arm. |
| `artifacts/runs/models/exp1_*`, `artifacts/runs/models/exp2_*`, `artifacts/runs/models/exp3_*` | Model configs, checkpoints, vocab sizes, representation metadata, and training logs. | Model inventory, vocab sizes, parameter counts, and pretraining-loss source. |
| `artifacts/runs/tokenized/mimiciv-3.1_meds_70-10-20/` and `artifacts/runs/exp3/` tokenized directories | Tokenized Parquet timelines and vocabularies. | Sequence lengths, overflow fractions, realized bin counts, and zero-frequency token checks. |
| `artifacts/runs/exp3/arms/meds_mapped/mappings/meta.json` | Train-split event remapping counts for Experiment 3. | LAB/VITAL remapping percentages reported in `paper.tex`. |
| `outputs/diagnostics_xval_zero_audit/xval_zero_out_results.json` | xVal near-zero suppression ratios and counts. | Supports the near-zero suppression claims. Regenerate with `slurm/xval_zero_out_audit_tier1q.sh`. |

### Builder and recompute entrypoints

| File or result | Builder / implementation |
| --- | --- |
| Aligned family metrics and CIs | `pipeline/scripts/regenerate_aligned_family_stats.py` |
| Baseline-centered pairwise stats | `paper/scripts/recompute_baseline_pairwise_view.py` |
| Appendix sweep and coverage tables | `paper/scripts/generate_mlhc_appendix_tables.py` |
| Appendix descriptive tables | `paper/scripts/generate_mlhc_appendix_outcome_descriptives.py` |
| Main-text and appendix figures | `paper/scripts/generate_mlhc_paper_figures.py` |
| Outcome extraction and clinical definitions | `pipeline/scripts/extract_outcomes_meds.py`, `pipeline/scripts/extract_extended_outcomes.py` |
| xVal zero-out check | `pipeline/scripts/diagnostics/diag_xval_zero_out.py`, `slurm/xval_zero_out_audit_tier1q.sh` |
| Soft-vs-discrete contextual CE check | `pipeline/scripts/diagnostics/diag_attention_washout.py` |
| Clinical boundary probe | `pipeline/scripts/diagnostics/diag_clinical_boundary_probe.py` |
| Embedding geometry / sparsity diagnostics | `pipeline/scripts/diagnostics/diag_embedding_geometry.py`, `pipeline/scripts/diagnostics/diag_potassium_geometry_grid.py`, `pipeline/scripts/diagnostics/diag_sparsity_distance.py` |
| Model-side tokenization and representation mechanics | `../fms-ehrs/fms_ehrs/scripts/tokenize_w_config.py`, `../fms-ehrs/fms_ehrs/framework/tokenizer.py`, `../fms-ehrs/fms_ehrs/framework/soft_discretization.py`, `../fms-ehrs/fms_ehrs/framework/xval.py`, `../fms-ehrs/fms_ehrs/framework/model_wrapper.py`, `../fms-ehrs/fms_ehrs/scripts/train_representation.py` |

### Where to look first

- For nearly every reported metric in the paper, start with
  `artifacts/runs/statistics/paper_stats_run_artifacts/all_family_metrics.csv`
  and `artifacts/runs/statistics/paper_stats_run_artifacts/all_family_pairwise_baseline.csv`.
- For Experiment 3 mapping coverage, start with
  `artifacts/runs/exp3/arms/meds_mapped/mappings/meta.json`.
- For tokenizer vocabulary size, token counts, and length distributions, start
  with `artifacts/runs/tokenized/mimiciv-3.1_meds_70-10-20/` and
  `artifacts/runs/exp3/`.
- For model inventory and training metadata, start with
  `artifacts/runs/models/exp1_*`, `artifacts/runs/models/exp2_*`, and
  `artifacts/runs/models/exp3_*`.

## Archived benchmark side paths

- `deprecated/scripts/xgboost_baseline.py`
- `deprecated/slurm/xgboost_baseline.sh`
- `deprecated/outputs/xgboost_baseline/xgboost_baseline_results.json`

## Statistics settings for release reruns

For Exp2/Exp3 stats regeneration:

- bootstrap samples: `2000`
- paired permutation tests: enabled with `permutation_n=2000` when requested
- baseline-only pairwise mode: pass `--baseline_handle` (`discrete_tt` for Exp2, `meds` for Exp3)

Use `slurm/15_submit_aligned_family_stats.sh` to generate the local stats rerun
jobfiles under `slurm/generated/statistics/`.
The completed strict-parity rerun sheets are archived under
`deprecated/slurm/strict_parity_exp23/generated/demo/`.

## Experiment orchestration tests and dry-runs

Use the shared paper environment:

```bash
conda env create -f environment.yml
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
| `paper/` | manuscript table and figure builders |
| `utilities/` | non-stage helper scripts and QC |
| `slurm/` | stage launchers and local generated jobfiles |
| `artifacts/` | run outputs |
| `deprecated/` | archived CLIF/UCMC/legacy material |

## Docs

- `PIPELINE.md`: stage-by-stage run notes
- `pipeline/tests/README.md`: orchestration test and dry-run details
- `slurm/README.md`: launcher layout
- `docs/layout.md`: repository layout notes
