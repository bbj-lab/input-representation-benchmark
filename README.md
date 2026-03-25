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
   - launchers: `slurm/11_run_stage3_tier2q_lr.sh`, `slurm/ref_qse/10_xfer_rep_based_preds.sh`, `slurm/ref_qse/11_diag_eval.sh`  
   - model repo call: `../fms-ehrs/fms_ehrs/scripts/transfer_rep_based_preds.py`
7. **Statistics refresh**  
   - `pipeline/scripts/regenerate_aligned_family_stats.py` + `slurm/15_submit_aligned_family_stats.sh`  
   - model repo backend: `../fms-ehrs/fms_ehrs/scripts/aggregate_version_preds.py`
8. **Paper refresh**  
   - `paper/scripts/generate_mlhc_appendix_tables.py`  
   - `paper/scripts/generate_mlhc_paper_figures.py`

For the full file-by-file stage walkthrough, use `PIPELINE.md`.

## Statistics settings for release reruns

For Exp2/Exp3 stats regeneration:

- bootstrap samples: `2000`
- paired permutation tests: enabled with `permutation_n=2000` when requested
- baseline-only pairwise mode: pass `--baseline_handle` (`discrete_tt` for Exp2, `meds` for Exp3)

The jobfiles under `slurm/generated/statistics/` and `slurm/strict_parity_exp23/generated/demo/` are wired for this workflow.

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
| `slurm/` | stage launchers and generated jobfiles |
| `artifacts/` | run outputs |
| `deprecated/` | archived CLIF/UCMC/legacy material |
| `scripts/`, `tests/`, `run_experiments.py` | compatibility entry paths kept during migration |

## Docs

- `PIPELINE.md`: stage-by-stage run notes
- `scripts/INDEX.md`: script inventory and compatibility map
- `pipeline/tests/README.md`: orchestration test and dry-run details
- `slurm/README.md`: launcher layout
- `docs/layout.md`: repository layout notes
- `docs/surface_inventory.md`: active vs utility vs archived surfaces

When docs and scripts disagree, follow the scripts and then update the docs.
