# Input Representation Benchmark

Codebase for the **MLHC 2026** input-representation benchmark: data prep,
SLURM orchestration, statistics, and manuscript tables/figures.

Model training and extraction scripts live in the sibling repo
[`../fms-ehrs`](../fms-ehrs). Read both README files to reproduce a run.

## Quick start

```bash
# clone fms-ehrs next to this repo, then:
conda env create -f environment.yml
conda activate input-rep
```

`environment.yml` installs both repos in editable mode and matches the Randi
cluster stack (CUDA, JAX, FlashAttention).

## Pipeline overview

Stages -1 through paper build. File-level detail: [`PIPELINE.md`](PIPELINE.md).

| Stage | Benchmark repo | `../fms-ehrs` |
| --- | --- | --- |
| -1 MEDS extraction | `benchmarks/mimic-meds-extraction/`, `slurm/01_*` | — |
| 0 tokenization + outcomes | `pipeline/run_experiments.py`, `slurm/03_*`, `slurm/04_*` | `tokenize_w_config.py` |
| 0.5 extended outcomes | `pipeline/scripts/extract_extended_outcomes.py`, `slurm/13_*` | — |
| 1 training | `slurm/04_*`, `slurm/07_*`, `slurm/08_*` | `tune_model.py`, `train_representation.py` |
| 2 extraction | `slurm/09_*`, `slurm/ref_qse/09_extract_reps.sh` | `extract_hidden_states.py` |
| 3 probes | `slurm/10_*`, `slurm/11_*`, `slurm/ref_qse/` | `transfer_rep_based_preds.py` |
| stats | `pipeline/scripts/regenerate_aligned_family_stats.py`, `slurm/15_*` | `aggregate_version_preds.py` |
| paper | `paper/scripts/generate_mlhc_*.py` | — |

Benchmark launchers set paths and submit jobs. `fms-ehrs` scripts read tokenized
data, write checkpoints, write `features-*.npy`, and write probe prediction
files.

**Paper stats root:** `outputs/runs/statistics/paper_stats_combined/`
(`all_family_metrics.csv`, `all_family_pairwise.csv`,
`all_family_pairwise_baseline.csv`).
Per-family rebuild outputs live under
`outputs/runs/statistics/paper_stats_run_outputs/`.

**Exp3 note:** upstream arm building rewrites several code families, but the
reported Exp3 tokenizer reads only `LAB` and `VITAL` blocks (see
`pipeline/scripts/build_exp3_meds_semantics_arms.py` and
`../fms-ehrs/fms_ehrs/config/mimic-meds-exp3-icu.yaml`).

## Where to look first

| Goal | Path |
| --- | --- |
| Paper metrics (Exp1–3) | `outputs/runs/statistics/paper_stats_combined/all_family_metrics.csv` |
| Qwen3 additional-run stats | `outputs/runs/statistics/generalizability_tests/qwen3_0p6b_llama10ep/`, `generalizability_tests/qwen3_0p6b_fused_only/`, `generalizability_tests/qwen3_scaled/` |
| Llama10ep additional-run stats | `outputs/runs/statistics/generalizability_tests/qwen3_0p6b_llama10ep/` (`llama10ep_*` family folders and combined CSVs) |
| Checkpoints and training logs | `outputs/runs/models/exp*_*`, `qwen3_*`, `llama10ep_*` |
| Tokenized timelines | `outputs/runs/tokenized/mimiciv-3.1_meds_70-10-20/` |
| Extracted features | `<data_version>_first_24h-tokenized/<split>/features-*.npy` |
| Probe outputs | `<data_version>_first_24h-tokenized/test/*-preds-*.pkl` |
| Exp3 mapping coverage | `outputs/runs/exp3/arms/meds_mapped/mappings/meta.json` |

On-disk tree policy: [`outputs/README.md`](outputs/README.md).

## Additional runs

These completed generalizability tests reuse the paper stage order but use
separate jobfiles, model prefixes, and statistics roots under `generalizability_tests/`.
Qwen3 tests stress architecture generalizability with 0.6B and scaled
depth8/depth16 fused-vs-unfused decile RoPE runs. Llama10ep tests stress
training-budget and seed generalizability with the scaled Llama 3.2 backbone
trained for 10 epochs across seeds 42–46 on decile discrete RoPE and centile
soft RoPE settings.

### Additional-run paths

| Item | Path |
| --- | --- |
| Prepare/submit | `slurm/17_prepare_submit_generalizability_tests.sh` |
| Jobfiles | `slurm/generated/generalizability_tests/` |
| Last job IDs | `slurm/generated/generalizability_tests/submit_state.env` |
| Serial Llama10ep runner | `slurm/18_run_gpu4_jobfile_serial.sh` |
| Resume markers | `slurm/state/*.last_completed` (local only; not source files) |

Model prefixes under `outputs/runs/models/`: `qwen3_0p6b_exp2_*`,
`qwen3_depth8_exp2_*`, `qwen3_depth16_exp2_*`, `llama10ep_exp2_*`.

Jobfile prefixes: `00` tokenize; `01*`/`02*` train; `03*`/`04` extract;
`05*`/`06` probes; `07*` stats.

**Feature filenames:** `<split>/features-<run_dir>-model-discrete-time_rope.npy`
(stem logic in `../fms-ehrs/fms_ehrs/scripts/extract_hidden_states.py`). Stage 3
must use the same stem.

**Llama10ep W&B checkpoints:** `slurm/07_exp2_stage1_train_representation.sh`
saves epoch-boundary checkpoints when `IRB_LLAMA10EP_WANDB_EPOCH_UPLOADS=true`
(default), avoiding step-based W&B uploads. Metrics still log normally.

**Additional-run extraction:** four `torchrun` ranks write shards, then rank 0 merges
to one `features-*.npy` per split.

### Qwen3 output map

**Models** (`outputs/runs/models/`):

| Prefix | Contents |
| --- | --- |
| `qwen3_0p6b_exp2_*` | 0.6B unfused/fused (`gpu4`, `gpu4-r2`, `gpu4-r3`) |
| `qwen3_depth8_exp2_*` | depth8 unfused/fused (`gpu4-r4`) |
| `qwen3_depth16_exp2_*` | depth16 unfused/fused (`gpu4-r4`) |

Each run dir has `checkpoint-*` and `model-discrete-time_rope/`. depth8/depth16
also have local `loss_perplexity_curve.csv`; 0.6B curves were recovered from W&B
and trainer state.

**Features** (base:
`outputs/runs/tokenized/mimiciv-3.1_meds_70-10-20/`):

| Condition | Split dir | Files |
| --- | --- | --- |
| unfused | `deciles_none_unfused_time_rope_first_24h-tokenized/{train,val,test}/` | `features-model-discrete-time_rope.npy` (0.6B legacy name); `features-qwen3_depth8_exp2_...`; `features-qwen3_depth16_exp2_...` |
| fused | `deciles_none_fused_time_rope_first_24h-tokenized/{train,val,test}/` | same pattern |

**Probes:** in each `test/` dir above; existing prediction filenames retain
legacy `revision-qwen3_*` tags and cover all four outcome families
(`primary_binary`, `additional_binary`, `length_of_stay`, `extended_regression`).

**Stats:**

| Root | Contents |
| --- | --- |
| `generalizability_tests/qwen3_0p6b_llama10ep/` | 0.6B Qwen3 and Llama10ep metrics, pairwise tables, and per-family folders |
| `generalizability_tests/qwen3_0p6b_fused_only/` | fused-only 0.6B subset |
| `generalizability_tests/qwen3_scaled/` | scaled Qwen per-family stats |

## Training stability outputs

Not paper results. Loss/LR/grad-norm plots and stratified validation-loss checks.

| Path | Contents |
| --- | --- |
| `outputs/runs/figures/qwen_loss_curves/` | Qwen train/eval loss |
| `outputs/runs/figures/llama10ep_loss_curves/` | Llama10ep train/eval loss |
| `outputs/runs/figures/qwen3_training_diagnostics/` | Qwen LR and grad norm |
| `outputs/runs/figures/llama10ep_training_diagnostics/` | Llama LR, grad norm, stratified eval loss |

Also: `loss_perplexity_curve.csv` in each model dir; W&B project
`input-rep-benchmark-generalizability-tests`. Stratified eval script:
`pipeline/scripts/llama10ep_stratified_eval_loss.py`.

## Model input coverage

The 28 paper transformers do not use every MEDS field.

| Source table | Key columns | Exp1–2 | Exp3 |
| --- | --- | --- | --- |
| `hosp/admissions` | admit/discharge metadata, race, insurance, … | yes | yes |
| `hosp/patients` | sex, age anchors | yes | yes |
| `hosp/labevents` | labs + timestamps | yes | yes |
| `hosp/emar` | medications | yes | no |
| `hosp/transfers` | transfers | yes | no |
| `icu/icustays` | ICU stay times | yes | cohort only |
| `icu/chartevents` | vitals | yes | yes |
| `icu/procedureevents` | procedures | yes | no |

Stage 1 trains on full timelines. Stages 2–3 use `_first_24h-tokenized` dirs.
Post-discharge billing tables are excluded to reduce leakage.

## Paper compute (reported runs)

| Stage | Resources |
| --- | --- |
| 0 tokenization | CPU, 8 cores, 300 GB |
| 1 training | 1× A100, 4 cores, 128 GB, FlashAttention-2 |
| 2 extraction | 1× A100, 4 cores, 32 GB |
| 3 probes | CPU, 8 cores, 256 GB |

All reported jobs were single-node.

## Paper-side checks and rebuild scripts

Manuscript checks: `pipeline/scripts/diagnostics/` (folder name is historical).

| Check | Script |
| --- | --- |
| xVal near-zero suppression | `diag_xval_zero_out.py` |
| soft vs discrete contextual CE | `diag_attention_washout.py` |
| clinical boundary probe | `diag_clinical_boundary_probe.py` |
| embedding geometry | `diag_embedding_geometry.py` |

Rebuild entry points: `regenerate_aligned_family_stats.py`, `recompute_baseline_pairwise_view.py`,
`generate_mlhc_*.py`, outcome extractors under `pipeline/scripts/`.

Stats reruns: bootstrap `2000`, permutation `2000` when enabled; baseline handles
`discrete_tt` (Exp2) or `meds` (Exp3). Jobfiles via `slurm/15_submit_aligned_family_stats.sh`.

## Tests

```bash
conda activate input-rep
pytest pipeline/tests/unit
bash pipeline/tests/dryrun/run_all.sh
```

Details: [`pipeline/tests/README.md`](pipeline/tests/README.md). Model-side tests:
[`../fms-ehrs/fms_ehrs/tests/`](../fms-ehrs/fms_ehrs/tests/).

## Directory map

| Path | Role |
| --- | --- |
| `pipeline/` | orchestration and paper-side checks |
| `paper/` | table and figure builders |
| `slurm/` | launchers; `generated/` holds local jobfiles |
| `outputs/runs/` | models, tokenized data, stats, figures |
| `benchmarks/mimic-meds-extraction/` | MEDS wrapper |
| `utilities/` | optional helpers outside the main chain |
| `deprecated/` | archived material |

Launcher numbering: [`slurm/README.md`](slurm/README.md).

## Related docs

| Doc | Contents |
| --- | --- |
| [`PIPELINE.md`](PIPELINE.md) | stage-by-stage walkthrough |
| [`../fms-ehrs/README.md`](../fms-ehrs/README.md) | model scripts and output contract |
| [`docs/layout.md`](docs/layout.md) | layout policy |
