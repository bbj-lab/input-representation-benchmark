# Slurm Batch Launchers

Slurm job scripts for the ML4H 2026 benchmark on the Randi cluster. Python entrypoints live in `../pipeline/`, `../paper/`, and `../utilities/`. See [`../PIPELINE.md`](../PIPELINE.md) for the stage-by-stage order.

## Script Prefixes

| Prefix | Stage | Purpose |
| --- | --- | --- |
| `00_preamble.sh` | Shared | Conda environment activation, Hugging Face scratch cache setup, and path defaults |
| `01_*` | Stage -1 | Raw MIMIC-IV v3.1 to MEDS extraction |
| `02_*` | Stage -0.5 | Experiment 3 matched ICU cohort alignment and CLIF 2.1.0 collation |
| `03_*`, `04_*` | Stage 0 | MEDS tokenization and base outcome extraction |
| `09*` | Calibration | Batch geometry calibration and campaign preflight checks |
| `10_*` | Stage 1 | 156-model pretraining campaign across `Llama-3.2` and `Qwen3` decoders |
| `11_*` | Stage 2 | 24-hour prefix hidden-state extraction |
| `21_*` | Stage 0.5 | 24-to-48-hour clinical outcome extraction |
| `22_*`, `26_*` | Stage 3 | Locked L2-regularized logistic and Ridge linear probes |
| `28_*` | Stats | Per-run admission-level bootstrap confidence intervals |
| `31_*`, `32_*`, `33_*` | Paper | Token counts, centile PCA and boundary probes, and table and figure compilation |
| `34_*` | Stats | Paired admission-swap permutation tests and hierarchical family-effect intervals |
| `35_*` | Stats | Patient-cluster bootstrap sensitivity intervals |
| `ref_qse/` | Stage 2 & 3 | Shared single-task extraction and linear probe worker wrappers |
