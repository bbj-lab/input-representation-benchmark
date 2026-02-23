# Benchmark Pipeline: What Was Trained and How

This document explains the complete experimental pipeline so any reader can
understand exactly which configurations were trained and which downstream
evaluations were run.

---

## Experimental Design

Three experiments, 22 total configurations, all trained on **MIMIC-IV v3.1** (MEDS format).

| Experiment | Question | Configurations |
|---|---|---|
| **Exp 1** | What quantization granularity and clinical anchoring works best? | 12 (4 granularities × {unfused, fused} × {time-token, no-time-token}) |
| **Exp 2** | How should numeric values and time be encoded? | 6 ({discrete, soft, xval} × {time-tokens, Time-Aware RoPE}) |
| **Exp 3** | Do vocabulary semantics matter, or is it just co-occurrence? | 4 (native MEDS, CLIF-mapped, randomized, frequency-matched) |

All configurations: single training seed (42), single epoch, 87M-parameter decoder-only transformer (Llama 3.2 architecture, 8 layers, 8 heads, 1024 hidden dim).

---

## Four-Stage Pipeline

```
Stage 0: Tokenize        →   Stage 1: Train        →   Stage 2: Extract       →   Stage 3: Probe
(MEDS → token seqs)          (causal LM, 1 epoch)       (last hidden state)         (LR + MLP probes)
```

### Stage 0 — Tokenize (`slurm/03_stage0_tokenize_and_outcomes_meds.sh`)
- Input: raw MEDS parquet shards (`benchmarks/mimic-meds-extraction/`)
- Runs: `fms-ehrs/fms_ehrs/scripts/tokenize_w_config.py`
- Output: tokenized timelines + outcomes in `outputs/<data_version>/`
- Exp 3 variant: `slurm/04_exp3_stage0_tokenize_and_outcomes.sh`

### Stage 1 — Train (`slurm/04_exp1_stage1_tune_packed.sh`, `07_exp2_stage1_train_representation.sh`, `08_exp3_stage1_train_representation.sh`)
- Runs: `fms-ehrs/fms_ehrs/scripts/train_representation.py` (via `torchrun`)
- All hyperparameters are centralized in `slurm/00_preamble.sh`
- Models saved to: `models/<model_version>/`
- Exp 2/3 use xVal/soft/discrete variants via `--representation` flag

### Stage 2 — Extract (`slurm/09_run_stage2_gpu2_extract.sh`)
- Runs: `fms-ehrs/fms_ehrs/scripts/extract_hidden_states.py`
- Extracts last hidden state at the 24h observation cutoff per admission
- Output: `<model_dir>/hidden_states_{split}.npy` (d=1024 per admission)

### Stage 3 — Probe (`slurm/11_run_stage3_tier2q_lr.sh`)
- Runs: `fms-ehrs/fms_ehrs/scripts/transfer_rep_based_preds.py`
- Primary probe: logistic regression (C-grid {0.01,0.1,1,10,100}, val-tuned on AUROC)
- Diagnostic probe: MLP (256 hidden units, ReLU, early stopping)
- Outputs: `outputs/<model_version>/preds_{split}.parquet` + metrics JSON

---

## Key Orchestration Scripts

| Script | Purpose |
|---|---|
| `run_experiments.py` | Master orchestration: generates SLURM job arrays for all 22 configurations; single source of truth for what was run |
| `slurm/00_preamble.sh` | All hyperparameter defaults (epoch budget, LR, RoPE scaling, xVal params, etc.) |
| `slurm/10_submit_stage2_3_after_train.sh` | Chains Stage 2+3 as SLURM dependencies after Stage 1 |
| `slurm/12_gate_submit_exp2_discrete_after_exp1_winner.sh` | Gated submission: Exp 2 uses Exp 1 winner's configuration |

---

## Mechanistic Diagnostics (Exp 2 only)

Run after Stage 2. Scripts in `scripts/diagnostics/`:

| Script | What it measures |
|---|---|
| `diag_embedding_geometry.py` | KNN boundary separation between discrete vs. soft embeddings |
| `diag_clinical_boundary_probe.py` | Can a linear probe decode which bin a value falls in? |
| `diag_xval_zero_out.py` | Hidden-state norm vs. \|z\| for xVal (detects representational collapse) |
| `diag_attention_washout.py` | Cross-entropy on tokens following numeric positions |
| `diag_sparsity_distance.py` | L2 distance analysis for sparse probe features |

Corresponding SLURM launchers: `slurm/12_run_diag_gpu1.sh`, `slurm/14_run_token_ce_gpuq.sh`

---

## Directory Layout

```
input-representation-benchmark/
├── run_experiments.py        ← master: generates all slurm jobs
├── slurm/                    ← pipeline scripts (numbered 00→14)
│   └── 00_preamble.sh        ← single source of truth for all hyperparameters
├── scripts/                  ← data prep, Exp3 arm construction, QC utilities
│   └── diagnostics/          ← mechanistic probe scripts
├── benchmarks/               ← MIMIC-IV MEDS extraction config
├── models/                   ← trained model checkpoints
├── outputs/                  ← hidden states, predictions, metrics
├── methods/MLHC2025-*/       ← paper LaTeX source
└── tests/                    ← unit tests
```

The `fms-ehrs` sibling repo provides all core Python: tokenizers, training loop,
extraction, probes. IRB adds its PYTHONPATH via `slurm/00_preamble.sh`.

---

## Reproducing from Scratch

```bash
# 1. Set up environment
bash scripts/setup_conda_env_input_rep.sh

# 2. Obtain MIMIC-IV v3.1 MEDS data (see benchmarks/mimic-meds-extraction/)

# 3. Generate all SLURM job scripts for all 22 configurations:
python run_experiments.py --generate-only

# 4. Submit the pipeline (stages 0-3) for all configs:
python run_experiments.py --submit
```

All hyperparameters are fixed in `slurm/00_preamble.sh`. No HPO search is needed
for Exp 2/3 (single training run per configuration). Exp 1 uses a small LR sweep
per granularity (centralized in `run_experiments.py`).
