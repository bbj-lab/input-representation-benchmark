# Run Output Tree

This directory holds tokenized datasets, pretrained model checkpoints, Stage 2 and Stage 3 job manifests, and aggregated evaluation tables for the ML4H 2026 benchmark.

## Directory Layout

```text
outputs/runs/
  tokenized/
    mimiciv-3.1_meds_70-10-20/    Stage 0 tokenized timelines, features, and probe predictions (Exp 1 & 2)
  exp3/
    cohort/                       Frozen ICU patient and admission split tables
    meds_icu/                     Native MIMIC-IV ICU MEDS events
    arms/                         Tokenized timelines, features, and probe predictions (Exp 3)
  ml4h_2026/
    models/                       156 pretrained checkpoints (Llama-3.2 and Qwen3 across 3 seeds)
    metrics/                      Aggregated 24-to-48-hour evaluation tables and trace sources
    bootstrap_24to48h_ci/         Admission-level bootstrap shards for per-run outcome CIs
    family_effect_bootstrap/      Paired admission-swap permutation tests and CIs (136 family contrasts)
    patient_cluster_bootstrap/    Patient-cluster bootstrap sensitivity intervals
    model_level_tests/            Model-level paired t-tests (df = 5) across 6 architecture-seed replicates
    stage2*/                      Stage 2 feature extraction jobfiles and manifests
    stage3*/                      Stage 3 linear probe jobfiles and manifests
```

## Key Evaluation Files

| Path | Description |
| --- | --- |
| `ml4h_2026/metrics/metrics_long.parquet` | Per-run outcome scores across 156 models and 37 tasks (`23,040` rows) |
| `ml4h_2026/metrics/metrics_long_with_ci.parquet` | Outcome scores with 1,000-resample bootstrap 95% confidence intervals |
| `ml4h_2026/family_effect_bootstrap/family_effect_intervals.parquet` | Paired admission-swap contrasts and Benjamini-Hochberg adjusted p-values |
| `ml4h_2026/patient_cluster_bootstrap/patient_cluster_intervals.parquet` | Patient-cluster bootstrap intervals and significance agreement summary |
| `ml4h_2026/model_level_tests/family_tests.csv` | Six-replicate paired t-test summary for all 136 printed family contrasts |

See [`runs/ml4h_2026/metrics/README.md`](runs/ml4h_2026/metrics/README.md) for column-level documentation.
