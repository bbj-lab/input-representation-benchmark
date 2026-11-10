# Output tree

Most of this tree stays out of git. Only the aggregated tables under `runs/ml4h_2026/` are in the repository.

```text
outputs/runs/
├── tokenized/mimiciv-3.1_meds_70-10-20/   Exp. 1–2 timelines, features, and probe predictions
├── exp3/
│   ├── cohort/                            frozen ICU cohort and patient split
│   ├── meds_icu/                          native MIMIC-IV arm: MEDS events, timelines, features, predictions
│   └── meds_clif/                         CLIF 2.1.0 arm: MEDS events, timelines, features, predictions
└── ml4h_2026/
    ├── models/                            156 trained checkpoints
    ├── stage2*/, stage3_24to48h_*/        extraction and probe jobfiles and manifests
    ├── outcomes_24to48h/, probe_logs/     Stage 0.5 and Stage 3 jobfiles and logs
    ├── metrics/                           aggregated scores                  (in git)
    ├── model_level_tests/                 six-pair and Welch t-tests         (in git)
    ├── patient_cluster_bootstrap/         patient-clustered intervals        (in git)
    └── bootstrap_24to48h_ci/              bootstrap shards
```

See [`runs/ml4h_2026/metrics/README.md`](runs/ml4h_2026/metrics/README.md) for the contents of each table.
