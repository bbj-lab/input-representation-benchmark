# Outcome Reruns

This directory contains the no-launch rerun sheet for outcome-definition
realignment and newly added literature-backed outcomes.

## Scope

These reruns cover:

- threshold fixes for existing additional binary outcomes
- newly added additional binary outcomes
- the current regression outcome panel

These reruns do **not** include:

- FM pretraining reruns
- Stage 2 hidden-state extraction reruns
- the held outcomes `RR > 30`, `bicarbonate < 10`, or creatinine-based AKI

## What Needs Rerun

`Stage 0`

- Refresh `tokens_timelines_extended_outcomes.parquet` for the `19` canonical
  tokenized roots.
- Preferred launcher:
  - `sbatch slurm/13_refresh_all_extended_outcomes.sh`

`Stage 3`

- Additional binary outcomes:
  - `24` checkpoint-level logistic reruns
  - jobfile: `10_additional_binary_logreg.jobfile`

- Regression outcomes:
  - `24` checkpoint-level ridge reruns
  - jobfile: `11_regression_ridge.jobfile`

Primary binary outcomes are already covered for all `24` pre-trained models and
do not need rerun for this label-change scope.

## Partition Order

Use this practical ordering:

1. `tier2q` first
2. `tier1q` if extra CPU capacity is needed
3. `tier3q` only if additional overflow is still needed

The Stage 3 runner already defaults to `tier2q`. To use a fallback partition
without editing the runner, override at submit time, for example:

```bash
sbatch --partition=tier1q --array=0-23 slurm/11_run_stage3_tier2q_lr.sh slurm/generated/outcome_reruns/10_additional_binary_logreg.jobfile
```

## No-Launch Commands

Stage 0 refresh:

```bash
sbatch slurm/13_refresh_all_extended_outcomes.sh
```

Stage 3 additional binary logistic reruns:

```bash
sbatch --array=0-23 slurm/11_run_stage3_tier2q_lr.sh slurm/generated/outcome_reruns/10_additional_binary_logreg.jobfile
```

Stage 3 regression ridge reruns:

```bash
sbatch --array=0-23 slurm/11_run_stage3_tier2q_lr.sh slurm/generated/outcome_reruns/11_regression_ridge.jobfile
```

## Outcome Panels

Additional binary outcomes:

- `hyperkalemia`
- `severe_hypokalemia`
- `severe_anemia`
- `hypoglycemia`
- `profound_hyponatremia`
- `severe_hypernatremia`
- `tachycardia_hr130`
- `severe_hypertension`
- `vasopressor_initiation`
- `hypotension`
- `crrt_initiation`
- `hemodialysis_initiation`

Regression outcomes:

- `peak_creatinine`
- `peak_troponin`
- `min_hemoglobin`
- `peak_potassium`
- `min_potassium`
- `min_glucose`
- `min_sodium`
- `max_sodium`
- `peak_bnp`
- `max_heart_rate`
- `max_sbp`
- `max_dbp`
