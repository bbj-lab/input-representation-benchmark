# Benchmark SLURM layout

Live benchmark launchers. Python entrypoints live in `../pipeline/`, `../paper/`,
and `../utilities/`. Run path: [`../README.md`](../README.md).

## Numbered runners

| Prefix | Role |
| --- | --- |
| `00_*` | shared preamble |
| `01_*` | MEDS extraction |
| `02_*`–`04_*` | Stage 0 |
| `04_*`, `07_*`, `08_*` | Stage 1 |
| `09_*` | Stage 2 array runner |
| `10_*`, `11_*` | Stage 3 |
| `12_*` | gated helpers |
| `13_*` | extended outcome refresh |
| `15_*` | stats regeneration |
| `17_*` | additional-run jobfile generation and high-GPU helpers |
| `18_*` | serial GPU runners and resume helpers |

## Local byproducts

| Path | Role |
| --- | --- |
| `generated/statistics/` | stats rerun jobfiles |
| `generated/outcome_reruns/` | one-off outcome refresh jobfiles |
| `generated/revision_gfkb/` | additional-run jobfiles (temporary `revision_*` name) |
| `state/*.last_completed` | serial-runner resume markers |

`ref_qse/` holds vendored Stage 2/3 wrappers. Retired launchers belong in
`../deprecated/slurm/`.
