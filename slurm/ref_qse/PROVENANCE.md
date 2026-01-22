# Vendored reference SLURM scripts (provenance)

This folder vendors selected SLURM scripts from the lab reference repository `Quantifying-Surprise-EHRs`.

This benchmark does **not** run UCMC transfer experiments; we vendor these scripts to preserve the
exact launcher semantics (DDP + flags) while adapting only:
- paths (`IRB_HOME`, `FMS_EHRS_HOME`, data/model dirs)
- resource scaling (LR runs on `tier2q` with scaled CPUs/RAM for our larger cohort)
- naming to match our `config_id` + `data_version` conventions

Each script contains a provenance header referencing the original file path.

