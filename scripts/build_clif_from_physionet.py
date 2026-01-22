#!/usr/bin/env python3
"""
Build CLIF (2.1) parquet tables from PhysioNet MIMIC-IV v3.1 raw CSVs using CLIF-MIMIC.

This script is the **Exp3 data preparation** entrypoint for this benchmark.

Pipeline (modular, evidence-driven)
----------------------------------
Model:
  - CLIF-MIMIC converts MIMIC-IV CSV -> CLIF-2.1 `clif_*.parquet` tables.
  - fms-ehrs (clifpy wrappers) adds derived tables required by our CLIF tokenizer config:
      - clif_respiratory_support_processed.parquet (waterfall)
      - clif_medication_admin_*_converted.parquet (dose unit conversion)
      - clif_sofa.parquet (SOFA scoring)
  - Benchmark scripts define a **patient-level split** (to prevent leakage) and
    then restrict to **ICU-eligible hospitalizations** within each patient split
    (ICU stay + LOS>=24h) so Exp3 isolates vocabulary semantics rather than cohort drift.

Outputs:
  - <exp3_root>/cohort_splits/ (patient_ids + hadm_ids per split)
  - <exp3_root>/clif_unsplit/  (single directory of clif_*.parquet + derived tables)
  - <clif_out_root>/raw/{train,val,test}/clif_*.parquet  (split CLIF tables for tokenization)

Notes:
  - This script does NOT create MEDS (Exp1/2). MEDS is already present.
  - You must have PhysioNet MIMIC-IV v3.1 CSVs available locally.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], *, cwd: Path | None = None) -> None:
    # Internal validity / reproducibility: ensure all subprocesses use the same Python
    # interpreter environment as the driver script (avoids missing deps like numpy).
    if cmd and cmd[0] == "python":
        cmd = [sys.executable] + cmd[1:]
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def _write_clif_mimic_config(
    *, clif_mimic_repo: Path, mimic_csv_dir: Path, mimic_parquet_dir: Path, clif_tables: list[str]
) -> Path:
    cfg_dir = clif_mimic_repo / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / "config.json"

    table_flags = {k: 0 for k in [
        "patient",
        "hospitalization",
        "adt",
        "vitals",
        "labs",
        "patient_assessments",
        "respiratory_support",
        "medication_admin_continuous",
        "medication_admin_intermittent",
        "position",
        "crrt_therapy",
        "ecmo_mcs",
        "code_status",
        "hospital_diagnosis",
        "patient_procedures",
    ]}
    for t in clif_tables:
        if t not in table_flags:
            raise ValueError(f"Unknown CLIF table '{t}' (not in CLIF-MIMIC config template).")
        table_flags[t] = 1

    cfg = {
        "current_workspace": "exp3",
        "exp3": {
            "mimic_csv_dir": str(mimic_csv_dir),
            "mimic_parquet_dir": str(mimic_parquet_dir),
        },
        # if parquet dir already exists/populated, CLIF-MIMIC can skip csv->parquet via this flag,
        # but we keep it explicit here to avoid hidden assumptions.
        "create_mimic_parquet_from_csv": 1,
        "overwrite_existing_mimic_parquet": 0,
        "clif_output_dir_name": "",  # default rclif-{clif_version}
        "mimic_version": "3.1",
        "clif_version": "2.1",
        "clif_tables": table_flags,
    }
    cfg_path.write_text(json.dumps(cfg, indent=2))
    return cfg_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build CLIF from PhysioNet MIMIC-IV v3.1 for Experiment 3.")
    p.add_argument(
        "--mimic_dir",
        type=Path,
        default=Path("physionet.org/files/mimiciv/3.1"),
        help="PhysioNet MIMIC-IV v3.1 root (default: physionet.org/files/mimiciv/3.1).",
    )
    p.add_argument(
        "--clif_mimic_repo",
        type=Path,
        default=Path("../CLIF-MIMIC"),
        help="Path to CLIF-MIMIC repo (default: ../CLIF-MIMIC).",
    )
    p.add_argument(
        "--clif_out_root",
        type=Path,
        default=Path("data/clif"),
        help="Output CLIF root for this benchmark (default: data/clif).",
    )
    p.add_argument(
        "--exp3_root",
        type=Path,
        default=Path("data/exp3"),
        help="Working directory for Exp3 preparation artifacts (default: data/exp3).",
    )
    p.add_argument("--data_seed", type=int, default=42, help="Seed for patient-level splitting.")
    p.add_argument("--min_los_hours", type=float, default=24.0, help="Min hospital LOS (hours) for ICU-eligibility.")
    p.add_argument(
        "--tables",
        type=str,
        nargs="+",
        default=[
            # tables required by fms-ehrs clif-21.yaml (or upstream for derived tables)
            "patient",
            "hospitalization",
            "adt",
            "vitals",
            "labs",
            "patient_assessments",
            "respiratory_support",
            "medication_admin_continuous",
            "medication_admin_intermittent",
            "position",
            "crrt_therapy",
            "code_status",
            "hospital_diagnosis",
            "patient_procedures",
        ],
        help="CLIF tables to build via CLIF-MIMIC (default: tables used by clif-21 tokenizer).",
    )
    p.add_argument(
        "--run_clifpy",
        action="store_true",
        help="Run fms-ehrs clifpy wrappers to create processed/converted tables needed by tokenization.",
    )
    args = p.parse_args(argv)

    irb_root = Path(__file__).resolve().parents[1]
    mimic_dir = args.mimic_dir.expanduser().resolve()
    clif_mimic_repo = args.clif_mimic_repo.expanduser().resolve()
    clif_out_root = (irb_root / args.clif_out_root).resolve() if not args.clif_out_root.is_absolute() else args.clif_out_root
    exp3_root = (irb_root / args.exp3_root).resolve() if not args.exp3_root.is_absolute() else args.exp3_root

    if not (mimic_dir / "hosp").exists() or not (mimic_dir / "icu").exists():
        raise FileNotFoundError(f"Invalid --mimic_dir (expected hosp/ and icu/): {mimic_dir}")
    if not (clif_mimic_repo / "main.py").exists():
        raise FileNotFoundError(f"Invalid --clif_mimic_repo (missing main.py): {clif_mimic_repo}")

    exp3_root.mkdir(parents=True, exist_ok=True)
    cohort_dir = exp3_root / "cohort_splits"
    clif_unsplit_dir = exp3_root / "clif_unsplit"
    clif_unsplit_dir.mkdir(parents=True, exist_ok=True)

    # 1) Cohort definition (patient-level split + ICU-eligible hadm list per split)
    _run(
        [
            "python",
            str(irb_root / "scripts" / "align_cohorts.py"),
            "--mimic_dir",
            str(mimic_dir),
            "--output_dir",
            str(cohort_dir),
            "--data_seed",
            str(args.data_seed),
            "--min_los_hours",
            str(args.min_los_hours),
        ]
    )

    # 2) Run CLIF-MIMIC
    mimic_parquet_dir = exp3_root / "mimic_parquet"
    cfg_path = _write_clif_mimic_config(
        clif_mimic_repo=clif_mimic_repo,
        mimic_csv_dir=mimic_dir,
        mimic_parquet_dir=mimic_parquet_dir,
        clif_tables=args.tables,
    )
    _run(["python", "main.py"], cwd=clif_mimic_repo)

    # 3) Copy CLIF-MIMIC output tables into clif_unsplit_dir
    # CLIF-MIMIC writes to <repo>/output/rclif-2.1/ by default.
    clif_mimic_out = clif_mimic_repo / "output" / "rclif-2.1"
    if not clif_mimic_out.exists():
        # fallback for alternative naming seen in docs
        clif_mimic_out = clif_mimic_repo / "output" / "rclif-2.1.0"
    if not clif_mimic_out.exists():
        raise FileNotFoundError(f"CLIF-MIMIC output directory not found under {clif_mimic_repo / 'output'}")

    for fp in sorted(clif_mimic_out.glob("clif_*.parquet")):
        shutil.copy2(fp, clif_unsplit_dir / fp.name)

    # 4) Optional: run clifpy wrappers to create derived tables needed by tokenization
    if args.run_clifpy:
        fms_repo = (irb_root / ".." / "fms-ehrs").resolve()
        run_clifpy = fms_repo / "fms_ehrs" / "scripts" / "run_clifpy.py"
        run_sofa = fms_repo / "fms_ehrs" / "scripts" / "run_sofa_scoring.py"
        if not run_clifpy.exists():
            raise FileNotFoundError(f"Missing fms-ehrs clifpy wrapper: {run_clifpy}")

        # respiratory_support_processed + med dose conversion outputs are written into data_dir
        _run(
            [
                "python",
                str(run_clifpy),
                "--data_dir",
                str(clif_unsplit_dir),
                "--tz",
                "UTC",
                "--waterfall",
                "--convert_doses_continuous",
                "--convert_doses_intermittent",
            ]
        )
        _run(["python", str(run_sofa), "--data_dir", str(clif_unsplit_dir), "--tz", "UTC"])

    # 5) Split by ICU-eligible hospitalizations within patient splits
    # This avoids non-ICU admissions from ICU patients contaminating Exp3.
    _run(
        [
            "python",
            str(irb_root / "scripts" / "split_clif_by_patient_splits.py"),
            "--clif_in_dir",
            str(clif_unsplit_dir),
            "--splits_dir",
            str(cohort_dir),
            "--clif_out_root",
            str(clif_out_root),
            "--filter_key",
            "hospitalization_id",
        ]
    )

    print("OK")
    print(f"  Wrote cohort splits: {cohort_dir}")
    print(f"  Wrote unsplit CLIF:  {clif_unsplit_dir}")
    print(f"  Wrote split CLIF:    {clif_out_root / 'raw'}")
    print(f"  Wrote CLIF-MIMIC config: {cfg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

