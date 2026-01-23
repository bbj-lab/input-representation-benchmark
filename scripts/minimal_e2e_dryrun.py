#!/usr/bin/env python3
"""
Minimal end-to-end dry run (local) for the MEDS → tokenization → outcomes path.

Goal
----
Before submitting any SLURM jobs on Randi, we want a cheap correctness check that:
1) fms-ehrs tokenization runs on MEDS sharded data
2) tokenization emits numeric_values aligned to tokens (and padded_* alignment)
3) MEDS outcome extraction runs and produces tokens_timelines_outcomes.parquet
4) MEDS tokenized layout can be normalized for fms-ehrs compatibility

This script samples a tiny cohort from an existing MEDS shard, writes it into a
temporary working directory (no `meds.parquet` required), runs tokenization, and
then runs outcome extraction.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import polars as pl


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Minimal local dry run for MEDS pipeline pieces.")
    # Repo roots (avoid hard-coded absolute paths; default to sibling layout).
    irb_home = Path(__file__).resolve().parents[1]
    p.add_argument(
        "--irb_home",
        type=Path,
        default=irb_home,
        help="Path to input-representation-benchmark repo root (default: inferred from this file).",
    )
    p.add_argument(
        "--fms_ehrs_home",
        type=Path,
        default=(irb_home / "../fms-ehrs").resolve(),
        help="Path to sibling fms-ehrs repo (default: ../fms-ehrs).",
    )
    p.add_argument(
        "--meds_shard",
        type=Path,
        default=Path(
            (irb_home / "benchmarks/mimic-meds-extraction/data/meds/data/train/0.parquet")
        ),
        help="Path to a MEDS shard parquet to sample from.",
    )
    p.add_argument(
        "--tokenizer_config",
        type=Path,
        default=Path((irb_home / "../fms-ehrs/fms_ehrs/config/mimic-meds-ed.yaml").resolve()),
        help="Tokenizer YAML config (fms-ehrs).",
    )
    p.add_argument("--n_hadm", type=int, default=5, help="Number of admissions (hadm_id) to sample.")
    p.add_argument(
        "--eval_max_padded_len",
        type=int,
        default=4096,
        help="Also run an eval-retokenization pass reusing the train vocab, with this max_padded_len.",
    )
    args = p.parse_args(argv)

    irb_home = args.irb_home.expanduser().resolve()
    fms_ehrs_home = args.fms_ehrs_home.expanduser().resolve()
    meds_shard = args.meds_shard.expanduser().resolve()
    cfg = args.tokenizer_config.expanduser().resolve()

    # Choose a few admissions with LOS >= 24h (tokenizer day_stay_filter requires this).
    meds = pl.scan_parquet(meds_shard).select(["subject_id", "hadm_id", "time", "code"])
    adm = (
        meds.filter(pl.col("hadm_id").is_not_null())
        .filter(pl.col("code").str.starts_with("HOSPITAL_ADMISSION"))
        .group_by("hadm_id")
        .agg(admission_time=pl.col("time").min(), subject_id=pl.col("subject_id").first())
    )
    dsc = (
        meds.filter(pl.col("hadm_id").is_not_null())
        .filter(pl.col("code").str.starts_with("HOSPITAL_DISCHARGE"))
        .group_by("hadm_id")
        .agg(discharge_time=pl.col("time").min())
    )
    cohort = (
        adm.join(dsc, on="hadm_id", how="inner")
        .with_columns(los_hours=(pl.col("discharge_time") - pl.col("admission_time")).dt.total_hours())
        .filter(pl.col("los_hours") >= 24)
        .head(args.n_hadm)
        .collect()
    )
    if cohort.is_empty():
        raise SystemExit(f"No LOS>=24h admissions found in {meds_shard}")

    hadm_ids = cohort["hadm_id"].to_list()
    subject_ids = cohort["subject_id"].to_list()

    subset = (
        pl.scan_parquet(meds_shard)
        .filter(
            (pl.col("hadm_id").is_in(hadm_ids))
            | (pl.col("hadm_id").is_null() & pl.col("subject_id").is_in(subject_ids))
        )
        .collect()
    )

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        split_dir = root / "train"
        split_dir.mkdir(parents=True)
        subset.write_parquet(split_dir / "0.parquet")

        # Tokenize (fms-ehrs)
        import sys

        sys.path.insert(0, str(fms_ehrs_home))
        from fms_ehrs.framework.tokenizer import Tokenizer21

        tkzr = Tokenizer21(data_dir=split_dir, config_file=cfg)
        tt = tkzr.get_tokens_timelines()
        assert "numeric_values" in tt.columns
        assert (tt.select(pl.col("tokens").list.len() == pl.col("numeric_values").list.len()).to_series()).all()

        tt = tkzr.pad_and_truncate(tt)
        assert "padded_numeric_values" in tt.columns
        assert "padded_times" in tt.columns

        tok_dir = root / "tokenized" / "train"
        tok_dir.mkdir(parents=True)
        tt.write_parquet(tok_dir / "tokens_timelines.parquet")
        tkzr.vocab.save(tok_dir / "vocab.gzip")

        # Eval-time retokenization (Stage0E analogue): reuse vocab, change max_padded_len.
        eval_L = int(args.eval_max_padded_len)
        if eval_L <= 1:
            raise ValueError("--eval_max_padded_len must be >= 2")
        tkzr_eval = Tokenizer21(
            data_dir=split_dir,
            config_file=cfg,
            vocab_path=(tok_dir / "vocab.gzip"),
            max_padded_len=eval_L,
        )
        tt_eval = tkzr_eval.get_tokens_timelines()
        tt_eval = tkzr_eval.pad_and_truncate(tt_eval)
        assert "padded" in tt_eval.columns
        assert int(tt_eval.select(pl.col("padded").list.len().max()).item()) == eval_L
        tok_eval_dir = root / f"tokenized_evalL{eval_L}" / "train"
        tok_eval_dir.mkdir(parents=True)
        tt_eval.write_parquet(tok_eval_dir / "tokens_timelines.parquet")
        tkzr_eval.vocab.save(tok_eval_dir / "vocab.gzip")

        # Compute outcomes (input-representation-benchmark)
        sys.path.insert(0, str(irb_home))
        from scripts.extract_outcomes_meds import main as extract_outcomes_main

        rc = extract_outcomes_main(
            [
                "--meds_events_dir",
                str(root),
                "--tokenized_dir",
                str(root / "tokenized"),
                "--splits",
                "train",
            ]
        )
        assert rc == 0
        out = pl.read_parquet(tok_dir / "tokens_timelines_outcomes.parquet")
        print(out.select(["hospitalization_id", "icu_admission", "imv_event"]).head())

        # Normalize layout
        from scripts.normalize_meds_tokenized_layout import main as norm_main

        rc2 = norm_main(["--input_dir", str(root / "tokenized"), "--output_dir", str(root / "tokenized_norm")])
        assert rc2 == 0

    print("[minimal_e2e_dryrun] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

