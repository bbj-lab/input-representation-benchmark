#!/usr/bin/env python3
"""
Archived XGBoost/LightGBM baseline using standard 24h feature aggregations.

This script is kept under `deprecated/` because the current manuscript no longer
uses the tabular baseline in the active paper path. It preserves the historical
implementation for audit or rerun purposes.

Usage (in input-rep conda env):
    python deprecated/scripts/xgboost_baseline.py \
        --meds_events_dir data/clif/raw \
        --output_dir deprecated/outputs/xgboost_baseline

SLURM submission:
    See deprecated/slurm/xgboost_baseline.sh
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score

try:
    import xgboost as xgb

    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb

    HAS_LGB = True
except ImportError:
    HAS_LGB = False

from pipeline.scripts.extract_outcomes_meds import (
    DEFAULT_IMV_ITEMIDS,
    _compute_outcomes_from_meds,
    _scan_meds_events,
)


BINARY_OUTCOMES = [
    ("same_admission_death", "Mortality"),
    ("long_length_of_stay", "Long LoS"),
    ("icu_admission", "ICU Admission"),
    ("imv_event", "IMV"),
]


def _extract_24h_features(meds: pl.LazyFrame) -> pl.DataFrame:
    """
    Extract first-24h feature aggregations from MEDS events.

    For each hospitalization:
      - Filter to first 24h (from admission time)
      - For numeric-valued events: compute min, max, mean, count per code
      - For categorical events: compute count per code

    Returns wide-format DataFrame: one row per hadm_id.
    """
    admissions = (
        meds.filter(pl.col("hadm_id").is_not_null())
        .filter(pl.col("code").str.starts_with("HOSPITAL_ADMISSION"))
        .group_by("hadm_id")
        .agg(admission_time=pl.col("time").min())
    ).collect()

    events = (
        meds.filter(pl.col("hadm_id").is_not_null())
        .collect()
        .join(admissions, on="hadm_id", how="inner")
    )

    obs_window = pl.duration(hours=24)
    events_24h = events.filter(
        (pl.col("time") - pl.col("admission_time")) <= obs_window
    )

    numeric_events = events_24h.filter(pl.col("numeric_value").is_not_null())

    if numeric_events.height > 0:
        numeric_agg = (
            numeric_events.group_by("hadm_id", "code").agg(
                val_min=pl.col("numeric_value").min(),
                val_max=pl.col("numeric_value").max(),
                val_mean=pl.col("numeric_value").mean(),
                val_count=pl.col("numeric_value").count(),
            )
        )

        top_codes = (
            numeric_agg.group_by("code")
            .agg(n=pl.col("hadm_id").count())
            .sort("n", descending=True)
            .head(200)
            .select("code")
        )
        numeric_agg = numeric_agg.join(top_codes, on="code", how="inner")

        features_list = []
        for stat_col, stat_name in [
            ("val_min", "min"),
            ("val_max", "max"),
            ("val_mean", "mean"),
            ("val_count", "count"),
        ]:
            pivoted = (
                numeric_agg.select("hadm_id", "code", stat_col)
                .pivot(on="code", index="hadm_id", values=stat_col)
                .rename(
                    {
                        c: f"{c}__{stat_name}"
                        for c in numeric_agg.select("code")
                        .unique()
                        .to_series()
                        .to_list()
                        if c != "hadm_id"
                    }
                )
            )
            rename_map = {}
            for c in pivoted.columns:
                if c != "hadm_id" and not c.endswith(f"__{stat_name}"):
                    rename_map[c] = f"{c}__{stat_name}"
            if rename_map:
                pivoted = pivoted.rename(rename_map)
            features_list.append(pivoted)

        features = features_list[0]
        for f in features_list[1:]:
            features = features.join(f, on="hadm_id", how="outer_coalesce")
    else:
        features = admissions.select("hadm_id")

    cat_events = events_24h.filter(pl.col("numeric_value").is_null())
    if cat_events.height > 0:
        cat_agg = (
            cat_events.group_by("hadm_id", "code").agg(cat_count=pl.col("time").count())
        )
        top_cat_codes = (
            cat_agg.group_by("code")
            .agg(n=pl.col("hadm_id").count())
            .sort("n", descending=True)
            .head(100)
            .select("code")
        )
        cat_agg = cat_agg.join(top_cat_codes, on="code", how="inner")
        cat_pivot = cat_agg.pivot(on="code", index="hadm_id", values="cat_count")
        rename_map = {}
        for c in cat_pivot.columns:
            if c != "hadm_id":
                rename_map[c] = f"{c}__cat_count"
        if rename_map:
            cat_pivot = cat_pivot.rename(rename_map)

        features = features.join(cat_pivot, on="hadm_id", how="outer_coalesce")

    features = features.fill_null(0)
    return features


def _bootstrap_ci(y_true, y_score, metric_fn, n_boot=1000, seed=42):
    """Bootstrap 95% CI for a metric."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(metric_fn(y_true[idx], y_score[idx]))
    if not scores:
        return 0.0, 0.0, 0.0
    point = metric_fn(y_true, y_score)
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return point, lo, hi


def main():
    parser = argparse.ArgumentParser(
        description="Archived XGBoost/LightGBM baseline for EHR prediction."
    )
    parser.add_argument(
        "--meds_events_dir",
        type=Path,
        required=True,
        help="MEDS events directory with train/val/test splits.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("deprecated/outputs/xgboost_baseline"),
        help="Output directory for archived results.",
    )
    parser.add_argument(
        "--model",
        choices=["xgboost", "lightgbm", "auto"],
        default="auto",
        help="Which model to use (default: auto = try xgboost, fallback lightgbm).",
    )
    parser.add_argument(
        "--n_bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap resamples for CIs.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.model == "auto":
        if HAS_XGB:
            model_name = "xgboost"
        elif HAS_LGB:
            model_name = "lightgbm"
        else:
            print("ERROR: Neither xgboost nor lightgbm is installed.")
            print("Install with: pip install xgboost  OR  pip install lightgbm")
            return 1
    else:
        model_name = args.model

    print(f"Using model: {model_name}")
    print("\n[1/4] Extracting 24h features...")

    splits = {}
    for split_name in ("train", "val", "test"):
        split_dir = args.meds_events_dir / split_name
        if not split_dir.exists():
            print(f"  SKIP: {split_dir} not found")
            continue

        print(f"  Processing {split_name}...")

        if (split_dir / "clif_hospitalization.parquet").exists():
            print("    Detected CLIF raw format (adapting to MEDS schema)")
            labs = pl.scan_parquet(split_dir / "clif_labs.parquet").select(
                pl.col("hospitalization_id").cast(pl.Int64).alias("hadm_id"),
                pl.col("lab_loinc_code").cast(pl.String).alias("code"),
                pl.col("lab_value_numeric").cast(pl.Float32).alias("numeric_value"),
                pl.col("lab_result_dttm").alias("time"),
            )
            vitals = pl.scan_parquet(split_dir / "clif_vitals.parquet").select(
                pl.col("hospitalization_id").cast(pl.Int64).alias("hadm_id"),
                pl.col("vital_category").cast(pl.String).alias("code"),
                pl.col("vital_value").cast(pl.Float32).alias("numeric_value"),
                pl.col("recorded_dttm").alias("time"),
            )
            hosp_adm = pl.scan_parquet(split_dir / "clif_hospitalization.parquet").select(
                pl.col("hospitalization_id").cast(pl.Int64).alias("hadm_id"),
                pl.lit("HOSPITAL_ADMISSION").cast(pl.String).alias("code"),
                pl.lit(None).cast(pl.Float32).alias("numeric_value"),
                pl.col("admission_dttm").alias("time"),
            )
            meds_adapted = pl.concat([labs, vitals, hosp_adm], how="diagonal")
            features = _extract_24h_features(meds_adapted)

            outcomes = (
                pl.scan_parquet(split_dir / "clif_hospitalization.parquet")
                .select(
                    pl.col("hospitalization_id").cast(pl.Int64).alias("hadm_id"),
                    same_admission_death=pl.col("discharge_category")
                    .str.contains("(?i)dead|expire|death")
                    .fill_null(False),
                    long_length_of_stay=(
                        (pl.col("discharge_dttm") - pl.col("admission_dttm")).dt.total_hours()
                        > 7 * 24
                    ).fill_null(False),
                    icu_admission=pl.lit(False),
                    imv_event=pl.lit(False),
                )
                .collect()
            )
        else:
            meds = _scan_meds_events(split_dir)
            features = _extract_24h_features(meds)
            outcomes = _compute_outcomes_from_meds(
                meds,
                imv_itemids=DEFAULT_IMV_ITEMIDS,
                los_days=7.0,
                observation_hours=24.0,
                icu_los_hours_threshold=48.0,
            ).collect()
            outcomes = outcomes.with_columns(
                pl.col("hospitalization_id").cast(pl.Int64).alias("hadm_id")
            )

        merged = features.join(
            outcomes.select(["hadm_id"] + [c for c, _ in BINARY_OUTCOMES]),
            on="hadm_id",
            how="inner",
        )

        print(
            f"    {split_name}: {merged.height:,} admissions, "
            f"{len(merged.columns) - 1 - len(BINARY_OUTCOMES)} features"
        )
        splits[split_name] = merged

    if "train" not in splits or "test" not in splits:
        print("ERROR: need at least train and test splits")
        return 1

    print("\n[2/4] Preparing feature matrices...")

    outcome_cols = [c for c, _ in BINARY_OUTCOMES]
    feature_cols = [c for c in splits["train"].columns if c not in ["hadm_id"] + outcome_cols]

    common_features = set(feature_cols)
    for split_name in splits:
        common_features &= set(
            c for c in splits[split_name].columns if c not in ["hadm_id"] + outcome_cols
        )
    feature_cols = sorted(common_features)

    print(f"  Common features: {len(feature_cols)}")

    X_train = splits["train"].select(feature_cols).to_numpy().astype(np.float32)
    X_test = splits["test"].select(feature_cols).to_numpy().astype(np.float32)
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    print("\n[3/4] Training and evaluating...")

    results = {}
    for col, display_name in BINARY_OUTCOMES:
        y_train = splits["train"][col].to_numpy().astype(int)
        y_test = splits["test"][col].to_numpy().astype(int)

        pos_rate = y_train.mean()
        print(f"\n  {display_name}: train prevalence = {pos_rate:.1%}")

        if len(np.unique(y_train)) < 2:
            print("    SKIP: single class in training set")
            continue

        scale_pos_weight = (1 - pos_rate) / max(pos_rate, 1e-6)

        if model_name == "xgboost":
            model = xgb.XGBClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss",
                use_label_encoder=False,
                random_state=42,
                n_jobs=-1,
                verbosity=0,
            )
        else:
            model = lgb.LGBMClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y_train)

        y_score = model.predict_proba(X_test)[:, 1]

        auroc, auroc_lo, auroc_hi = _bootstrap_ci(
            y_test, y_score, roc_auc_score, args.n_bootstrap
        )
        auprc, auprc_lo, auprc_hi = _bootstrap_ci(
            y_test, y_score, average_precision_score, args.n_bootstrap
        )

        results[display_name] = {
            "AUROC": f"{auroc:.3f} [{auroc_lo:.3f}, {auroc_hi:.3f}]",
            "AUPRC": f"{auprc:.3f} [{auprc_lo:.3f}, {auprc_hi:.3f}]",
            "auroc_point": float(auroc),
            "auprc_point": float(auprc),
            "test_prevalence": float(y_test.mean()),
            "train_n": int(len(y_train)),
            "test_n": int(len(y_test)),
        }

        print(f"    AUROC: {auroc:.3f} [{auroc_lo:.3f}, {auroc_hi:.3f}]")
        print(f"    AUPRC: {auprc:.3f} [{auprc_lo:.3f}, {auprc_hi:.3f}]")

    print("\n[4/4] Saving results...")

    out_json = args.output_dir / "xgboost_baseline_results.json"
    with open(out_json, "w") as f:
        json.dump({"model": model_name, "features": len(feature_cols), "results": results}, f, indent=2)
    print(f"  Results saved to: {out_json}")

    print("\n" + "=" * 60)
    print("SUMMARY TABLE (archived xgBoost baseline)")
    print("=" * 60)
    header = f"{'Outcome':20s}  {'AUROC':>10s}  {'AUPRC':>10s}"
    print(header)
    print("-" * len(header))
    for name, r in results.items():
        print(f"{name:20s}  {r['auroc_point']:>10.3f}  {r['auprc_point']:>10.3f}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
