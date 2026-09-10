#!/usr/bin/env python3
"""Post-hoc percentile bootstrap CIs for 24–48h ML4H probe metrics.

Reads existing prediction pickles and `metrics_long.parquet`. Writes a sibling
CI table and never overwrites `metrics_long`. Intermediate shards live under a
dedicated work root so array tasks cannot clobber each other or the live table.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn import metrics as skl_metrics


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METRICS_LONG = (
    REPO_ROOT / "outputs/runs/ml4h_2026/metrics/metrics_long.parquet"
)
DEFAULT_WORK_ROOT = (
    REPO_ROOT / "outputs/runs/ml4h_2026/bootstrap_24to48h_ci"
)
DEFAULT_OUTPUT_PARQUET = (
    REPO_ROOT / "outputs/runs/ml4h_2026/metrics/metrics_long_with_ci.parquet"
)
BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 42
ALPHA = 0.05
EXPECTED_RUNS = 156
EXPECTED_CI_ROWS = 20_400
SKIP_METRICS = frozenset({"average_precision"})
BINARY_METRICS = ("roc_auc", "brier", "ece_15")
REGRESSION_METRICS = ("spearman_rho", "r2", "mae", "rmse")
RUN_KEYS = ("experiment", "config_id", "backbone", "seed")
ROW_KEYS = (
    "experiment",
    "config_id",
    "backbone",
    "seed",
    "task_type",
    "outcome",
    "metric",
)
PROTECTED_METRIC_NAMES = frozenset({"metrics_long.parquet", "metrics_long.csv"})


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ece(y_true: np.ndarray, y_score: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    bucket = np.clip(np.digitize(y_score, edges[1:-1]), 0, bins - 1)
    value = 0.0
    for index in range(bins):
        mask = bucket == index
        if mask.any():
            value += mask.mean() * abs(y_true[mask].mean() - y_score[mask].mean())
    return float(value)


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(skl_metrics.roc_auc_score(y_true, y_score))


def brier(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(skl_metrics.brier_score_loss(y_true, y_score))


def spearman_rho(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(stats.spearmanr(y_true, y_pred).statistic)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(skl_metrics.r2_score(y_true, y_pred))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(skl_metrics.mean_absolute_error(y_true, y_pred))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(skl_metrics.mean_squared_error(y_true, y_pred)))


METRIC_FNS = {
    "roc_auc": roc_auc,
    "brier": brier,
    "ece_15": ece,
    "spearman_rho": spearman_rho,
    "r2": r2_score,
    "mae": mae,
    "rmse": rmse,
}


def assert_unique_ci_output(path: Path, metrics_long: Path) -> Path:
    resolved = path.expanduser().resolve()
    metrics_resolved = metrics_long.expanduser().resolve()
    if resolved == metrics_resolved:
        raise ValueError(f"Refusing to overwrite metrics_long: {resolved}")
    if resolved.name in PROTECTED_METRIC_NAMES:
        raise ValueError(f"Refusing protected filename: {resolved.name}")
    if "with_ci" not in resolved.name:
        raise ValueError(f"CI output name must contain 'with_ci': {resolved}")
    return resolved


def assert_unique_work_root(path: Path, metrics_long: Path, output_parquet: Path) -> Path:
    resolved = path.expanduser().resolve()
    metrics_resolved = metrics_long.expanduser().resolve()
    output_resolved = output_parquet.expanduser().resolve()
    if resolved == metrics_resolved:
        raise ValueError(f"Work root cannot be metrics_long: {resolved}")
    if resolved == output_resolved:
        raise ValueError(f"Work root cannot be the CI parquet: {resolved}")
    if resolved == metrics_resolved.parent:
        raise ValueError(f"Work root cannot be the dump 00_data directory: {resolved}")
    if "bootstrap_24to48h_ci" not in resolved.as_posix():
        raise ValueError(
            "Work root must contain 'bootstrap_24to48h_ci' so shards stay isolated: "
            f"{resolved}"
        )
    return resolved


def shard_paths(work_root: Path, run_id: str) -> tuple[Path, Path, Path]:
    shard_dir = work_root / "shards"
    tmp_dir = work_root / "tmp"
    shard_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    shard = shard_dir / f"{run_id}.parquet"
    done = shard_dir / f"{run_id}.done.json"
    tmp = tmp_dir / f"{run_id}.{os.getpid()}.parquet"
    if shard.resolve() == done.resolve():
        raise ValueError(f"Shard and done paths collided for {run_id}")
    return shard, done, tmp


def write_parquet_atomic(frame: pd.DataFrame, destination: Path, tmp: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp.parent.mkdir(parents=True, exist_ok=True)
    if tmp.resolve() == destination.resolve():
        raise ValueError(f"Temp path collides with destination: {destination}")
    if tmp.exists():
        tmp.unlink()
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, destination)


def run_id_for(record: pd.Series) -> str:
    safe_config = re.sub(r"[^A-Za-z0-9._-]+", "_", str(record["config_id"]))
    return (
        f"{int(record['run_index']):03d}__{record['experiment']}__"
        f"{record['backbone']}__s{int(record['seed'])}__{safe_config}"
    )


def load_ci_metrics(metrics_long: Path) -> pd.DataFrame:
    frame = pd.read_parquet(metrics_long)
    frame = frame.loc[~frame["metric"].astype(str).isin(SKIP_METRICS)].copy()
    if frame.empty:
        raise ValueError(f"No CI-eligible rows in {metrics_long}")
    unknown = set(frame["metric"].astype(str)) - set(METRIC_FNS)
    if unknown:
        raise ValueError(f"Unexpected metrics after dropping AUPRC: {sorted(unknown)}")
    return frame.reset_index(drop=True)


def list_runs(metrics: pd.DataFrame) -> pd.DataFrame:
    artifacts = (
        metrics.groupby([*RUN_KEYS, "task_type"], sort=False)["source_artifact"]
        .first()
        .unstack("task_type")
    )
    missing = {"binary", "regression"} - set(artifacts.columns)
    if missing:
        raise ValueError(f"Each run needs binary and regression pickles; missing {missing}")
    runs = artifacts.reset_index().rename(
        columns={"binary": "binary_artifact", "regression": "regression_artifact"}
    )
    runs = runs.sort_values(list(RUN_KEYS), kind="stable").reset_index(drop=True)
    runs["run_index"] = np.arange(len(runs), dtype=int)
    runs["run_id"] = [run_id_for(row) for _, row in runs.iterrows()]
    if runs["run_id"].duplicated().any():
        raise ValueError("run_id values are not unique.")
    return runs


def aligned_arrays(payload: dict, outcome: str) -> tuple[np.ndarray, np.ndarray]:
    raw_y_true = np.asarray(payload["labels"][outcome])
    y_pred = np.asarray(payload["predictions"][outcome], dtype=float)
    qualifier = np.asarray(payload["qualifiers"][outcome], dtype=bool)
    y_true = raw_y_true
    if len(y_true) != len(y_pred):
        if len(qualifier) != len(y_true) or int(qualifier.sum()) != len(y_pred):
            raise ValueError(f"Cannot align labels and predictions for {outcome}")
        y_true = y_true[qualifier]
    y_true = np.asarray(y_true, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[mask], y_pred[mask]


def percentile_interval(values: list[float]) -> tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    array = np.asarray(values, dtype=float)
    return (
        float(np.quantile(array, ALPHA / 2)),
        float(np.quantile(array, 1.0 - ALPHA / 2)),
    )


def bootstrap_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_names: tuple[str, ...],
    *,
    n_samples: int,
    seed: int,
    require_two_classes: bool,
) -> dict[str, dict[str, float | int]]:
    rng = np.random.default_rng(seed)
    n = int(len(y_true))
    collected: dict[str, list[float]] = {name: [] for name in metric_names}
    tries = 0
    max_tries = max(n_samples * 20, 100)
    accepted = 0
    while accepted < n_samples and tries < max_tries:
        tries += 1
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        ys = y_score[idx]
        if require_two_classes and np.unique(yt).size < 2:
            continue
        draw: dict[str, float] = {}
        valid = True
        for name in metric_names:
            try:
                value = float(METRIC_FNS[name](yt, ys))
            except Exception:
                valid = False
                break
            if not np.isfinite(value):
                valid = False
                break
            draw[name] = value
        if not valid:
            continue
        for name, value in draw.items():
            collected[name].append(value)
        accepted += 1
    out: dict[str, dict[str, float | int]] = {}
    for name in metric_names:
        lo, hi = percentile_interval(collected[name])
        out[name] = {
            "ci_lo": lo,
            "ci_hi": hi,
            "n_bootstrap_accepted": int(len(collected[name])),
        }
    return out


def load_pickle(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle)


def bootstrap_run(
    *,
    run: pd.Series,
    metrics: pd.DataFrame,
    n_samples: int,
    seed: int,
) -> pd.DataFrame:
    subset = metrics.loc[
        (metrics["experiment"] == run["experiment"])
        & (metrics["config_id"] == run["config_id"])
        & (metrics["backbone"] == run["backbone"])
        & (metrics["seed"] == int(run["seed"]))
    ].copy()
    n_binary = subset.loc[subset["task_type"] == "binary", "outcome"].nunique()
    n_regression = subset.loc[subset["task_type"] == "regression", "outcome"].nunique()
    expected = n_binary * len(BINARY_METRICS) + n_regression * len(REGRESSION_METRICS)
    if len(subset) != expected:
        raise ValueError(
            f"{run['run_id']}: expected {expected} CI rows from "
            f"{n_binary} binary and {n_regression} regression outcomes; found {len(subset)}"
        )
    payloads = {
        "binary": load_pickle(Path(str(run["binary_artifact"]))),
        "regression": load_pickle(Path(str(run["regression_artifact"]))),
    }
    records: list[dict[str, object]] = []
    for task_type, metric_names in (
        ("binary", BINARY_METRICS),
        ("regression", REGRESSION_METRICS),
    ):
        payload = payloads[task_type]
        outcomes = sorted(
            set(subset.loc[subset["task_type"] == task_type, "outcome"].astype(str))
        )
        for outcome in outcomes:
            y_true, y_pred = aligned_arrays(payload, outcome)
            intervals = bootstrap_metrics(
                y_true,
                y_pred,
                metric_names,
                n_samples=n_samples,
                seed=seed,
                require_two_classes=task_type == "binary",
            )
            for metric in metric_names:
                records.append(
                    {
                        "experiment": run["experiment"],
                        "config_id": run["config_id"],
                        "backbone": run["backbone"],
                        "seed": int(run["seed"]),
                        "task_type": task_type,
                        "outcome": outcome,
                        "metric": metric,
                        "run_id": run["run_id"],
                        "bootstrap_n": int(n_samples),
                        "bootstrap_seed": int(seed),
                        **intervals[metric],
                    }
                )
    boot = pd.DataFrame(records)
    merged = subset.merge(boot, on=list(ROW_KEYS), how="left", validate="one_to_one")
    if merged["ci_lo"].isna().all() and n_samples > 0:
        raise ValueError(f"{run['run_id']}: bootstrap produced no intervals.")
    return merged


def shard_is_complete(shard: Path, done: Path) -> bool:
    if not shard.is_file() or not done.is_file():
        return False
    try:
        payload = json.loads(done.read_text(encoding="utf-8"))
        frame = pd.read_parquet(shard)
    except Exception:
        return False
    expected = int(payload.get("n_rows", -1))
    return expected > 0 and len(frame) == expected and "ci_lo" in frame.columns


def write_run_index(
    runs: pd.DataFrame,
    work_root: Path,
    metrics_long: Path,
    *,
    bootstrap_n: int,
    bootstrap_seed: int,
) -> Path:
    (work_root / "tmp").mkdir(parents=True, exist_ok=True)
    index_path = work_root / "runs.csv"
    jobfile_path = work_root / "bootstrap_runs.jobfile"
    tmp_index = work_root / "tmp" / f"runs.{os.getpid()}.csv"
    runs.to_csv(tmp_index, index=False)
    os.replace(tmp_index, index_path)
    python = Path(sys.executable)
    script = Path(__file__).resolve()
    lines = []
    for _, run in runs.iterrows():
        lines.append(
            " ".join(
                [
                    str(python),
                    str(script),
                    "--mode",
                    "run",
                    "--run-index",
                    str(int(run["run_index"])),
                    "--metrics-long",
                    str(metrics_long),
                    "--work-root",
                    str(work_root),
                    "--bootstrap-n",
                    str(int(bootstrap_n)),
                    "--bootstrap-seed",
                    str(int(bootstrap_seed)),
                ]
            )
        )
    tmp_job = work_root / "tmp" / f"jobfile.{os.getpid()}.txt"
    tmp_job.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(tmp_job, jobfile_path)
    return jobfile_path


def concat_shards(
    *,
    metrics: pd.DataFrame,
    runs: pd.DataFrame,
    work_root: Path,
    output_parquet: Path,
    metrics_long: Path,
    strict_grid: bool,
) -> dict[str, object]:
    frames = []
    for _, run in runs.iterrows():
        shard, done, _tmp = shard_paths(work_root, str(run["run_id"]))
        if not shard_is_complete(shard, done):
            raise FileNotFoundError(f"Incomplete shard: {shard}")
        frames.append(pd.read_parquet(shard))
    if strict_grid:
        if len(runs) != EXPECTED_RUNS:
            raise ValueError(f"Expected {EXPECTED_RUNS} runs; found {len(runs)}")
        if len(metrics) != EXPECTED_CI_ROWS:
            raise ValueError(
                f"Expected {EXPECTED_CI_ROWS} CI-eligible metrics_long rows; found {len(metrics)}"
            )
    table = pd.concat(frames, ignore_index=True)
    if (table["metric"].astype(str) == "average_precision").any():
        raise ValueError("CI table contains average_precision rows.")
    expected_rows = int(len(metrics))
    if len(table) != expected_rows:
        raise ValueError(f"Expected {expected_rows} CI rows; found {len(table)}")
    ci_cols = [
        "ci_lo",
        "ci_hi",
        "bootstrap_n",
        "bootstrap_seed",
        "n_bootstrap_accepted",
        "run_id",
    ]
    left = metrics.sort_values(list(ROW_KEYS), kind="stable").reset_index(drop=True)
    right = table[list(ROW_KEYS) + ci_cols].sort_values(
        list(ROW_KEYS), kind="stable"
    ).reset_index(drop=True)
    merged = left.merge(right, on=list(ROW_KEYS), how="left", validate="one_to_one")
    merged = merged.sort_values(list(ROW_KEYS), kind="stable").reset_index(drop=True)
    if len(merged) != expected_rows:
        raise ValueError("Join to metrics_long changed the CI row count.")
    if merged[ci_cols].isna().any().any():
        raise ValueError("Join dropped CI values for at least one metrics_long row.")
    lo = merged["ci_lo"].to_numpy(dtype=float)
    hi = merged["ci_hi"].to_numpy(dtype=float)
    point = merged["point"].to_numpy(dtype=float)
    finite = np.isfinite(lo) & np.isfinite(hi) & np.isfinite(point)
    inside = finite & (lo - 1e-12 <= point) & (point <= hi + 1e-12)
    merged["point_in_interval"] = inside
    outside = merged.loc[~merged["point_in_interval"]]
    if not outside.empty:
        unexpected = outside.loc[outside["metric"].astype(str) != "ece_15"]
        if not unexpected.empty:
            raise ValueError(
                "Non-ECE intervals miss the point estimate: "
                f"{unexpected[list(ROW_KEYS)].head(5).to_dict(orient='records')}"
            )
    if (merged["n_bootstrap_accepted"] < int(merged["bootstrap_n"].iloc[0])).any():
        raise ValueError("At least one row finished fewer than the requested draws.")

    tmp = work_root / "tmp" / f"{output_parquet.name}.{os.getpid()}.parquet"
    write_parquet_atomic(merged, output_parquet, tmp)
    metadata = {
        "created_at": utc_now(),
        "analysis": "24to48h",
        "bootstrap_n": int(merged["bootstrap_n"].iloc[0]),
        "bootstrap_seed": int(merged["bootstrap_seed"].iloc[0]),
        "interval": "equal_tailed_percentile",
        "alpha": ALPHA,
        "n_rows": int(len(merged)),
        "n_runs": int(len(runs)),
        "n_point_outside_interval": int((~merged["point_in_interval"]).sum()),
        "ece_point_outside_interval": int(
            ((merged["metric"].astype(str) == "ece_15") & (~merged["point_in_interval"])).sum()
        ),
        "strict_grid": bool(strict_grid),
        "metrics": list(BINARY_METRICS + REGRESSION_METRICS),
        "excluded_metrics": sorted(SKIP_METRICS),
        "source_metrics_long": str(metrics_long),
        "source_metrics_long_sha256": sha256(metrics_long),
        "output_parquet": str(output_parquet),
        "output_sha256": sha256(output_parquet),
        "work_root": str(work_root),
        "generator": str(Path(__file__).resolve()),
    }
    metadata_path = output_parquet.with_suffix(".json")
    if metadata_path.resolve() == metrics_long.resolve():
        raise ValueError("Metadata path collided with metrics_long.")
    tmp_meta = work_root / "tmp" / f"{metadata_path.name}.{os.getpid()}.json"
    tmp_meta.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp_meta, metadata_path)
    return metadata


def mode_list(args: argparse.Namespace) -> int:
    metrics = load_ci_metrics(args.metrics_long)
    runs = list_runs(metrics)
    if args.strict_grid and len(runs) != EXPECTED_RUNS:
        raise ValueError(f"Expected {EXPECTED_RUNS} runs; found {len(runs)}")
    jobfile = write_run_index(
        runs,
        args.work_root,
        args.metrics_long,
        bootstrap_n=int(args.bootstrap_n),
        bootstrap_seed=int(args.bootstrap_seed),
    )
    print(
        json.dumps(
            {
                "n_runs": int(len(runs)),
                "n_ci_rows": int(len(metrics)),
                "work_root": str(args.work_root),
                "jobfile": str(jobfile),
                "runs_csv": str(args.work_root / "runs.csv"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def mode_run(args: argparse.Namespace) -> int:
    metrics = load_ci_metrics(args.metrics_long)
    runs = list_runs(metrics)
    if args.run_index < 0 or args.run_index >= len(runs):
        raise IndexError(f"run-index {args.run_index} is outside 0..{len(runs) - 1}")
    run = runs.iloc[args.run_index]
    shard, done, tmp = shard_paths(args.work_root, str(run["run_id"]))
    if shard_is_complete(shard, done):
        print(json.dumps({"status": "skipped", "run_id": run["run_id"], "shard": str(shard)}))
        return 0
    table = bootstrap_run(
        run=run,
        metrics=metrics,
        n_samples=int(args.bootstrap_n),
        seed=int(args.bootstrap_seed),
    )
    write_parquet_atomic(table, shard, tmp)
    done.write_text(
        json.dumps(
            {
                "run_id": run["run_id"],
                "run_index": int(run["run_index"]),
                "n_rows": int(len(table)),
                "bootstrap_n": int(args.bootstrap_n),
                "bootstrap_seed": int(args.bootstrap_seed),
                "shard_sha256": sha256(shard),
                "created_at": utc_now(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "wrote",
                "run_id": run["run_id"],
                "n_rows": int(len(table)),
                "shard": str(shard),
            }
        )
    )
    return 0


def mode_concat(args: argparse.Namespace) -> int:
    metrics = load_ci_metrics(args.metrics_long)
    runs = list_runs(metrics)
    metadata = concat_shards(
        metrics=metrics,
        runs=runs,
        work_root=args.work_root,
        output_parquet=args.output_parquet,
        metrics_long=args.metrics_long,
        strict_grid=bool(args.strict_grid),
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("list", "run", "concat"),
        required=True,
        help="list writes the unique run index; run bootstraps one model; concat merges shards.",
    )
    parser.add_argument("--metrics-long", type=Path, default=DEFAULT_METRICS_LONG)
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    parser.add_argument("--output-parquet", type=Path, default=DEFAULT_OUTPUT_PARQUET)
    parser.add_argument("--run-index", type=int, default=None)
    parser.add_argument("--bootstrap-n", type=int, default=BOOTSTRAP_N)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument(
        "--strict-grid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require the full 156-run / 20,400-row 24–48h grid.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.metrics_long = args.metrics_long.expanduser().resolve()
    args.output_parquet = assert_unique_ci_output(args.output_parquet, args.metrics_long)
    args.work_root = assert_unique_work_root(
        args.work_root, args.metrics_long, args.output_parquet
    )
    args.work_root.mkdir(parents=True, exist_ok=True)
    (args.work_root / "shards").mkdir(exist_ok=True)
    (args.work_root / "tmp").mkdir(exist_ok=True)
    (args.work_root / "logs").mkdir(exist_ok=True)
    if not args.metrics_long.is_file():
        raise FileNotFoundError(args.metrics_long)
    if args.mode == "list":
        return mode_list(args)
    if args.mode == "run":
        if args.run_index is None:
            raise ValueError("--run-index is required for --mode run")
        return mode_run(args)
    return mode_concat(args)


if __name__ == "__main__":
    raise SystemExit(main())
