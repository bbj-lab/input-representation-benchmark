from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "pipeline"
    / "scripts"
    / "bootstrap_ml4h_24to48h_metrics.py"
)


def load_module():
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("bootstrap_ml4h_24to48h_metrics", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _payload(kind: str, outcomes: tuple[str, ...], n: int, rng: np.random.Generator) -> dict:
    labels: dict[str, np.ndarray] = {}
    preds: dict[str, np.ndarray] = {}
    quals: dict[str, np.ndarray] = {}
    for outcome in outcomes:
        if kind == "binary":
            y = rng.integers(0, 2, size=n).astype(float)
            y[0] = 0.0
            y[1] = 1.0
            p = np.clip(0.2 + 0.6 * y + 0.1 * rng.random(n), 0.0, 1.0)
        else:
            y = rng.normal(size=n)
            p = y + 0.15 * rng.normal(size=n)
        labels[outcome] = y
        preds[outcome] = p
        quals[outcome] = np.ones(n, dtype=bool)
    return {
        "labels": labels,
        "predictions": preds,
        "qualifiers": quals,
        "metadata": {"outcomes": list(outcomes)},
    }


def _metric_rows(boot, y_true, y_pred, kind: str) -> list[tuple[str, float]]:
    if kind == "binary":
        return [
            ("roc_auc", boot.roc_auc(y_true, y_pred)),
            ("average_precision", float(average_precision_score(y_true, y_pred))),
            ("brier", boot.brier(y_true, y_pred)),
            ("ece_15", boot.ece(y_true, y_pred)),
        ]
    return [(name, boot.METRIC_FNS[name](y_true, y_pred)) for name in boot.REGRESSION_METRICS]


def test_refuses_to_overwrite_metrics_long(tmp_path: Path) -> None:
    boot = load_module()
    metrics_long = tmp_path / "metrics_long.parquet"
    metrics_long.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="overwrite"):
        boot.assert_unique_ci_output(metrics_long, metrics_long)
    with pytest.raises(ValueError, match="with_ci"):
        boot.assert_unique_ci_output(tmp_path / "other.parquet", metrics_long)
    with pytest.raises(ValueError, match="00_data|bootstrap_24to48h_ci"):
        boot.assert_unique_work_root(
            tmp_path, metrics_long, tmp_path / "metrics_long_with_ci.parquet"
        )


def test_toy_run_concat_keeps_unique_paths(tmp_path: Path) -> None:
    boot = load_module()
    rng = np.random.default_rng(0)
    binary_outcomes = ("icu_admission",)
    regression_outcomes = ("min_heart_rate",)
    work_root = tmp_path / "bootstrap_24to48h_ci"
    dump_dir = tmp_path / "00_data"
    dump_dir.mkdir()
    metrics_long = dump_dir / "metrics_long.parquet"
    output_parquet = dump_dir / "metrics_long_with_ci.parquet"
    binary_pkl = tmp_path / "binary.pkl"
    regression_pkl = tmp_path / "regression.pkl"
    binary_payload = _payload("binary", binary_outcomes, n=48, rng=rng)
    regression_payload = _payload("regression", regression_outcomes, n=48, rng=rng)
    binary_pkl.write_bytes(pickle.dumps(binary_payload))
    regression_pkl.write_bytes(pickle.dumps(regression_payload))

    rows = []
    for kind, payload, artifact, outcomes in (
        ("binary", binary_payload, binary_pkl, binary_outcomes),
        ("regression", regression_payload, regression_pkl, regression_outcomes),
    ):
        for outcome in outcomes:
            y_true, y_pred = boot.aligned_arrays(payload, outcome)
            for metric, point in _metric_rows(boot, y_true, y_pred, kind):
                rows.append(
                    {
                        "experiment": "toy",
                        "config_id": "cfg_a",
                        "config_label": "A",
                        "backbone": "llama",
                        "seed": 42,
                        "task_type": kind,
                        "outcome": outcome,
                        "outcome_label": outcome,
                        "metric": metric,
                        "point": point,
                        "n_test": len(y_true),
                        "test_cohort_rows": len(y_true),
                        "quantizer": "deciles",
                        "anchoring": "none",
                        "fusion": "unfused",
                        "representation": "discrete",
                        "temporal": "time_tokens",
                        "effective_logreg_C": 0.01 if kind == "binary" else np.nan,
                        "effective_ridge_alpha": 100.0 if kind == "regression" else np.nan,
                        "source_artifact": str(artifact),
                        "source_sha256": "abc",
                    }
                )
    pd.DataFrame(rows).to_parquet(metrics_long, index=False)
    original = pd.read_parquet(metrics_long)

    common = [
        "--metrics-long",
        str(metrics_long),
        "--work-root",
        str(work_root),
        "--output-parquet",
        str(output_parquet),
        "--bootstrap-n",
        "40",
        "--no-strict-grid",
    ]
    assert boot.main(["--mode", "list", *common]) == 0
    assert boot.main(["--mode", "run", "--run-index", "0", *common]) == 0
    assert boot.main(["--mode", "concat", *common]) == 0

    after = pd.read_parquet(metrics_long)
    pd.testing.assert_frame_equal(original, after)
    result = pd.read_parquet(output_parquet)
    assert len(result) == 7
    assert "average_precision" not in set(result["metric"])
    assert result["run_id"].nunique() == 1
    assert (result["n_bootstrap_accepted"] == 40).all()
    finite = result[["ci_lo", "point", "ci_hi"]].dropna()
    assert (finite["ci_lo"] <= finite["point"] + 1e-12).all()
    assert (finite["point"] <= finite["ci_hi"] + 1e-12).all()
    shards = list((work_root / "shards").glob("*.parquet"))
    assert len(shards) == 1
    assert shards[0].resolve() != output_parquet.resolve()
    assert shards[0].resolve() != metrics_long.resolve()
    assert not (dump_dir / "metrics_long.csv").exists()
