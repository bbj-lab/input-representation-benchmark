"""Main-text table contrasts for paired family-mean Δ tests.

These conditions match the five printed tables. They are independent of
``EXPERIMENT_CONDITIONS`` in ``generate_ml4h_submission_previews``, whose
Experiment~2 temporal contrast averages across numeric encodings.

Each printed family Δ is a paired permutation test of that arm versus the
table baseline. Stars are Benjamini--Hochberg significant within the table.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


TASK_ORDER = ("binary", "regression")
TASK_METRIC = {"binary": "roc_auc", "regression": "spearman_rho"}
FAMILY_ORDER = {
    "binary": ("Hospital outcomes", "Organ support", "Laboratory", "Vitals"),
    "regression": ("Electrolytes", "Biomarkers", "Hematology", "Vitals"),
}
LEAF_ORDER = tuple(
    (task_type, family)
    for task_type in TASK_ORDER
    for family in FAMILY_ORDER[task_type]
)
BH_Q = 0.05
COLUMN_COUNT = 17
LEAF_COUNT = 136
ROW_COUNT = LEAF_COUNT
TABLE_FAMILY = {
    ("exp1", "granularity"): "exp1_quantization",
    ("exp1", "fusion"): "exp1_fusion",
    ("exp2", "numeric"): "exp2_numeric",
    ("exp2", "temporal"): "exp2_temporal",
    ("exp3", "schema"): "exp3",
}
EXPECTED_TESTS = {
    "exp1_quantization": 40,
    "exp1_fusion": 48,
    "exp2_numeric": 24,
    "exp2_temporal": 16,
    "exp3": 8,
}


@dataclass(frozen=True)
class ContrastLevel:
    label: str
    filters: dict[str, str]
    baseline_filters: dict[str, str] | None = None


@dataclass(frozen=True)
class Contrast:
    key: str
    baseline_filters: dict[str, str]
    levels: tuple[ContrastLevel, ...]


def _fusion_level(label: str, quantizer: str, anchoring: str) -> ContrastLevel:
    grain = {"quantizer": quantizer, "anchoring": anchoring}
    return ContrastLevel(
        label,
        {**grain, "fusion": "fused"},
        baseline_filters={**grain, "fusion": "unfused"},
    )


MAIN_TABLE_CONDITIONS: dict[str, tuple[Contrast, ...]] = {
    "exp1": (
        Contrast(
            key="granularity",
            baseline_filters={
                "quantizer": "deciles",
                "anchoring": "none",
                "fusion": "unfused",
            },
            levels=(
                ContrastLevel(
                    "Ventiles · population",
                    {
                        "quantizer": "ventiles",
                        "anchoring": "none",
                        "fusion": "unfused",
                    },
                ),
                ContrastLevel(
                    "Ventiles · reference",
                    {
                        "quantizer": "ventiles",
                        "anchoring": "5-10-5",
                        "fusion": "unfused",
                    },
                ),
                ContrastLevel(
                    "Trentiles · population",
                    {
                        "quantizer": "trentiles",
                        "anchoring": "none",
                        "fusion": "unfused",
                    },
                ),
                ContrastLevel(
                    "Trentiles · reference",
                    {
                        "quantizer": "trentiles",
                        "anchoring": "10-10-10",
                        "fusion": "unfused",
                    },
                ),
                ContrastLevel(
                    "Centiles · population",
                    {
                        "quantizer": "centiles",
                        "anchoring": "none",
                        "fusion": "unfused",
                    },
                ),
            ),
        ),
        Contrast(
            key="fusion",
            baseline_filters={
                "quantizer": "deciles",
                "anchoring": "none",
                "fusion": "unfused",
            },
            levels=(
                _fusion_level("Fused · deciles", "deciles", "none"),
                _fusion_level("Fused · ventiles", "ventiles", "none"),
                _fusion_level("Fused · 5-10-5", "ventiles", "5-10-5"),
                _fusion_level("Fused · trentiles", "trentiles", "none"),
                _fusion_level("Fused · 10-10-10", "trentiles", "10-10-10"),
                _fusion_level("Fused · centiles", "centiles", "none"),
            ),
        ),
    ),
    "exp2": (
        Contrast(
            key="numeric",
            baseline_filters={
                "representation": "discrete",
                "temporal": "time_tokens",
            },
            levels=(
                ContrastLevel(
                    "Soft discretization",
                    {"representation": "soft", "temporal": "time_tokens"},
                ),
                ContrastLevel(
                    "Code-normalized xVal",
                    {"representation": "xval", "temporal": "time_tokens"},
                ),
                ContrastLevel(
                    "Affine xVal",
                    {"representation": "xval_affine", "temporal": "time_tokens"},
                ),
            ),
        ),
        Contrast(
            key="temporal",
            baseline_filters={
                "representation": "discrete",
                "temporal": "time_tokens",
            },
            levels=(
                ContrastLevel(
                    "Event order",
                    {"representation": "discrete", "temporal": "event_order"},
                ),
                ContrastLevel(
                    "Admission-relative RoPE",
                    {"representation": "discrete", "temporal": "time_rope"},
                ),
            ),
        ),
    ),
    "exp3": (
        Contrast(
            key="schema",
            baseline_filters={"schema": "Native MIMIC"},
            levels=(
                ContrastLevel(
                    "CLIF 2.1.0",
                    {"schema": "CLIF harmonized"},
                ),
            ),
        ),
    ),
}


def column_specs() -> list[dict[str, str | int]]:
    columns: list[dict[str, str | int]] = []
    for experiment, conditions in MAIN_TABLE_CONDITIONS.items():
        for condition in conditions:
            for level in condition.levels:
                columns.append(
                    {
                        "task_index": len(columns),
                        "experiment": experiment,
                        "condition": condition.key,
                        "level": level.label,
                    }
                )
    if len(columns) != COLUMN_COUNT:
        raise ValueError(f"Expected {COLUMN_COUNT} column tasks, found {len(columns)}")
    return columns


def leaf_specs() -> list[dict[str, str | int]]:
    leaves: list[dict[str, str | int]] = []
    for column in column_specs():
        for task_type, family in LEAF_ORDER:
            leaves.append(
                {
                    **column,
                    "task_type": task_type,
                    "family": family,
                    "role": "family",
                }
            )
    if len(leaves) != LEAF_COUNT:
        raise ValueError(f"Expected {LEAF_COUNT} leaf tests, found {len(leaves)}")
    return leaves


def task_specs() -> list[dict[str, str | int]]:
    return column_specs()


def find_condition(experiment: str, key: str) -> Contrast:
    matches = [
        condition
        for condition in MAIN_TABLE_CONDITIONS[experiment]
        if condition.key == key
    ]
    if len(matches) != 1:
        raise ValueError(f"Could not identify condition {experiment}/{key}")
    return matches[0]


def find_level(condition: Contrast, label: str) -> ContrastLevel:
    matches = [level for level in condition.levels if level.label == label]
    if len(matches) != 1:
        raise ValueError(f"Could not identify level {condition.key}/{label}")
    return matches[0]


def level_baseline_filters(condition: Contrast, level: ContrastLevel) -> dict[str, str]:
    if level.baseline_filters is not None:
        return level.baseline_filters
    return condition.baseline_filters


def add_schema(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["schema"] = np.where(
        out["config_id"].astype(str).str.startswith("meds_icu_"),
        "Native MIMIC",
        np.where(
            out["config_id"].astype(str).str.startswith("meds_clif_"),
            "CLIF harmonized",
            "",
        ),
    )
    return out


def filter_rows(frame: pd.DataFrame, filters: dict[str, str]) -> pd.DataFrame:
    out = add_schema(frame)
    for column, value in filters.items():
        out = out.loc[out[column].astype(str) == value]
    return out


def six_model_family_mean(
    metrics: pd.DataFrame,
    *,
    experiment: str,
    task_type: str,
    family: str,
    filters: dict[str, str],
) -> float:
    from generate_ml4h_appendix_result_tables import OUTCOME_FAMILIES

    rows = metrics.loc[
        (metrics["experiment"] == experiment)
        & (metrics["task_type"] == task_type)
        & (metrics["metric"] == TASK_METRIC[task_type])
    ].copy()
    rows["outcome_family"] = rows["outcome"].map(OUTCOME_FAMILIES)
    rows = rows.loc[rows["outcome_family"] == family]
    rows = filter_rows(rows, filters)
    per_model = rows.groupby(["backbone", "seed"], as_index=False, sort=False).agg(
        point=("point", "mean")
    )
    if len(per_model) != 6:
        raise ValueError(
            f"{experiment}/{task_type}/{family}/{filters}: "
            f"expected 6 models, found {len(per_model)}"
        )
    return float(per_model["point"].mean())


def unrounded_table_delta(
    metrics: pd.DataFrame,
    *,
    experiment: str,
    condition_key: str,
    level_label: str,
    task_type: str,
    family: str,
) -> float:
    condition = find_condition(experiment, condition_key)
    level = find_level(condition, level_label)
    comparison = six_model_family_mean(
        metrics,
        experiment=experiment,
        task_type=task_type,
        family=family,
        filters=level.filters,
    )
    baseline = six_model_family_mean(
        metrics,
        experiment=experiment,
        task_type=task_type,
        family=family,
        filters=level_baseline_filters(condition, level),
    )
    return float(comparison - baseline)


def permutation_pvalue(observed: float, null_values: np.ndarray) -> float:
    finite = np.asarray(null_values, dtype=float)
    finite = finite[np.isfinite(finite)]
    n = int(len(finite))
    if n == 0:
        raise ValueError("No finite permutation draws for a p-value")
    if not np.isfinite(observed):
        raise ValueError("Observed Δ is not finite")
    p_value = float(np.sum(np.abs(finite) >= abs(float(observed)))) / n
    if p_value == 0.0:
        p_value = 1.0 / n
    return float(min(1.0, p_value))


def benjamini_hochberg(p_values: np.ndarray, q: float = BH_Q) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(p_values, dtype=float)
    n = int(len(p))
    if n == 0:
        return p, np.zeros(0, dtype=bool)
    order = np.argsort(p, kind="mergesort")
    ranked = p[order]
    adjusted = np.empty(n, dtype=float)
    running = 1.0
    for index in range(n, 0, -1):
        running = min(running, ranked[index - 1] * n / index)
        adjusted[index - 1] = running
    p_adj = np.empty(n, dtype=float)
    p_adj[order] = np.minimum(adjusted, 1.0)
    reject = p_adj <= q
    return p_adj, reject


def attach_within_table_bh(
    intervals: pd.DataFrame, q: float = BH_Q
) -> pd.DataFrame:
    out = intervals.copy()
    out["table_family"] = [
        TABLE_FAMILY[(str(row.experiment), str(row.condition))]
        for row in out.itertuples()
    ]
    out["p_adj"] = np.nan
    out["bh_significant"] = False
    if len(out) != ROW_COUNT:
        raise ValueError(f"Expected {ROW_COUNT} family rows, found {len(out)}")
    if "role" in out.columns and not out["role"].astype(str).isin(("family", "leaf")).all():
        raise ValueError("Interval table has unexpected roles")

    for family, index in out.groupby("table_family", sort=False).groups.items():
        expected = EXPECTED_TESTS[family]
        if len(index) != expected:
            raise ValueError(f"{family}: expected {expected} tests, found {len(index)}")
        p_adj, significant = benjamini_hochberg(out.loc[index, "p_value"].to_numpy(), q=q)
        out.loc[index, "p_adj"] = p_adj
        out.loc[index, "bh_significant"] = significant
    return out


def attach_bh_decisions(intervals: pd.DataFrame, q: float = BH_Q) -> pd.DataFrame:
    return attach_within_table_bh(intervals, q=q)
