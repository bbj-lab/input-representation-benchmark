#!/usr/bin/env python3
"""Model-level paired t-tests for the printed ML4H family-mean differences.

The admission-level prediction-swap test holds the six fitted models fixed.
This script holds the test set fixed and treats the six fitted models as the
replicates. It pairs each alternative with its baseline within architecture and
seed, runs a two-sided one-sample t-test on the six paired differences (five
degrees of freedom), and applies Benjamini-Hochberg adjustment inside each
printed table, as the prediction-swap analysis does. It also writes
outcome-level differences with agreement between architectures and experiments.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
import ml4h_main_table_contrasts as contrasts  # noqa: E402
from generate_ml4h_appendix_result_tables import OUTCOME_FAMILIES  # noqa: E402

RUN_ROOT = REPO_ROOT / "outputs/runs/ml4h_2026"
METRICS = RUN_ROOT / "metrics/metrics_long.parquet"
SWAP = RUN_ROOT / "family_effect_bootstrap/family_effect_intervals.parquet"
OUTPUT = RUN_ROOT / "model_level_tests"
Q = contrasts.BH_Q
T_CRIT = float(stats.t.ppf(0.975, 5))
HEADLINE = {
    "fusion": lambda c: c["condition"] == "fusion" and c["level"].endswith("deciles"),
    "soft": lambda c: c["level"] == "Soft discretization",
    "xval": lambda c: c["level"] == "Code-normalized xVal",
    "xval_affine": lambda c: c["level"] == "Affine xVal",
    "event_order": lambda c: c["level"] == "Event order",
    "rope": lambda c: c["level"] == "Admission-relative RoPE",
    "clif": lambda c: c["level"] == "CLIF 2.1.0",
}
FAMILY_COLUMNS = [
    "table", "level", "task_type", "family", "delta", "ci_lo", "ci_hi",
    "delta_min", "delta_max", "n_positive", "p_swap", "p_t", "p_t_adj",
    "star_swap", "star_t",
]


def outcome_matrix(metrics, experiment, task_type, filters, cache) -> pd.DataFrame:
    """Rows are (architecture, seed). Columns are outcomes. Values are AUROC or Spearman."""
    key = (experiment, task_type, tuple(sorted(filters.items())))
    if key not in cache:
        rows = metrics.loc[
            (metrics["experiment"] == experiment)
            & (metrics["task_type"] == task_type)
            & (metrics["metric"] == contrasts.TASK_METRIC[task_type])
        ]
        rows = contrasts.filter_rows(rows, filters)
        if rows.duplicated(["backbone", "seed", "outcome"]).any():
            raise ValueError(f"More than one configuration matches {key}")
        wide = rows.pivot(index=["backbone", "seed"], columns="outcome", values="point")
        if len(wide) != 6 or wide.isna().any().any():
            raise ValueError(f"Expected six complete models for {key}")
        cache[key] = wide
    return cache[key]


def summarize(diff: pd.Series) -> dict:
    """Summarize six paired differences indexed by (architecture, seed)."""
    values = diff.to_numpy(float)
    llama = diff.xs("llama", level="backbone").to_numpy(float)
    qwen = diff.xs("qwen", level="backbone").to_numpy(float)
    mean, sd = float(values.mean()), float(values.std(ddof=1))
    se = sd / np.sqrt(len(values))
    return {
        "delta": mean,
        "delta_llama": float(llama.mean()),
        "delta_qwen": float(qwen.mean()),
        "delta_min": float(values.min()),
        "delta_max": float(values.max()),
        "n_positive": int((values > 0).sum()),
        "ci_lo": mean - T_CRIT * se,
        "ci_hi": mean + T_CRIT * se,
        "t": mean / se if se > 0 else np.nan,
        "p_t": float(stats.ttest_1samp(values, 0.0).pvalue),
        "p_arch": float(stats.ttest_ind(llama, qwen, equal_var=False).pvalue),
    }


def family_table(metrics, swap) -> pd.DataFrame:
    cache, rows = {}, []
    for spec in contrasts.leaf_specs():
        condition = contrasts.find_condition(spec["experiment"], spec["condition"])
        level = contrasts.find_level(condition, spec["level"])
        args = (metrics, spec["experiment"], spec["task_type"])
        alt = outcome_matrix(*args, level.filters, cache)
        base = outcome_matrix(*args, contrasts.level_baseline_filters(condition, level), cache)
        members = [o for o in alt.columns if OUTCOME_FAMILIES.get(o) == spec["family"]]
        diff = (alt[members].mean(axis=1) - base[members].mean(axis=1)).sort_index()
        rows.append(
            {
                **{k: spec[k] for k in ("task_index", "experiment", "condition", "level", "task_type", "family")},
                "table": contrasts.TABLE_FAMILY[(spec["experiment"], spec["condition"])],
                "n_outcomes": len(members),
                **summarize(diff),
            }
        )
    fam = pd.DataFrame(rows)
    keys = ["experiment", "condition", "level", "task_type", "family"]
    swap = swap[keys + ["point", "p_value", "p_adj", "bh_significant"]].rename(
        columns={"point": "delta_swap", "p_value": "p_swap", "p_adj": "p_swap_adj", "bh_significant": "star_swap"}
    )
    fam = fam.merge(swap, on=keys, how="left", validate="one_to_one")
    if fam["star_swap"].isna().any():
        raise ValueError("Some printed differences have no prediction-swap row")
    gap = float((fam["delta"] - fam["delta_swap"]).abs().max())
    print(f"max |delta here - delta in swap file| = {gap:.2e}")
    if gap > 1e-9:
        raise ValueError("Paired differences do not reproduce the printed differences")
    fam["star_swap"] = fam["star_swap"].astype(bool)
    for column in ("p_t_adj", "p_arch_adj"):
        fam[column] = np.nan
    for _, part in fam.groupby("table", sort=False):
        fam.loc[part.index, "p_t_adj"] = contrasts.benjamini_hochberg(part["p_t"].to_numpy(), q=Q)[0]
        fam.loc[part.index, "p_arch_adj"] = contrasts.benjamini_hochberg(part["p_arch"].to_numpy(), q=Q)[0]
    fam["star_t"] = fam["p_t_adj"] <= Q
    return fam


def outcome_table(metrics) -> pd.DataFrame:
    cache, rows = {}, []
    for column in contrasts.column_specs():
        condition = contrasts.find_condition(column["experiment"], column["condition"])
        level = contrasts.find_level(condition, column["level"])
        for task_type in contrasts.TASK_ORDER:
            args = (metrics, column["experiment"], task_type)
            alt = outcome_matrix(*args, level.filters, cache)
            base = outcome_matrix(*args, contrasts.level_baseline_filters(condition, level), cache)
            diff = (alt - base).sort_index()
            for outcome in diff.columns:
                rows.append(
                    {
                        **{k: column[k] for k in ("task_index", "experiment", "condition", "level")},
                        "task_type": task_type,
                        "outcome": outcome,
                        "family": OUTCOME_FAMILIES[outcome],
                        "base": float(base[outcome].mean()),
                        "alt": float(alt[outcome].mean()),
                        **summarize(diff[outcome]),
                    }
                )
    out = pd.DataFrame(rows)
    out["p_t_adj"] = np.nan
    for _, part in out.groupby("task_index"):
        out.loc[part.index, "p_t_adj"] = contrasts.benjamini_hochberg(part["p_t"].to_numpy(), q=Q)[0]
    out["star_t"] = out["p_t_adj"] <= Q
    return out


def show(title: str, frame: pd.DataFrame, columns=None) -> None:
    print(f"\n== {title}")
    shown = frame if columns is None else frame[columns]
    print(shown.to_string(index=False, float_format=lambda v: f"{v:.4g}"))


def report_family(fam: pd.DataFrame) -> None:
    fam["both"] = fam["star_swap"] & fam["star_t"]
    fam["swap_only"] = fam["star_swap"] & ~fam["star_t"]
    fam["t_only"] = ~fam["star_swap"] & fam["star_t"]
    counts = fam.groupby("table", sort=False)[
        ["star_swap", "star_t", "both", "swap_only", "t_only"]
    ].sum()
    counts.insert(0, "tests", fam.groupby("table", sort=False).size())
    counts.loc["all"] = counts.sum()
    print("\n== Stars by test (Benjamini-Hochberg within table, q = 0.05)")
    print(counts.astype(int).to_string())
    print(f"unadjusted model-level p < 0.05 in {int((fam['p_t'] < 0.05).sum())} of {len(fam)} tests")

    for task_type in ("binary", "regression"):
        starred = fam[(fam["task_type"] == task_type) & fam["star_swap"]]
        show(f"Smallest |delta| starred by the swap test, {task_type}",
             starred.assign(abs_delta=starred["delta"].abs()).nsmallest(4, "abs_delta"), FAMILY_COLUMNS)
    show("Fusion at matched deciles (Table 2, first column)",
         fam[(fam["condition"] == "fusion") & fam["level"].str.endswith("deciles")], FAMILY_COLUMNS)
    show("Swap-starred, model-level not starred", fam[fam["swap_only"]].sort_values("delta", key=abs), FAMILY_COLUMNS)
    show("Model-level starred, swap not starred", fam[fam["t_only"]], FAMILY_COLUMNS)
    show("Code-normalized and affine xVal (Table 6)", fam[fam["level"].str.contains("xVal")], FAMILY_COLUMNS)
    show("Soft discretization (Table 6)", fam[fam["level"] == "Soft discretization"], FAMILY_COLUMNS)
    show("Temporal encodings (Table 7)", fam[fam["condition"] == "temporal"], FAMILY_COLUMNS)
    show("CLIF versus native MIMIC (Table 3)", fam[fam["experiment"] == "exp3"], FAMILY_COLUMNS)

    fam["same_sign"] = np.sign(fam["delta_llama"]) == np.sign(fam["delta_qwen"])
    fam["llama_larger"] = fam["delta_llama"].abs() > fam["delta_qwen"].abs()
    arch = fam.groupby("table", sort=False).apply(
        lambda g: pd.Series(
            {
                "tests": len(g),
                "same_sign": int(g["same_sign"].sum()),
                "llama_larger_if_same_sign": int((g["llama_larger"] & g["same_sign"]).sum()),
                "sum_abs_llama_over_qwen": g["delta_llama"].abs().sum() / g["delta_qwen"].abs().sum(),
                "arch_p_below_0.05": int((g["p_arch"] < 0.05).sum()),
                "arch_bh": int((g["p_arch_adj"] <= Q).sum()),
            }
        ),
        include_groups=False,
    )
    print("\n== Architecture agreement in the 136 family-level differences")
    print(arch.to_string(float_format=lambda v: f"{v:.3f}"))
    both = fam[fam["both"]]
    print(f"among {len(both)} tests starred by both tests: same sign in both architectures in {int(both['same_sign'].sum())}")


def report_outcomes(out: pd.DataFrame) -> None:
    mortality = out[out["outcome"] == "same_admission_death"]
    show("Hospital mortality in hours 24-48 in every contrast", mortality,
         ["experiment", "level", "base", "alt", "delta", "ci_lo", "ci_hi", "p_t", "p_t_adj", "delta_llama", "delta_qwen"])

    fusion_id = next(c["task_index"] for c in contrasts.column_specs() if HEADLINE["fusion"](c))
    fusion = out[out["task_index"] == fusion_id].copy()
    show("Fusion at deciles, every outcome by |delta|", fusion.sort_values("delta", key=abs, ascending=False),
         ["task_type", "outcome", "base", "alt", "delta", "ci_lo", "ci_hi", "p_t", "p_t_adj",
          "n_positive", "delta_llama", "delta_qwen"])
    for task_type, part in fusion.groupby("task_type"):
        same = int((np.sign(part["delta_llama"]) == np.sign(part["delta_qwen"])).sum())
        rho = stats.spearmanr(part["delta_llama"], part["delta_qwen"])[0]
        print(f"\nfusion {task_type}: {len(part)} outcomes, six-model delta > 0 in {int((part['delta'] > 0).sum())}, "
              f"BH-starred (model level) {int(part['star_t'].sum())}, same sign in both architectures {same}, "
              f"Spearman between architectures {rho:.3f}")
    hospital = fusion[fusion["family"] == "Hospital outcomes"]
    others = hospital[hospital["outcome"] != "same_admission_death"]
    print(f"hospital family, fusion at deciles: all four outcomes {hospital['delta'].mean():.4f}, "
          f"without mortality {others['delta'].mean():.4f}")

    ids = {name: next(c["task_index"] for c in contrasts.column_specs() if rule(c)) for name, rule in HEADLINE.items()}
    inverse = {v: k for k, v in ids.items()}
    chosen = out[out["task_index"].isin(ids.values())].assign(name=lambda d: d["task_index"].map(inverse))
    for task_type in ("regression", "binary"):
        part = chosen[chosen["task_type"] == task_type]
        delta = part.pivot(index="outcome", columns="name", values="delta")[list(HEADLINE)]
        tvalue = part.pivot(index="outcome", columns="name", values="t")[list(HEADLINE)]
        print(f"\n== Rank agreement across contrasts, {task_type} outcomes (n = {len(delta)}; Experiment 3 omits ICU admission)")
        print("Spearman of |delta|:\n" + delta.abs().corr(method="spearman").round(2).to_string())
        print("Spearman of signed delta:\n" + delta.corr(method="spearman").round(2).to_string())
        print("Spearman of |t|:\n" + tvalue.abs().corr(method="spearman").round(2).to_string())
        for name in HEADLINE:
            top = part[part["name"] == name].assign(a=lambda d: d["delta"].abs()).nlargest(6, "a")
            print(f"top 6 by |delta|, {name}: " + ", ".join(
                f"{r.outcome} {r.delta:+.3f}{'*' if r.star_t else ''}" for r in top.itertuples()))


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)
    metrics = pd.read_parquet(METRICS)
    fam = family_table(metrics, pd.read_parquet(SWAP))
    out = outcome_table(metrics)
    report_family(fam)
    report_outcomes(out)
    fam.to_csv(OUTPUT / "family_tests.csv", index=False)
    out.to_csv(OUTPUT / "outcome_tests.csv", index=False)
    print(f"\nwrote {OUTPUT}")


if __name__ == "__main__":
    main()
