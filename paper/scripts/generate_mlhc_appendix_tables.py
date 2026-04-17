#!/usr/bin/env python3
"""Generate appendix tables for the MLHC manuscript from aligned statistics files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METRICS = (
    ROOT
    / "artifacts"
    / "runs"
    / "statistics"
    / "paper_stats_combined"
    / "all_family_metrics.csv"
)
DEFAULT_PAIRWISE = (
    ROOT
    / "artifacts"
    / "runs"
    / "statistics"
    / "paper_stats_combined"
    / "all_family_pairwise_baseline.csv"
)
DEFAULT_OUT_DIR = ROOT.parent / "MLHC2026" / "MLHC" / "generated"

OUTCOME_LABELS = {
    "same_admission_death": "Hospital mortality",
    "long_length_of_stay": "Hospital LOS $>$ 7 d",
    "icu_admission": "ICU admission",
    "prolonged_icu_stay": "ICU LOS $>$ 48 h",
    "imv_event": "IMV",
    "hyperkalemia": "Hyperkalemia",
    "severe_hypokalemia": "Severe hypokalemia",
    "severe_anemia": "Severe anemia",
    "hypoglycemia": "Hypoglycemia",
    "profound_hyponatremia": "Profound hyponatremia",
    "severe_hypernatremia": "Severe hypernatremia",
    "tachycardia_hr130": "Tachycardia",
    "severe_hypertension": "Severe hypertension",
    "vasopressor_initiation": "Vasopressor initiation",
    "hypotension": "Hypotension",
    "crrt_initiation": "CRRT initiation",
    "hemodialysis_initiation": "Hemodialysis initiation",
    "length_of_stay": "Hospital LOS (h)",
    "peak_creatinine": "Peak creatinine",
    "min_hemoglobin": "Minimum hemoglobin",
    "peak_potassium": "Peak potassium",
    "min_potassium": "Minimum potassium",
    "min_glucose": "Minimum glucose",
    "min_sodium": "Minimum sodium",
    "max_sodium": "Maximum sodium",
    "peak_troponin": "Peak troponin",
    "peak_bnp": "Peak BNP",
    "max_heart_rate": "Maximum heart rate",
    "max_sbp": "Maximum SBP",
    "max_dbp": "Maximum DBP",
}

HANDLE_LABELS = {
    "deciles_unfused": "Deciles, unfused",
    "deciles_fused": "Deciles, fused",
    "ventiles_unfused": "Ventiles, unfused",
    "ventiles_fused": "Ventiles, fused",
    "ventiles_5_10_5_unfused": "Ventiles (clin.), unfused",
    "ventiles_5_10_5_fused": "Ventiles (clin.), fused",
    "trentiles_unfused": "Trentiles, unfused",
    "trentiles_fused": "Trentiles, fused",
    "trentiles_10_10_10_unfused": "Trentiles (clin.), unfused",
    "trentiles_10_10_10_fused": "Trentiles (clin.), fused",
    "centiles_unfused": "Centiles, unfused",
    "centiles_fused": "Centiles, fused",
    "discrete_none": "Discrete + event order only",
    "discrete_tt": "Discrete + time tokens",
    "discrete_rope": "Discrete + admission-relative RoPE",
    "soft_none": "Soft + event order only",
    "soft_tt": "Soft + time tokens",
    "soft_rope": "Soft + admission-relative RoPE",
    "xval_none": "xVal (code-normalized) + event order only",
    "xval_tt": "xVal (code-normalized) + time tokens",
    "xval_rope": "xVal (code-normalized) + admission-relative RoPE",
    "xval_affine_none": "xVal-affine (code-normalized + affine shift) + event order only",
    "xval_affine_tt": "xVal-affine (code-normalized + affine shift) + time tokens",
    "xval_affine_rope": "xVal-affine (code-normalized + affine shift) + admission-relative RoPE",
    "meds": "Native MIMIC codes",
    "mapped": "CLIF-mapped",
    "randomized": "Randomized mapped codes",
    "freqmatched": "Frequency-matched mapped codes",
}

BINARY_OUTCOME_ORDER = [
    "same_admission_death",
    "long_length_of_stay",
    "icu_admission",
    "prolonged_icu_stay",
    "imv_event",
    "hyperkalemia",
    "severe_hypokalemia",
    "severe_anemia",
    "hypoglycemia",
    "profound_hyponatremia",
    "severe_hypernatremia",
    "tachycardia_hr130",
    "severe_hypertension",
    "vasopressor_initiation",
    "hypotension",
    "crrt_initiation",
    "hemodialysis_initiation",
]

REGRESSION_OUTCOME_ORDER = [
    "length_of_stay",
    "peak_creatinine",
    "min_hemoglobin",
    "peak_potassium",
    "min_potassium",
    "min_glucose",
    "min_sodium",
    "max_sodium",
    "peak_troponin",
    "peak_bnp",
    "max_heart_rate",
    "max_sbp",
    "max_dbp",
]


def _fmt_ci(row: pd.Series) -> str:
    handle = HANDLE_LABELS.get(str(row["handle"]), str(row["handle"]))
    return f"{handle}; {row['point']:.3f} [{row['ci_lo']:.3f}, {row['ci_hi']:.3f}]"


def _best_by_outcome(
    metrics: pd.DataFrame,
    *,
    families: list[str],
    metric: str,
) -> pd.DataFrame:
    sub = metrics[
        metrics["family_name"].isin(families) & (metrics["metric"] == metric)
    ].copy()
    return (
        sub.sort_values(["outcome", "point"], ascending=[True, False])
        .groupby("outcome", as_index=False)
        .head(1)
        .set_index("outcome")
    )


def _binary_best_table(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for outcome in BINARY_OUTCOME_ORDER:
        row = {"Outcome": OUTCOME_LABELS[outcome]}
        for exp in [1, 2, 3]:
            best = _best_by_outcome(
                metrics,
                families=[f"exp{exp}_primary_binary", f"exp{exp}_additional_binary"],
                metric="roc_auc",
            )
            row[f"Exp{exp}"] = _fmt_ci(best.loc[outcome]) if outcome in best.index else "---"
        rows.append(row)
    return pd.DataFrame(rows)


def _regression_best_table(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for outcome in REGRESSION_OUTCOME_ORDER:
        row = {"Outcome": OUTCOME_LABELS[outcome]}
        for exp in [1, 2, 3]:
            best = _best_by_outcome(
                metrics,
                families=[f"exp{exp}_length_of_stay", f"exp{exp}_extended_regression"],
                metric="spearman_rho",
            )
            row[f"Exp{exp}"] = _fmt_ci(best.loc[outcome]) if outcome in best.index else "---"
        rows.append(row)
    return pd.DataFrame(rows)


def _coverage_table(metrics: pd.DataFrame, pairwise: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for exp in [1, 2, 3]:
        exp_rows = metrics[metrics["family_name"].str.startswith(f"exp{exp}_")].copy()
        pair_rows = pairwise[
            pairwise["family_name"].isin(
                [
                    f"exp{exp}_primary_binary",
                    f"exp{exp}_additional_binary",
                    f"exp{exp}_length_of_stay",
                    f"exp{exp}_extended_regression",
                ]
            )
        ].copy()
        alignment_modes = ", ".join(sorted(pair_rows["alignment_mode"].dropna().unique()))
        rows.append(
            {
                "Experiment": str(exp),
                "Models": str(exp_rows["handle"].nunique()),
                "Binary LR sweep": "16 outcomes x AUROC/AUPRC/Brier/ECE",
                "Regression sweep": "13 outcomes x Spearman/R2/MAE/RMSE",
                "Pairwise alignment": alignment_modes,
            }
        )
    return pd.DataFrame(rows)


def _df_to_latex(
    df: pd.DataFrame,
    *,
    caption: str,
    label: str,
    colspec: str,
    size_cmd: str = "\\scriptsize",
    tabcolsep: int = 4,
) -> str:
    def _latex_escape(text: str) -> str:
        for src, dst in (
            ("\\", "\\textbackslash{}"),
            ("&", "\\&"),
            ("%", "\\%"),
            ("_", "\\_"),
            ("#", "\\#"),
            ("{", "\\{"),
            ("}", "\\}"),
        ):
            text = text.replace(src, dst)
        return text

    header = " & ".join(_latex_escape(str(col)) for col in df.columns) + " \\\\"
    body = []
    for _, row in df.iterrows():
        values = [_latex_escape(str(row[col])) for col in df.columns]
        body.append(" & ".join(values) + " \\\\")
    return "\n".join(
        [
            "\\begin{table*}[t]",
            "  \\centering",
            f"  \\caption{{{caption}}}",
            f"  \\label{{{label}}}",
            f"  {size_cmd}",
            f"  \\setlength{{\\tabcolsep}}{{{tabcolsep}pt}}",
            f"  \\begin{{tabular}}{{{colspec}}}",
            "    \\toprule",
            f"    {header}",
            "    \\midrule",
            *[f"    {line}" for line in body],
            "    \\bottomrule",
            "  \\end{tabular}",
            "\\end{table*}",
        ]
    )


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_csv", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--pairwise_csv", type=Path, default=DEFAULT_PAIRWISE)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    metrics = pd.read_csv(args.metrics_csv)
    pairwise = pd.read_csv(args.pairwise_csv)
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    coverage = _coverage_table(metrics, pairwise)
    binary = _binary_best_table(metrics)
    regression = _regression_best_table(metrics)

    coverage.to_csv(out_dir / "appendix_stats_coverage.csv", index=False)
    binary.to_csv(out_dir / "appendix_binary_sweep.csv", index=False)
    regression.to_csv(out_dir / "appendix_regression_sweep.csv", index=False)

    coverage_tex = _df_to_latex(
        coverage.rename(columns={"Experiment": "Exp."}),
        caption=(
            "\\textbf{Statistical coverage across the reported benchmark sweeps.} "
            "The full CI and baseline-centered pairwise-testing audit covers the complete logistic-regression "
            "binary sweep and Ridge-regression sweep for every experiment, i.e., all 30 "
            "benchmark outcomes at the benchmark level (17 binary outcomes and 13 regression "
            "outcomes, with each experiment using 16 binary outcomes because the ICU "
            "endpoint differs between Experiments~1--2 and 3). The final column refers to those "
            "full LR/Ridge sweeps only."
        ),
        label="tab:evaluation_coverage",
        colspec="p{0.08\\textwidth}p{0.08\\textwidth}p{0.27\\textwidth}p{0.27\\textwidth}p{0.20\\textwidth}",
        tabcolsep=3,
    )
    binary_tex = _df_to_latex(
        binary,
        caption=(
            "\\textbf{Full binary-outcome sweep} (best AUROC with 95\\% bootstrap CI for each "
            "experiment). Each cell reports the best-performing configuration within that "
            "experiment on the named binary outcome. AUPRC, Brier score, ECE-15, and the "
            "baseline-centered BH-corrected paired permutation results for the same outcomes are "
            "reported in the aligned statistics files. ICU admission appears only in Experiments~1--2; ICU LOS "
            "$>48$h appears only in Experiment~3."
        ),
        label="tab:appendix_binary_sweep",
        colspec=(
            ">{\\raggedright\\arraybackslash}p{0.17\\textwidth}"
            ">{\\raggedright\\arraybackslash}p{0.265\\textwidth}"
            ">{\\raggedright\\arraybackslash}p{0.265\\textwidth}"
            ">{\\raggedright\\arraybackslash}p{0.265\\textwidth}"
        ),
        tabcolsep=2,
    )
    regression_tex = _df_to_latex(
        regression,
        caption=(
            "\\textbf{Full regression-outcome sweep} (best Spearman $\\rho$ with 95\\% bootstrap CI "
            "for each experiment). Each cell reports the best-performing configuration within that "
            "experiment on the named regression outcome. $R^2$, MAE, RMSE, and the baseline-centered "
            "BH-corrected paired permutation results for the same outcomes are reported in the aligned "
            "statistics files."
        ),
        label="tab:appendix_regression_sweep",
        colspec=(
            ">{\\raggedright\\arraybackslash}p{0.17\\textwidth}"
            ">{\\raggedright\\arraybackslash}p{0.265\\textwidth}"
            ">{\\raggedright\\arraybackslash}p{0.265\\textwidth}"
            ">{\\raggedright\\arraybackslash}p{0.265\\textwidth}"
        ),
        tabcolsep=2,
    )

    _write_text(out_dir / "appendix_stats_coverage.tex", coverage_tex + "\n")
    _write_text(out_dir / "appendix_binary_sweep.tex", binary_tex + "\n")
    _write_text(out_dir / "appendix_regression_sweep.tex", regression_tex + "\n")
    print(f"Wrote appendix tables to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
