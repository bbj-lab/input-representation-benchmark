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
    "imv_event": "Invasive mechanical ventilation",
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
    "discrete_none": "Discrete + order",
    "discrete_tt": "Discrete + tokens",
    "discrete_rope": "Discrete + RoPE",
    "soft_none": "Soft + order",
    "soft_tt": "Soft + tokens",
    "soft_rope": "Soft + RoPE",
    "xval_none": "xVal + order",
    "xval_tt": "xVal + tokens",
    "xval_rope": "xVal + RoPE",
    "xval_affine_none": "xVal-affine + order",
    "xval_affine_tt": "xVal-affine + tokens",
    "xval_affine_rope": "xVal-affine + RoPE",
    "meds": "Native MIMIC",
    "mapped": "CLIF-mapped",
    "randomized": "Randomized mapped",
    "freqmatched": "Freq-matched mapped",
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

BINARY_GROUPS = [
    ("Hospital", ["same_admission_death", "long_length_of_stay"]),
    ("ICU endpoints", ["icu_admission", "prolonged_icu_stay"]),
    (
        "Interventions",
        ["imv_event", "vasopressor_initiation", "crrt_initiation", "hemodialysis_initiation"],
    ),
    (
        "Post-24h physiologic thresholds",
        [
            "hyperkalemia",
            "severe_hypokalemia",
            "severe_anemia",
            "hypoglycemia",
            "profound_hyponatremia",
            "severe_hypernatremia",
            "tachycardia_hr130",
            "severe_hypertension",
            "hypotension",
        ],
    ),
]

REGRESSION_GROUPS = [
    ("Hospital", ["length_of_stay"]),
    (
        "Laboratory extrema",
        [
            "peak_creatinine",
            "min_hemoglobin",
            "peak_potassium",
            "min_potassium",
            "min_glucose",
            "min_sodium",
            "max_sodium",
            "peak_troponin",
            "peak_bnp",
        ],
    ),
    ("Vital extrema", ["max_heart_rate", "max_sbp", "max_dbp"]),
]


def _fmt_ci(row: pd.Series) -> str:
    handle = HANDLE_LABELS.get(str(row["handle"]), str(row["handle"]))
    return f"{handle}; {row['point']:.3f} [{row['ci_lo']:.3f}, {row['ci_hi']:.3f}]"


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


def _fmt_ci_latex(cell: str) -> str:
    if cell == "---":
        return cell
    handle, rest = cell.split("; ", 1)
    point, ci_tail = rest.split(" [", 1)
    ci = "[" + ci_tail
    return f"\\shortstack[l]{{{_latex_escape(handle)}; {point}\\\\{{}}{ci}}}"


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
        row = {"OutcomeKey": outcome, "Outcome": OUTCOME_LABELS[outcome]}
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
        row = {"OutcomeKey": outcome, "Outcome": OUTCOME_LABELS[outcome]}
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


def _grouped_sweep_to_latex(
    df: pd.DataFrame,
    *,
    groups: list[tuple[str, list[str]]],
    caption: str,
    label: str,
    first_col_width: str = "0.17\\textwidth",
    exp_col_width: str = "0.24\\textwidth",
    first_exp_col_width: str | None = None,
    size_cmd: str = "\\scriptsize",
    tabcolsep: int = 3,
    note: str | None = None,
) -> str:
    rows_by_key = {
        str(row["OutcomeKey"]): row
        for _, row in df.iterrows()
    }
    exp1_w = first_exp_col_width if first_exp_col_width else exp_col_width
    colspec = (
        f">{{\\raggedright\\arraybackslash}}p{{{first_col_width}}}"
        f">{{\\raggedright\\arraybackslash}}p{{{exp1_w}}}"
        f">{{\\raggedright\\arraybackslash}}p{{{exp_col_width}}}"
        f">{{\\raggedright\\arraybackslash}}p{{{exp_col_width}}}"
    )
    lines = [
        "\\begin{table*}[t]",
        "  \\centering",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        f"  {size_cmd}",
        "  \\renewcommand{\\arraystretch}{1.12}",
        f"  \\setlength{{\\tabcolsep}}{{{tabcolsep}pt}}",
        f"  \\begin{{tabular}}{{{colspec}}}",
        "    \\toprule",
        "    Outcome & Exp1 & Exp2 & Exp3 \\\\",
        "    \\midrule",
    ]
    for group_idx, (group_name, outcome_keys) in enumerate(groups):
        bold_italic = "\\textbf{\\textit{" + group_name + "}}"
        lines.append(f"    \\multicolumn{{4}}{{l}}{{{bold_italic}}} \\\\")
        lines.append("    \\addlinespace[1pt]")
        for outcome_key in outcome_keys:
            row = rows_by_key[outcome_key]
            values = [
                _latex_escape(str(row["Outcome"])),
                _fmt_ci_latex(str(row["Exp1"])),
                _fmt_ci_latex(str(row["Exp2"])),
                _fmt_ci_latex(str(row["Exp3"])),
            ]
            lines.append("    " + " & ".join(values) + " \\\\")
        if group_idx != len(groups) - 1:
            lines.append("    \\addlinespace[2pt]")
    lines.extend(
        [
            "    \\bottomrule",
            "  \\end{tabular}",
        ]
    )
    if note:
        lines.extend(
            [
                "  \\vspace{1pt}",
                "  \\begin{minipage}{0.98\\textwidth}",
                f"  \\tiny {note}",
                "  \\end{minipage}",
            ]
        )
    lines.append("\\end{table*}")
    return "\n".join(lines)


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
    binary.drop(columns=["OutcomeKey"]).to_csv(out_dir / "appendix_binary_sweep.csv", index=False)
    regression.drop(columns=["OutcomeKey"]).to_csv(out_dir / "appendix_regression_sweep.csv", index=False)

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
    abbrev_note = (
        "\\textit{Abbreviations.} ICU = intensive care unit; LOS = length of stay; "
        "IMV = invasive mechanical ventilation; CRRT = continuous renal replacement therapy; "
        "SBP = systolic blood pressure; DBP = diastolic blood pressure; BNP = B-type natriuretic peptide / NT-proBNP; "
        "clin. = reference-range anchored clinical bins; order = event order only; tokens = inserted time tokens; "
        "RoPE = admission-relative rotary position embeddings; Freq-matched = frequency-matched mapped codes."
    )
    binary_tex = _grouped_sweep_to_latex(
        binary,
        groups=BINARY_GROUPS,
        caption=(
            "\\textbf{Binary outcome sweep across experiments.} Each cell lists the best configuration "
            "in that experiment, followed by AUROC and its 95\\% bootstrap CI. AUPRC, Brier score, "
            "ECE-15, and baseline-centered BH-adjusted paired tests for the same outcomes are reported in the "
            "aligned statistics files. ICU admission appears only in Experiments~1--2, ICU LOS "
            "$>48$h appears only in Experiment~3, and cells shown as --- are not applicable."
        ),
        label="tab:appendix_binary_sweep",
        first_exp_col_width="0.255\\textwidth",
        note=abbrev_note,
    )
    regression_tex = _grouped_sweep_to_latex(
        regression,
        groups=REGRESSION_GROUPS,
        caption=(
            "\\textbf{Regression outcome sweep across experiments.} Each cell lists the best "
            "configuration in that experiment, followed by Spearman $\\rho$ and its 95\\% bootstrap CI. "
            "$R^2$, MAE, RMSE, and baseline-centered BH-adjusted paired tests for the same outcomes "
            "are reported in the aligned statistics files."
        ),
        label="tab:appendix_regression_sweep",
        note=abbrev_note,
    )

    _write_text(out_dir / "appendix_stats_coverage.tex", coverage_tex + "\n")
    _write_text(out_dir / "appendix_binary_sweep.tex", binary_tex + "\n")
    _write_text(out_dir / "appendix_regression_sweep.tex", regression_tex + "\n")
    print(f"Wrote appendix tables to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
