#!/usr/bin/env python3
"""Generate appendix outcome-descriptive tables for the MLHC manuscript."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from paper.scripts.generate_mlhc_appendix_tables import (
    BINARY_OUTCOME_ORDER,
    DEFAULT_OUT_DIR,
    OUTCOME_LABELS,
    REGRESSION_OUTCOME_ORDER,
    _df_to_latex,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXP12_EXTENDED = (
    ROOT
    / "artifacts"
    / "runs"
    / "tokenized"
    / "mimiciv-3.1_meds_70-10-20"
    / "deciles_none_unfused_time_tokens_first_24h-tokenized"
    / "test"
    / "tokens_timelines_extended_outcomes.parquet"
)
DEFAULT_EXP3_EXTENDED = (
    ROOT
    / "artifacts"
    / "runs"
    / "exp3"
    / "meds_icu"
    / "deciles_none_unfused_time_rope_first_24h-tokenized"
    / "test"
    / "tokens_timelines_extended_outcomes.parquet"
)

EXP12_BINARY_OUTCOMES = {
    "same_admission_death",
    "long_length_of_stay",
    "icu_admission",
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
}

EXP3_BINARY_OUTCOMES = {
    "same_admission_death",
    "long_length_of_stay",
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
}


def _binary_summary(df: pd.DataFrame, outcome: str) -> dict[str, int] | None:
    values = pd.to_numeric(df[outcome], errors="coerce")
    eligible = values.notna()
    outcome_24h = outcome + "_24h"
    if outcome_24h in df.columns:
        eligible &= ~df[outcome_24h].fillna(False).astype(bool)
    eligible_n = int(eligible.sum())
    if eligible_n == 0:
        return None
    positives = int((values[eligible] > 0).sum())
    negatives = eligible_n - positives
    return {
        "eligible_n": eligible_n,
        "positive_n": positives,
        "negative_n": negatives,
    }


def _regression_summary(df: pd.DataFrame, outcome: str) -> dict[str, float] | None:
    values = pd.to_numeric(df[outcome], errors="coerce")
    values = values[np.isfinite(values)]
    if values.empty:
        return None
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return {
        "eligible_n": int(len(values)),
        "mean": float(values.mean()),
        "std": std,
    }


def _fmt_binary_counts(summary: dict[str, int] | None) -> tuple[str, str]:
    if summary is None:
        return "---", "---"
    return str(summary["eligible_n"]), f"{summary['positive_n']} / {summary['negative_n']}"


def _fmt_regression_stats(summary: dict[str, float] | None) -> tuple[str, str]:
    if summary is None:
        return "---", "---"
    return str(summary["eligible_n"]), f"{summary['mean']:.2f} ({summary['std']:.2f})"


def _binary_tables(exp12: pd.DataFrame, exp3: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    csv_rows: list[dict[str, object]] = []
    tex_rows: list[dict[str, str]] = []
    for outcome in BINARY_OUTCOME_ORDER:
        exp12_summary = (
            _binary_summary(exp12, outcome) if outcome in EXP12_BINARY_OUTCOMES and outcome in exp12.columns else None
        )
        exp3_summary = (
            _binary_summary(exp3, outcome) if outcome in EXP3_BINARY_OUTCOMES and outcome in exp3.columns else None
        )
        exp12_n, exp12_counts = _fmt_binary_counts(exp12_summary)
        exp3_n, exp3_counts = _fmt_binary_counts(exp3_summary)
        tex_rows.append(
            {
                "Outcome": OUTCOME_LABELS[outcome],
                "Exp1-2 eligible N": exp12_n,
                "Exp1-2 pos / neg": exp12_counts,
                "Exp3 eligible N": exp3_n,
                "Exp3 pos / neg": exp3_counts,
            }
        )
        csv_rows.append(
            {
                "outcome": outcome,
                "label": OUTCOME_LABELS[outcome],
                "exp12_eligible_n": None if exp12_summary is None else exp12_summary["eligible_n"],
                "exp12_positive_n": None if exp12_summary is None else exp12_summary["positive_n"],
                "exp12_negative_n": None if exp12_summary is None else exp12_summary["negative_n"],
                "exp3_eligible_n": None if exp3_summary is None else exp3_summary["eligible_n"],
                "exp3_positive_n": None if exp3_summary is None else exp3_summary["positive_n"],
                "exp3_negative_n": None if exp3_summary is None else exp3_summary["negative_n"],
            }
        )
    return pd.DataFrame(csv_rows), pd.DataFrame(tex_rows)


def _regression_tables(exp12: pd.DataFrame, exp3: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    csv_rows: list[dict[str, object]] = []
    tex_rows: list[dict[str, str]] = []
    for outcome in REGRESSION_OUTCOME_ORDER:
        exp12_summary = _regression_summary(exp12, outcome) if outcome in exp12.columns else None
        exp3_summary = _regression_summary(exp3, outcome) if outcome in exp3.columns else None
        exp12_n, exp12_stats = _fmt_regression_stats(exp12_summary)
        exp3_n, exp3_stats = _fmt_regression_stats(exp3_summary)
        tex_rows.append(
            {
                "Outcome": OUTCOME_LABELS[outcome],
                "Exp1-2 eligible N": exp12_n,
                "Exp1-2 mean (SD)": exp12_stats,
                "Exp3 eligible N": exp3_n,
                "Exp3 mean (SD)": exp3_stats,
            }
        )
        csv_rows.append(
            {
                "outcome": outcome,
                "label": OUTCOME_LABELS[outcome],
                "exp12_eligible_n": None if exp12_summary is None else exp12_summary["eligible_n"],
                "exp12_mean": None if exp12_summary is None else exp12_summary["mean"],
                "exp12_std": None if exp12_summary is None else exp12_summary["std"],
                "exp3_eligible_n": None if exp3_summary is None else exp3_summary["eligible_n"],
                "exp3_mean": None if exp3_summary is None else exp3_summary["mean"],
                "exp3_std": None if exp3_summary is None else exp3_summary["std"],
            }
        )
    return pd.DataFrame(csv_rows), pd.DataFrame(tex_rows)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp12_extended_outcomes", type=Path, default=DEFAULT_EXP12_EXTENDED)
    parser.add_argument("--exp3_extended_outcomes", type=Path, default=DEFAULT_EXP3_EXTENDED)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    exp12 = pd.read_parquet(args.exp12_extended_outcomes)
    exp3 = pd.read_parquet(args.exp3_extended_outcomes)
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    binary_csv, binary_tex_df = _binary_tables(exp12, exp3)
    regression_csv, regression_tex_df = _regression_tables(exp12, exp3)

    binary_csv.to_csv(out_dir / "appendix_binary_outcome_descriptives.csv", index=False)
    regression_csv.to_csv(out_dir / "appendix_regression_outcome_descriptives.csv", index=False)

    binary_tex = _df_to_latex(
        binary_tex_df,
        caption=(
            "\\textbf{Held-out binary outcome counts by evaluation cohort.} "
            "Each row reports the eligible test-set denominator and the positive/negative counts "
            "used by the binary Stage~3 probes. Whenever a parallel 24-hour exclusion flag exists, "
            "admissions that already met the outcome during the first 24 hours are excluded from "
            "the eligible denominator to match the benchmark evaluation protocol. Exp1--2 use the "
            "all-admission cohort; Exp3 uses the ICU-only cohort, so ICU admission is left blank for "
            "Exp3 and ICU LOS $>48$h is left blank for Exp1--2."
        ),
        label="tab:appendix_binary_outcome_descriptives",
        colspec="p{0.22\\textwidth}p{0.155\\textwidth}p{0.155\\textwidth}p{0.155\\textwidth}p{0.155\\textwidth}",
        tabcolsep=3,
    )
    regression_tex = _df_to_latex(
        regression_tex_df,
        caption=(
            "\\textbf{Held-out regression outcome summaries by evaluation cohort.} "
            "Each row reports the eligible test-set denominator and the mean with standard deviation "
            "for the raw regression labels used by the Stage~3 Ridge probes. Eligibility matches the "
            "evaluation tasks: rows with missing targets are excluded. Exp1--2 use the all-admission "
            "cohort; Exp3 uses the ICU-only cohort."
        ),
        label="tab:appendix_regression_outcome_descriptives",
        colspec="p{0.22\\textwidth}p{0.155\\textwidth}p{0.155\\textwidth}p{0.155\\textwidth}p{0.155\\textwidth}",
        tabcolsep=3,
    )

    _write_text(out_dir / "appendix_binary_outcome_descriptives.tex", binary_tex + "\n")
    _write_text(out_dir / "appendix_regression_outcome_descriptives.tex", regression_tex + "\n")
    print(f"Wrote appendix outcome descriptives to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
