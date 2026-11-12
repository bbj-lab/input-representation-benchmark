#!/usr/bin/env python3
"""Write main-text contrast tables and appendix six-model summary tables.

Reads metrics_long.csv and family_tests.csv. Each appendix cell reports the mean
across the six trained models (3 seeds × 2 architectures), with the min and max
of those six values. Family means first average member outcomes on each model,
then summarize the six family means.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ml4h_main_table_contrasts import (
    ROW_COUNT,
    attach_within_table_bh,
    leaf_specs,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
METRICS_CSV = REPO_ROOT / "outputs/runs/ml4h_2026/metrics/metrics_long.csv"
INTERVALS_CSV = (
    REPO_ROOT / "outputs/runs/ml4h_2026/model_level_tests/family_tests.csv"
)
OUTPUT_DIR = REPO_ROOT.parent / "ML4H2026/ML4H/generated"

OUTCOME_FAMILIES = {
    "same_admission_death": "Hospital outcomes",
    "long_length_of_stay": "Hospital outcomes",
    "icu_admission": "Hospital outcomes",
    "prolonged_icu_stay": "Hospital outcomes",
    "imv_event": "Organ support",
    "vasopressor_initiation": "Organ support",
    "crrt_initiation": "Organ support",
    "hemodialysis_initiation": "Organ support",
    "hyperkalemia": "Laboratory",
    "severe_hypokalemia": "Laboratory",
    "profound_hyponatremia": "Laboratory",
    "severe_hypernatremia": "Laboratory",
    "hypoglycemia": "Laboratory",
    "severe_anemia": "Laboratory",
    "tachycardia_hr130": "Vitals",
    "severe_hypertension": "Vitals",
    "hypotension": "Vitals",
    "min_potassium": "Electrolytes",
    "peak_potassium": "Electrolytes",
    "min_sodium": "Electrolytes",
    "max_sodium": "Electrolytes",
    "min_creatinine": "Biomarkers",
    "peak_creatinine": "Biomarkers",
    "min_troponin": "Biomarkers",
    "peak_troponin": "Biomarkers",
    "min_bnp": "Biomarkers",
    "peak_bnp": "Biomarkers",
    "min_glucose": "Biomarkers",
    "max_glucose": "Biomarkers",
    "min_hemoglobin": "Hematology",
    "max_hemoglobin": "Hematology",
    "min_heart_rate": "Vitals",
    "max_heart_rate": "Vitals",
    "min_sbp": "Vitals",
    "max_sbp": "Vitals",
    "min_dbp": "Vitals",
    "max_dbp": "Vitals",
}
FAMILY_ORDER = {
    "binary": ("Hospital outcomes", "Organ support", "Laboratory", "Vitals"),
    "regression": ("Electrolytes", "Biomarkers", "Hematology", "Vitals"),
}
OUTCOME_FAMILY_DISPLAY = {
    "Hospital outcomes": "Hospital",
    "Organ support": "Organ support",
    "Laboratory": "Laboratory",
    "Vitals": "Vital signs",
    "Electrolytes": "Electrolytes",
    "Biomarkers": "Biomarkers",
    "Hematology": "Hematology",
}
GRANULARITY_LABEL = {
    ("deciles", "none"): "Deciles · population",
    ("ventiles", "none"): "Ventiles · population",
    ("ventiles", "5-10-5"): "Ventiles · 5--10--5 reference",
    ("trentiles", "none"): "Trentiles · population",
    ("trentiles", "10-10-10"): "Trentiles · 10--10--10 reference",
    ("centiles", "none"): "Centiles · population",
}
REPRESENTATION_LABEL = {
    "discrete": "Discrete deciles",
    "soft": "Soft discretization",
    "xval": "Code-normalized xVal",
    "xval_affine": "Affine xVal",
}
TEMPORAL_LABEL = {
    "time_tokens": "time tokens",
    "event_order": "event order",
    "time_rope": "admission-relative RoPE",
}
FAMILY_COLUMN = {
    ("binary", "Hospital outcomes"): "Hospital AUROC",
    ("binary", "Organ support"): "Organ-support AUROC",
    ("binary", "Laboratory"): "Laboratory AUROC",
    ("binary", "Vitals"): "Vital signs AUROC",
    ("regression", "Electrolytes"): r"Electrolytes $\rho$",
    ("regression", "Biomarkers"): r"Biomarkers $\rho$",
    ("regression", "Hematology"): r"Hematology $\rho$",
    ("regression", "Vitals"): r"Vital signs $\rho$",
}
FAMILY_CAPTION_NOTE = (
    "Each cell reports the mean across six trained models "
    r"(three seeds $\times$ two architectures), with min/max in brackets. "
    "Member outcomes receive equal weight within each model before the "
    "six-model summary."
)
OUTCOME_CAPTION_NOTE = (
    "Mean, Min, and Max summarize six trained models "
    r"(three seeds $\times$ two architectures)."
)
OUTCOME_DISPLAY_LABELS = {
    "same_admission_death": "Hospital mortality in hours 24--48",
    "long_length_of_stay": "Still hospitalized at hour 48",
    "icu_admission": "First ICU admission in hours 24--48",
    "prolonged_icu_stay": "Still in ICU at hour 48",
    "crrt_initiation": "Continuous renal replacement therapy initiation",
    "peak_potassium": "Maximum potassium",
    "peak_creatinine": "Maximum creatinine",
    "peak_troponin": "Maximum troponin",
    "min_bnp": "Minimum NT-proBNP",
    "peak_bnp": "Maximum NT-proBNP",
}
FAMILY_ROWS = [
    ("binary", "Hospital outcomes", "Hospital AUROC"),
    ("binary", "Organ support", "Organ-support AUROC"),
    ("binary", "Laboratory", "Laboratory AUROC"),
    ("binary", "Vitals", "Vital signs AUROC"),
    ("regression", "Electrolytes", r"Electrolytes $\rho$"),
    ("regression", "Biomarkers", r"Biomarkers $\rho$"),
    ("regression", "Hematology", r"Hematology $\rho$"),
    ("regression", "Vitals", r"Vital signs $\rho$"),
]
MAIN_FAMILY_LABELS = (
    "Hospital",
    "Organ support",
    "Laboratory",
    "Vital signs",
    "Electrolytes",
    "Biomarkers",
    "Hematology",
    "Vital signs",
)


def panel_line(n_cols: int, title: str) -> str:
    return rf"    \multicolumn{{{n_cols}}}{{@{{}}l}}{{\textit{{{title}}}}} \\"


def append_family_panels(lines: list[str], n_cols: int, write_row) -> None:
    if len(FAMILY_ROWS) != len(MAIN_FAMILY_LABELS):
        raise ValueError("Family label count mismatch")
    for index, ((task, family_name, _), short) in enumerate(
        zip(FAMILY_ROWS, MAIN_FAMILY_LABELS)
    ):
        if index == 0:
            lines.append(panel_line(n_cols, r"Binary families, AUROC"))
        elif index == 4:
            lines.append(r"    \midrule")
            lines.append(panel_line(n_cols, r"Regression families, Spearman $\rho$"))
        lines.append(r"    \addlinespace[1pt]")
        write_row(lines, task, family_name, short)


def performance_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    selected = metrics.loc[
        (
            ((metrics["task_type"] == "binary") & (metrics["metric"] == "roc_auc"))
            | (
                (metrics["task_type"] == "regression")
                & (metrics["metric"] == "spearman_rho")
            )
        )
    ].copy()
    if selected.empty:
        raise ValueError("No AUROC or Spearman rows.")
    selected["outcome_family"] = selected["outcome"].map(OUTCOME_FAMILIES)
    missing = selected["outcome_family"].isna()
    if missing.any():
        unknown = sorted(set(selected.loc[missing, "outcome"].astype(str)))
        raise ValueError(f"Outcomes missing family mapping: {unknown}")
    return selected


def config_label(experiment: str, row: pd.Series) -> str:
    if experiment == "exp1":
        grain = GRANULARITY_LABEL[(str(row["quantizer"]), str(row["anchoring"]))]
        return f"{grain}, {row['fusion']}"
    if experiment == "exp2":
        return (
            f"{REPRESENTATION_LABEL[str(row['representation'])]} + "
            f"{TEMPORAL_LABEL[str(row['temporal'])]}"
        )
    label = str(row["config_label"])
    if label == "CLIF harmonized":
        return "CLIF 2.1.0"
    return label


def six_model_outcome(selected: pd.DataFrame) -> pd.DataFrame:
    keys = ["experiment", "config_id", "task_type", "metric", "outcome", "outcome_family"]
    per_model = selected.groupby(
        [*keys, "backbone", "seed"], as_index=False, sort=False
    ).agg(point=("point", "mean"))
    counts = per_model.groupby(keys, sort=False)["point"].transform("size")
    if not (counts == 6).all():
        raise ValueError("Every outcome cell must have six trained models.")
    return per_model.groupby(keys, as_index=False, sort=False).agg(
        mean=("point", "mean"),
        lo=("point", "min"),
        hi=("point", "max"),
    )


def six_model_family(selected: pd.DataFrame) -> pd.DataFrame:
    keys = ["experiment", "config_id", "task_type", "metric", "outcome_family"]
    per_model = selected.groupby(
        [*keys, "backbone", "seed"], as_index=False, sort=False
    ).agg(point=("point", "mean"))
    counts = per_model.groupby(keys, sort=False)["point"].transform("size")
    if not (counts == 6).all():
        raise ValueError("Every family cell must have six trained models.")
    return per_model.groupby(keys, as_index=False, sort=False).agg(
        mean=("point", "mean"),
        lo=("point", "min"),
        hi=("point", "max"),
    )


def experiment_name(experiment: str) -> str:
    return f"Experiment~{experiment[-1]}"


def format_cell(mean: float, lo: float, hi: float) -> str:
    return rf"\shortstack[c]{{{mean:.3f}\\{{}}[{lo:.3f}, {hi:.3f}]}}"


def tex_escape(text: str) -> str:
    return text.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


def config_order(experiment: str, rows: pd.DataFrame) -> list[str]:
    unique = rows.drop_duplicates("config_id")
    if experiment == "exp1":
        grain_order = list(GRANULARITY_LABEL.keys())
        fusion_order = ("unfused", "fused")

        def key(row: pd.Series) -> tuple[int, int]:
            return (
                grain_order.index((str(row["quantizer"]), str(row["anchoring"]))),
                fusion_order.index(str(row["fusion"])),
            )

        return [
            str(row["config_id"])
            for _, row in sorted(unique.iterrows(), key=lambda item: key(item[1]))
        ]
    if experiment == "exp2":
        rep_order = list(REPRESENTATION_LABEL)
        temp_order = list(TEMPORAL_LABEL)

        def key(row: pd.Series) -> tuple[int, int]:
            return (
                temp_order.index(str(row["temporal"])),
                rep_order.index(str(row["representation"])),
            )

        return [
            str(row["config_id"])
            for _, row in sorted(unique.iterrows(), key=lambda item: key(item[1]))
        ]
    preferred = ("Native MIMIC", "CLIF harmonized")
    label_to_id = {
        str(row["config_label"]): str(row["config_id"])
        for _, row in unique.iterrows()
    }
    missing = [label for label in preferred if label not in label_to_id]
    if missing:
        raise ValueError(f"Unexpected Experiment~3 labels: {sorted(label_to_id)}")
    return [label_to_id[label] for label in preferred]


def write_family_table(
    experiment: str,
    selected: pd.DataFrame,
    family: pd.DataFrame,
    path: Path,
) -> None:
    labels = {
        str(row["config_id"]): config_label(experiment, row)
        for _, row in selected.drop_duplicates("config_id").iterrows()
    }
    order = config_order(experiment, selected)
    columns = [
        FAMILY_COLUMN[(task, family_name)]
        for task, families in FAMILY_ORDER.items()
        for family_name in families
    ]
    lookup = {
        (str(row["config_id"]), str(row["task_type"]), str(row["outcome_family"])): row
        for _, row in family.iterrows()
    }
    lines = [
        r"\begin{table}[htbp]",
        r"\floatconts",
        f"  {{tab:appendix_family_means_{experiment}}}",
        "  {\\caption{Family-mean AUROC and Spearman~$\\rho$ for every "
        + experiment_name(experiment)
        + " configuration. "
        + FAMILY_CAPTION_NOTE
        + "}}",
        r"  {\resizebox{\linewidth}{!}{%",
        r"  \scriptsize",
        r"  \setlength{\tabcolsep}{3pt}",
        r"  \begin{tabular}{l" + "c" * len(columns) + "}",
        r"    \toprule",
        "    Configuration & " + " & ".join(columns) + r" \\",
        r"    \midrule",
    ]
    for config_id in order:
        cells = [tex_escape(labels[config_id])]
        for task, families in FAMILY_ORDER.items():
            for family_name in families:
                row = lookup.get((config_id, task, family_name))
                if row is None:
                    cells.append("---")
                else:
                    cells.append(format_cell(row["mean"], row["lo"], row["hi"]))
        lines.append("    " + " & ".join(cells) + r" \\")
    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}}}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_outcome_table(
    experiment: str,
    selected: pd.DataFrame,
    outcomes: pd.DataFrame,
    path: Path,
) -> None:
    labels = {
        str(row["config_id"]): config_label(experiment, row)
        for _, row in selected.drop_duplicates("config_id").iterrows()
    }
    name = {
        str(row["outcome"]): OUTCOME_DISPLAY_LABELS.get(
            str(row["outcome"]), str(row["outcome_label"])
        )
        for _, row in selected.drop_duplicates("outcome").iterrows()
    }
    order = config_order(experiment, selected)
    family_sequence = [
        (task, family_name)
        for task, families in FAMILY_ORDER.items()
        for family_name in families
    ]
    lines = [
        r"\begin{longtable}{>{\raggedright\arraybackslash}p{0.20\textwidth}"
        r">{\raggedright\arraybackslash}p{0.34\textwidth}ccc}",
        r"  \caption{Outcome-level AUROC and Spearman~$\rho$ for every "
        f"{experiment_name(experiment)} configuration. "
        f"{OUTCOME_CAPTION_NOTE}"
        r"} \label{tab:appendix_outcome_means_" + experiment + r"} \\",
        r"  \toprule",
        r"  Outcome & Configuration & Mean & Min & Max \\",
        r"  \midrule",
        r"  \endfirsthead",
        r"  \multicolumn{5}{l}{\textbf{Table~\thetable\ continued from previous page}} \\",
        r"  \toprule",
        r"  Outcome & Configuration & Mean & Min & Max \\",
        r"  \midrule",
        r"  \endhead",
        r"  \midrule",
        r"  \multicolumn{5}{r}{\textit{Continued on next page}} \\",
        r"  \endfoot",
        r"  \bottomrule",
        r"  \endlastfoot",
    ]
    for task, family_name in family_sequence:
        members = outcomes.loc[
            (outcomes["task_type"] == task) & (outcomes["outcome_family"] == family_name)
        ]
        if members.empty:
            continue
        metric = "AUROC" if task == "binary" else r"Spearman~$\rho$"
        display_family = OUTCOME_FAMILY_DISPLAY[family_name]
        lines.append(
            r"  \multicolumn{5}{l}{\textbf{"
            + f"{display_family} ({metric})"
            + r"}} \\"
        )
        for outcome in members["outcome"].drop_duplicates():
            block = members.loc[members["outcome"] == outcome]
            by_config = {str(row["config_id"]): row for _, row in block.iterrows()}
            first = True
            for config_id in order:
                row = by_config.get(config_id)
                if row is None:
                    continue
                outcome_cell = tex_escape(name[str(outcome)]) if first else ""
                first = False
                lines.append(
                    "  "
                    + " & ".join(
                        [
                            outcome_cell,
                            tex_escape(labels[config_id]),
                            f"{row['mean']:.3f}",
                            f"{row['lo']:.3f}",
                            f"{row['hi']:.3f}",
                        ]
                    )
                    + r" \\"
                )
    lines.extend([r"\end{longtable}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def attach_config_attrs(family: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    attrs = selected.drop_duplicates("config_id")[
        [
            "config_id",
            "config_label",
            "quantizer",
            "anchoring",
            "fusion",
            "representation",
            "temporal",
        ]
    ]
    return family.merge(attrs, on="config_id", how="left")


def family_mean(family: pd.DataFrame, task: str, family_name: str, **attrs: str) -> float:
    mask = (family["task_type"] == task) & (family["outcome_family"] == family_name)
    for key, value in attrs.items():
        mask &= family[key].astype(str) == value
    hit = family.loc[mask]
    if len(hit) != 1:
        raise ValueError(f"Expected one family cell for {task}/{family_name}/{attrs}, found {len(hit)}")
    return float(hit.iloc[0]["mean"])


def fmt_point(value: float) -> str:
    return f"${value:.3f}$"


def fmt_signed(value: float) -> str:
    rounded = round(value, 3)
    if abs(rounded) == 0.0:
        return "0.000"
    return f"{rounded:+.3f}"


def fmt_delta(value: float, significant: bool = False) -> str:
    text = fmt_signed(value)
    if significant:
        return rf"{{\boldmath${text}\rlap{{$^{{*}}$}}$}}"
    return f"${text}$"


def fmt_ci(ci_lo: float, ci_hi: float, significant: bool = False) -> str:
    def fmt_bound(bound: float) -> str:
        rounded = round(bound, 3)
        if abs(rounded) == 0.0:
            return "0.000"
        if rounded < 0:
            return rf"-\mkern-1.5mu {abs(rounded):.3f}"
        return f"+{rounded:.3f}"

    bold = r"\boldmath" if significant else ""
    return rf"{{\scriptsize{bold}$({fmt_bound(ci_lo)}{{,}}{{{fmt_bound(ci_hi)}}})$}}"


def interval_row(
    intervals: pd.DataFrame,
    *,
    experiment: str,
    condition: str,
    level: str,
    task_type: str,
    family: str,
) -> pd.Series:
    hit = intervals.loc[
        (intervals["experiment"] == experiment)
        & (intervals["condition"] == condition)
        & (intervals["level"] == level)
        & (intervals["task_type"] == task_type)
        & (intervals["family"] == family)
    ]
    if "role" in intervals.columns:
        hit = hit.loc[hit["role"].astype(str).isin(("family", "leaf"))]
    if len(hit) != 1:
        raise ValueError(
            f"Expected one interval for {experiment}/{condition}/{level}/"
            f"{task_type}/{family}, found {len(hit)}"
        )
    return hit.iloc[0]


def delta_significant(row: pd.Series) -> bool:
    if "bh_significant" not in row.index:
        raise ValueError("Interval row is missing the within-table BH decision")
    return bool(row["bh_significant"])


def starred_delta_and_ci(
    intervals: pd.DataFrame,
    *,
    experiment: str,
    condition: str,
    level: str,
    task_type: str,
    family: str,
    value: float,
) -> tuple[str, str]:
    row = interval_row(
        intervals,
        experiment=experiment,
        condition=condition,
        level=level,
        task_type=task_type,
        family=family,
    )
    expected = float(row["point"])
    if abs(expected - value) > 1e-12:
        raise ValueError(
            f"{experiment}/{condition}/{level}/{task_type}/{family}: "
            f"table Δ {value} does not match interval point {expected}"
        )
    sig = delta_significant(row)
    return (
        fmt_delta(value, significant=sig),
        fmt_ci(float(row["ci_lo"]), float(row["ci_hi"]), significant=sig),
    )


def starred_delta(
    intervals: pd.DataFrame,
    *,
    experiment: str,
    condition: str,
    level: str,
    task_type: str,
    family: str,
    value: float,
) -> str:
    row = interval_row(
        intervals,
        experiment=experiment,
        condition=condition,
        level=level,
        task_type=task_type,
        family=family,
    )
    expected = float(row["point"])
    if abs(expected - value) > 1e-12:
        raise ValueError(
            f"{experiment}/{condition}/{level}/{task_type}/{family}: "
            f"table Δ {value} does not match interval point {expected}"
        )
    return fmt_delta(value, significant=delta_significant(row))


def write_main_exp1_quantization_table(
    family: pd.DataFrame, path: Path, intervals: pd.DataFrame
) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\floatconts",
        r"  {tab:exp1_quantization}",
        r"  {\caption{Experiment~1: Unfused quantization relative to equal-frequency deciles. Across \tableref{tab:exp1_quantization}--\tableref{tab:exp3_family}, entries report six-model family means and signed differences ($\Delta$) with unadjusted 95\% confidence intervals ($n=6$, $\mathrm{df}=5$) in parentheses. Bold and {\boldmath$^{*}$} show BH-adjusted $p<0.05$.}}",
        r"  {\footnotesize",
        r"  \renewcommand{\arraystretch}{1.0}",
        r"  \setlength{\tabcolsep}{4.0pt}",
        r"  \begin{tabular}{@{}lcccccc@{}}",
        r"    \toprule",
        r"    & Reference & \multicolumn{2}{c}{Ventiles} & \multicolumn{2}{c}{Trentiles} & Centiles \\",
        r"    \cmidrule(lr){2-2} \cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-7}",
        r"    Family & Mean & $\Delta$ equal-frequency & $\Delta$ 5--10--5 & $\Delta$ equal-frequency & $\Delta$ 10--10--10 & $\Delta$ equal-frequency \\",
        r"    \midrule",
    ]
    alts = [
        (("ventiles", "none"), "Ventiles · population"),
        (("ventiles", "5-10-5"), "Ventiles · reference"),
        (("trentiles", "none"), "Trentiles · population"),
        (("trentiles", "10-10-10"), "Trentiles · reference"),
        (("centiles", "none"), "Centiles · population"),
    ]

    def write_row(lines: list[str], task: str, family_name: str, short: str) -> None:
        ref = family_mean(
            family,
            task,
            family_name,
            quantizer="deciles",
            anchoring="none",
            fusion="unfused",
        )
        top_cells = [short, fmt_point(ref)]
        ci_cells = []
        for (quantizer, anchoring), level in alts:
            alt = family_mean(
                family,
                task,
                family_name,
                quantizer=quantizer,
                anchoring=anchoring,
                fusion="unfused",
            )
            d_cell, ci_cell = starred_delta_and_ci(
                intervals,
                experiment="exp1",
                condition="granularity",
                level=level,
                task_type=task,
                family=family_name,
                value=alt - ref,
            )
            top_cells.append(d_cell)
            ci_cells.append(ci_cell)
        lines.append("    " + " & ".join(top_cells) + r" \\[-2pt]")
        lines.append("    & & " + " & ".join(ci_cells) + r" \\")

    append_family_panels(lines, 7, write_row)
    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}}",
            r"\end{table*}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_main_exp1_fusion_table(
    family: pd.DataFrame, path: Path, intervals: pd.DataFrame
) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\floatconts",
        r"  {tab:exp1_fusion}",
        r"  {\caption{Experiment~1: Effect of code--value fusion ($\Delta$, fused minus unfused) at matched quantization granularity.}}",
        r"  {\footnotesize",
        r"  \renewcommand{\arraystretch}{1.0}",
        r"  \setlength{\tabcolsep}{4.0pt}",
        r"  \begin{tabular}{@{}lcccccc@{}}",
        r"    \toprule",
        r"    Family & Deciles & Ventiles & 5--10--5 & Trentiles & 10--10--10 & Centiles \\",
        r"    \midrule",
    ]
    grains = [
        (("deciles", "none"), "Fused · deciles"),
        (("ventiles", "none"), "Fused · ventiles"),
        (("ventiles", "5-10-5"), "Fused · 5-10-5"),
        (("trentiles", "none"), "Fused · trentiles"),
        (("trentiles", "10-10-10"), "Fused · 10-10-10"),
        (("centiles", "none"), "Fused · centiles"),
    ]

    def write_row(lines: list[str], task: str, family_name: str, short: str) -> None:
        top_cells = [short]
        ci_cells = []
        for (quantizer, anchoring), level in grains:
            unfused = family_mean(
                family,
                task,
                family_name,
                quantizer=quantizer,
                anchoring=anchoring,
                fusion="unfused",
            )
            fused = family_mean(
                family,
                task,
                family_name,
                quantizer=quantizer,
                anchoring=anchoring,
                fusion="fused",
            )
            d_cell, ci_cell = starred_delta_and_ci(
                intervals,
                experiment="exp1",
                condition="fusion",
                level=level,
                task_type=task,
                family=family_name,
                value=fused - unfused,
            )
            top_cells.append(d_cell)
            ci_cells.append(ci_cell)
        lines.append("    " + " & ".join(top_cells) + r" \\[-2pt]")
        lines.append("    & " + " & ".join(ci_cells) + r" \\")

    append_family_panels(lines, 7, write_row)
    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}}",
            r"\end{table*}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_main_exp2_table(
    family: pd.DataFrame,
    path: Path,
    intervals: pd.DataFrame,
) -> None:
    numeric_specs = [
        ("soft", "Soft discretization"),
        ("xval", "Code-normalized xVal"),
        ("xval_affine", "Affine xVal"),
    ]
    temporal_specs = [
        ("event_order", "Event order"),
        ("time_rope", "Admission-relative RoPE"),
    ]
    lines = [
        r"\begin{table*}[t]",
        r"\floatconts",
        r"  {tab:exp2_numeric}",
        r"  {\caption{Experiment~2: Numeric and temporal encodings ($\Delta$ relative to discrete deciles with explicit time tokens). Numeric encoding columns vary the value representation with explicit time tokens fixed. Temporal encoding columns vary the time representation with discrete deciles fixed.}}",
        r"  {\footnotesize",
        r"  \renewcommand{\arraystretch}{1.0}",
        r"  \setlength{\tabcolsep}{4.0pt}",
        r"  \begin{tabular}{@{}lcccccc@{}}",
        r"    \toprule",
        r"    & Reference & \multicolumn{3}{c}{Numeric encoding $\Delta$} & \multicolumn{2}{c}{Temporal encoding $\Delta$} \\",
        r"    \cmidrule(lr){2-2} \cmidrule(lr){3-5} \cmidrule(lr){6-7}",
        r"    & & Soft & Code-normalized & Affine & Event & Admission-relative \\[-1pt]",
        r"    Family & Mean & discretization & xVal & xVal & order & RoPE \\",
        r"    \midrule",
    ]

    def write_row(lines: list[str], task: str, family_name: str, short: str) -> None:
        ref = family_mean(
            family,
            task,
            family_name,
            representation="discrete",
            temporal="time_tokens",
        )
        top_cells = [short, fmt_point(ref)]
        ci_cells = []
        for representation, level in numeric_specs:
            value = family_mean(
                family,
                task,
                family_name,
                representation=representation,
                temporal="time_tokens",
            )
            d_cell, ci_cell = starred_delta_and_ci(
                intervals,
                experiment="exp2",
                condition="numeric",
                level=level,
                task_type=task,
                family=family_name,
                value=value - ref,
            )
            top_cells.append(d_cell)
            ci_cells.append(ci_cell)
        for temporal, level in temporal_specs:
            value = family_mean(
                family,
                task,
                family_name,
                representation="discrete",
                temporal=temporal,
            )
            d_cell, ci_cell = starred_delta_and_ci(
                intervals,
                experiment="exp2",
                condition="temporal",
                level=level,
                task_type=task,
                family=family_name,
                value=value - ref,
            )
            top_cells.append(d_cell)
            ci_cells.append(ci_cell)
        lines.append("    " + " & ".join(top_cells) + r" \\[-2pt]")
        lines.append("    & & " + " & ".join(ci_cells) + r" \\")

    append_family_panels(lines, 7, write_row)
    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}}",
            r"\end{table*}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_main_exp3_table(
    family: pd.DataFrame, path: Path, intervals: pd.DataFrame
) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\floatconts",
        r"  {tab:exp3_family}",
        r"  {\caption{Experiment~3: Effect of schema remapping ($\Delta$, CLIF~2.1.0 minus native MIMIC).}}",
        r"  {\footnotesize",
        r"  \renewcommand{\arraystretch}{1.0}",
        r"  \setlength{\tabcolsep}{7.0pt}",
        r"  \begin{tabular}{@{}lcc@{}}",
        r"    \toprule",
        r"    & Reference & $\Delta$ \\",
        r"    \cmidrule(lr){2-2} \cmidrule(lr){3-3}",
        r"    Family & Native MIMIC & CLIF 2.1.0 \\",
        r"    \midrule",
    ]

    def write_row(lines: list[str], task: str, family_name: str, short: str) -> None:
        native = family_mean(family, task, family_name, config_label="Native MIMIC")
        clif = family_mean(family, task, family_name, config_label="CLIF harmonized")
        d_cell, ci_cell = starred_delta_and_ci(
            intervals,
            experiment="exp3",
            condition="schema",
            level="CLIF 2.1.0",
            task_type=task,
            family=family_name,
            value=clif - native,
        )
        lines.append(
            "    " + " & ".join([short, fmt_point(native), d_cell]) + r" \\[-2pt]"
        )
        lines.append("    & & " + ci_cell + r" \\")

    append_family_panels(lines, 3, write_row)
    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def load_intervals(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing family test table: {path}")
    intervals = (
        pd.read_csv(path)
        if path.suffix.lower() == ".csv"
        else pd.read_parquet(path)
    )
    if "delta" in intervals.columns and "point" not in intervals.columns:
        intervals = intervals.rename(columns={"delta": "point"})
    if "p_t" in intervals.columns and "p_value" not in intervals.columns:
        intervals = intervals.rename(columns={"p_t": "p_value"})
    if "star_t" in intervals.columns and "bh_significant" not in intervals.columns:
        intervals = intervals.rename(columns={"star_t": "bh_significant"})
    if len(intervals) != ROW_COUNT:
        raise ValueError(f"Expected {ROW_COUNT} interval rows, found {len(intervals)}")
    if "p_value" not in intervals.columns or intervals["p_value"].isna().any():
        raise ValueError("Family test rows need two-sided p-values")
    if "ci_lo" not in intervals.columns or "ci_hi" not in intervals.columns:
        raise ValueError("Family test rows need ci_lo and ci_hi columns")
    if "bh_significant" not in intervals.columns:
        intervals = attach_within_table_bh(intervals)
    for task in leaf_specs():
        interval_row(
            intervals,
            experiment=str(task["experiment"]),
            condition=str(task["condition"]),
            level=str(task["level"]),
            task_type=str(task["task_type"]),
            family=str(task["family"]),
        )
    return intervals


def generate(
    *,
    metrics_path: Path,
    output_dir: Path,
    intervals_path: Path,
) -> dict[str, str]:
    metrics = pd.read_csv(metrics_path)
    intervals = load_intervals(intervals_path)
    selected = performance_rows(metrics)
    expected = {"exp1": 12, "exp2": 12, "exp3": 2}
    written: dict[str, str] = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    for experiment, n_config in expected.items():
        subset = selected.loc[selected["experiment"] == experiment].copy()
        n_found = int(subset["config_id"].nunique())
        if n_found != n_config:
            raise ValueError(f"{experiment}: expected {n_config} configs, found {n_found}")
        family = six_model_family(subset)
        outcomes = six_model_outcome(subset)
        family_path = output_dir / f"appendix_family_means_{experiment}.tex"
        outcome_path = output_dir / f"appendix_outcome_means_{experiment}.tex"
        write_family_table(experiment, subset, family, family_path)
        write_outcome_table(experiment, subset, outcomes, outcome_path)
        written[f"{experiment}_family"] = str(family_path)
        written[f"{experiment}_outcome"] = str(outcome_path)
        family_attrs = attach_config_attrs(family, subset)
        if experiment == "exp1":
            quantization_path = output_dir / "main_exp1_quantization_table.tex"
            fusion_path = output_dir / "main_exp1_fusion_table.tex"
            write_main_exp1_quantization_table(family_attrs, quantization_path, intervals)
            write_main_exp1_fusion_table(family_attrs, fusion_path, intervals)
            written["exp1_main_quantization"] = str(quantization_path)
            written["exp1_main_fusion"] = str(fusion_path)
        elif experiment == "exp2":
            numeric_path = output_dir / "main_exp2_numeric_table.tex"
            write_main_exp2_table(family_attrs, numeric_path, intervals)
            written["exp2_main_numeric"] = str(numeric_path)
        else:
            main_path = output_dir / "main_exp3_family_table.tex"
            write_main_exp3_table(family_attrs, main_path, intervals)
            written["exp3_main"] = str(main_path)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-long", type=Path, default=METRICS_CSV)
    parser.add_argument("--intervals", type=Path, default=INTERVALS_CSV)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    written = generate(
        metrics_path=args.metrics_long.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        intervals_path=args.intervals.expanduser().resolve(),
    )
    for key, path in written.items():
        print(f"{key}\t{path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
