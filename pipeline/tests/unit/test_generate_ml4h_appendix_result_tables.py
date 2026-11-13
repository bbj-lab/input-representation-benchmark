from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "paper"
    / "scripts"
    / "generate_ml4h_appendix_result_tables.py"
)
CONTRASTS = Path(__file__).resolve().parents[3] / "paper" / "scripts"


def load_module():
    sys.path.insert(0, str(CONTRASTS))
    spec = importlib.util.spec_from_file_location(
        "generate_ml4h_appendix_result_tables", SCRIPT
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_fmt_delta_and_ci_format_significance_and_zero() -> None:
    tables = load_module()
    assert tables.fmt_delta(0.012) == "$+0.012$"
    assert tables.fmt_delta(-0.0004) == "$0.000$"
    assert tables.fmt_delta(-0.012, significant=True) == (
        r"{\boldmath$-0.012\rlap{$^{*}$}$}"
    )
    assert tables.fmt_ci(-0.0042, 0.0031, significant=False) == (
        r"{\scriptsize$(-\mkern-1.5mu 0.004{,}{+0.003})$}"
    )
    assert tables.fmt_ci(0.0002, 0.0074, significant=True) == (
        r"{\scriptsize\boldmath$(0.000{,}{+0.007})$}"
    )


def test_starred_delta_and_ci_requires_matching_point_and_bh_flag() -> None:
    tables = load_module()
    intervals = pd.DataFrame(
        [
            {
                "experiment": "exp3",
                "condition": "schema",
                "level": "CLIF 2.1.0",
                "task_type": "binary",
                "family": "Hospital outcomes",
                "role": "family",
                "point": 0.01,
                "ci_lo": 0.007,
                "ci_hi": 0.012,
                "bh_significant": True,
            }
        ]
    )
    d_cell, ci_cell = tables.starred_delta_and_ci(
        intervals,
        experiment="exp3",
        condition="schema",
        level="CLIF 2.1.0",
        task_type="binary",
        family="Hospital outcomes",
        value=0.01,
    )
    assert d_cell == r"{\boldmath$+0.010\rlap{$^{*}$}$}"
    assert ci_cell == r"{\scriptsize\boldmath$(+0.007{,}{+0.012})$}"
    with pytest.raises(ValueError, match="does not match"):
        tables.starred_delta(
            intervals,
            experiment="exp3",
            condition="schema",
            level="CLIF 2.1.0",
            task_type="binary",
            family="Hospital outcomes",
            value=0.02,
        )


def test_appendix_table_formatting_helpers(tmp_path: Path) -> None:
    tables = load_module()
    exp1_row = pd.Series(
        {"quantizer": "trentiles", "anchoring": "10-10-10", "fusion": "unfused"}
    )
    assert (
        tables.config_label("exp1", exp1_row)
        == "Trentiles (10--10--10 reference), unfused"
    )
    assert tables.format_cell(0.884, 0.880, 0.887) == (
        r"\begin{tabular}[c]{@{}c@{}}0.884\\[-1pt][0.880, 0.887]\end{tabular}"
    )

    selected = pd.DataFrame(
        [
            {
                "config_id": "native",
                "config_label": "Native MIMIC",
                "task_type": "binary",
                "outcome_family": "Hospital outcomes",
                "outcome": "same_admission_death",
                "outcome_label": "Same-admission death",
            },
            {
                "config_id": "clif",
                "config_label": "CLIF harmonized",
                "task_type": "binary",
                "outcome_family": "Hospital outcomes",
                "outcome": "same_admission_death",
                "outcome_label": "Same-admission death",
            },
            {
                "config_id": "native",
                "config_label": "Native MIMIC",
                "task_type": "regression",
                "outcome_family": "Vitals",
                "outcome": "min_hr",
                "outcome_label": "Minimum heart rate",
            },
            {
                "config_id": "clif",
                "config_label": "CLIF harmonized",
                "task_type": "regression",
                "outcome_family": "Vitals",
                "outcome": "min_hr",
                "outcome_label": "Minimum heart rate",
            },
        ]
    )
    outcomes = pd.DataFrame(
        [
            {
                "config_id": "native",
                "task_type": "binary",
                "outcome_family": "Hospital outcomes",
                "outcome": "same_admission_death",
                "mean": 0.901,
                "lo": 0.895,
                "hi": 0.906,
            },
            {
                "config_id": "clif",
                "task_type": "binary",
                "outcome_family": "Hospital outcomes",
                "outcome": "same_admission_death",
                "mean": 0.912,
                "lo": 0.908,
                "hi": 0.916,
            },
            {
                "config_id": "native",
                "task_type": "regression",
                "outcome_family": "Vitals",
                "outcome": "min_hr",
                "mean": 0.510,
                "lo": 0.502,
                "hi": 0.518,
            },
            {
                "config_id": "clif",
                "task_type": "regression",
                "outcome_family": "Vitals",
                "outcome": "min_hr",
                "mean": 0.667,
                "lo": 0.660,
                "hi": 0.674,
            },
        ]
    )
    out_path = tmp_path / "outcome_test.tex"
    tables.write_outcome_table("exp3", selected, outcomes, out_path)
    tex = out_path.read_text(encoding="utf-8")
    assert r"\multicolumn{5}{@{}l}{\textbf{Hospital (AUROC)}} \\*[2pt]" in tex
    assert r"\multicolumn{5}{@{}l}{\textbf{Vital signs (Spearman~$\rho$)}} \\*[2pt]" in tex
    assert r"& Native MIMIC & 0.901 & 0.895 & 0.906 \\*" in tex
    assert r"& CLIF 2.1.0 & 0.912 & 0.908 & 0.916 \\[5pt]" in tex
    assert r"& CLIF 2.1.0 & 0.667 & 0.660 & 0.674 \\" in tex
