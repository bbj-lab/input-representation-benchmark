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
