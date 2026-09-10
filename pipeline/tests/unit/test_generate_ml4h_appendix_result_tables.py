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
CONTRASTS = (
    Path(__file__).resolve().parents[3] / "paper" / "scripts"
)


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


def test_fmt_delta_stars_only_when_requested() -> None:
    tables = load_module()
    assert tables.fmt_delta(0.012) == "+0.012"
    assert tables.fmt_delta(-0.012, significant=True) == "{-0.012$^{*}$}"


def test_starred_delta_requires_matching_point_and_bh_flag() -> None:
    tables = load_module()
    intervals = pd.DataFrame(
        [
            {
                "experiment": "exp3",
                "condition": "schema",
                "level": "CLIF",
                "task_type": "binary",
                "family": "Hospital outcomes",
                "role": "family",
                "point": 0.01,
                "bh_significant": True,
            }
        ]
    )
    text = tables.starred_delta(
        intervals,
        experiment="exp3",
        condition="schema",
        level="CLIF",
        task_type="binary",
        family="Hospital outcomes",
        value=0.01,
    )
    assert text == "{+0.010$^{*}$}"
    with pytest.raises(ValueError, match="does not match"):
        tables.starred_delta(
            intervals,
            experiment="exp3",
            condition="schema",
            level="CLIF",
            task_type="binary",
            family="Hospital outcomes",
            value=0.02,
        )
