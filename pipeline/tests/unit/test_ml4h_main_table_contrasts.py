from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "paper"
    / "scripts"
    / "ml4h_main_table_contrasts.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("ml4h_main_table_contrasts", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_main_table_counts_match_printed_grid() -> None:
    contrasts = load_module()
    columns = contrasts.column_specs()
    leaves = contrasts.leaf_specs()
    assert len(columns) == contrasts.COLUMN_COUNT == 17
    assert len(leaves) == contrasts.LEAF_COUNT == 136
    assert sum(contrasts.EXPECTED_TESTS.values()) == contrasts.LEAF_COUNT
    families = {leaf["table_family"] for leaf in (
        {**leaf, "table_family": contrasts.TABLE_FAMILY[(leaf["experiment"], leaf["condition"])]}
        for leaf in leaves
    )}
    assert families == set(contrasts.EXPECTED_TESTS)


def test_permutation_pvalue_floors_at_one_over_draws() -> None:
    contrasts = load_module()
    observed = 0.4
    null = np.array([-0.01, 0.02, -0.03, 0.01])
    assert contrasts.permutation_pvalue(observed, null) == pytest.approx(0.25)
    with pytest.raises(ValueError, match="finite"):
        contrasts.permutation_pvalue(float("nan"), null)
    with pytest.raises(ValueError, match="finite"):
        contrasts.permutation_pvalue(0.1, np.array([np.nan, np.nan]))


def test_within_table_bh_is_independent_across_tables() -> None:
    contrasts = load_module()
    rows = []
    for leaf in contrasts.leaf_specs():
        rows.append(
            {
                **leaf,
                "p_value": 1e-6
                if leaf["experiment"] == "exp3"
                else 0.9,
            }
        )
    intervals = contrasts.attach_within_table_bh(pd.DataFrame(rows))
    exp3 = intervals.loc[intervals["table_family"] == "exp3"]
    other = intervals.loc[intervals["table_family"] != "exp3"]
    assert bool(exp3["bh_significant"].all())
    assert not bool(other["bh_significant"].any())
    assert len(intervals) == contrasts.ROW_COUNT
