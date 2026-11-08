from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "paper"
    / "scripts"
    / "generate_ml4h_24to48h_provisional.py"
)


def load_renderer():
    spec = importlib.util.spec_from_file_location(
        "generate_ml4h_24to48h_provisional", SCRIPT
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_provisional_renderer_uses_37_37_36_task_contract() -> None:
    renderer = load_renderer()

    assert tuple(map(len, renderer.outcome_contract("exp1"))) == (17, 20)
    assert tuple(map(len, renderer.outcome_contract("exp2"))) == (17, 20)
    assert tuple(map(len, renderer.outcome_contract("exp3"))) == (16, 20)
    assert renderer.DEFAULT_OUTPUT_ROOT.parts[-3:] == ("runs", "ml4h_2026", "metric_bundle")


def test_partial_render_skips_exp1_and_scales_metric_rows() -> None:
    renderer = load_renderer()

    assert tuple(renderer.selected_specs(("exp2", "exp3"))) == ("exp2", "exp3")
    assert renderer.expected_figure_count(("exp2", "exp3")) == 8
    assert renderer.expected_metric_rows(("exp2", "exp3")) == (
        72 * (17 * 4 + 20 * 4) + 12 * (16 * 4 + 20 * 4)
    )
    assert renderer.expected_metric_rows(("exp1", "exp2", "exp3")) == (
        72 * (17 * 4 + 20 * 4) * 2 + 12 * (16 * 4 + 20 * 4)
    )
