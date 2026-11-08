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
    assert renderer.DEFAULT_OUTPUT_ROOT.parts[-3:] == ("runs", "ml4h_2026", "metrics")


def test_experiment_specs_pin_completed_cohort_sizes() -> None:
    renderer = load_renderer()

    assert renderer.EXPERIMENT_SPECS["exp1"].stage3_dir == "stage3_24to48h_exp1"
    assert renderer.EXPERIMENT_SPECS["exp1"].expected_runs == 72
    assert renderer.EXPERIMENT_SPECS["exp1"].test_rows == 85_057
    assert renderer.EXPERIMENT_SPECS["exp2"].stage3_dir == "stage3_24to48h_exp2"
    assert renderer.EXPERIMENT_SPECS["exp2"].expected_runs == 72
    assert renderer.EXPERIMENT_SPECS["exp2"].test_rows == 85_057
    assert renderer.EXPERIMENT_SPECS["exp3"].stage3_dir == "stage3_24to48h_exp3"
    assert renderer.EXPERIMENT_SPECS["exp3"].expected_runs == 12
    assert renderer.EXPERIMENT_SPECS["exp3"].test_rows == 16_529


def test_exp2_labels_preserve_value_and_temporal_mechanics() -> None:
    renderer = load_renderer()
    spec = renderer.EXPERIMENT_SPECS["exp2"]
    config_id = "meds_deciles_none_fusedFalse_xval_affine_time_rope"

    assert renderer._exp2_representation_and_temporal(config_id) == (
        "xval_affine",
        "time_rope",
    )
    assert renderer.experiment_config_label(spec, config_id) == "xVal-affine · Time RoPE"


def test_nested_model_locations_keep_distinct_artifact_stems() -> None:
    renderer = load_renderer()

    first = "/runs/exp2_a-discrete-time_rope-s42/model-discrete-time_rope"
    second = "/runs/exp2_b-discrete-time_rope-s42/model-discrete-time_rope"
    assert renderer.artifact_stem(first) != renderer.artifact_stem(second)
    assert renderer.artifact_stem(first).endswith("model-discrete-time_rope")


def test_partial_render_skips_exp1_and_scales_metric_rows() -> None:
    renderer = load_renderer()

    assert tuple(renderer.selected_specs(("exp2", "exp3"))) == ("exp2", "exp3")
    assert renderer.expected_metric_rows(("exp2", "exp3")) == (
        72 * (17 * 4 + 20 * 4) + 12 * (16 * 4 + 20 * 4)
    )
    assert renderer.expected_metric_rows(("exp1", "exp2", "exp3")) == (
        72 * (17 * 4 + 20 * 4) * 2 + 12 * (16 * 4 + 20 * 4)
    )
