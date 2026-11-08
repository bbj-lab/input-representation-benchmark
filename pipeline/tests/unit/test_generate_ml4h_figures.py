from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "paper"
    / "scripts"
    / "generate_ml4h_figures.py"
)


def load_renderer():
    spec = importlib.util.spec_from_file_location("generate_ml4h_figures", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_experiment_specs_pin_completed_cohort_sizes() -> None:
    renderer = load_renderer()

    assert renderer.EXPERIMENT_SPECS["exp1"].stage3_dir == "stage3_padded_repair"
    assert renderer.EXPERIMENT_SPECS["exp1"].expected_runs == 72
    assert renderer.EXPERIMENT_SPECS["exp1"].test_rows == 85_057
    assert renderer.EXPERIMENT_SPECS["exp2"].expected_runs == 72
    assert renderer.EXPERIMENT_SPECS["exp2"].test_rows == 85_057
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
    assert renderer.experiment_config_marker(spec, config_id) == "^"


def test_nested_model_locations_keep_distinct_artifact_stems() -> None:
    renderer = load_renderer()

    first = "/runs/exp2_a-discrete-time_rope-s42/model-discrete-time_rope"
    second = "/runs/exp2_b-discrete-time_rope-s42/model-discrete-time_rope"
    assert renderer.artifact_stem(first) != renderer.artifact_stem(second)
    assert renderer.artifact_stem(first).endswith("model-discrete-time_rope")


def test_exp3_outcome_contract_drops_degenerate_icu_admission() -> None:
    renderer = load_renderer()
    spec = renderer.EXPERIMENT_SPECS["exp3"]
    binary = tuple(f"binary_{index}" for index in range(16))
    regression = tuple(f"regression_{index}" for index in range(22))

    renderer.validate_outcome_contract(spec, binary, regression)
    assert renderer.expected_metric_row_count(spec, binary, regression) == 1_824
    with pytest.raises(ValueError, match="icu_admission"):
        renderer.validate_outcome_contract(
            spec, binary + ("icu_admission",), regression
        )


def test_stage3_task_parser_requires_matching_prediction_tag(tmp_path: Path) -> None:
    renderer = load_renderer()
    task = {
        "model_loc": "/runs/exp2-discrete-s42/model-discrete-time_tokens",
        "data_version": "deciles_none_unfused_time_tokens_first_24h",
    }
    jobfile = tmp_path / "binary.jobfile"
    jobfile.write_text(
        "data_version=deciles_none_unfused_time_tokens_first_24h "
        "model_loc=/runs/exp2-discrete-s42/model-discrete-time_tokens "
        "IRB_PREDS_TAG=ml4h-exp2-binary bash run.sh\n",
        encoding="utf-8",
    )

    parsed = renderer.stage3_task_list(
        jobfile,
        {task["model_loc"]: task},
        expected_runs=1,
        expected_preds_tag="ml4h-exp2-binary",
    )
    assert parsed == [task]

    with pytest.raises(ValueError, match="Unexpected prediction tag"):
        renderer.stage3_task_list(
            jobfile,
            {task["model_loc"]: task},
            expected_runs=1,
            expected_preds_tag="wrong-tag",
        )
