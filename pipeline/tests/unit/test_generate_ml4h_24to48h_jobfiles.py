from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OUTCOME_GENERATOR = (
    ROOT / "utilities" / "scripts" / "generate_ml4h_24to48h_outcome_jobfiles.py"
)
STAGE3_GENERATOR = (
    ROOT / "utilities" / "scripts" / "generate_ml4h_24to48h_stage3_jobfiles.py"
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_outcome_job_uses_separate_bounded_outcome_files() -> None:
    generator = load_module("generate_24to48h_outcomes", OUTCOME_GENERATOR)

    command = generator.outcome_command(
        meds_events_dir=Path("/events"),
        tokenized_dir=Path("/tokenized"),
    )

    assert "--observation_hours 24" in command
    assert "--followup_end_hours 48" in command
    assert "--outcomes_only" in command
    assert "tokens_timelines_outcomes_24to48h.parquet" in command
    assert "tokens_timelines_extended_outcomes_24to48h.parquet" in command
    assert " && " in command


def test_stage3_commands_keep_main_artifacts_isolated() -> None:
    generator = load_module("generate_24to48h_stage3", STAGE3_GENERATOR)
    task = {
        "data_version": "deciles_none_unfused_time_tokens_first_24h",
        "model_loc": "/models/model-a",
    }

    binary = generator.command(
        experiment="exp1",
        task=task,
        shared_tokenized_root=Path("/tokenized"),
        fms_ehrs_home=Path("/fms-ehrs"),
        task_kind="binary",
    )
    regression = generator.command(
        experiment="exp3",
        task={**task, "data_dir": "/exp3"},
        shared_tokenized_root=Path("/tokenized"),
        fms_ehrs_home=Path("/fms-ehrs"),
        task_kind="regression",
    )

    assert "IRB_PREDS_TAG=ml4h-exp1-24to48h-binary" in binary
    assert "IRB_PREDS_TAG=ml4h-exp3-24to48h-regression" in regression
    assert "tokens_timelines_extended_outcomes_24to48h.parquet" in binary
    assert "tokens_timelines_extended_outcomes_24to48h.parquet" in regression
    assert "length_of_stay" not in regression
    assert len(generator.outcomes_for("exp1", "binary")) == 17
    assert len(generator.outcomes_for("exp1", "regression")) == 20
    assert len(generator.outcomes_for("exp3", "binary")) == 16
