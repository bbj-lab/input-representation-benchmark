from __future__ import annotations

from pathlib import Path

from pipeline.run_experiments import (
    DEFAULT_PATHS,
    ExperimentConfig,
    generate_exp1_stage0_job_file,
    generate_exp2_stage0_job_file,
    make_exp2_configs,
)


def test_stage0_job_generation_includes_data_dir_and_tokenizer_overrides(tmp_path: Path) -> None:
    original_paths = dict(DEFAULT_PATHS)
    try:
        DEFAULT_PATHS["meds_data_dir"] = "custom-meds-root"
        DEFAULT_PATHS["meds_tokenizer_config"] = "custom-tokenizer.yaml"

        exp1_job = tmp_path / "exp1.jobfile"
        exp2_job = tmp_path / "exp2.jobfile"

        generate_exp1_stage0_job_file(
            [ExperimentConfig("deciles", "none", False, "discrete", "time_tokens", "meds")],
            exp1_job,
        )
        generate_exp2_stage0_job_file(
            [ExperimentConfig("ventiles", "5-10-5", False, "xval", "time_rope", "meds")],
            exp2_job,
        )

        exp1_text = exp1_job.read_text()
        exp2_text = exp2_job.read_text()

        assert "--data_dir custom-meds-root" in exp1_text
        assert "--tokenizer_config custom-tokenizer.yaml" in exp1_text
        assert "--data_dir custom-meds-root" in exp2_text
        assert "--tokenizer_config custom-tokenizer.yaml" in exp2_text
    finally:
        DEFAULT_PATHS.clear()
        DEFAULT_PATHS.update(original_paths)


def test_make_exp2_configs_can_include_auxiliary_xval_affine() -> None:
    configs = make_exp2_configs(
        include_discrete=False,
        discrete_only=False,
        include_soft=False,
        include_continuous=False,
        include_affine_continuous=True,
        quantizer="ventiles",
        clinical_anchoring="5-10-5",
        fused_discrete=False,
    )

    assert [(cfg.representation, cfg.temporal) for cfg in configs] == [
        ("xval_affine", "time_tokens"),
        ("xval_affine", "time_rope"),
    ]
