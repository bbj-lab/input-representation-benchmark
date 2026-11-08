from __future__ import annotations

from pathlib import Path

from pipeline.run_experiments import (
    DEFAULT_PATHS,
    ExperimentConfig,
    generate_exp1_rep_pred_job_file,
    generate_exp1_stage0_job_file,
    generate_exp2_rep_pred_job_file,
    generate_exp2_stage0_job_file,
    generate_exp2_stage1_job_file,
    generate_exp3_rep_pred_job_file,
    generate_exp3_stage0_job_file,
    generate_exp3_stage1_job_file,
    get_data_version,
    make_exp2_configs,
    make_exp3_configs,
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


def test_exp23_training_jobfiles_pin_batch_size_four(tmp_path: Path) -> None:
    exp2_config = ExperimentConfig(
        "deciles", "none", False, "discrete", "time_tokens", "meds"
    )
    exp3_config = ExperimentConfig(
        "deciles", "none", False, "discrete", "time_tokens", "meds_icu"
    )
    exp2_job = tmp_path / "exp2-stage1.jobfile"
    exp3_job = tmp_path / "exp3-stage1.jobfile"

    generate_exp2_stage1_job_file([exp2_config], [42], exp2_job, mode="demo")
    generate_exp3_stage1_job_file([exp3_config], [42], exp3_job, mode="demo")

    for jobfile in (exp2_job, exp3_job):
        text = jobfile.read_text(encoding="utf-8")
        assert "IRB_PER_DEVICE_TRAIN_BATCH_SIZE=4" in text
        assert "IRB_PER_DEVICE_EVAL_BATCH_SIZE=4" in text
        assert "IRB_GRADIENT_ACCUMULATION_STEPS=1" in text


def test_exp2_crosses_every_encoder_with_all_three_temporal_settings() -> None:
    configs = make_exp2_configs(
        include_discrete=True,
        discrete_only=False,
        include_soft=True,
        include_continuous=True,
        quantizer="deciles",
        clinical_anchoring="none",
        fused_discrete=False,
    )

    assert len(configs) == 12
    assert not any(cfg.fused for cfg in configs)
    for representation in ("discrete", "soft", "xval", "xval_affine"):
        temporal = {
            cfg.temporal for cfg in configs if cfg.representation == representation
        }
        assert temporal == {"event_order", "time_tokens", "time_rope"}


def test_exp2_can_omit_affine_without_dropping_the_other_encoders() -> None:
    configs = make_exp2_configs(
        include_discrete=True,
        discrete_only=False,
        include_soft=True,
        include_continuous=True,
        include_affine_continuous=False,
        quantizer="deciles",
        clinical_anchoring="none",
        fused_discrete=False,
    )
    assert len(configs) == 9
    assert {cfg.representation for cfg in configs} == {"discrete", "soft", "xval"}


def test_xval_affine_requires_numeric_values() -> None:
    affine = ExperimentConfig(
        "deciles", "none", False, "xval_affine", "time_tokens", "meds"
    )
    xval = ExperimentConfig(
        "deciles", "none", False, "xval", "time_tokens", "meds"
    )
    discrete = ExperimentConfig(
        "deciles", "none", False, "discrete", "time_tokens", "meds"
    )
    assert affine.needs_numeric_values is True
    assert xval.needs_numeric_values is True
    assert discrete.needs_numeric_values is False


def test_event_order_reuses_the_time_rope_tokenization() -> None:
    shared = {
        "quantizer": "deciles",
        "clinical_anchoring": "none",
        "fused": False,
        "representation": "discrete",
        "data_format": "meds",
    }
    event_order = ExperimentConfig(temporal="event_order", **shared)
    time_rope = ExperimentConfig(temporal="time_rope", **shared)
    time_tokens = ExperimentConfig(temporal="time_tokens", **shared)

    # Neither inserts spacing tokens, so they share one tokenized dataset while
    # remaining distinct configurations.
    assert get_data_version(event_order) == get_data_version(time_rope)
    assert get_data_version(event_order) != get_data_version(time_tokens)
    assert event_order.config_id != time_rope.config_id


def test_rep_pred_jobfiles_enforce_fixed_probe_hyperparameters(tmp_path: Path) -> None:
    exp1_config = ExperimentConfig(
        "deciles", "none", False, "discrete", "time_tokens", "meds"
    )
    exp2_config = ExperimentConfig(
        "ventiles", "none", False, "xval", "time_tokens", "meds"
    )
    exp3_config = ExperimentConfig(
        "deciles", "none", False, "discrete", "time_tokens", "meds_icu"
    )
    jobfiles = (
        tmp_path / "exp1-preds.jobfile",
        tmp_path / "exp2-preds.jobfile",
        tmp_path / "exp3-preds.jobfile",
    )

    generate_exp1_rep_pred_job_file(
        [exp1_config], [42], jobfiles[0], mode="demo"
    )
    generate_exp2_rep_pred_job_file(
        [exp2_config], [42], jobfiles[1], mode="demo"
    )
    generate_exp3_rep_pred_job_file(
        [exp3_config], [42], jobfiles[2], mode="demo"
    )

    for jobfile in jobfiles:
        text = jobfile.read_text()
        assert "--no-tune_logreg_C" in text
        assert "--logreg_C_grid [0.01]" in text
        assert "--ridge_alpha_grid [100]" in text


def test_exp3_uses_fixed_anchor_with_matched_raw_and_full_clif_arms(
    tmp_path: Path,
) -> None:
    original_paths = dict(DEFAULT_PATHS)
    try:
        DEFAULT_PATHS["exp3_meds_icu_data_dir"] = "raw-arm"
        DEFAULT_PATHS["exp3_meds_clif_data_dir"] = "clif-arm"
        configs = make_exp3_configs()
        assert [config.data_format for config in configs] == ["meds_icu", "meds_clif"]
        assert {
            (
                config.quantizer,
                config.clinical_anchoring,
                config.fused,
                config.representation,
                config.temporal,
            )
            for config in configs
        } == {("deciles", "none", False, "discrete", "time_tokens")}

        jobfile = tmp_path / "exp3.jobfile"
        generate_exp3_stage0_job_file(configs, jobfile)
        text = jobfile.read_text()
        assert "--data_format meds_clif" in text
        assert "--outcomes_meds_events_dir raw-arm" in text
    finally:
        DEFAULT_PATHS.clear()
        DEFAULT_PATHS.update(original_paths)
