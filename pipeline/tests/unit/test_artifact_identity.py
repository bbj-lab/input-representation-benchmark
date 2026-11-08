#!/usr/bin/env python3
"""Regression tests for per-run artifact identity.

Guards the failure mode where distinct training runs published as symlinks to a
shared checkpoint collapsed onto one feature filename and overwrote each other.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from fms_ehrs.framework.artifacts import (
    assert_unique_model_stems,
    build_provenance,
    model_artifact_stem,
    provenance_mismatch,
    read_provenance,
    write_provenance,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline.run_experiments import (  # noqa: E402
    EXP1_CONFIGS,
    _exp1_best_model_loc,
    _exp2_model_loc,
    _exp3_model_loc,
    make_exp2_configs,
    make_exp3_configs,
)

SEEDS = [42, 43, 44]

# One selected Exp1 regime feeds Exp2; one selected Exp2 method feeds Exp3.
EXP2_CONFIGS = make_exp2_configs(
    include_discrete=True,
    discrete_only=False,
    quantizer="deciles",
    clinical_anchoring="none",
    fused_discrete=False,
)
EXP3_CONFIGS = make_exp3_configs()


def test_runs_sharing_a_checkpoint_keep_distinct_stems() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        checkpoint = root / "checkpoint-372270"
        checkpoint.mkdir()
        first = root / "ml4h_llama_exp1_deciles-s42-hp-dv"
        second = root / "ml4h_qwen_exp1_deciles-s44-hp-dv"
        first.symlink_to(checkpoint, target_is_directory=True)
        second.symlink_to(checkpoint, target_is_directory=True)

        assert first.resolve() == second.resolve()
        assert model_artifact_stem(first) != model_artifact_stem(second)


def test_run_names_containing_dots_are_not_truncated() -> None:
    assert model_artifact_stem(Path("/runs/exp1_v1.5-s42")) == "exp1_v1.5-s42"


def test_wrapper_checkpoints_are_qualified_by_their_run_directory() -> None:
    first = Path("/runs/exp2_a-discrete-time_rope-s42/model-discrete-time_rope")
    second = Path("/runs/exp2_b-discrete-time_rope-s42/model-discrete-time_rope")
    assert model_artifact_stem(first) != model_artifact_stem(second)


def test_colliding_plan_is_rejected() -> None:
    try:
        assert_unique_model_stems(["/runs/a/model-x", "/other/a/model-x"])
    except ValueError as error:
        assert "collide" in str(error)
    else:
        raise AssertionError("expected colliding model locations to raise")


def test_every_experiment_plan_has_unique_stems() -> None:
    for configs, model_loc_fn in (
        (EXP1_CONFIGS, _exp1_best_model_loc),
        (EXP2_CONFIGS, _exp2_model_loc),
        (EXP3_CONFIGS, _exp3_model_loc),
    ):
        assert_unique_model_stems(
            model_loc_fn(config, seed) for config in configs for seed in SEEDS
        )


def test_provenance_detects_a_foreign_feature_file() -> None:
    with tempfile.TemporaryDirectory() as directory:
        feature_path = Path(directory) / "features-run_a.npy"
        feature_path.write_bytes(b"")
        recorded = build_provenance(
            model_loc=Path(directory) / "run_a",
            data_version="dv",
            split="train",
            all_layers=False,
            shape=(4, 2),
            dtype="float16",
        )
        write_provenance(feature_path, recorded)

        same = build_provenance(
            model_loc=Path(directory) / "run_a",
            data_version="dv",
            split="train",
            all_layers=False,
            shape=(4, 2),
            dtype="float16",
        )
        other = build_provenance(
            model_loc=Path(directory) / "run_b",
            data_version="dv",
            split="train",
            all_layers=False,
            shape=(4, 2),
            dtype="float16",
        )

        assert provenance_mismatch(read_provenance(feature_path), same) is None
        assert provenance_mismatch(read_provenance(feature_path), other) is not None
        assert provenance_mismatch(None, same) is not None


if __name__ == "__main__":
    failures = 0
    for name, test in sorted(globals().items()):
        if not name.startswith("test_") or not callable(test):
            continue
        try:
            test()
        except Exception as error:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {error}")
        else:
            print(f"ok   {name}")
    raise SystemExit(1 if failures else 0)
