#!/usr/bin/env python3
"""Unit tests for ML4H batch-geometry selection helpers."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _load_module():
    path = (
        Path(__file__).resolve().parents[3]
        / "utilities"
        / "scripts"
        / "calibrate_ml4h_batch_geometry.py"
    )
    spec = importlib.util.spec_from_file_location("calibrate_ml4h_batch_geometry", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_campaign_generator():
    path = (
        Path(__file__).resolve().parents[3]
        / "utilities"
        / "scripts"
        / "generate_ml4h_campaign.py"
    )
    spec = importlib.util.spec_from_file_location("generate_ml4h_campaign", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_select_microbatch_prefers_batch_four_when_efficient():
    mod = _load_module()
    probes = [
        {
            "ok": True,
            "oom": False,
            "per_device_train_batch_size": 1,
            "tokens_per_second": 100.0,
            "peak_allocated_bytes": 10,
        },
        {
            "ok": True,
            "oom": False,
            "per_device_train_batch_size": 2,
            "tokens_per_second": 180.0,
            "peak_allocated_bytes": 18,
        },
        {
            "ok": True,
            "oom": False,
            "per_device_train_batch_size": 4,
            "tokens_per_second": 200.0,
            "peak_allocated_bytes": 30,
        },
        {
            "ok": False,
            "oom": True,
            "per_device_train_batch_size": 8,
            "tokens_per_second": None,
            "peak_allocated_bytes": None,
        },
    ]
    selected = mod.select_microbatch(
        probes, gpu_total_bytes=100, memory_safety_fraction=0.9
    )
    # Batch 4 adds 11.1% throughput over batch 2, clearing the 10% threshold
    # for fewer optimizer updates per epoch.
    assert selected["selected_batch_size"] == 4


def test_build_report_shares_conservative_batch():
    mod = _load_module()
    report = mod.build_report(
        memory_ranking=[
            {
                "ok": True,
                "representation": "soft",
                "peak_allocated_bytes": 20,
            },
            {
                "ok": True,
                "representation": "xval",
                "peak_allocated_bytes": 40,
            },
        ],
        sweeps={
            "llama:discrete": [
                {
                    "ok": True,
                    "oom": False,
                    "per_device_train_batch_size": 4,
                    "tokens_per_second": 200.0,
                    "peak_allocated_bytes": 20,
                    "progress": {},
                }
            ],
            "llama:xval": [
                {
                    "ok": True,
                    "oom": False,
                    "per_device_train_batch_size": 4,
                    "tokens_per_second": 150.0,
                    "peak_allocated_bytes": 30,
                    "progress": {},
                }
            ],
        },
        clip_checks=[
            {
                "ok": True,
                "max_grad_norm": 1.0,
                "per_device_train_batch_size": 4,
                "progress": {"finite_loss": True},
            }
        ],
        window_diag={"gate_errors": []},
        batch_checks={"gate_errors": []},
        gpu_total_bytes=100,
        memory_safety_fraction=0.9,
    )
    assert report["selected"]["per_device_train_batch_size"] == 4
    assert report["selected"]["gradient_accumulation_steps"] == 1
    assert report["gate_passed"] is True
    assert report["worst_case_representation"]["representation"] == "xval"


def test_campaign_generation_rejects_non_four_calibration_batch(tmp_path: Path):
    generator = _load_campaign_generator()
    report = {
        "gate_errors": [],
        "selected": {
            "per_device_train_batch_size": 4,
            "per_device_eval_batch_size": 4,
            "max_grad_norm": 1.0,
        },
    }
    report_path = tmp_path / "batch_geometry_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    calibration = generator._load_calibration(report_path)
    assert calibration["selected"] == report["selected"]
    environment = generator._environment(
        backbone="llama",
        calibration=calibration,
        model_dir=tmp_path / "models",
        project="test",
    )
    assert "IRB_PER_DEVICE_TRAIN_BATCH_SIZE=4" in environment
    assert "IRB_PER_DEVICE_EVAL_BATCH_SIZE=4" in environment
    assert "WANDB_MODE=offline" in environment

    report["selected"]["per_device_train_batch_size"] = 2
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(SystemExit, match="fixed ML4H batch contract"):
        generator._load_calibration(report_path)


def test_exp3_only_campaign_generation_writes_two_stage0_and_twelve_stage1_tasks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    generator = _load_campaign_generator()
    report = {
        "gate_errors": [],
        "selected": {
            "per_device_train_batch_size": 4,
            "per_device_eval_batch_size": 4,
            "max_grad_norm": 1.0,
        },
    }
    report_path = tmp_path / "batch_geometry_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    output_dir = tmp_path / "generated"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_ml4h_campaign.py",
            "--exp3-only",
            "--calibration-report",
            str(report_path),
            "--output-dir",
            str(output_dir),
            "--model-dir",
            str(tmp_path / "models"),
        ],
    )

    assert generator.main() == 0
    stage0 = (output_dir / "04_exp3_stage0.jobfile").read_text().splitlines()
    stage1 = (output_dir / "05_exp3_stage1.jobfile").read_text().splitlines()

    assert len(stage0) == 2
    assert len(stage1) == 12
    assert all("--only_24h_cut" not in line for line in stage0)
    assert all("--temporal time_tokens" in line for line in stage1)
    assert all("IRB_PER_DEVICE_TRAIN_BATCH_SIZE=4" in line for line in stage1)
    assert all("IRB_STAGE1_EARLY_STOPPING_PATIENCE=3" in line for line in stage1)
    assert sum("IRB_MODEL_VERSION_PREFIX=ml4h_llama_exp3" in line for line in stage1) == 6
    assert sum("IRB_MODEL_VERSION_PREFIX=ml4h_qwen_exp3" in line for line in stage1) == 6
    assert not (output_dir / "02_exp1_stage1.jobfile").exists()
    assert not (output_dir / "03_exp2_stage1.jobfile").exists()
