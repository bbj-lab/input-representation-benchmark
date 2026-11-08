#!/usr/bin/env python3
"""Generate ML4H Stage-1 jobfiles without submitting them."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline.run_experiments import (
    EXP1_CONFIGS,
    get_data_dir,
    get_data_version,
    get_tokenizer_config,
    make_exp2_configs,
    make_exp3_configs,
)


MODEL_NAMES = {
    "llama": "meta-llama/Llama-3.2-1B",
    "qwen": "Qwen/Qwen3-0.6B",
}

FIXED_CONTRACT = {
    "optimizer": "adamw",
    "learning_rate": 1e-4,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "rope_theta": 10000.0,
    "seconds_per_position": 30.0,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "lr_scheduler_type": "linear",
    "warmup_ratio": 0.01,
    "hidden_size": 1024,
    "intermediate_size": 2048,
    "num_hidden_layers": 8,
    "num_attention_heads": 8,
    "max_position_embeddings": 4096,
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 4,
    "gradient_accumulation_steps": 1,
    "evaluations_per_epoch": 2,
    "early_stopping_patience": 3,
}


def _load_calibration(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"Calibration report not found: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("gate_errors"):
        raise SystemExit(
            "Batch-geometry calibration gate is not satisfied:\n  - "
            + "\n  - ".join(report["gate_errors"])
        )
    required = {"per_device_train_batch_size", "per_device_eval_batch_size", "max_grad_norm"}
    missing = required - set(report.get("selected", {}))
    if missing:
        raise SystemExit(f"Calibration report missing selected fields: {sorted(missing)}")
    if report.get("selected", {}).get("per_device_train_batch_size") is None:
        raise SystemExit("Calibration report did not select a train microbatch.")
    selected = report["selected"]
    expected_batch = {
        "per_device_train_batch_size": FIXED_CONTRACT["per_device_train_batch_size"],
        "per_device_eval_batch_size": FIXED_CONTRACT["per_device_eval_batch_size"],
    }
    observed_batch = {key: selected.get(key) for key in expected_batch}
    if observed_batch != expected_batch:
        raise SystemExit(
            "Calibration report does not validate the fixed ML4H batch contract: "
            f"expected {expected_batch}, found {observed_batch}."
        )
    return report


def _environment(
    *,
    backbone: str,
    calibration: dict,
    model_dir: Path,
    project: str,
) -> str:
    selected = calibration["selected"]
    model_name = MODEL_NAMES[backbone]
    return " ".join(
        (
            f"MODEL_DIR={model_dir}",
            f"IRB_MODEL_NAME={model_name}",
            "IRB_USE_MODEL_CONFIG_OVERRIDES=true",
            f"IRB_MODEL_HIDDEN_SIZE={FIXED_CONTRACT['hidden_size']}",
            f"IRB_MODEL_INTERMEDIATE_SIZE={FIXED_CONTRACT['intermediate_size']}",
            f"IRB_MODEL_NUM_HIDDEN_LAYERS={FIXED_CONTRACT['num_hidden_layers']}",
            f"IRB_MODEL_NUM_ATTENTION_HEADS={FIXED_CONTRACT['num_attention_heads']}",
            f"IRB_MODEL_MAX_POSITION_EMBEDDINGS={FIXED_CONTRACT['max_position_embeddings']}",
            f"IRB_ROPE_THETA={FIXED_CONTRACT['rope_theta']}",
            f"IRB_SECONDS_PER_POSITION={FIXED_CONTRACT['seconds_per_position']}",
            f"IRB_STAGE1_OPTIMIZER={FIXED_CONTRACT['optimizer']}",
            f"IRB_STAGE1_LR={FIXED_CONTRACT['learning_rate']}",
            f"IRB_ADAM_BETA1={FIXED_CONTRACT['adam_beta1']}",
            f"IRB_ADAM_BETA2={FIXED_CONTRACT['adam_beta2']}",
            f"IRB_STAGE1_WEIGHT_DECAY={FIXED_CONTRACT['weight_decay']}",
            f"IRB_STAGE1_MAX_GRAD_NORM={selected['max_grad_norm']}",
            f"IRB_STAGE1_LR_SCHEDULER={FIXED_CONTRACT['lr_scheduler_type']}",
            f"IRB_STAGE1_WARMUP_RATIO={FIXED_CONTRACT['warmup_ratio']}",
            f"IRB_PER_DEVICE_TRAIN_BATCH_SIZE={FIXED_CONTRACT['per_device_train_batch_size']}",
            f"IRB_PER_DEVICE_EVAL_BATCH_SIZE={FIXED_CONTRACT['per_device_eval_batch_size']}",
            f"IRB_GRADIENT_ACCUMULATION_STEPS={FIXED_CONTRACT['gradient_accumulation_steps']}",
            f"IRB_STAGE1_EVALUATIONS_PER_EPOCH={FIXED_CONTRACT['evaluations_per_epoch']}",
            f"IRB_STAGE1_EARLY_STOPPING_PATIENCE={FIXED_CONTRACT['early_stopping_patience']}",
            "IRB_EXP1_STAGE1_EPOCHS=5",
            "IRB_EXP23_STAGE1_EPOCHS=5",
            "IRB_RESUME_FROM_CHECKPOINT=auto",
            "IRB_STAGE1_SAVE_STRATEGY=epoch",
            "IRB_STAGE1_EVAL_STRATEGY=epoch",
            "IRB_STAGE1_CHECKPOINT_INTERVAL_SECONDS=600",
            "IRB_STAGE1_SAVE_TOTAL_LIMIT=4",
            "IRB_LOAD_BEST_MODEL_AT_END=true",
            f"IRB_WANDB_PROJECT={project}",
            "WANDB_MODE=offline",
            "WANDB_LOG_MODEL=false",
        )
    )


def _exp3_stage0_lines() -> list[str]:
    """Build the two full-tokenization commands required before Exp3 Stage 1."""
    lines: list[str] = []
    for config in make_exp3_configs():
        outcomes_arg = (
            " --outcomes_meds_events_dir outputs/runs/exp3/meds_icu"
            if config.data_format == "meds_clif"
            else ""
        )
        lines.append(
            "bash slurm/04_exp3_stage0_tokenize_and_outcomes.sh "
            f"--data_format {config.data_format} "
            f"--data_dir {get_data_dir(config)} "
            f"--data_version_out {get_data_version(config)} "
            f"--tokenizer_config {get_tokenizer_config(config)} "
            "--max_padded_len ${IRB_MAX_PADDED_LEN} "
            f"--quantizer {config.quantizer} "
            f"--clinical_anchoring {config.clinical_anchoring} "
            "--include_ref_ranges false "
            "--include_time_spacing_tokens true "
            "--fused_category_values false "
            "--numeric_encoding quantile"
            f"{outcomes_arg}"
        )
    return lines


def _exp3_stage1_lines(
    *,
    calibration: dict,
    model_dir: Path,
    project: str,
    seeds: tuple[int, ...] | list[int],
) -> list[str]:
    """Build 2 arms × 2 backbones × 3 seeds at the fixed Exp3 anchor."""
    lines: list[str] = []
    for backbone in MODEL_NAMES:
        environment = _environment(
            backbone=backbone,
            calibration=calibration,
            model_dir=model_dir,
            project=project,
        )
        for seed in seeds:
            for config in make_exp3_configs():
                lines.append(
                    f"{environment} IRB_MODEL_VERSION_PREFIX=ml4h_{backbone}_exp3 "
                    "bash slurm/08_exp3_stage1_train_representation.sh "
                    f"--config_id {config.config_id} "
                    f"--data_dir {get_data_dir(config)} "
                    f"--data_version {get_data_version(config)} "
                    f"--representation {config.representation} "
                    f"--temporal {config.temporal} "
                    f"--num_bins {config.num_bins} "
                    f"--seed {seed}"
                )
    return lines


def _write_jobfile(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(lines)} commands to {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calibration-report",
        type=Path,
        default=Path("outputs/runs/ml4h_2026/calibration/batch_geometry_report.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("slurm/generated/ml4h_2026"),
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("outputs/runs/ml4h_2026/models"),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=(42, 43, 44))
    parser.add_argument("--wandb-project", default="input-rep-benchmark-ml4h")
    parser.add_argument(
        "--exp3-only",
        action="store_true",
        help=(
            "Write only the standalone Exp3 Stage-0 and 12-run Stage-1 jobfiles. "
            "This leaves active Exp1/Exp2 campaign files untouched."
        ),
    )
    args = parser.parse_args()

    calibration = _load_calibration(args.calibration_report)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.exp3_only:
        stage0_lines = _exp3_stage0_lines()
        stage1_lines = _exp3_stage1_lines(
            calibration=calibration,
            model_dir=args.model_dir,
            project=args.wandb_project,
            seeds=args.seeds,
        )
        if len(stage0_lines) != 2 or len(stage1_lines) != 12:
            raise SystemExit(
                "Exp3 generation must emit 2 Stage-0 and 12 Stage-1 commands."
            )
        _write_jobfile(args.output_dir / "04_exp3_stage0.jobfile", stage0_lines)
        _write_jobfile(args.output_dir / "05_exp3_stage1.jobfile", stage1_lines)
        manifest = {
            "experiment": "exp3",
            "stage0_tasks": len(stage0_lines),
            "stage1_tasks": len(stage1_lines),
            "seeds": list(args.seeds),
            "backbones": MODEL_NAMES,
            "fixed_contract": FIXED_CONTRACT,
            "calibration_report": str(args.calibration_report),
            "selected_batch_geometry": calibration["selected"],
            "anchor": (
                "population deciles, unfused, discrete values, explicit time tokens"
            ),
        }
        (args.output_dir / "05_exp3_stage1_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return 0

    exp1_lines, exp2_lines = [], []
    exp2_configs = make_exp2_configs(
        include_discrete=True,
        discrete_only=False,
        include_soft=True,
        include_continuous=True,
        quantizer="deciles",
        clinical_anchoring="none",
        fused_discrete=False,
    )
    for backbone in MODEL_NAMES:
        environment = _environment(
            backbone=backbone,
            calibration=calibration,
            model_dir=args.model_dir,
            project=args.wandb_project,
        )
        for seed in args.seeds:
            for config in EXP1_CONFIGS:
                exp1_lines.append(
                    f"bash -lc 'export {environment} IRB_MODEL_VERSION_PREFIX=ml4h_{backbone}_exp1 "
                    f"config_id={config.config_id} data_version={get_data_version(config)} seed={seed} "
                    "IRB_TOKENIZED_DATASET_DIR=outputs/runs/tokenized/mimiciv-3.1_meds_70-10-20; "
                    "bash slurm/04_exp1_stage1_tune_packed.sh'"
                )
            for config in exp2_configs:
                exp2_lines.append(
                    f"{environment} IRB_MODEL_VERSION_PREFIX=ml4h_{backbone}_exp2 "
                    "IRB_TOKENIZED_DATASET_DIR=outputs/runs/tokenized/mimiciv-3.1_meds_70-10-20 "
                    "bash slurm/07_exp2_stage1_train_representation.sh "
                    f"--config_id {config.config_id} --data_version {get_data_version(config)} "
                    f"--representation {config.representation} --temporal {config.temporal} "
                    f"--num_bins {config.num_bins} --seed {seed}"
                )

    for name, lines in (
        ("02_exp1_stage1.jobfile", exp1_lines),
        ("03_exp2_stage1.jobfile", exp2_lines),
    ):
        _write_jobfile(args.output_dir / name, lines)

    manifest = {
        "seeds": list(args.seeds),
        "fixed_contract": FIXED_CONTRACT,
        "calibration_report": str(args.calibration_report),
        "selected_batch_geometry": calibration["selected"],
        "epochs_per_run": 5,
        "evaluations_per_epoch": FIXED_CONTRACT["evaluations_per_epoch"],
        "architecture_comparison": {
            "llama_expected_parameters": 123534336,
            "qwen_expected_parameters": 103720960,
            "max_position_embeddings": 4096,
            "note": "Architecture comparison, not capacity-matched.",
        },
        "token_exposure": {
            "contract": "five data passes per arm",
            "note": "Packed Exp1 and windowed Exp2 are not token-matched; report both update counts and non-pad tokens.",
        },
        "recovery": {
            "checkpoint_interval_seconds": 600,
            "opportunistic_checkpoint_interval_seconds": 300,
            "checkpoint_signals": ["SIGUSR1", "SIGTERM"],
            "resume_from_checkpoint": "auto",
            "save_total_limit": 4,
            "slurm_time_limit": "1-12:00:00",
            "exp1_and_exp2": True,
        },
        "exp1_runs": len(exp1_lines),
        "exp2_runs": len(exp2_lines),
        "exp3_anchor": (
            "Standalone study, not gated on Exp2. Generate raw-MIMIC versus full-CLIF at the "
            "predeclared transport anchor: population deciles, unfused, discrete values, time tokens."
        ),
    }
    (args.output_dir / "campaign_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
