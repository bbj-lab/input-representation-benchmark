#!/usr/bin/env python3
"""Calibrate Stage-1 microbatch geometry with gradient_accumulation_steps=1.

Phases:
  1. Peak-memory ranking of soft / xval at batch 1.
  2. Power-of-two microbatch sweeps for discrete RoPE and the measured worst case.
  3. Short clip-threshold check at the selected geometry.
  4. Gate report consumed by generate_ml4h_campaign.py.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


BACKBONES = {
    "llama": "meta-llama/Llama-3.2-1B",
    "qwen": "Qwen/Qwen3-0.6B",
}

MEMORY_PROBE_REPS = ("soft", "xval")
FIXED = {
    "optimizer": "adamw",
    "learning_rate": 1e-4,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "rope_theta": 10000.0,
    "seconds_per_position": 30.0,
    "weight_decay": 0.01,
    "warmup_ratio": 0.01,
    "hidden_size": 1024,
    "intermediate_size": 2048,
    "num_hidden_layers": 8,
    "num_attention_heads": 8,
    "max_position_embeddings": 4096,
    "gradient_accumulation_steps": 1,
}


def _data_version(representation: str) -> str:
    if representation.startswith("xval"):
        return "deciles_none_unfused_time_rope_numencxval"
    return "deciles_none_unfused_time_rope"


def _oom_like(text: str) -> bool:
    patterns = (
        r"out of memory",
        r"CUDA out of memory",
        r"cudaErrorMemoryAllocation",
        r"CUBLAS_STATUS_ALLOC_FAILED",
        r"torch.OutOfMemoryError",
    )
    lower = text.lower()
    return any(re.search(p, lower) for p in patterns)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _summarize_progress(progress_path: Path) -> dict[str, Any]:
    losses: list[float] = []
    grad_norms: list[float] = []
    tokens_seen = None
    steps = 0
    train_elapsed = None
    if not progress_path.is_file():
        return {
            "steps": 0,
            "finite_loss": True,
            "mean_loss": None,
            "mean_grad_norm": None,
            "max_grad_norm_raw": None,
            "tokens_seen": None,
            "train_elapsed_seconds": None,
        }
    for line in progress_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        metrics = event.get("latest_train_metrics") or event.get("metrics") or {}
        if "loss" in metrics:
            try:
                losses.append(float(metrics["loss"]))
            except (TypeError, ValueError):
                pass
        if "grad_norm" in metrics:
            try:
                grad_norms.append(float(metrics["grad_norm"]))
            except (TypeError, ValueError):
                pass
        if event.get("global_step") is not None:
            steps = max(steps, int(event["global_step"]))
        if event.get("num_input_tokens_seen") is not None:
            tokens_seen = int(event["num_input_tokens_seen"])
        # Prefer the Trainer-local elapsed clock so cache materialization does
        # not dilute tokens/second for the first probe of a new mapped cache.
        if event.get("event") in ("train_end", "heartbeat", "metrics") and event.get(
            "elapsed_seconds"
        ) is not None:
            try:
                train_elapsed = float(event["elapsed_seconds"])
            except (TypeError, ValueError):
                pass
        if "num_input_tokens_seen" in metrics:
            try:
                tokens_seen = int(metrics["num_input_tokens_seen"])
            except (TypeError, ValueError):
                pass
    finite = all(math.isfinite(v) for v in losses) and all(
        math.isfinite(v) for v in grad_norms
    )
    return {
        "steps": steps,
        "finite_loss": finite and bool(losses),
        "mean_loss": (sum(losses) / len(losses)) if losses else None,
        "mean_grad_norm": (sum(grad_norms) / len(grad_norms)) if grad_norms else None,
        "max_grad_norm_raw": max(grad_norms) if grad_norms else None,
        "tokens_seen": tokens_seen,
        "train_elapsed_seconds": train_elapsed,
        "n_loss_events": len(losses),
        "n_grad_events": len(grad_norms),
    }


def _gpu_total_bytes() -> int | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        props = torch.cuda.get_device_properties(0)
        return int(props.total_memory)
    except Exception:
        return None


def run_probe(
    *,
    repo_root: Path,
    fms_home: Path,
    tokenized_dir: Path,
    model_root: Path,
    backbone: str,
    representation: str,
    batch_size: int,
    max_steps: int,
    max_grad_norm: float,
    seed: int,
) -> dict[str, Any]:
    model_name = BACKBONES[backbone]
    tag = (
        f"calib_{backbone}_{representation}_bs{batch_size}"
        f"_clip{str(max_grad_norm).replace('.', 'p')}_s{seed}"
    )
    out_dir = model_root / tag
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "WANDB_MODE": "offline",
            "WANDB_LOG_MODEL": "false",
            "IRB_REQUIRE_WANDB": "false",
            "IRB_USE_BF16": env.get("IRB_USE_BF16", "true"),
            "IRB_ATTN_IMPL": env.get("IRB_ATTN_IMPL", "sdpa"),
            "IRB_USE_MAPPED_DATASET_CACHE": env.get(
                "IRB_USE_MAPPED_DATASET_CACHE", "true"
            ),
            "IRB_PER_DEVICE_TRAIN_BATCH_SIZE": str(batch_size),
            "IRB_PER_DEVICE_EVAL_BATCH_SIZE": str(batch_size),
            "IRB_GRADIENT_ACCUMULATION_STEPS": "1",
            "IRB_STAGE1_MAX_STEPS": str(max_steps),
            "IRB_STAGE1_LOGGING_STEPS": "1",
            "IRB_STAGE1_SAVE_STRATEGY": "no",
            "IRB_STAGE1_EVAL_STRATEGY": "no",
            "IRB_LOAD_BEST_MODEL_AT_END": "false",
            "IRB_STAGE1_CHECKPOINT_INTERVAL_SECONDS": "3600",
            "IRB_PROGRESS_INTERVAL_SECONDS": "5",
            "IRB_STAGE1_OPTIMIZER": "adamw",
            "IRB_STAGE1_LR": str(FIXED["learning_rate"]),
            "IRB_ADAM_BETA1": str(FIXED["adam_beta1"]),
            "IRB_ADAM_BETA2": str(FIXED["adam_beta2"]),
            "IRB_STAGE1_WEIGHT_DECAY": str(FIXED["weight_decay"]),
            "IRB_STAGE1_MAX_GRAD_NORM": str(max_grad_norm),
            "IRB_STAGE1_WARMUP_RATIO": str(FIXED["warmup_ratio"]),
            "IRB_ROPE_THETA": str(FIXED["rope_theta"]),
            "IRB_SECONDS_PER_POSITION": str(FIXED["seconds_per_position"]),
            "PYTHONUNBUFFERED": "1",
        }
    )

    cmd = [
        sys.executable,
        str(fms_home / "fms_ehrs" / "scripts" / "train_representation.py"),
        "--data_dir",
        str(tokenized_dir),
        "--data_version",
        _data_version(representation),
        "--model_dir",
        str(model_root),
        "--model_version",
        tag,
        "--model_name",
        model_name,
        "--representation",
        representation,
        "--temporal",
        "time_rope",
        "--num_bins",
        "10",
        "--n_epochs",
        "1",
        "--windowed_padded",
        "true",
        "--window_stride",
        "4096",
        "--max_windows_per_admission",
        "128",
        "--add_cont_token",
        "true",
        "--max_seq_length",
        "4096",
        "--learning_rate",
        str(FIXED["learning_rate"]),
        "--adam_beta1",
        str(FIXED["adam_beta1"]),
        "--adam_beta2",
        str(FIXED["adam_beta2"]),
        "--jid",
        "calib",
        "--seed",
        str(seed),
        "--wandb_project",
        "input-rep-benchmark-ml4h-calib",
        "--hidden_size",
        str(FIXED["hidden_size"]),
        "--intermediate_size",
        str(FIXED["intermediate_size"]),
        "--num_hidden_layers",
        str(FIXED["num_hidden_layers"]),
        "--num_attention_heads",
        str(FIXED["num_attention_heads"]),
        "--max_position_embeddings",
        str(FIXED["max_position_embeddings"]),
    ]

    started = time.monotonic()
    print(
        f"[calib] starting probe backbone={backbone} rep={representation} "
        f"bs={batch_size} clip={max_grad_norm} steps={max_steps}",
        flush=True,
    )
    log_path = out_dir / "probe_subprocess.log"
    # Stream child output live so mapped-cache builds are visible in Slurm logs.
    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write(
            f"cmd={' '.join(cmd)}\nstarted_at={time.strftime('%Y-%m-%dT%H:%M:%S')}\n\n"
        )
        log_handle.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(repo_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        combined_chunks: list[str] = []
        for line in proc.stdout:
            combined_chunks.append(line)
            log_handle.write(line)
            log_handle.flush()
            # Keep Slurm stdout readable without drowning it in HF progress bars.
            if (
                line.startswith("[")
                or "ERROR" in line
                or "Traceback" in line
                or "out of memory" in line.lower()
                or "Building mapped" in line
                or "Loading mapped" in line
                or "Starting training" in line
                or "IRB_PROGRESS" in line
            ):
                print(line, end="", flush=True)
        returncode = proc.wait()
    elapsed = time.monotonic() - started
    combined = "".join(combined_chunks)
    with log_path.open("a", encoding="utf-8") as log_handle:
        log_handle.write(f"\nexit={returncode}\nelapsed_seconds={elapsed:.1f}\n")

    run_dirs = sorted(model_root.glob(f"{tag}-*"))
    run_dir = run_dirs[-1] if run_dirs else out_dir
    run_record = _read_json(run_dir / "run_record.json") or {}
    training_rec = run_record.get("training") or {}
    model_rec = run_record.get("model") or {}
    progress = _summarize_progress(run_dir / "progress.jsonl")
    # Prefer Trainer log_history via run_record when progress callbacks lacked losses.
    if progress.get("mean_loss") is None and run_dir.joinpath("loss_perplexity_curve.json").is_file():
        curve = _read_json(run_dir / "loss_perplexity_curve.json") or {}
        train_losses = [
            float(p["loss"])
            for p in curve.get("train", [])
            if p.get("loss") is not None and math.isfinite(float(p["loss"]))
        ]
        if train_losses:
            progress["mean_loss"] = sum(train_losses) / len(train_losses)
            progress["finite_loss"] = True
            progress["n_loss_events"] = len(train_losses)
    oom = returncode != 0 and _oom_like(combined)
    peak_alloc = training_rec.get("gpu_max_memory_allocated_bytes")
    peak_reserved = training_rec.get("gpu_max_memory_reserved_bytes")
    tokens = progress.get("tokens_seen") or training_rec.get("max_input_tokens_seen")
    steps = max(
        int(progress.get("steps") or 0),
        int(training_rec.get("global_step") or 0),
    )
    updates = max(1, steps)
    train_elapsed = progress.get("train_elapsed_seconds")
    rate_elapsed = float(train_elapsed) if train_elapsed and train_elapsed > 0 else elapsed
    finite_ok = bool(progress.get("finite_loss")) or (
        returncode == 0 and training_rec.get("global_step")
    )
    result = {
        "backbone": backbone,
        "representation": representation,
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": 1,
        "max_grad_norm": max_grad_norm,
        "max_steps_requested": max_steps,
        "exit_code": returncode,
        "oom": oom,
        "ok": returncode == 0 and bool(finite_ok),
        "elapsed_seconds": elapsed,
        "train_elapsed_seconds": train_elapsed,
        "updates_per_second": updates / rate_elapsed if rate_elapsed > 0 else None,
        "tokens_per_second": (
            float(tokens) / rate_elapsed if tokens and rate_elapsed > 0 else None
        ),
        "peak_allocated_bytes": peak_alloc,
        "peak_reserved_bytes": peak_reserved,
        "parameter_count": model_rec.get("parameter_count"),
        "clip_fraction": training_rec.get("gradient_norm_clipped_fraction"),
        "run_dir": str(run_dir),
        "progress": progress,
    }
    print(
        f"[calib] finished probe ok={result['ok']} oom={result['oom']} "
        f"peak_alloc={result['peak_allocated_bytes']} "
        f"tok/s={result['tokens_per_second']} elapsed={elapsed:.1f}s",
        flush=True,
    )
    return result


def refresh_probe_throughput(probe: dict[str, Any]) -> dict[str, Any]:
    """Recompute tokens/s using Trainer train-time, not cache-build wall time."""
    out = dict(probe)
    run_dir = Path(str(probe.get("run_dir") or ""))
    progress = _summarize_progress(run_dir / "progress.jsonl") if run_dir.is_dir() else {}
    run_record = _read_json(run_dir / "run_record.json") if run_dir.is_dir() else None
    training_rec = (run_record or {}).get("training") or {}
    tokens = (
        progress.get("tokens_seen")
        or training_rec.get("max_input_tokens_seen")
        or out.get("progress", {}).get("tokens_seen")
    )
    steps = max(
        int(progress.get("steps") or 0),
        int(training_rec.get("global_step") or 0),
        int((out.get("progress") or {}).get("steps") or 0),
        1,
    )
    train_elapsed = progress.get("train_elapsed_seconds")
    wall = out.get("elapsed_seconds")
    rate_elapsed = (
        float(train_elapsed)
        if train_elapsed and float(train_elapsed) > 0
        else (float(wall) if wall else None)
    )
    if rate_elapsed and rate_elapsed > 0:
        out["train_elapsed_seconds"] = train_elapsed
        out["updates_per_second"] = steps / rate_elapsed
        out["tokens_per_second"] = (
            float(tokens) / rate_elapsed if tokens is not None else out.get("tokens_per_second")
        )
    if progress:
        merged = dict(out.get("progress") or {})
        merged.update({k: v for k, v in progress.items() if v is not None})
        out["progress"] = merged
    if training_rec.get("gpu_max_memory_allocated_bytes") is not None:
        out["peak_allocated_bytes"] = training_rec["gpu_max_memory_allocated_bytes"]
    if training_rec.get("gpu_max_memory_reserved_bytes") is not None:
        out["peak_reserved_bytes"] = training_rec["gpu_max_memory_reserved_bytes"]
    if training_rec.get("gradient_norm_clipped_fraction") is not None:
        out["clip_fraction"] = training_rec["gradient_norm_clipped_fraction"]
    return out


def load_probe_artifacts(output_dir: Path) -> tuple[list[dict], dict[str, list[dict]], list[dict]]:
    memory_ranking = []
    for representation in MEMORY_PROBE_REPS:
        path = output_dir / f"memory_rank_{representation}.json"
        payload = _read_json(path)
        if payload:
            memory_ranking.append(refresh_probe_throughput(payload))
    sweeps: dict[str, list[dict]] = {}
    for path in sorted(output_dir.glob("sweep_*.json")):
        payload = _read_json(path)
        if not payload:
            continue
        scope = f"{payload['backbone']}:{payload['representation']}"
        sweeps.setdefault(scope, []).append(refresh_probe_throughput(payload))
    for scope, probes in sweeps.items():
        sweeps[scope] = sorted(
            probes, key=lambda p: int(p.get("per_device_train_batch_size") or 0)
        )
    clip_checks = []
    for path in sorted(output_dir.glob("clip_check_*.json")):
        payload = _read_json(path)
        if payload:
            clip_checks.append(refresh_probe_throughput(payload))
    return memory_ranking, sweeps, clip_checks


def select_microbatch(
    probes: list[dict[str, Any]],
    *,
    gpu_total_bytes: int | None,
    memory_safety_fraction: float,
    min_relative_gain: float = 0.10,
) -> dict[str, Any]:
    viable = [p for p in probes if p.get("ok") and not p.get("oom")]
    if not viable:
        return {"selected_batch_size": None, "reason": "no_viable_probe"}

    mem_cap = None
    if gpu_total_bytes is not None:
        mem_cap = int(gpu_total_bytes * memory_safety_fraction)

    def fits(p: dict[str, Any]) -> bool:
        peak = p.get("peak_allocated_bytes")
        if mem_cap is None or peak is None:
            return True
        return int(peak) <= mem_cap

    candidates = [p for p in viable if fits(p)]
    if not candidates:
        candidates = viable

    # Stop at the throughput knee. Every doubling of the microbatch halves the
    # optimizer updates per epoch, so we only pay that price when the larger
    # batch buys at least `min_relative_gain` more tokens/second.
    ordered = sorted(
        candidates, key=lambda p: int(p["per_device_train_batch_size"])
    )
    best_tps = max((p.get("tokens_per_second") or 0.0) for p in candidates)
    chosen = ordered[0]
    gains: list[dict[str, Any]] = []
    for previous, current in zip(ordered, ordered[1:]):
        prev_tps = previous.get("tokens_per_second") or 0.0
        cur_tps = current.get("tokens_per_second") or 0.0
        gain = (cur_tps / prev_tps - 1.0) if prev_tps > 0 else 0.0
        gains.append(
            {
                "from_batch_size": int(previous["per_device_train_batch_size"]),
                "to_batch_size": int(current["per_device_train_batch_size"]),
                "relative_throughput_gain": gain,
            }
        )
        if gain < min_relative_gain:
            break
        chosen = current
    return {
        "selected_batch_size": int(chosen["per_device_train_batch_size"]),
        "reason": "throughput_knee_preserving_update_count",
        "min_relative_gain": min_relative_gain,
        "throughput_gains": gains,
        "best_tokens_per_second": best_tps,
        "selected_tokens_per_second": chosen.get("tokens_per_second"),
        "memory_cap_bytes": mem_cap,
        "chosen_probe": chosen,
    }


def project_campaign_walltime(
    *,
    sweeps: dict[str, list[dict[str, Any]]],
    shared_batch: int | None,
    window_diag: dict[str, Any] | None,
    n_epochs: int = 5,
) -> dict[str, Any] | None:
    """Project per-run hours from measured throughput and train-window counts."""
    if shared_batch is None or not window_diag:
        return None
    stats = window_diag.get("raw_admission_length_stats") or {}
    admissions = stats.get("admissions_total")
    windows_mean = (stats.get("required_windows") or {}).get("mean")
    max_seq = stats.get("max_seq_length") or FIXED["max_position_embeddings"]
    if not admissions or not windows_mean:
        return None
    windows_per_epoch = float(admissions) * float(windows_mean)
    padded_tokens_per_epoch = windows_per_epoch * float(max_seq)
    # Raw (non-pad) exposure, so Exp1 packed and Exp2 windowed arms can be
    # compared on real tokens rather than padded positions.
    token_mean = (stats.get("token_length") or {}).get("mean")
    raw_tokens_per_epoch = (
        float(admissions) * float(token_mean) if token_mean else None
    )

    per_scope: dict[str, Any] = {}
    for scope, probes in sweeps.items():
        match = [
            p
            for p in probes
            if p.get("ok")
            and int(p.get("per_device_train_batch_size") or 0) == int(shared_batch)
        ]
        tps = match[0].get("tokens_per_second") if match else None
        if not tps:
            continue
        per_scope[scope] = {
            "tokens_per_second": float(tps),
            "updates_per_epoch": windows_per_epoch / int(shared_batch),
            "projected_hours_per_run": padded_tokens_per_epoch
            * n_epochs
            / float(tps)
            / 3600.0,
        }
    slowest = (
        max(per_scope.items(), key=lambda kv: kv[1]["projected_hours_per_run"])
        if per_scope
        else None
    )
    return {
        "n_epochs": n_epochs,
        "note": (
            "Window counts come from the sampled window-exposure diagnostic, so "
            "these are estimates for the full MIMIC train split; padded tokens "
            "include pad positions that carry compute but no loss."
        ),
        "train_windows_per_epoch_estimate": windows_per_epoch,
        "padded_tokens_per_epoch_estimate": padded_tokens_per_epoch,
        "raw_tokens_per_epoch_estimate": raw_tokens_per_epoch,
        "nonpad_fraction_estimate": (
            raw_tokens_per_epoch / padded_tokens_per_epoch
            if raw_tokens_per_epoch and padded_tokens_per_epoch
            else None
        ),
        "per_scope": per_scope,
        "slowest_scope": slowest[0] if slowest else None,
        "projected_hours_per_run_worst": (
            slowest[1]["projected_hours_per_run"] if slowest else None
        ),
    }


def build_report(
    *,
    memory_ranking: list[dict[str, Any]],
    sweeps: dict[str, list[dict[str, Any]]],
    clip_checks: list[dict[str, Any]],
    window_diag: dict[str, Any] | None,
    batch_checks: dict[str, Any] | None,
    gpu_total_bytes: int | None,
    memory_safety_fraction: float,
) -> dict[str, Any]:
    gate_errors: list[str] = []
    worst = None
    if memory_ranking:
        ok_rank = [p for p in memory_ranking if p.get("ok")]
        if not ok_rank:
            gate_errors.append("Peak-memory ranking produced no successful batch-1 probes.")
        else:
            worst = max(
                ok_rank,
                key=lambda p: int(p.get("peak_allocated_bytes") or 0),
            )

    selected_by_scope: dict[str, Any] = {}
    for scope, probes in sweeps.items():
        selected_by_scope[scope] = select_microbatch(
            probes,
            gpu_total_bytes=gpu_total_bytes,
            memory_safety_fraction=memory_safety_fraction,
        )
        if selected_by_scope[scope]["selected_batch_size"] is None:
            gate_errors.append(f"No viable microbatch for sweep scope {scope}.")

    # Shared conservative batch across discrete and worst-case scopes.
    shared_sizes = [
        v["selected_batch_size"]
        for v in selected_by_scope.values()
        if v.get("selected_batch_size") is not None
    ]
    shared_batch = min(shared_sizes) if shared_sizes else None
    if shared_batch is None:
        gate_errors.append("Could not determine a shared campaign microbatch.")

    clip_ok = [c for c in clip_checks if c.get("ok")]
    retained_clip = 1.0
    if not clip_ok:
        gate_errors.append("Clip-threshold check did not produce a finite run.")
    elif shared_batch is not None:
        # Gradient norms depend on batch size, so a clip decision measured at a
        # different microbatch than the campaign will use is not evidence.
        mismatched = sorted(
            {
                int(c.get("per_device_train_batch_size") or 0)
                for c in clip_ok
                if int(c.get("per_device_train_batch_size") or 0) != int(shared_batch)
            }
        )
        if mismatched:
            gate_errors.append(
                "Clip-threshold checks ran at microbatch "
                f"{mismatched} but the campaign selects {shared_batch}; "
                "rerun with --clip-only at the selected microbatch."
            )
    else:
        # Prefer the stricter of the finite checks when both work; keep 1.0 if
        # the looser threshold does not improve update health.
        retained_clip = 1.0
        loose = [c for c in clip_ok if float(c["max_grad_norm"]) > 1.0]
        tight = [c for c in clip_ok if float(c["max_grad_norm"]) == 1.0]
        if tight:
            retained_clip = 1.0
        elif loose:
            retained_clip = float(loose[0]["max_grad_norm"])

    if window_diag is not None and window_diag.get("gate_errors"):
        gate_errors.extend(window_diag["gate_errors"])
    if batch_checks is not None and batch_checks.get("gate_errors"):
        gate_errors.extend(batch_checks["gate_errors"])

    report = {
        "fixed_contract": FIXED,
        "gpu_total_bytes": gpu_total_bytes,
        "memory_safety_fraction": memory_safety_fraction,
        "memory_ranking": memory_ranking,
        "worst_case_representation": (
            None
            if worst is None
            else {
                "representation": worst["representation"],
                "peak_allocated_bytes": worst.get("peak_allocated_bytes"),
            }
        ),
        "sweeps": sweeps,
        "selected_by_scope": selected_by_scope,
        "clip_checks": clip_checks,
        "campaign_projection": project_campaign_walltime(
            sweeps=sweeps,
            shared_batch=shared_batch,
            window_diag=window_diag,
        ),
        "window_diagnostics": window_diag,
        "representation_batch_checks": batch_checks,
        "selected": {
            "per_device_train_batch_size": shared_batch,
            "per_device_eval_batch_size": shared_batch,
            "gradient_accumulation_steps": 1,
            "max_grad_norm": retained_clip,
            "evaluations_per_epoch": 2,
            "max_position_embeddings": 4096,
        },
        "gate_errors": gate_errors,
        "gate_passed": not gate_errors,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument(
        "--fms-home",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--tokenized-dir",
        type=Path,
        default=Path("outputs/runs/tokenized/mimiciv-3.1_meds_70-10-20"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/runs/ml4h_2026/calibration"),
    )
    parser.add_argument("--backbones", nargs="+", default=("llama", "qwen"))
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=(1, 2, 4, 8))
    parser.add_argument("--probe-steps", type=int, default=20)
    parser.add_argument("--clip-probe-steps", type=int, default=40)
    parser.add_argument("--memory-safety-fraction", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-gpu-probes",
        action="store_true",
        help="Only assemble a report skeleton / selection helpers (for unit tests).",
    )
    parser.add_argument(
        "--rebuild-report-only",
        action="store_true",
        help="Rebuild the gate report from existing probe JSON files without new GPU runs.",
    )
    parser.add_argument(
        "--clip-only",
        action="store_true",
        help=(
            "Reuse existing memory/sweep probes and rerun only the clipping "
            "comparison, e.g. for a longer loss-curve comparison."
        ),
    )
    parser.add_argument("--clip-values", nargs="+", type=float, default=None)
    parser.add_argument("--clip-batch-size", type=int, default=None)
    parser.add_argument("--clip-backbone", default="llama")
    parser.add_argument("--clip-representation", default=None)
    parser.add_argument(
        "--window-diagnostics",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--batch-checks",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    fms_home = (args.fms_home or (repo_root.parent / "fms-ehrs")).resolve()
    tokenized_dir = (
        args.tokenized_dir
        if args.tokenized_dir.is_absolute()
        else (repo_root / args.tokenized_dir)
    ).resolve()
    output_dir = (
        args.output_dir
        if args.output_dir.is_absolute()
        else (repo_root / args.output_dir)
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_root = output_dir / "probe_models"
    model_root.mkdir(parents=True, exist_ok=True)

    window_diag = (
        _read_json(args.window_diagnostics) if args.window_diagnostics else None
    )
    batch_checks = _read_json(args.batch_checks) if args.batch_checks else None
    gpu_total = _gpu_total_bytes()
    if gpu_total is None:
        # Report rebuilds usually run on a login node with no GPU visible; keep
        # the capacity measured by the probe job so the memory cap stays real.
        previous = _read_json(output_dir / "batch_geometry_report.json") or {}
        gpu_total = previous.get("gpu_total_bytes")

    memory_ranking: list[dict[str, Any]] = []
    sweeps: dict[str, list[dict[str, Any]]] = {}
    clip_checks: list[dict[str, Any]] = []

    if args.rebuild_report_only or args.clip_only:
        memory_ranking, sweeps, clip_checks = load_probe_artifacts(output_dir)
        if not window_diag:
            window_diag = _read_json(output_dir / "window_exposure.json")
        if not batch_checks:
            batch_checks = _read_json(output_dir / "representation_batch_checks.json")

    if args.clip_only:
        if not sweeps:
            raise SystemExit(
                "--clip-only needs existing sweep_*.json probes in the output directory."
            )
        sizes = [
            v["selected_batch_size"]
            for v in (
                select_microbatch(
                    probes,
                    gpu_total_bytes=gpu_total,
                    memory_safety_fraction=args.memory_safety_fraction,
                )
                for probes in sweeps.values()
            )
            if v.get("selected_batch_size") is not None
        ]
        clip_bs = int(args.clip_batch_size or (min(sizes) if sizes else 1))
        clip_rep = args.clip_representation or (
            max(
                (p for p in memory_ranking if p.get("ok")),
                key=lambda p: int(p.get("peak_allocated_bytes") or 0),
            )["representation"]
            if any(p.get("ok") for p in memory_ranking)
            else "soft"
        )
        for stale in output_dir.glob("clip_check_*.json"):
            stale.unlink()
        clip_checks = []
        for clip in args.clip_values or (1.0, 32.0):
            probe = run_probe(
                repo_root=repo_root,
                fms_home=fms_home,
                tokenized_dir=tokenized_dir,
                model_root=model_root,
                backbone=args.clip_backbone,
                representation=clip_rep,
                batch_size=clip_bs,
                max_steps=args.clip_probe_steps,
                max_grad_norm=float(clip),
                seed=args.seed,
            )
            clip_checks.append(probe)
            (output_dir / f"clip_check_{clip}.json").write_text(
                json.dumps(probe, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    elif not args.rebuild_report_only and not args.skip_gpu_probes:
        # Use the larger backbone (Llama) for representation memory ranking.
        rank_backbone = "llama" if "llama" in args.backbones else args.backbones[0]
        for representation in MEMORY_PROBE_REPS:
            probe = run_probe(
                repo_root=repo_root,
                fms_home=fms_home,
                tokenized_dir=tokenized_dir,
                model_root=model_root,
                backbone=rank_backbone,
                representation=representation,
                batch_size=1,
                max_steps=args.probe_steps,
                max_grad_norm=1.0,
                seed=args.seed,
            )
            memory_ranking.append(probe)
            (output_dir / f"memory_rank_{representation}.json").write_text(
                json.dumps(probe, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

        ok_rank = [p for p in memory_ranking if p.get("ok")]
        worst_rep = (
            max(ok_rank, key=lambda p: int(p.get("peak_allocated_bytes") or 0))[
                "representation"
            ]
            if ok_rank
            else "xval"
        )

        for backbone in args.backbones:
            for representation in ("discrete", worst_rep):
                scope = f"{backbone}:{representation}"
                probes: list[dict[str, Any]] = []
                for batch_size in args.batch_sizes:
                    probe = run_probe(
                        repo_root=repo_root,
                        fms_home=fms_home,
                        tokenized_dir=tokenized_dir,
                        model_root=model_root,
                        backbone=backbone,
                        representation=representation,
                        batch_size=batch_size,
                        max_steps=args.probe_steps,
                        max_grad_norm=1.0,
                        seed=args.seed,
                    )
                    probes.append(probe)
                    (output_dir / f"sweep_{backbone}_{representation}_bs{batch_size}.json").write_text(
                        json.dumps(probe, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    if probe.get("oom"):
                        break
                sweeps[scope] = probes

        # Provisional selection for clip checks.
        provisional = build_report(
            memory_ranking=memory_ranking,
            sweeps=sweeps,
            clip_checks=[],
            window_diag=window_diag,
            batch_checks=batch_checks,
            gpu_total_bytes=gpu_total,
            memory_safety_fraction=args.memory_safety_fraction,
        )
        selected_bs = provisional["selected"]["per_device_train_batch_size"] or 1
        clip_backbone = "llama" if "llama" in args.backbones else args.backbones[0]
        clip_rep = (
            provisional["worst_case_representation"]["representation"]
            if provisional.get("worst_case_representation")
            else "discrete"
        )
        observed = []
        for scope_probes in sweeps.values():
            for p in scope_probes:
                raw = (p.get("progress") or {}).get("max_grad_norm_raw")
                if raw is not None:
                    observed.append(float(raw))
        loose_clip = max(16.0, max(observed) * 1.05) if observed else 24.0
        for clip in (1.0, loose_clip):
            probe = run_probe(
                repo_root=repo_root,
                fms_home=fms_home,
                tokenized_dir=tokenized_dir,
                model_root=model_root,
                backbone=clip_backbone,
                representation=clip_rep,
                batch_size=int(selected_bs),
                max_steps=args.clip_probe_steps,
                max_grad_norm=float(clip),
                seed=args.seed,
            )
            clip_checks.append(probe)
            (output_dir / f"clip_check_{clip}.json").write_text(
                json.dumps(probe, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

    report = build_report(
        memory_ranking=memory_ranking,
        sweeps=sweeps,
        clip_checks=clip_checks,
        window_diag=window_diag,
        batch_checks=batch_checks,
        gpu_total_bytes=gpu_total,
        memory_safety_fraction=args.memory_safety_fraction,
    )
    report_path = output_dir / "batch_geometry_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"report": str(report_path), "gate_passed": report["gate_passed"], "selected": report["selected"]}, indent=2))
    return 0 if report["gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
