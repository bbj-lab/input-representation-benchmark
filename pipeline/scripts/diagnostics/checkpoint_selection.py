#!/usr/bin/env python3
"""Resolve HuggingFace Trainer ``best_model_checkpoint`` for a run directory."""

from __future__ import annotations

import argparse
import json
import pathlib


def _checkpoint_step(path: pathlib.Path) -> int:
    try:
        return int(path.name.split("-")[-1])
    except ValueError as exc:
        raise ValueError(f"Not a checkpoint directory: {path}") from exc


def _search_roots(model_dir: pathlib.Path) -> list[pathlib.Path]:
    roots = [model_dir]
    for run_dir in sorted(model_dir.glob("run-*")):
        if run_dir.is_dir():
            roots.append(run_dir)
    return roots


def _trainer_state_paths(model_dir: pathlib.Path) -> list[tuple[int, pathlib.Path]]:
    found: list[tuple[int, pathlib.Path]] = []
    for root in _search_roots(model_dir):
        root_state = root / "trainer_state.json"
        if root_state.is_file():
            found.append((-1, root_state))
        for ckpt in root.glob("checkpoint-*"):
            if not ckpt.is_dir():
                continue
            state = ckpt / "trainer_state.json"
            if state.is_file():
                found.append((_checkpoint_step(ckpt), state))
    return found


def _has_weights(checkpoint: pathlib.Path) -> bool:
    return (checkpoint / "model.safetensors").is_file() or (
        checkpoint / "pytorch_model.bin"
    ).is_file()


def find_best_model_checkpoint(model_dir: pathlib.Path) -> pathlib.Path:
    """Return the validation-selected checkpoint, not the highest step.

    The latest ``trainer_state.json`` records ``best_model_checkpoint``. That
    path may be an absolute training-host path, so the basename is resolved
    under the local run directory.
    """
    model_dir = model_dir.resolve()
    states = _trainer_state_paths(model_dir)
    if not states:
        raise FileNotFoundError(f"No trainer_state.json under {model_dir}")

    _, state_path = max(states, key=lambda item: item[0])
    recorded = json.loads(state_path.read_text()).get("best_model_checkpoint")
    if not recorded:
        raise FileNotFoundError(
            f"{state_path} has no best_model_checkpoint"
        )

    recorded_path = pathlib.Path(recorded)
    candidates = [recorded_path]
    for root in _search_roots(model_dir):
        candidates.append(root / recorded_path.name)

    for candidate in candidates:
        if candidate.is_dir() and _has_weights(candidate):
            return candidate.resolve()

    raise FileNotFoundError(
        f"best_model_checkpoint={recorded!r} from {state_path} "
        f"was not found under {model_dir}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find the best model checkpoint for a training run folder."
    )
    parser.add_argument(
        "model_dir",
        type=pathlib.Path,
        help="Path to the model run folder that has trainer_state.json.",
    )
    args = parser.parse_args()
    print(find_best_model_checkpoint(args.model_dir))


if __name__ == "__main__":
    main()

