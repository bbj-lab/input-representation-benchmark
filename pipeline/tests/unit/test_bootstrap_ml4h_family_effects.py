from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "pipeline"
    / "scripts"
    / "bootstrap_ml4h_family_effects.py"
)
CONTRASTS = (
    Path(__file__).resolve().parents[3] / "paper" / "scripts"
)


def load_module():
    sys.path.insert(0, str(CONTRASTS))
    spec = importlib.util.spec_from_file_location("bootstrap_ml4h_family_effects", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _context() -> dict:
    n = 8
    y = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    strong = np.array([0.1, 0.15, 0.12, 0.18, 0.85, 0.9, 0.88, 0.92])
    weak = np.array([0.45, 0.55, 0.4, 0.6, 0.5, 0.48, 0.52, 0.47])
    run_keys = (("llama", 42), ("qwen", 42))
    return {
        "families": [
            {
                "task_type": "binary",
                "family": "Hospital outcomes",
                "outcomes": ("same_admission_death",),
                "point": 0.0,
            }
        ],
        "n_rows": n,
        "labels": {"same_admission_death": y},
        "valid": {"same_admission_death": np.ones(n, dtype=bool)},
        "predictions": {
            "level": {
                "cfg_level": {
                    run_key: {"same_admission_death": strong} for run_key in run_keys
                }
            },
            "baseline": {
                "cfg_base": {
                    run_key: {"same_admission_death": weak} for run_key in run_keys
                }
            },
        },
        "config_ids": {"level": ("cfg_level",), "baseline": ("cfg_base",)},
        "run_keys": run_keys,
    }


def test_task_count_matches_main_table_columns() -> None:
    boot = load_module()
    assert len(boot.task_specs()) == 17


def test_permutation_to_reorders_to_target_ids() -> None:
    boot = load_module()
    source = np.array(["a", "b", "c"])
    target = np.array(["c", "a", "b"])
    order = boot.permutation_to(source, target)
    assert np.array_equal(source[order], target)
    assert np.array_equal(boot.permutation_to(source, source), np.arange(3))
    with pytest.raises(ValueError, match="differ"):
        boot.permutation_to(source, np.array(["a", "b", "z"]))


def test_family_delta_flips_when_every_admission_is_swapped() -> None:
    boot = load_module()
    context = _context()
    none = np.zeros(context["n_rows"], dtype=bool)
    all_swap = np.ones(context["n_rows"], dtype=bool)
    observed = boot.family_delta_with_swap(context, context["families"][0], none)
    flipped = boot.family_delta_with_swap(context, context["families"][0], all_swap)
    assert observed > 0.2
    assert flipped == pytest.approx(-observed)
