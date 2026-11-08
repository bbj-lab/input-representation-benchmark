from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "pipeline"
    / "scripts"
    / "bootstrap_ml4h_patient_clusters.py"
)
SCRIPT_DIR = SCRIPT.parent


def load_module():
    sys.path.insert(0, str(SCRIPT_DIR))
    spec = importlib.util.spec_from_file_location(
        "bootstrap_ml4h_patient_clusters",
        SCRIPT,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_cluster_draw_includes_and_repeats_all_patient_admissions() -> None:
    bootstrap = load_module()
    cluster_ids = np.array(["p1", "p1", "p2", "p3", "p3"])
    sampled = np.array(["p3", "p1", "p1"])

    indices = bootstrap.cluster_bootstrap_indices(cluster_ids, sampled)

    assert np.array_equal(indices, np.array([0, 0, 1, 1, 3, 4]))
    assert np.array_equal(cluster_ids[indices], np.array(["p1", "p1", "p1", "p1", "p3", "p3"]))
