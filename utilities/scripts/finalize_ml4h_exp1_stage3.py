#!/usr/bin/env python3
"""Validate Exp1 features, remove stale collisions, and launch Stage 3 once."""

from __future__ import annotations

import fcntl
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "fms-ehrs"))
from fms_ehrs.framework.artifacts import feature_filename, model_artifact_stem


IRB_HOME = Path(__file__).resolve().parents[2]
STAGE2_MANIFEST = IRB_HOME / "outputs/runs/ml4h_2026/stage2/exp1_extract_manifest.json"
STAGE3_DIR = IRB_HOME / "outputs/runs/ml4h_2026/stage3"
TOKENIZED_ROOT = IRB_HOME / "outputs/runs/tokenized/mimiciv-3.1_meds_70-10-20"
FINALIZE_LOCK = STAGE3_DIR / ".exp1_stage3_finalize.lock"
FINALIZE_RECORD = STAGE3_DIR / "exp1_stage3_finalize.json"


def expected_feature_paths(tasks: list[dict]) -> set[Path]:
    paths = {
        TOKENIZED_ROOT
        / f"{task['data_version']}-tokenized"
        / split
        / feature_filename(model_artifact_stem(task["model_loc"]))
        for task in tasks
        for split in ("train", "val", "test")
    }
    if len(paths) != 216:
        raise RuntimeError(
            f"Expected 216 unique feature destinations (72 models × 3 splits); found {len(paths)}."
        )
    return paths


def write_record(payload: dict) -> None:
    temporary = FINALIZE_RECORD.with_name(f".{FINALIZE_RECORD.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(FINALIZE_RECORD)


def main() -> int:
    STAGE3_DIR.mkdir(parents=True, exist_ok=True)
    with FINALIZE_LOCK.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)

        # This generator is the validation gate: it checks all 216 arrays for
        # expected row count, hidden size, and float16 dtype, checks all outcome
        # columns, and refuses colliding model stems.
        validation = subprocess.run(
            [
                "python",
                "utilities/scripts/generate_ml4h_exp1_stage3_jobfiles.py",
                "--output-dir",
                str(STAGE3_DIR),
            ],
            cwd=IRB_HOME,
            check=True,
            capture_output=True,
            text=True,
        )

        manifest = json.loads(STAGE2_MANIFEST.read_text(encoding="utf-8"))
        tasks = manifest["tasks"]
        if len(tasks) != 72:
            raise RuntimeError(f"Expected 72 models in the Stage-2 manifest; found {len(tasks)}.")
        expected = expected_feature_paths(tasks)
        missing = sorted(str(path) for path in expected if not path.is_file())
        if missing:
            raise RuntimeError(
                "Stage-3 generator passed but expected feature files are absent:\n  - "
                + "\n  - ".join(missing)
            )

        # User-authorized cleanup, deliberately constrained to the obsolete
        # basename pattern inside the tokenized root. Training checkpoints and
        # run-specific features cannot match this pattern.
        stale_features = sorted(TOKENIZED_ROOT.rglob("features-checkpoint-*.npy"))
        stale_sidecars = sorted(TOKENIZED_ROOT.rglob("features-checkpoint-*.npy.provenance.json"))
        stale_bytes = sum(path.stat().st_size for path in stale_features)
        for path in (*stale_features, *stale_sidecars):
            path.unlink()
        remaining_stale = list(TOKENIZED_ROOT.rglob("features-checkpoint-*.npy"))
        if remaining_stale:
            raise RuntimeError(f"Failed to remove {len(remaining_stale)} stale feature file(s).")

        # The launcher has its own lock and per-family launch state. Re-running
        # this finalizer cannot submit duplicate binary or regression arrays.
        launch = subprocess.run(
            ["bash", "slurm/12_submit_ml4h_exp1_stage3_cpu.sh"],
            cwd=IRB_HOME,
            check=True,
            capture_output=True,
            text=True,
        )

        payload = {
            "completed_at": datetime.now().astimezone().isoformat(),
            "validated_models": 72,
            "validated_feature_files": len(expected),
            "removed_stale_feature_files": len(stale_features),
            "removed_stale_sidecars": len(stale_sidecars),
            "removed_stale_bytes": stale_bytes,
            "validation_output": validation.stdout.strip(),
            "launch_output": launch.stdout.strip(),
        }
        write_record(payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
