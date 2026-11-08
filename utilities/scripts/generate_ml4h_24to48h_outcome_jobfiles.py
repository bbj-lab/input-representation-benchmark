#!/usr/bin/env python3
"""Generate isolated 24-to-48-hour ML4H outcome-materialization jobfiles."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline.provisional_24to48_task_registry import (
    HORIZON_END_HOURS,
    HORIZON_START_HOURS,
    OUTCOMES_FILENAME,
)


BASE_OUTCOMES_FILENAME = "tokens_timelines_outcomes_24to48h.parquet"
DEFAULT_SHARED_TOKENIZED_ROOT = Path(
    "outputs/runs/tokenized/mimiciv-3.1_meds_70-10-20"
)
DEFAULT_MEDS_EVENTS_DIR = Path(
    "benchmarks/mimic-meds-extraction/data/meds/data"
)
DEFAULT_EXP3_MEDS_EVENTS_DIR = Path("outputs/runs/exp3/meds_icu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exp1-stage2-manifest",
        type=Path,
        default=Path(
            "outputs/runs/ml4h_2026/stage2_repaired_test/exp1_extract_manifest.json"
        ),
    )
    parser.add_argument(
        "--exp2-stage2-manifest",
        type=Path,
        default=Path("outputs/runs/ml4h_2026/stage2_exp2/exp2_extract_manifest.json"),
    )
    parser.add_argument(
        "--exp3-stage2-manifest",
        type=Path,
        default=Path("outputs/runs/ml4h_2026/stage2_exp3/exp3_extract_manifest.json"),
    )
    parser.add_argument(
        "--shared-tokenized-root",
        type=Path,
        default=DEFAULT_SHARED_TOKENIZED_ROOT,
    )
    parser.add_argument(
        "--shared-meds-events-dir",
        type=Path,
        default=DEFAULT_MEDS_EVENTS_DIR,
    )
    parser.add_argument(
        "--exp3-meds-events-dir",
        type=Path,
        default=DEFAULT_EXP3_MEDS_EVENTS_DIR,
        help=(
            "Native ICU MEDS events used to label both Exp3 source arms on the "
            "matched ICU cohort."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/runs/ml4h_2026/provisional_24to48h"),
    )
    return parser.parse_args()


def load_tasks(path: Path, *, expected_count: int) -> list[dict]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    tasks = manifest["tasks"]
    if len(tasks) != expected_count:
        raise SystemExit(
            f"{path}: expected {expected_count} Stage-2 tasks, found {len(tasks)}."
        )
    return tasks


def outcome_command(*, meds_events_dir: Path, tokenized_dir: Path) -> str:
    common = (
        f"--meds_events_dir {shlex.quote(str(meds_events_dir))} "
        f"--tokenized_dir {shlex.quote(str(tokenized_dir))} "
        "--splits train,val,test "
        f"--observation_hours {HORIZON_START_HOURS:g} "
        f"--followup_end_hours {HORIZON_END_HOURS:g} --strict"
    )
    return " && ".join(
        [
            "python pipeline/scripts/extract_outcomes_meds.py "
            + common
            + " --outcomes_only"
            + f" --output_filename {BASE_OUTCOMES_FILENAME}",
            "python pipeline/scripts/extract_extended_outcomes.py "
            + common
            + f" --base_outcomes_filename {BASE_OUTCOMES_FILENAME}"
            + f" --output_filename {OUTCOMES_FILENAME}",
        ]
    )


def main() -> int:
    args = parse_args()
    shared_tokenized_root = args.shared_tokenized_root.expanduser().resolve()
    shared_meds_events_dir = args.shared_meds_events_dir.expanduser().resolve()
    exp3_meds_events_dir = args.exp3_meds_events_dir.expanduser().resolve()
    if not shared_meds_events_dir.is_dir():
        raise SystemExit(f"Shared MEDS events directory does not exist: {shared_meds_events_dir}")
    if not exp3_meds_events_dir.is_dir():
        raise SystemExit(f"Exp3 MEDS events directory does not exist: {exp3_meds_events_dir}")

    manifests = {
        "exp1": (args.exp1_stage2_manifest.resolve(), 72),
        "exp2": (args.exp2_stage2_manifest.resolve(), 72),
        "exp3": (args.exp3_stage2_manifest.resolve(), 12),
    }
    materializations: dict[tuple[str, str], dict] = {}
    for experiment, (manifest_path, expected_count) in manifests.items():
        tasks = load_tasks(manifest_path, expected_count=expected_count)
        for task in tasks:
            if experiment == "exp3":
                tokenized_root = Path(task["data_dir"]).expanduser().resolve()
                # Exp3 compares native MIMIC against CLIF harmonization on a
                # matched ICU cohort. The native events are the shared outcome
                # source for both arms; CLIF supplies representation inputs.
                meds_events_dir = exp3_meds_events_dir
            else:
                tokenized_root = shared_tokenized_root
                meds_events_dir = shared_meds_events_dir
            tokenized_dir = tokenized_root / f"{task['data_version']}-tokenized"
            key = (str(meds_events_dir), str(tokenized_dir))
            record = materializations.setdefault(
                key,
                {
                    "meds_events_dir": str(meds_events_dir),
                    "tokenized_dir": str(tokenized_dir),
                    "data_version": task["data_version"],
                    "experiments": [],
                },
            )
            if experiment not in record["experiments"]:
                record["experiments"].append(experiment)

    records = sorted(
        materializations.values(),
        key=lambda item: (item["tokenized_dir"], item["meds_events_dir"]),
    )
    for record in records:
        tokenized_dir = Path(record["tokenized_dir"])
        missing = [
            str(tokenized_dir / split / "tokens_timelines.parquet")
            for split in ("train", "val", "test")
            if not (tokenized_dir / split / "tokens_timelines.parquet").is_file()
        ]
        if missing:
            raise SystemExit(
                "Cannot materialize provisional outcomes; missing timelines:\n  - "
                + "\n  - ".join(missing)
            )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    commands = [
        outcome_command(
            meds_events_dir=Path(record["meds_events_dir"]),
            tokenized_dir=Path(record["tokenized_dir"]),
        )
        for record in records
    ]
    jobfile = output_dir / "materialize_outcomes_24to48h.jobfile"
    jobfile.write_text(
        "\n".join(
            [
                "# ML4H provisional 24-to-48-hour outcome materialization",
                "# Each task writes new horizon-specific outcome files only.",
                *commands,
                "",
            ]
        ),
        encoding="utf-8",
    )
    (output_dir / "outcome_materialization_manifest.json").write_text(
        json.dumps(
            {
                "analysis": "24to48h",
                "observation_hours": HORIZON_START_HOURS,
                "followup_end_hours": HORIZON_END_HOURS,
                "base_outcomes_filename": BASE_OUTCOMES_FILENAME,
                "extended_outcomes_filename": OUTCOMES_FILENAME,
                "task_count": len(records),
                "tasks": records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(records)} outcome-materialization tasks to {jobfile}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
