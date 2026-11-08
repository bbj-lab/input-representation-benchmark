#!/usr/bin/env python3
"""Inventory and remove only local Weights & Biases data from this benchmark.

The script never contacts W&B and never deletes credentials.  A run is eligible
only when its local W&B metadata proves that it originated from
``input-representation-benchmark`` or uses an ``input-rep-benchmark*`` project.
It writes a JSON manifest before deletion; use ``--apply`` to remove the
manifested run directories.

Examples
--------
python utilities/scripts/cleanup_wandb_local.py
python utilities/scripts/cleanup_wandb_local.py --apply
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_NAME = "input-representation-benchmark"
PROJECT_PREFIX = "input-rep-benchmark"


def _default_roots() -> list[Path]:
    roots = [
        Path(os.environ.get("WANDB_DIR", f"/scratch/{os.environ['USER']}/wandb")),
        Path(os.environ.get("WANDB_CACHE_DIR", Path.home() / ".cache" / "wandb")),
    ]
    return list(dict.fromkeys(root.expanduser() for root in roots))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _config_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _is_benchmark_run(run_dir: Path) -> tuple[bool, str]:
    files_dir = run_dir / "files"
    metadata = _read_json(files_dir / "wandb-metadata.json")
    config = _config_text(files_dir / "config.yaml")

    serialized_metadata = json.dumps(metadata, sort_keys=True)
    if REPO_NAME in serialized_metadata:
        return True, "metadata references benchmark repository"
    if REPO_NAME in config:
        return True, "config references benchmark repository"
    if PROJECT_PREFIX in config:
        return True, "config references benchmark W&B project"
    return False, ""


def _directory_size_bytes(path: Path) -> int:
    total = 0
    for child in path.rglob("*"):
        try:
            if child.is_file() and not child.is_symlink():
                total += child.stat().st_size
        except OSError:
            continue
    return total


def _eligible_runs(roots: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for root in roots:
        if not root.is_dir():
            continue
        for run_dir in sorted(root.iterdir()):
            if not run_dir.is_dir() or not (
                run_dir.name.startswith("run-") or run_dir.name.startswith("offline-run-")
            ):
                continue
            eligible, reason = _is_benchmark_run(run_dir)
            if eligible:
                records.append(
                    {
                        "path": str(run_dir.resolve()),
                        "root": str(root.resolve()),
                        "reason": reason,
                        "size_bytes": _directory_size_bytes(run_dir),
                    }
                )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        action="append",
        type=Path,
        help="Local W&B directory to scan. Repeat to scan multiple roots.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/runs/ml4h_2026/cleanup/wandb_local_cleanup_manifest.json"),
        help="Path for the inventory/deletion manifest.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Delete only the eligible run directories recorded in the manifest.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Report eligible runs without writing or replacing the manifest.",
    )
    args = parser.parse_args()
    if args.apply and args.verify:
        parser.error("--apply and --verify cannot be used together")

    roots = [root.expanduser() for root in args.root] if args.root else _default_roots()
    records = _eligible_runs(roots)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "apply" if args.apply else "dry_run",
        "scope": {
            "repository": REPO_NAME,
            "project_prefix": PROJECT_PREFIX,
            "roots": [str(root) for root in roots],
        },
        "candidate_runs": records,
        "candidate_size_bytes": sum(record["size_bytes"] for record in records),
    }

    if not args.verify:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Manifest: {args.manifest.resolve()}")
    print(f"Eligible local benchmark runs: {len(records)}")
    print(f"Eligible bytes: {manifest['candidate_size_bytes']}")
    for record in records:
        print(f"  {record['path']} ({record['reason']}; {record['size_bytes']} bytes)")

    if not args.apply:
        print("Dry run only. Re-run with --apply to delete these local run directories.")
        return 0

    for record in records:
        path = Path(record["path"])
        if path.is_dir() and (path.name.startswith("run-") or path.name.startswith("offline-run-")):
            shutil.rmtree(path)
            print(f"Deleted: {path}")

    for root in roots:
        latest_run = root / "latest-run"
        if latest_run.is_symlink() and not latest_run.exists():
            latest_run.unlink()
            print(f"Deleted dangling symlink: {latest_run}")

    manifest["deleted_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["deleted_run_count"] = len(records)
    manifest["post_delete_remaining_candidates"] = len(_eligible_runs(roots))
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
