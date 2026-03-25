#!/usr/bin/env python3
"""
Build a primary run-artifact tree under ``artifacts/runs``.

The goal is to consolidate the real non-deprecated run artifacts while preserving
backward compatibility through symlinks at the old paths. This script is meant
to be run only after review because it moves real directories.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def move_with_symlink(src: Path, dst: Path, *, dry_run: bool) -> None:
    if not src.exists() and not src.is_symlink():
        if dry_run:
            print(f"SKIP missing source in dry-run: {src}")
            return
        raise FileNotFoundError(f"Source path does not exist: {src}")
    if dst.exists() or dst.is_symlink():
        if dry_run:
            print(f"SKIP existing destination in dry-run: {dst}")
            return
        raise FileExistsError(f"Destination already exists: {dst}")

    print(f"MOVE {src} -> {dst}")
    print(f"LINK {src} -> {dst}")
    if dry_run:
        return

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    src.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(dst, src, target_is_directory=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the primary run artifact tree with compatibility symlinks."
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=repo_root() / "artifacts" / "runs",
        help="Primary run artifact root (default: artifacts/runs).",
    )
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Preview moves without modifying the filesystem (default: true).",
    )
    args = parser.parse_args()

    root = repo_root()
    run_root = args.run_root.expanduser().resolve()
    dry_run = bool(args.dry_run)

    print(f"Repository root: {root}")
    print(f"Run root:        {run_root}")
    print(f"Dry run:         {dry_run}")

    move_specs = [
        (
            root / ".cache" / "tokenized" / "mimiciv-3.1_meds_70-10-20",
            run_root / "tokenized" / "mimiciv-3.1_meds_70-10-20",
        ),
        (
            root / "models",
            run_root / "models",
        ),
        (
            root / "data" / "exp3",
            run_root / "exp3",
        ),
    ]

    if not dry_run:
        run_root.mkdir(parents=True, exist_ok=True)

    for src, dst in move_specs:
        move_with_symlink(src, dst, dry_run=dry_run)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
