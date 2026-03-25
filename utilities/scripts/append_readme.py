#!/usr/bin/env python3
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    note = root / "deprecated" / "docs" / "paper_audit_trail.md"
    print(f"Moved note: {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
