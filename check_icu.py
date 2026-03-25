#!/usr/bin/env python3
from pathlib import Path
import runpy

_ROOT = Path(__file__).resolve().parent
_TARGET = _ROOT / "deprecated" / "scripts" / "legacy_misc" / "check_icu.py"
if __name__ == "__main__":
    runpy.run_path(str(_TARGET), run_name="__main__")
