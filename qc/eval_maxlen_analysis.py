#!/usr/bin/env python3
from pathlib import Path
import runpy

_ROOT = Path(__file__).resolve().parents[1]
_TARGET = _ROOT / "utilities" / "qc" / "eval_maxlen_analysis.py"
if __name__ == "__main__":
    runpy.run_path(str(_TARGET), run_name="__main__")
