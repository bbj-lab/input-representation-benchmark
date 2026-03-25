#!/usr/bin/env python3
from pathlib import Path
import importlib
import runpy
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_TARGET_FILE = _ROOT / "utilities/scripts/generate_calibration_plot.py"
_mod = importlib.import_module("utilities.scripts.generate_calibration_plot")
for _name in dir(_mod):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_mod, _name)

if __name__ == "__main__":
    runpy.run_path(str(_TARGET_FILE), run_name="__main__")
