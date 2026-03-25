from __future__ import annotations

import ast
import json
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _pipeline_scripts(root: Path) -> list[Path]:
    top_level = [root / "pipeline" / "run_experiments.py"]
    scripts_dir = root / "pipeline" / "scripts"
    script_files = sorted(p for p in scripts_dir.rglob("*.py") if p.name != "__init__.py")
    return top_level + script_files


def _dryrun_name(rel_path: str) -> str:
    return f"run_{rel_path.replace('/', '__').replace('.py', '')}.sh"


def test_every_pipeline_script_has_main_and_entry_guard() -> None:
    root = _repo_root()
    for script_path in _pipeline_scripts(root):
        tree = ast.parse(script_path.read_text(encoding="utf-8"))
        has_main = any(isinstance(node, ast.FunctionDef) and node.name == "main" for node in tree.body)
        text = script_path.read_text(encoding="utf-8")
        has_guard = '__name__' in text and '__main__' in text
        assert has_main, f"missing main() in {script_path.relative_to(root)}"
        assert has_guard, f"missing __main__ guard in {script_path.relative_to(root)}"


def test_every_pipeline_script_has_dryrun_wrapper() -> None:
    root = _repo_root()
    dryrun_root = root / "pipeline" / "tests" / "dryrun"
    for script_path in _pipeline_scripts(root):
        rel = script_path.relative_to(root).as_posix()
        dryrun = dryrun_root / _dryrun_name(rel)
        assert dryrun.exists(), f"missing dry-run wrapper for {rel}"


def test_dryrun_manifest_matches_pipeline_scripts() -> None:
    root = _repo_root()
    scripts = [p.relative_to(root).as_posix() for p in _pipeline_scripts(root)]
    manifest_path = root / "pipeline" / "tests" / "dryrun" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    listed = sorted(entry["script"] for entry in manifest["scripts"])
    assert listed == scripts
