#!/bin/bash
# =============================================================================
# Cleanup helper (safe-by-default) for input-representation-benchmark
# =============================================================================
#
# Philosophy:
# - Default actions remove only universally-safe noise:
#   - macOS .DS_Store
#   - Python __pycache__ directories / .pyc files
#   - slurm/generated/* (jobfiles are regenerated deterministically)
#
# - Anything compute-expensive to rebuild (e.g., tokenization cache) is NEVER
#   deleted unless explicitly requested with an opt-in flag.
#
# Usage:
#   bash scripts/cleanup_artifacts.sh --dry-run
#   bash scripts/cleanup_artifacts.sh --apply
#
# Optional (risky) flags:
#   --delete-token-cache   # deletes IRB token cache root (expensive to rebuild)
#   --archive-slurm-output # move slurm/output to a timestamped dir under slurm/output_archive/
#
# =============================================================================

set -euo pipefail

DRY_RUN=1
DELETE_TOKEN_CACHE=0
ARCHIVE_SLURM_OUTPUT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --apply) DRY_RUN=0; shift ;;
    --delete-token-cache) DELETE_TOKEN_CACHE=1; shift ;;
    --archive-slurm-output) ARCHIVE_SLURM_OUTPUT=1; shift ;;
    -h|--help)
      sed -n '1,120p' "$0"
      exit 0
      ;;
    *)
      echo "ERROR: unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

find_repo_root() {
  local d="$1"
  while [[ "$d" != "/" ]]; do
    if [[ -f "$d/run_experiments.py" && -d "$d/slurm" && -d "$d/scripts" ]]; then
      echo "$d"
      return 0
    fi
    d="$(dirname "$d")"
  done
  return 1
}

IRB_HOME="$(find_repo_root "$(pwd)")"
cd "${IRB_HOME}"

run() {
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[dry-run] $*"
  else
    echo "[apply] $*"
    "$@"
  fi
}

echo "=============================================="
echo "Cleanup artifacts (safe-by-default)"
echo "=============================================="
echo "IRB_HOME: ${IRB_HOME}"
echo "Mode: $([[ ${DRY_RUN} -eq 1 ]] && echo DRY-RUN || echo APPLY)"
echo ""

# 1) .DS_Store
echo "== Removing .DS_Store =="
while IFS= read -r p; do
  run rm -f "$p"
done < <(find "${IRB_HOME}" -name '.DS_Store' -print)
echo ""

# 2) Python bytecode
echo "== Removing __pycache__ and *.pyc =="
while IFS= read -r d; do
  run rm -rf "$d"
done < <(find "${IRB_HOME}" -type d -name '__pycache__' -print)
while IFS= read -r f; do
  run rm -f "$f"
done < <(find "${IRB_HOME}" -type f -name '*.pyc' -print)
echo ""

# 3) Generated jobfiles (safe: always regeneratable)
echo "== Removing slurm/generated/* =="
if [[ -d "${IRB_HOME}/slurm/generated" ]]; then
  # Keep directory itself (gitignored) but remove its contents.
  while IFS= read -r child; do
    run rm -rf "$child"
  done < <(find "${IRB_HOME}/slurm/generated" -mindepth 1 -maxdepth 1 -print)
fi
echo ""

# 4) Optionally archive SLURM output logs
if [[ "${ARCHIVE_SLURM_OUTPUT}" -eq 1 ]]; then
  echo "== Archiving slurm/output (optional) =="
  if [[ -d "${IRB_HOME}/slurm/output" ]]; then
    ts="$(date +%Y%m%d_%H%M%S)"
    dest="${IRB_HOME}/slurm/output_archive/${ts}"
    run mkdir -p "${IRB_HOME}/slurm/output_archive"
    run mv "${IRB_HOME}/slurm/output" "${dest}"
    run mkdir -p "${IRB_HOME}/slurm/output"
  else
    echo "No slurm/output directory found; skipping."
  fi
  echo ""
fi

# 5) Optionally delete token cache (EXPENSIVE)
if [[ "${DELETE_TOKEN_CACHE}" -eq 1 ]]; then
  echo "== Deleting token cache (risky/expensive; opt-in) =="
  CACHE_ROOT="${IRB_HOME}/.cache/tokenized"
  if [[ -d "${CACHE_ROOT}" ]]; then
    run rm -rf "${CACHE_ROOT}"
  else
    echo "No cache root found at: ${CACHE_ROOT}"
  fi
  echo ""
fi

echo "Done."

