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
#   bash utilities/scripts/cleanup_outputs.sh --dry-run
#   bash utilities/scripts/cleanup_outputs.sh --apply
#
# Optional (risky) flags:
#   --delete-token-cache   # deletes IRB token cache root (expensive to rebuild)
#   --delete-tokenized-symlinks # deletes *-tokenized symlinks under DATA_DIR (safe-ish; recreatable)
#   --delete-stage2-stage3-outputs # deletes features-*.npy and *-preds-*.pkl under token cache + DATA_DIR symlinks
#   --archive-slurm-output # move slurm/output to a timestamped dir under slurm/output_archive/
#
# =============================================================================

set -euo pipefail

DRY_RUN=1
DELETE_TOKEN_CACHE=0
DELETE_TOKENIZED_SYMLINKS=0
DELETE_STAGE2_STAGE3_ARTIFACTS=0
ARCHIVE_SLURM_OUTPUT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --apply) DRY_RUN=0; shift ;;
    --delete-token-cache) DELETE_TOKEN_CACHE=1; shift ;;
    --delete-tokenized-symlinks) DELETE_TOKENIZED_SYMLINKS=1; shift ;;
    --delete-stage2-stage3-outputs) DELETE_STAGE2_STAGE3_ARTIFACTS=1; shift ;;
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
    if [[ -f "$d/pipeline/run_experiments.py" && -d "$d/slurm" ]]; then
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
echo "Cleanup outputs (safe-by-default)"
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
  CACHE_ROOT="${IRB_HOME}/outputs/runs/tokenized"
  if [[ -d "${CACHE_ROOT}" ]]; then
    run rm -rf "${CACHE_ROOT}"
  else
    echo "No cache root found at: ${CACHE_ROOT}"
  fi
  echo ""
fi

# 6) Optionally delete tokenized symlinks under DATA_DIR (helps avoid Stage0 collisions)
if [[ "${DELETE_TOKENIZED_SYMLINKS}" -eq 1 ]]; then
  echo "== Deleting *-tokenized symlinks under DATA_DIR (opt-in) =="
  source "${IRB_HOME}/slurm/00_preamble.sh" >/dev/null 2>&1 || true
  DATA_DIR="${DATA_DIR:-${IRB_HOME}/benchmarks/mimic-meds-extraction/data/meds/data}"
  if [[ -d "${DATA_DIR}" ]]; then
    while IFS= read -r link; do
      # Only delete symlinks; refuse to delete real directories.
      if [[ -L "${link}" ]]; then
        run rm -f "${link}"
      fi
    done < <(find "${DATA_DIR}" -maxdepth 1 -type l -name '*-tokenized' -print)
  else
    echo "No DATA_DIR found at: ${DATA_DIR}"
  fi
  echo ""
fi

# 7) Optionally delete Stage2/Stage3 outputs (features/preds)
if [[ "${DELETE_STAGE2_STAGE3_ARTIFACTS}" -eq 1 ]]; then
  echo "== Deleting Stage2/Stage3 outputs (opt-in) =="
  source "${IRB_HOME}/slurm/00_preamble.sh" >/dev/null 2>&1 || true
  DATA_DIR="${DATA_DIR:-${IRB_HOME}/benchmarks/mimic-meds-extraction/data/meds/data}"
  # Stage2/Stage3 outputs are written into the main tokenized run tree.
  # Compatibility symlinks under DATA_DIR may still exist during transition.
  #
  # We delete in both places:
  # - main tokenized root (preferred; actual storage)
  # - DATA_DIR (compatibility-link view; safe if symlinks exist, no-op otherwise)
  CACHE_ROOT_LOCAL="${IRB_HOME}/outputs/runs/tokenized"
  CACHE_ROOT_SCRATCH=""
  if [[ -n "${hm:-}" && -d "${hm}" ]]; then
    CACHE_ROOT_SCRATCH="${hm}/irb_scratch/tokenized"
  fi

  for ROOT in "${CACHE_ROOT_LOCAL}" "${CACHE_ROOT_SCRATCH}" "${DATA_DIR}"; do
    if [[ -z "${ROOT}" ]]; then
      continue
    fi
    if [[ -d "${ROOT}" ]]; then
      echo "  - scanning: ${ROOT}"
      while IFS= read -r f; do
        run rm -f "${f}"
      done < <(find "${ROOT}" -type f \( -name 'features-*.npy' -o -name '*-preds-*.pkl' \) -print)
    fi
  done
  echo ""
fi

echo "Done."

