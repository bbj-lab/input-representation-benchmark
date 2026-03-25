#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"
run_script_dryrun "pipeline/scripts/create_xval_rep_mechanics.py" "execute" --dry-run --repo_root $REPO_ROOT
