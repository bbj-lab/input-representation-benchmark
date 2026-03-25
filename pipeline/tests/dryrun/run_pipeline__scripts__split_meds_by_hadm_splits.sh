#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"
run_script_dryrun "pipeline/scripts/split_meds_by_hadm_splits.py" "execute" --help
