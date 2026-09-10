#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"
run_script_dryrun "pipeline/scripts/bootstrap_ml4h_24to48h_metrics.py" "execute" --help
