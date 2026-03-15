#!/bin/bash

# Backward-compatible wrapper. Prefer:
#   slurm/13_refresh_all_extended_outcomes.sh

set -euo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/13_refresh_all_extended_outcomes.sh" "$@"
