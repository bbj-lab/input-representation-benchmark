#!/usr/bin/env bash
set -euo pipefail

_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "${_ROOT}/utilities/scripts/setup_conda_env_meds_extract.sh" "$@"
