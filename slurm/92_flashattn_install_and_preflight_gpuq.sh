#!/bin/bash
#SBATCH --job-name=irb-flashattn
#SBATCH --output=./slurm/output/%j-%x.stdout
#SBATCH --partition=gpudev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96GB
# NOTE: Source builds of flash-attn can take >2 hours on shared nodes.
# Keep this generous to avoid wasting compile time due to TIMEOUT.
#SBATCH --time=0-06:00:00
#
# Install flash-attn (if possible) and run a preflight report.
#
# Notes:
# - This must run on a node with a CUDA toolchain (nvcc + CUDA_HOME) OR a matching prebuilt wheel.
# - If no matching wheel exists and nvcc/CUDA_HOME are missing, installation will fail.

set -euo pipefail

find_repo_root() {
  local d="$1"
  while [[ "$d" != "/" ]]; do
    if [[ -f "$d/run_experiments.py" && -d "$d/slurm" ]]; then
      echo "$d"
      return 0
    fi
    d="$(dirname "$d")"
  done
  return 1
}

# NOTE: In SLURM, the executing script path may be under /var/spool/...,
# so we MUST anchor repo discovery on the submission directory.
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
IRB_HOME="$(find_repo_root "${SUBMIT_DIR}")"
source "${IRB_HOME}/slurm/00_preamble.sh"
cd "${IRB_HOME}"

echo "=== flash-attn install attempt ==="
echo "Host: $(hostname)"
echo "Python: $(python -V)"
python -c "import torch; print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available(), 'cuda', torch.version.cuda)"

# Use a newer GCC toolchain (PyTorch requires GCC >= 9 for C++17).
if command -v module >/dev/null 2>&1; then
  module load gcc/12.1.0
fi

# Avoid cross-filesystem hardlink issues during wheel caching by keeping pip cache/temp
# on the same filesystem as the repo (GPFS).
PIP_CACHE_DIR="${IRB_HOME}/.cache/pip"
TMPDIR="${IRB_HOME}/.cache/tmp"
mkdir -p "${PIP_CACHE_DIR}" "${TMPDIR}"
export PIP_CACHE_DIR TMPDIR
echo "PIP_CACHE_DIR=${PIP_CACHE_DIR}"
echo "TMPDIR=${TMPDIR}"

# Best-effort CUDA_HOME inference if nvcc exists.
if command -v nvcc >/dev/null 2>&1; then
  CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
  export CUDA_HOME
  echo "CUDA_HOME inferred from nvcc: ${CUDA_HOME}"
else
  echo "nvcc not found; will rely on prebuilt wheels (if available)."
fi

FLASH_ATTN_VER="${IRB_FLASH_ATTN_VERSION:-2.8.3}"

echo "=== system diagnostics (glibc + toolchain) ==="
getconf GNU_LIBC_VERSION || true
ldd --version 2>/dev/null | head -n 1 || true
command -v gcc >/dev/null 2>&1 && gcc --version | head -n 1 || true
command -v g++ >/dev/null 2>&1 && g++ --version | head -n 1 || true
python - <<'PY'
import sysconfig
print("python CC:", sysconfig.get_config_var("CC"))
print("python CXX:", sysconfig.get_config_var("CXX"))
PY

echo "=== source install flash-attn==${FLASH_ATTN_VER} (RHEL8-compatible) ==="
python -m pip uninstall -y flash-attn || true

# Force a source build (avoid prebuilt wheels which can be GLIBC-incompatible on RHEL8 nodes).
export MAX_JOBS="2"
export CC="gcc"
export CXX="g++"
export CUDAHOSTCXX="g++"
export FLASH_ATTENTION_FORCE_BUILD="TRUE"
# Reduce build-time RAM/compile volume:
# - Force building only for A100 (sm80) unless you explicitly override.
# - Use fewer nvcc threads.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export FLASH_ATTENTION_FORCE_SINGLE_THREAD="TRUE"
echo "MAX_JOBS=${MAX_JOBS}"
echo "CC=${CC}"
echo "CXX=${CXX}"
echo "CUDAHOSTCXX=${CUDAHOSTCXX}"
echo "FLASH_ATTENTION_FORCE_BUILD=${FLASH_ATTENTION_FORCE_BUILD}"
echo "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
echo "FLASH_ATTENTION_FORCE_SINGLE_THREAD=${FLASH_ATTENTION_FORCE_SINGLE_THREAD}"

python -m pip install --upgrade --no-cache-dir ninja

python -m pip install --upgrade --no-build-isolation --no-cache-dir --no-binary flash-attn "flash-attn==${FLASH_ATTN_VER}"

echo "=== verify import ==="
python -c "import flash_attn; print('flash_attn ok:', flash_attn.__version__)"

echo "=== preflight (explicitly testing flash_attention_2) ==="
IRB_ATTN_IMPL="flash_attention_2" python scripts/preflight_perf_knobs.py

