#!/usr/bin/env python3
"""
Preflight check for optional performance settings (bf16 + attention backend).

This script is intended to be run *on a GPU node* (e.g., via gpuq) to verify:
- CUDA visibility
- bf16 support
- whether `flash_attn` is importable (required for flash_attention_2)
- versions of torch/transformers/accelerate
- requested attention implementation (IRB_ATTN_IMPL) and bf16 toggle (IRB_USE_BF16)

It does not run training and does not download model weights. It is safe to run
before submitting long jobs.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys


def _v(mod: str) -> str:
    try:
        m = importlib.import_module(mod)
        return str(getattr(m, "__version__", "unknown"))
    except Exception as e:
        return f"NOT_IMPORTABLE ({e!r})"


def _bool_env(name: str, default: str) -> bool:
    s = str(os.getenv(name, default)).strip().lower()
    return s in ("1", "true", "t", "yes", "y", "on")


def main() -> int:
    use_bf16 = _bool_env("IRB_USE_BF16", "true")
    attn_impl = os.getenv("IRB_ATTN_IMPL", "sdpa")

    print("=== IRB performance preflight ===")
    print("python:", sys.version.replace("\n", " "))
    print("torch:", _v("torch"))
    print("transformers:", _v("transformers"))
    print("trl:", _v("trl"))
    print("accelerate:", _v("accelerate"))
    print("IRB_USE_BF16:", use_bf16)
    print("IRB_ATTN_IMPL:", attn_impl)

    try:
        import torch

        print("cuda_available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("cuda_version:", torch.version.cuda)
            print("device_count:", torch.cuda.device_count())
            print("device0:", torch.cuda.get_device_name(0))
            try:
                print("bf16_supported:", torch.cuda.is_bf16_supported())
            except Exception as e:
                print("bf16_supported: error", repr(e))

            # SDPA backend flags (if present)
            for name in ("flash_sdp_enabled", "mem_efficient_sdp_enabled", "math_sdp_enabled"):
                if hasattr(torch.backends.cuda, name):
                    print(f"{name}:", getattr(torch.backends.cuda, name)())
        else:
            print("NOTE: CUDA not available in this environment; run on a GPU node for full validation.")
    except Exception as e:
        print("torch import failed:", repr(e))
        return 2

    flash_attn_present = importlib.util.find_spec("flash_attn") is not None
    print("flash_attn_present:", flash_attn_present)
    if attn_impl == "flash_attention_2" and not flash_attn_present:
        print("ERROR: IRB_ATTN_IMPL=flash_attention_2 but flash-attn is not installed.")
        return 3

    print("=== done ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

