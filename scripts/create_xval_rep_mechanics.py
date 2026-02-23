#!/usr/bin/env python3
"""Create representation_mechanics.pt for xVal models that lack it (rescue script artifact).

Run this script once to create the missing metadata files. This is needed because
the rescue script saved only pytorch_model.bin + vocab.gzip, omitting the
representation mechanics metadata that extract_hidden_states.py uses for auto-detection.

Usage:
    python3 scripts/create_xval_rep_mechanics.py
"""
import torch

configs = [
    {
        "path": "models/exp2_meds_deciles_none_fusedFalse_xval_time_tokens-xval-time_tokens-s42/model-xval-time_tokens/representation_mechanics.pt",
        "representation": "xval",
        "temporal": "time_tokens",
    },
    {
        "path": "models/exp2_meds_deciles_none_fusedFalse_xval_time_rope-xval-time_rope-s42/model-xval-time_rope/representation_mechanics.pt",
        "representation": "xval",
        "temporal": "time_rope",
    },
]

for cfg in configs:
    rep_state = {
        "representation": cfg["representation"],
        "temporal": cfg["temporal"],
        "num_bins": 10,
        "time_rope_scaling": 60.0,
        "value_encoder_state": None,  # xVal doesn't have a separate value encoder
    }
    torch.save(rep_state, cfg["path"])
    print(f"Created: {cfg['path']}")

print("Done. All representation_mechanics.pt files created.")
