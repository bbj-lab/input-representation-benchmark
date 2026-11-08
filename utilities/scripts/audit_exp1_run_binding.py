#!/usr/bin/env python
"""Verify every Exp1 run trained on the tokenization its configuration names.

A run whose recorded `data_version` disagrees with the version implied by its
config id would invalidate that arm and require retraining, so this compares the
two independently rather than trusting the directory name. It also checks the
`-hp-<data_version>` alias symlinks, which downstream stages use to name feature
artifacts, and confirms the seed and batch geometry actually used.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

MODELS_ROOT = Path("outputs/runs/ml4h_2026/models")
RUN_PATTERN = re.compile(
    r"^ml4h_(?P<backbone>llama|qwen)_exp1_meds_"
    r"(?P<quantizer>deciles|ventiles|trentiles|centiles)_"
    r"(?P<anchoring>none|5-10-5|10-10-10)_"
    r"fused(?P<fused>True|False)_discrete_time_tokens-s(?P<seed>4[234])$"
)
EXPECTED_RUNS = 72
EXPECTED_BATCH = 4


def main() -> int:
    problems: list[str] = []
    seen: list[str] = []
    seeds: Counter[str] = Counter()
    arms: Counter[str] = Counter()

    for path in sorted(MODELS_ROOT.iterdir()):
        if path.is_symlink() or not path.is_dir():
            continue
        match = RUN_PATTERN.match(path.name)
        if not match:
            continue
        seen.append(path.name)
        fields = match.groupdict()
        layout = "fused" if fields["fused"] == "True" else "unfused"
        expected_version = (
            f"{fields['quantizer']}_{fields['anchoring']}_{layout}_time_tokens"
        )
        arms[f"{fields['quantizer']}/{fields['anchoring']}/{layout}"] += 1
        seeds[fields["seed"]] += 1

        record_path = path / "run_record.json"
        if not record_path.is_file():
            problems.append(f"{path.name}: missing run_record.json")
            continue
        record = json.loads(record_path.read_text(encoding="utf-8"))

        recorded_version = record.get("data_version")
        if recorded_version != expected_version:
            problems.append(
                f"{path.name}: trained on '{recorded_version}', "
                f"config implies '{expected_version}'"
            )

        if str(record.get("seed")) != fields["seed"]:
            problems.append(
                f"{path.name}: recorded seed {record.get('seed')} "
                f"!= name seed {fields['seed']}"
            )

        args = (record.get("training") or {}).get("arguments") or {}
        batch = args.get("per_device_train_batch_size")
        accum = args.get("gradient_accumulation_steps")
        if batch is not None and batch != EXPECTED_BATCH:
            problems.append(f"{path.name}: train batch {batch} != {EXPECTED_BATCH}")
        if accum is not None and accum != 1:
            problems.append(f"{path.name}: accumulation {accum} != 1")

        # The alias points at the selected checkpoint inside the run directory,
        # not the directory itself. Checkpoint basenames repeat across runs, so
        # downstream artifacts must be named from the alias, never the target.
        alias = MODELS_ROOT / f"{path.name}-hp-{expected_version}"
        if not alias.is_symlink():
            problems.append(f"{path.name}: missing alias '{alias.name}'")
        else:
            target = alias.resolve()
            if target.parent.resolve() != path.resolve():
                problems.append(
                    f"{alias.name}: resolves into {target.parent.name}, expected {path.name}"
                )
            elif not target.name.startswith("checkpoint-"):
                problems.append(f"{alias.name}: target '{target.name}' is not a checkpoint")

    print(f"Exp1 runs inspected : {len(seen)}/{EXPECTED_RUNS}")
    print(f"distinct arms       : {len(arms)} (expect 12)")
    print(f"seeds               : {dict(sorted(seeds.items()))}")
    duplicated = [arm for arm, count in arms.items() if count != 6]
    if duplicated:
        problems.append(f"arms without exactly 6 runs (2 backbones x 3 seeds): {duplicated}")
    if len(seen) != EXPECTED_RUNS:
        problems.append(f"expected {EXPECTED_RUNS} runs, found {len(seen)}")

    if problems:
        print("\nproblems:")
        for line in problems:
            print(f"  {line}")
        print("\nRESULT: ACTION NEEDED — run/tokenization binding is not consistent")
        return 1

    print("\nRESULT: every Exp1 run matches its configured tokenization, seed, and batch geometry")
    return 0


if __name__ == "__main__":
    sys.exit(main())
