#!/usr/bin/env python
"""Audit that every Exp1 tokenization arm is materially distinct on disk.

Exp1 varies three tokenization knobs: quantizer granularity, reference-range
anchoring, and code-value fusion. Each knob can silently no-op (anchoring falls
back to population quantiles when reference ranges are missing; fusion changes
only the vocabulary layout). If two arms produced identical bin boundaries, the
runs trained on them would not be a valid comparison, so this check compares the
saved vocabularies rather than trusting the job-generation flags.
"""

from __future__ import annotations

import gzip
import pickle
import statistics
import sys
from pathlib import Path

TOKENIZED_ROOT = Path(
    "outputs/runs/tokenized/mimiciv-3.1_meds_70-10-20"
)
EXPECTED_BINS = {"deciles": 10, "ventiles": 20, "trentiles": 30, "centiles": 100}
ARMS = [
    ("deciles", "none"),
    ("ventiles", "none"),
    ("ventiles", "5-10-5"),
    ("trentiles", "none"),
    ("trentiles", "10-10-10"),
    ("centiles", "none"),
]


def _load_vocab(path: Path) -> dict:
    with gzip.open(path, mode="rb") as handle:
        return pickle.load(handle)


def _version(quantizer: str, anchoring: str, fused: bool) -> str:
    layout = "fused" if fused else "unfused"
    return f"{quantizer}_{anchoring}_{layout}_time_tokens"


def _vocab_path(version: str) -> Path:
    return TOKENIZED_ROOT / f"{version}-tokenized" / "train" / "vocab.gzip"


def _boundaries(aux: dict) -> dict[str, tuple]:
    """Map each numeric code to its bin cut points.

    `aux` stores the cut points directly as a list per code.
    """
    out: dict[str, tuple] = {}
    for word, data in aux.items():
        if not isinstance(data, (list, tuple)) or not data:
            continue
        try:
            out[str(word)] = tuple(round(float(b), 6) for b in data)
        except (TypeError, ValueError):
            continue
    return out


def main() -> int:
    summaries: dict[str, dict] = {}
    problems: list[str] = []

    for quantizer, anchoring in ARMS:
        for fused in (False, True):
            version = _version(quantizer, anchoring, fused)
            path = _vocab_path(version)
            if not path.is_file():
                problems.append(f"missing vocabulary: {path}")
                continue
            payload = _load_vocab(path)
            lookup = payload.get("lookup", {})
            aux = payload.get("aux", {})
            bounds = _boundaries(aux)
            widths = [len(b) for b in bounds.values()]
            summaries[version] = {
                "vocab_size": len(lookup),
                "coded_numerics": len(bounds),
                "median_breaks": statistics.median(widths) if widths else 0,
                "boundaries": bounds,
            }

    if problems:
        for line in problems:
            print(f"MISSING: {line}")
        return 1

    print(f"{'version':<48}{'vocab':>8}{'codes':>8}{'breaks':>8}")
    for version, info in summaries.items():
        print(
            f"{version:<48}{info['vocab_size']:>8}"
            f"{info['coded_numerics']:>8}{info['median_breaks']:>8}"
        )

    findings: list[str] = []

    # 1. Anchoring re-bins only codes carrying reference ranges, so the changed
    # set must be non-empty, confined to lab codes, and identical across the two
    # anchored quantizers (both target the same reference-range codes).
    changed_sets: dict[str, set[str]] = {}
    for quantizer, anchoring in ARMS:
        if anchoring == "none":
            continue
        for fused in (False, True):
            anchored = summaries[_version(quantizer, anchoring, fused)]["boundaries"]
            population = summaries[_version(quantizer, "none", fused)]["boundaries"]
            shared = set(anchored) & set(population)
            changed = {code for code in shared if anchored[code] != population[code]}
            changed_sets[f"{quantizer}/{anchoring}/fused={fused}"] = changed
            non_lab = {code for code in changed if not code.startswith("LAB")}
            status = "OK" if changed and not non_lab else "SUSPECT"
            findings.append(
                f"[{status}] {quantizer}/{anchoring} fused={fused}: "
                f"{len(changed)}/{len(shared)} codes re-binned, "
                f"{len(non_lab)} non-lab codes affected"
            )

    distinct_targets = {frozenset(s) for s in changed_sets.values()}
    status = "OK" if len(distinct_targets) == 1 else "SUSPECT"
    findings.append(
        f"[{status}] anchored arms target an identical code set "
        f"({len(distinct_targets)} distinct set(s) across {len(changed_sets)} arms)"
    )

    # 2. Finer quantizers must yield more bins than coarser ones.
    for fused in (False, True):
        observed = {
            quantizer: summaries[_version(quantizer, "none", fused)]["median_breaks"]
            for quantizer in ("deciles", "ventiles", "trentiles", "centiles")
        }
        ordered = (
            observed["deciles"]
            < observed["ventiles"]
            < observed["trentiles"]
            < observed["centiles"]
        )
        status = "OK" if ordered else "SUSPECT"
        findings.append(
            f"[{status}] fused={fused} granularity ladder (median breaks): {observed}"
        )

    # 3. Fusion must change the vocabulary, not merely the directory name.
    for quantizer, anchoring in ARMS:
        unfused = summaries[_version(quantizer, anchoring, False)]["vocab_size"]
        fused = summaries[_version(quantizer, anchoring, True)]["vocab_size"]
        status = "OK" if fused != unfused else "SUSPECT"
        findings.append(
            f"[{status}] {quantizer}/{anchoring} fusion: "
            f"unfused vocab={unfused}, fused vocab={fused}"
        )

    print("\nfindings:")
    for line in findings:
        print(f"  {line}")

    suspect = [line for line in findings if line.startswith("[SUSPECT]")]
    print(
        "\nRESULT:",
        "every Exp1 arm is materially distinct"
        if not suspect
        else f"ACTION NEEDED — {len(suspect)} suspect comparison(s)",
    )
    return 0 if not suspect else 1


if __name__ == "__main__":
    sys.exit(main())
