#!/usr/bin/env python3
"""Regenerate aligned family-level benchmark summaries from saved prediction pickles.

This script reads an existing family-spec CSV (for example the current
`all_family_metrics.csv` file), groups rows by `family_name`, and reruns the
reusable aggregation layer with hospitalization-id alignment. It writes one
directory per family plus combined `all_family_metrics.csv` and
`all_family_pairwise.csv` tables under the chosen output root.
"""

from __future__ import annotations

import argparse
import collections
import csv
import sys
from pathlib import Path

import pandas as pd


def _load_families(source_csv: Path) -> collections.OrderedDict[str, dict[str, object]]:
    families: collections.OrderedDict[str, dict[str, object]] = collections.OrderedDict()
    with source_csv.open() as f:
        reader = csv.DictReader(f)
        required = {"family_name", "task_type", "handle", "pred_path", "outcomes_parquet"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns in {source_csv}: {sorted(missing)}")
        for row in reader:
            family = row["family_name"]
            spec = families.setdefault(
                family,
                {
                    "task_type": row["task_type"],
                    "handles": collections.OrderedDict(),
                    "outcomes_paths": collections.OrderedDict(),
                },
            )
            if spec["task_type"] != row["task_type"]:
                raise ValueError(f"task_type mismatch within family {family!r}")
            spec["handles"].setdefault(row["handle"], row["pred_path"])
            spec["outcomes_paths"].setdefault(row["handle"], row["outcomes_parquet"])
    return families


def _combine_outputs(
    *,
    out_root: Path,
    families: collections.OrderedDict[str, dict[str, object]],
) -> None:
    metric_frames: list[pd.DataFrame] = []
    pairwise_frames: list[pd.DataFrame] = []
    for family in families:
        metrics_path = out_root / family / f"{family}-metrics.csv"
        pairwise_path = out_root / family / f"{family}-pairwise.csv"
        if metrics_path.exists():
            metric_frames.append(pd.read_csv(metrics_path))
        if pairwise_path.exists():
            pairwise_frames.append(pd.read_csv(pairwise_path))

    if metric_frames:
        pd.concat(metric_frames, ignore_index=True).to_csv(
            out_root / "all_family_metrics.csv", index=False
        )
    if pairwise_frames:
        pd.concat(pairwise_frames, ignore_index=True).to_csv(
            out_root / "all_family_pairwise.csv", index=False
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_metrics_csv",
        type=Path,
        required=True,
        help="CSV with at least family_name, task_type, handle, and pred_path columns.",
    )
    parser.add_argument(
        "--out_root",
        type=Path,
        required=True,
        help="Output directory for regenerated family summaries.",
    )
    parser.add_argument("--bootstrap_n", type=int, default=2000)
    parser.add_argument("--permutation_n", type=int, default=2000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--fdr", choices=["bh", "none"], default="bh")
    parser.add_argument(
        "--baseline_handle",
        type=str,
        default=None,
        help="Optional baseline handle; if set, only compute pairwise tests baseline-vs-others.",
    )
    parser.add_argument(
        "--family_name",
        type=str,
        default=None,
        help="Optional single family to regenerate.",
    )
    parser.add_argument(
        "--outcomes",
        type=str,
        nargs="*",
        default=None,
        help="Optional subset of outcomes to regenerate within each selected family.",
    )
    parser.add_argument(
        "--skip_combine",
        action="store_true",
        help="Skip writing combined all-family CSVs.",
    )
    parser.add_argument(
        "--combine_only",
        action="store_true",
        help="Skip family regeneration and only combine existing family outputs.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    fms_repo = repo_root.parent / "fms-ehrs"
    sys.path.insert(0, str(fms_repo))
    from fms_ehrs.scripts.aggregate_version_preds import main as summarize_main

    source_csv = args.source_metrics_csv.expanduser().resolve()
    out_root = args.out_root.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if args.family_name is not None and not args.combine_only and not args.skip_combine:
        raise ValueError(
            "--family_name only regenerates a subset of families. Pass --skip_combine "
            "for partial reruns, then invoke a separate --combine_only step once all "
            "target families are refreshed."
        )

    families = _load_families(source_csv)
    if args.family_name is not None:
        if args.family_name not in families:
            raise ValueError(f"Unknown family_name {args.family_name!r}")
        families = collections.OrderedDict([(args.family_name, families[args.family_name])])

    if not args.combine_only:
        for idx, (family, spec) in enumerate(families.items(), start=1):
            handles = list(spec["handles"].keys())
            pred_paths = list(spec["handles"].values())
            outcomes_paths = [spec["outcomes_paths"][handle] for handle in handles]
            family_out = out_root / family
            family_out.mkdir(parents=True, exist_ok=True)
            print(f"[{idx}/{len(families)}] {family} -> {family_out}", flush=True)
            sys.argv = [
                "aggregate_version_preds.py",
                "--family_name",
                family,
                "--task_type",
                str(spec["task_type"]),
                "--out_dir",
                str(family_out),
                "--bootstrap_n",
                str(int(args.bootstrap_n)),
                "--permutation_n",
                str(int(args.permutation_n)),
                "--alpha",
                str(float(args.alpha)),
                "--fdr",
                str(args.fdr),
                "--handles",
                *handles,
                "--pred_paths",
                *pred_paths,
                "--outcomes_paths",
                *outcomes_paths,
            ]
            if args.baseline_handle is not None:
                sys.argv.extend(["--baseline_handle", str(args.baseline_handle)])
            if args.outcomes:
                sys.argv.extend(["--outcomes", *map(str, args.outcomes)])
            rc = summarize_main()
            if rc != 0:
                return int(rc)

    if not args.skip_combine:
        families_for_combine = families if args.family_name is not None else _load_families(source_csv)
        _combine_outputs(out_root=out_root, families=families_for_combine)

    print(f"DONE {out_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
