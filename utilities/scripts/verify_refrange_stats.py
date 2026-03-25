#!/usr/bin/env python3
"""Verify reference-range statistics from MIMIC-IV labevents."""
import csv, gzip

LAB_FILE = "/gpfs/data/bbj-lab/users/daniel/input-representation-benchmark/physionet.org/files/mimiciv/3.1/hosp/labevents.csv.gz"

total_rows = 0
rows_with_ref = 0
codes_with_ref = set()
all_codes = set()

with gzip.open(LAB_FILE, "rt", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        total_rows += 1
        itemid = row["itemid"]
        all_codes.add(itemid)
        has_lower = row.get("ref_range_lower", "").strip() not in ("", ".")
        has_upper = row.get("ref_range_upper", "").strip() not in ("", ".")
        if has_lower or has_upper:
            rows_with_ref += 1
            codes_with_ref.add(itemid)

print(f"Total lab events:                 {total_rows:>15,}")
print(f"Events with reference ranges:     {rows_with_ref:>15,}  ({rows_with_ref/total_rows*100:.1f}%)")
print(f"Unique lab item IDs (total):      {len(all_codes):>15,}")
print(f"Unique codes with ref ranges:     {len(codes_with_ref):>15,}  ({len(codes_with_ref)/len(all_codes)*100:.1f}%)")
