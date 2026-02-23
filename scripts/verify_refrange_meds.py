#!/usr/bin/env python3
"""Verify reference-range stats from MEDS-extracted labevents (requires hadm_id & storetime)."""
import csv, gzip

LAB_FILE = "/gpfs/data/bbj-lab/users/daniel/input-representation-benchmark/physionet.org/files/mimiciv/3.1/hosp/labevents.csv.gz"

total_raw = 0
total_filtered = 0  # rows with hadm_id AND storetime
rows_with_ref = 0
codes_with_ref = set()
all_codes_filtered = set()

with gzip.open(LAB_FILE, "rt", newline="") as f:
    reader = csv.DictReader(f)
    cols = reader.fieldnames
    print("Columns:", cols[:10])
    for row in reader:
        total_raw += 1
        hadm_id = row.get("hadm_id", "").strip()
        storetime = row.get("storetime", "").strip()
        # Apply MEDS filter: must have hadm_id AND storetime
        if not hadm_id or not storetime or hadm_id == "" or storetime == "":
            continue
        total_filtered += 1
        itemid = row.get("itemid", "").strip()
        all_codes_filtered.add(itemid)
        has_lower = row.get("ref_range_lower", "").strip() not in ("", ".")
        has_upper = row.get("ref_range_upper", "").strip() not in ("", ".")
        if has_lower or has_upper:
            rows_with_ref += 1
            codes_with_ref.add(itemid)

print(f"Total raw lab events:                    {total_raw:>15,}")
print(f"MEDS-filtered (hadm_id & storetime):     {total_filtered:>15,}")
print(f"Events with reference ranges (filtered): {rows_with_ref:>15,}  ({rows_with_ref/total_filtered*100:.1f}%)")
print(f"Unique codes (filtered total):           {len(all_codes_filtered):>15,}")
print(f"Unique codes with ref ranges (filtered): {len(codes_with_ref):>15,}  ({len(codes_with_ref)/len(all_codes_filtered)*100:.1f}%)")
