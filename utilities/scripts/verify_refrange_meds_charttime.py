#!/usr/bin/env python3
"""MEDS-filtered ref-range stats: requires hadm_id, but uses storetime OR charttime (like MEDS pipeline)."""
import csv, gzip

LAB_FILE = "/gpfs/data/bbj-lab/users/daniel/input-representation-benchmark/physionet.org/files/mimiciv/3.1/hosp/labevents.csv.gz"

total_raw = 0
total_filtered = 0
rows_with_ref = 0
codes_with_ref = set()
all_codes_filtered = set()

with gzip.open(LAB_FILE, "rt", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        total_raw += 1
        hadm_id = row.get("hadm_id", "").strip()
        storetime = row.get("storetime", "").strip()
        charttime = row.get("charttime", "").strip()
        # MEDS filter: must have hadm_id AND (storetime OR charttime)
        has_time = bool(storetime) or bool(charttime)
        if not hadm_id or not has_time:
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
print(f"MEDS-filtered (hadm_id + time):          {total_filtered:>15,}")
print(f"Events with reference ranges:            {rows_with_ref:>15,}  ({rows_with_ref/total_filtered*100:.1f}%)")
print(f"Unique codes (filtered total):           {len(all_codes_filtered):>15,}")
print(f"Unique codes with ref ranges:            {len(codes_with_ref):>15,}  ({len(codes_with_ref)/len(all_codes_filtered)*100:.1f}%)")
