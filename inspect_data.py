
import polars as pl
import sys

def inspect(path, f):
    f.write(f"--- Inspecting {path} ---\n")
    try:
        df = pl.scan_parquet(path)
        f.write(f"Schema: {df.collect_schema()}\n")
        f.write(f"First 5 rows: {df.limit(5).collect()}\n")
    except Exception as e:
        f.write(f"Error: {e}\n")

if __name__ == "__main__":
    with open("inspect_schema.log", "w") as f:
        inspect("data/clif/raw/test/clif_hospitalization.parquet", f)
        inspect("data/clif/raw/test/clif_labs.parquet", f)
