import polars as pl
from pathlib import Path
import sys

def analyze_non_numeric_labs():
    # Define path to labevents
    # Assuming script is run from project root or benchmarks/ventile-quantization
    # Try to find the file relative to the script location
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent.parent
    labevents_path = project_root / "physionet.org/files/mimiciv/3.1/hosp/labevents.csv.gz"
    
    if not labevents_path.exists():
        print(f"Error: labevents.csv.gz not found at {labevents_path}")
        sys.exit(1)
        
    print(f"Reading {labevents_path}...")
    
    # Scan the file (lazy evaluation)
    q = pl.scan_csv(labevents_path)
    
    # Filter for non-numeric values (valuenum is null)
    # We are interested in what kind of text values exist when valuenum is null
    non_numeric_q = q.filter(pl.col("valuenum").is_null())
    
    # Count total rows
    total_rows = q.select(pl.len()).collect().item()
    non_numeric_rows = non_numeric_q.select(pl.len()).collect().item()
    
    print(f"Total Lab Events: {total_rows:,}")
    print(f"Non-Numeric Events: {non_numeric_rows:,} ({non_numeric_rows/total_rows*100:.2f}%)")
    
    # Analyze the 'value' column for non-numeric entries
    print("\nTop 20 Non-Numeric Values:")
    value_counts = (
        non_numeric_q
        .group_by("value")
        .len()
        .sort("len", descending=True)
        .limit(20)
        .collect()
    )
    
    print(value_counts)
    
    # Analyze by itemid (lab code)
    print("\nTop 20 Lab Codes with Non-Numeric Values:")
    code_counts = (
        non_numeric_q
        .group_by("itemid")
        .len()
        .sort("len", descending=True)
        .limit(20)
        .collect()
    )
    
    # Join with d_labitems to get names if possible
    d_labitems_path = project_root / "physionet.org/files/mimiciv/3.1/hosp/d_labitems.csv.gz"
    if d_labitems_path.exists():
        d_labitems = pl.read_csv(d_labitems_path)
        code_counts = code_counts.join(d_labitems, on="itemid", how="left")
        print(code_counts.select(["itemid", "label", "len"]))
    else:
        print(code_counts)

    # Categorize types of non-numeric values
    print("\nCategorization of Non-Numeric Values:")
    
    # Define some categories
    categories = [
        ("Specimen/Source", ["ART.", "VEN.", "CAPILLARY", "URINE", "BLOOD"]),
        ("Comparison", ["GREATER THAN", "LESS THAN", ">", "<"]),
        ("Textual", ["NEG", "POS", "NONE", "NORMAL", "ABNORMAL", "SEE COMMENT"]),
        ("Null/Empty", ["", "___", "ERROR"]),
    ]
    
    # We can't easily regex everything in SQL/Lazy, so let's take a sample or just describe based on top counts
    # For this script, the top counts are sufficient to document "what kinds of data types or values we have"
    
    # Check for specific patterns mentioned in the prompt
    print("\nSpecific Patterns Check:")
    patterns = [
        "HOLD", "DISCARD", "ERROR", "CANCELLED"
    ]
    
    for pat in patterns:
        count = non_numeric_q.filter(pl.col("value").str.contains(pat)).select(pl.len()).collect().item()
        print(f"Contains '{pat}': {count:,}")

if __name__ == "__main__":
    analyze_non_numeric_labs()

