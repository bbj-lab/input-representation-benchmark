#!/usr/bin/env python3
"""Run custom QuantizationVentile on existing tokenized data.

This script applies our reference range-aware ventile quantization to
data that has already been preprocessed by the standard ethos pipeline.

Usage:
    python 03_run_ventile_quantization.py
"""

import sys
from pathlib import Path

# Add input_representation to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from input_representation.tokenize.common.quantization import QuantizationVentile
from loguru import logger


def main():
    """Run QuantizationVentile on preprocessed data."""
    
    experiment_dir = Path(__file__).parent.parent
    
    # Input: Stage 04 output (after make_quantiles creates LAB//Q// codes with ref_range columns)
    # We assume this script is run from benchmarks/ventile-quantization/
    # and data is in data/tokenized/train/
    
    # Adjust path relative to CWD (benchmarks/ventile-quantization/)
    # cwd = Path.cwd()
    # input_dir = cwd / "data/tokenized/train/04_preprocessing_with_counts"
    # output_file = cwd / "data/tokenized/train/ventile_breaks.json"
    
    # Use path relative to this script to be robust
    input_dir = experiment_dir / "data/tokenized/train/04_preprocessing_with_counts"
    output_file = experiment_dir / "data/tokenized/train/ventile_breaks.json"
    
    # Get all input files
    input_files = sorted(input_dir.glob("*.parquet"))
    
    if not input_files:
        logger.error(f"No input files found in {input_dir}")
        logger.error("Run standard tokenization up to stage 06 first")
        return 1
    
    logger.info(f"Found {len(input_files)} input files")
    logger.info(f"Running QuantizationVentile with reference range-aware binning")
    
    # Run QuantizationVentile with code prefixes
    quantizator = QuantizationVentile
    
    # Filter files to only include codes we want to quantize
    code_prefixes = ["LAB//Q//", "BMI//Q", "VITAL//Q//", "SUBJECT_FLUID_OUTPUT//Q//"]
    
    logger.info(f"Filtering to code prefixes: {code_prefixes}")
    
    try:
        quantizator.agg(
            in_fps=input_files,
            out_fp=output_file,
            num_quantiles=20,
            code_prefixes=code_prefixes,
        )
        logger.info(f"SUCCESS: Ventile breaks saved to {output_file}")
        logger.info("All labs should now have exactly 20 bins (with or without ref ranges)")
        return 0
    except Exception as e:
        logger.error(f"Error running QuantizationVentile: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

