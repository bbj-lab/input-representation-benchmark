#!/bin/bash
# Input Representation Benchmark - Ventile Quantization Pipeline
set -e

cd "$(dirname "$0")/benchmarks/quantization-ventile"

echo ">>> MEDS Extraction"
bash scripts/01_extract_meds.sh 3

echo ">>> Standard Tokenization"
ethos_tokenize -m worker='range(0,3)' \
    input_dir=data/meds/data/train \
    output_dir=data/tokenized \
    out_fn=train dataset=mimic num_quantiles=20

echo ">>> Ventile Quantization"
python scripts/02_run_ventile_quantization.py

echo ">>> Analysis"
python scripts/03_analyze_ventiles.py

echo ">>> Testing"
python test_ventile_pipeline.py

echo "✓ Pipeline complete. Analysis: benchmarks/quantization-ventile/analysis/"
