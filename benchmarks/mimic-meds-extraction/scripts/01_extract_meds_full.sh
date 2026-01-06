#!/bin/bash
set -e

# =============================================================================
# MEDS Extraction with Full ICU Events (chartevents, inputevents, outputevents)
# =============================================================================
# This script extracts MEDS data including:
#   - All hosp module tables (same as quantization-ventile)
#   - icu/icustays
#   - icu/chartevents (vital signs)
#   - icu/inputevents (infusions)
#   - icu/outputevents (fluid outputs)
#   - icu/procedureevents (ICU procedures)
#
# The MEDS extraction pipeline is adapted from ETHOS-ARES:
#   Repository: https://github.com/ipolharvard/ethos-ares
#   Citation: Renc, P., et al. (2025). Foundation Model of Electronic Medical
#             Records for Adaptive Risk Estimation. GigaScience, 14, giaf107.
#   License: MIT (Copyright (c) 2024 Paweł Renc)
#
# =============================================================================

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BENCHMARK_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$(dirname "$BENCHMARK_DIR")")"

# MEDS pipeline (now local to this repository)
MEDS_PIPELINE_DIR="${BENCHMARK_DIR}/meds_pipeline"

N_WORKERS=${1:-3}

# Set paths
export MIMICIV_RAW_DIR="${PROJECT_ROOT}/physionet.org/files/mimiciv/3.1"
export MIMICIV_PRE_MEDS_DIR="${BENCHMARK_DIR}/data/pre_meds"
export MIMICIV_MEDS_COHORT_DIR="${BENCHMARK_DIR}/data/meds"

# CRITICAL: Use OUR custom event config with ICU events enabled and storetime semantics
export EVENT_CONVERSION_CONFIG_FP="${BENCHMARK_DIR}/configs/event_configs_v3.1_full.yaml"

# Pipeline config (now local)
export PIPELINE_CONFIG_FP="${MEDS_PIPELINE_DIR}/configs/extract_MIMIC.yaml"
export PRE_MEDS_PY_FP="${MEDS_PIPELINE_DIR}/pre_MEDS.py"
export N_WORKERS="${N_WORKERS}"

echo "=============================================="
echo "MEDS Extraction - FULL ICU Events Version"
echo "=============================================="
echo ""
echo "MEDS Pipeline Attribution:"
echo "  Source: ETHOS-ARES (https://github.com/ipolharvard/ethos-ares)"
echo "  License: MIT (Copyright (c) 2024 Paweł Renc)"
echo "  Citation: Renc, P., et al. (2025). GigaScience, 14, giaf107."
echo ""
echo "Configuration:"
echo "  MIMIC-IV Source:    ${MIMICIV_RAW_DIR}"
echo "  Pre-MEDS Output:    ${MIMICIV_PRE_MEDS_DIR}"
echo "  MEDS Output:        ${MIMICIV_MEDS_COHORT_DIR}"
echo "  Event Config:       ${EVENT_CONVERSION_CONFIG_FP}"
echo "  Pipeline Config:    ${PIPELINE_CONFIG_FP}"
echo "  Workers:            ${N_WORKERS}"
echo ""
echo "Included ICU Tables:"
echo "  - icu/icustays"
echo "  - icu/chartevents (VITAL signs)"
echo "  - icu/inputevents (INFUSION_START/END)"
echo "  - icu/outputevents (FLUID_OUTPUT)"
echo "  - icu/procedureevents (PROCEDURE events)"
echo ""
echo "Key Features:"
echo "  - Uses 'storetime' instead of 'charttime' for event ordering"
echo "  - Excludes post-discharge billing codes (ICD, CPT, DRG)"
echo "  - 70/10/20 train/val/test split (patient-level)"
echo ""

# Create output directories
mkdir -p "${MIMICIV_PRE_MEDS_DIR}"
mkdir -p "${MIMICIV_MEDS_COHORT_DIR}"

# Check required files
if [ ! -f "${EVENT_CONVERSION_CONFIG_FP}" ]; then
    echo "Error: Event config not found at ${EVENT_CONVERSION_CONFIG_FP}"
    exit 1
fi

if [ ! -f "${PIPELINE_CONFIG_FP}" ]; then
    echo "Error: Pipeline config not found at ${PIPELINE_CONFIG_FP}"
    exit 1
fi

if [ ! -f "${PRE_MEDS_PY_FP}" ]; then
    echo "Error: pre_MEDS.py not found at ${PRE_MEDS_PY_FP}"
    exit 1
fi

echo "Unsetting SLURM_CPU_BIND in case you're running this on a slurm interactive node with slurm parallelism"
unset SLURM_CPU_BIND

# =============================================================================
# Step 1: Run pre-MEDS conversion
# =============================================================================
echo ""
echo "=== Step 1: Running pre-MEDS conversion ==="
python "$PRE_MEDS_PY_FP" input_dir="$MIMICIV_RAW_DIR" cohort_dir="$MIMICIV_PRE_MEDS_DIR"

# =============================================================================
# Step 2: Run MEDS extraction pipeline
# =============================================================================
echo ""
echo "=== Step 2: Running MEDS extraction pipeline ==="
echo "Using custom event config: ${EVENT_CONVERSION_CONFIG_FP}"

# Change to meds_pipeline directory for proper relative path resolution
cd "${MEDS_PIPELINE_DIR}"

MEDS_transform-pipeline "pipeline_config_fp=$PIPELINE_CONFIG_FP" \
    stage_runner_fp=configs/local_parallelism_runner.yaml

echo ""
echo "=============================================="
echo "MEDS Extraction Complete!"
echo "=============================================="
echo "Output: ${MIMICIV_MEDS_COHORT_DIR}"
