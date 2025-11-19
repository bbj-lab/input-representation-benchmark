#!/bin/bash
set -e

# Wrapper for ethos-ares MEDS extraction with custom event config

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BENCHMARK_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$(dirname "$BENCHMARK_DIR")")"
# ethos-ares repo located as sibling to input-representation-benchmark
ETHOS_DIR="$(dirname "$PROJECT_ROOT")/ethos-ares"

N_WORKERS=${1:-3}

# Set paths
export MIMICIV_RAW_DIR="${PROJECT_ROOT}/physionet.org/files/mimiciv/3.1"
export MIMICIV_PRE_MEDS_DIR="${BENCHMARK_DIR}/data/pre_meds"
export MIMICIV_MEDS_COHORT_DIR="${BENCHMARK_DIR}/data/meds"
export EVENT_CONVERSION_CONFIG_FP="${BENCHMARK_DIR}/configs/event_configs_v3.1.yaml"
export N_WORKERS="${N_WORKERS}"

echo "MEDS Extraction with Reference Ranges"
echo "--------------------------------------"
echo "MIMIC-IV: ${MIMICIV_RAW_DIR}"
echo "Output: ${MIMICIV_MEDS_COHORT_DIR}"
echo "Event Config: ${EVENT_CONVERSION_CONFIG_FP}"
echo "Workers: ${N_WORKERS}"
echo ""

# Create output directories
mkdir -p "${MIMICIV_PRE_MEDS_DIR}"
mkdir -p "${MIMICIV_MEDS_COHORT_DIR}"

# Run ethos-ares MEDS extraction
if [ -d "${ETHOS_DIR}/scripts/meds" ]; then
    cd "${ETHOS_DIR}/scripts/meds"
    bash run_mimic.sh \
        "${MIMICIV_RAW_DIR}" \
        "${MIMICIV_PRE_MEDS_DIR}" \
        "${MIMICIV_MEDS_COHORT_DIR}" \
        ""  # No extension (base MIMIC-IV)
else
    echo "Error: ethos-ares scripts not found at ${ETHOS_DIR}/scripts/meds"
    echo "Please ensure ethos-ares is cloned at ../ethos-ares"
    exit 1
fi

