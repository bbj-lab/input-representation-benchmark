#!/bin/bash

# =============================================================================
# Provenance: Quantifying-Surprise-EHRs/slurm/10_xfer_rep_based_preds.sh
# Source: /gpfs/data/bbj-lab/users/daniel/Quantifying-Surprise-EHRs/slurm/10_xfer_rep_based_preds.sh
# =============================================================================
#
# Vendored to preserve the LR evaluation semantics and flags.
#
# - single-dataset mode: use same dir for orig/new (we're not doing mimic→ucmc transfer here)
# - scale CPU resources via a dedicated tier2q runner (outside this script)
# - jobfile sets: data_dir, data_version, model_loc
#
# =============================================================================

#SBATCH --job-name=xfer-rep-preds
#SBATCH --output=./slurm/output/%A_%a-%x.stdout
#SBATCH --partition=tier2q
#SBATCH --mem=10GB
#SBATCH --time=1:00:00

set -euo pipefail

source slurm/ref_qse/preamble.sh

: "${FMS_EHRS_HOME:?Set FMS_EHRS_HOME (sibling fms-ehrs repo)}"

# Expect these from the jobfile command
: "${data_dir:?Set data_dir (tokenized data root)}"
: "${data_version:?Set data_version (should end with _first_24h)}"
: "${model_loc:?Set model_loc (HF model directory)}"
IRB_PREDS_TAG="${IRB_PREDS_TAG:-base-primary_binary}"

echo "Generating representation-based predictions..."
python3 "${FMS_EHRS_HOME}/fms_ehrs/scripts/transfer_rep_based_preds.py" \
  --data_dir_orig "${data_dir}" \
  --data_dir_new "${data_dir}" \
  --data_version "${data_version}" \
  --model_loc "${model_loc}" \
  --classifier logistic_regression \
  --preds_tag "${IRB_PREDS_TAG}" \
  --save_preds \
  ${IRB_XFER_PRED_ARGS:-}

