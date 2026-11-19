#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="${SCRIPT_DIR}/ml_baselines.py"
OUTDIR="results/traditional_ml"
IMG_ROOT="dataset/ADNI/ADNI"

mkdir -p "${OUTDIR}"

# Metadata baselines (Age/Sex)
for MODEL in svm logreg random_forest; do
  "${PYTHON}" "${SCRIPT_DIR}/metadata_ml_baseline.py" \
    --model "${MODEL}" \
    --output-report "${OUTDIR}/metadata_${MODEL}.txt"
done

# Three-view image baselines
for MODEL in svm logreg random_forest; do
  "${PYTHON}" "${SCRIPT}" \
    --source image \
    --model "${MODEL}" \
    --image-root "${IMG_ROOT}" \
    --pca-components 256 \
    --output-report "${OUTDIR}/image_pca256_${MODEL}.txt"
done

# Single-slice image baselines
for MODEL in svm logreg random_forest; do
  "${PYTHON}" "${SCRIPT}" \
    --source image \
    --model "${MODEL}" \
    --image-root "${IMG_ROOT}" \
    --single-slice \
    --pca-components 256 \
    --output-report "${OUTDIR}/image_single_slice_pca256_${MODEL}.txt"
done

echo "All traditional ML baselines finished."
