#!/usr/bin/env bash
set -e

SCRIPT="ml.py"
OUTDIR="results/traditional_ml"
IMG_ROOT="dataset/ADNI/ADNI"

mkdir -p "${OUTDIR}"

############################
# 2. IMAGE BASELINES (3-view)
############################

for MODEL in svm logreg random_forest; do
  python ${SCRIPT} \
    --source image \
    --model ${MODEL} \
    --image-root ${IMG_ROOT} \
    --pca-components 256 \
    --output-report ${OUTDIR}/image_pca256_${MODEL}.txt
done

############################
# 3. IMAGE BASELINES (single slice)
############################

for MODEL in svm logreg random_forest; do
  python ${SCRIPT} \
    --source image \
    --model ${MODEL} \
    --image-root ${IMG_ROOT} \
    --single-slice \
    --pca-components 256 \
    --output-report ${OUTDIR}/image_single_slice_pca256_${MODEL}.txt
done

echo "All traditional ML baselines finished."
