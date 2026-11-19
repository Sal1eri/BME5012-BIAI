#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
SEEDS="${SEEDS:-42 43 44 45 46}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
MODELS="${MODELS:-resnet vgg16 hcct_vit vit_3d swin_3d_cnn mvst}"
GPUS="${GPUS:-3}"
PARALLEL="${PARALLEL:-9}"
OUTDIR="${OUTDIR:-results/runs}"

mkdir -p "${OUTDIR}" logs

run_one() {
  local model="$1"
  local seed="$2"
  local gpu="$3"
  local ckpt

  if [ "${seed}" = "42" ]; then
    ckpt="checkpoints/${model}/${model}_best.pth"
  else
    ckpt="checkpoints/${model}/${model}_seed${seed}.pth"
  fi

  "${PYTHON}" train.py \
    --model "${model}" \
    --seed "${seed}" \
    --device "cuda:${gpu}" \
    --num-threads 1 \
    --save-path "${ckpt}" \
    > "${OUTDIR}/${model}_seed${seed}_train.log" 2>&1

  "${PYTHON}" test.py \
    --model "${model}" \
    --checkpoint "${ckpt}" \
    --split test \
    --device "cuda:${gpu}" \
    > "${OUTDIR}/${model}_seed${seed}_test.log" 2>&1
}
export -f run_one
export PYTHON OUTDIR

jobs=()
i=0
for model in ${MODELS}; do
  for seed in ${SEEDS}; do
    jobs+=("${model} ${seed} $((i % GPUS))")
    i=$((i + 1))
  done
done

printf '%s\n' "${jobs[@]}" | xargs -P "${PARALLEL}" -n 3 bash -c 'set -euo pipefail; run_one "$@"' _
echo "All runs finished."
