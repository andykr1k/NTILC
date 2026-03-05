#!/usr/bin/env bash
set -euo pipefail

# Edit this line to choose GPUs.
GPU_IDS="${GPU_IDS:-0,1,2,3}"
IFS=',' read -r -a _gpu_array <<< "${GPU_IDS}"
NPROC_PER_NODE="${#_gpu_array[@]}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
EXTRA_ARGS=("$@")

if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE}" \
    training/train_intent_embedding.py \
    --data-path data/man/nl_command_pairs_flat_train_v2.jsonl \
    --val-ratio 0.1 \
    --gpu-ids "${GPU_IDS}" \
    "${EXTRA_ARGS[@]}"
else
  python training/train_intent_embedding.py \
    --data-path data/man/nl_command_pairs_flat_train_v2.jsonl \
    --val-ratio 0.1 \
    --device cuda:0 \
    --gpu-ids "${GPU_IDS}" \
    "${EXTRA_ARGS[@]}"
fi
