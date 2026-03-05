#!/usr/bin/env bash
set -euo pipefail

# Edit this line to choose GPUs.
GPU_IDS="0,1,2,3"
IFS=',' read -r -a _gpu_array <<< "${GPU_IDS}"
NPROC_PER_NODE="${#_gpu_array[@]}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE}" \
    training/train_lora_nl_command.py \
    --train-data data/man/nl_command_pairs_flat_train_v2.jsonl \
    --mode full \
    --model-name Qwen/Qwen3.5-9B \
    --output-dir checkpoints/lora_nl_command_full \
    --gpu-ids "${GPU_IDS}"
else
  python training/train_lora_nl_command.py \
    --train-data data/man/nl_command_pairs_flat_train_v2.jsonl \
    --mode full \
    --model-name Qwen/Qwen3.5-9B \
    --output-dir checkpoints/lora_nl_command_full \
    --gpu-ids "${GPU_IDS}"
fi
