#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

python -m evaluation.run_tool_calling_eval \
  --clean-data data/man/nl_command_pairs_flat_clean_v2.json \
  --train-data data/man/nl_command_pairs_flat_train_v2.json \
  --raw-tools-json data/man/raw_ai.json \
  --intent-embedder-path checkpoints/intent_embedder/best_model.pt \
  --query-encoder-path checkpoints/cluster_retrieval/best_model.pt \
  --lora-adapter-path checkpoints/lora_nl_command_full \
  --qwen-model Qwen/Qwen3.5-9B \
  --baseline-device cuda:0 \
  --ntilc-device cuda:1 \
  --num-samples 25 \
  "$@"
