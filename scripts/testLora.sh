#!/usr/bin/env bash
set -euo pipefail

python testing/test_lora_nl_command.py \
  --base-model Qwen/Qwen3.5-9B \
  --adapter-path checkpoints/lora_nl_command_full \
  --mode full \
  --eval-data data/man/nl_command_pairs_flat_clean_v2.jsonl \
  --num-samples 500
