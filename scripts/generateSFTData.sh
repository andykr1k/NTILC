#!/usr/bin/env bash
set -euo pipefail

python utils/generate_protocol_sft_data.py \
  --input-data data/man/nl_command_pairs_flat_train_v2.jsonl \
  --output-dir data/protocol \
  --planner-max-rows 120000 \
  --planner-multi-step-ratio 0.35 \
  --planner-max-actions-per-plan 3
