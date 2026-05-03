#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "/scratch4/home/akrik/base/bin/python" ]]; then
    PYTHON_BIN="/scratch4/home/akrik/base/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi
DATASET_GLOB="${DATASET_GLOB:-data/*/tool_embedding_dataset_train.jsonl}"
OUTPUT_PATH="${OUTPUT_PATH:-analysis/tool_blur_summary.json}"
export CUDA_VISIBLE_DEVICES=7
DEVICE="${DEVICE:-cuda:0}"

"${PYTHON_BIN}" -m analysis.tool_blur_pairs \
  --dataset-glob "${DATASET_GLOB}" \
  --output-path "${OUTPUT_PATH}" \
  --device "${DEVICE}" \
  "$@"
