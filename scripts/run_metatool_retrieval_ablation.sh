#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "/scratch4/home/akrik/base/bin/python" ]]; then
    PYTHON_BIN="/scratch4/home/akrik/base/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
DATASET_ROOT="${DATASET_ROOT:-${ROOT_DIR}/data/MetaTool}"
ABLATION_ROOT="${ABLATION_ROOT:-${ROOT_DIR}/data/ablations/MetaTool}"
MODEL_ROOT="${MODEL_ROOT:-${ABLATION_ROOT}/models}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ABLATION_ROOT}/outputs}"
TRAIN_MISSING="${TRAIN_MISSING:-1}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"
SKIP_DENSE="${SKIP_DENSE:-0}"
SKIP_CHECKPOINTS="${SKIP_CHECKPOINTS:-0}"
DENSE_LOCAL_FILES_ONLY="${DENSE_LOCAL_FILES_ONLY:-0}"
QWEN_EMBEDDING_MODEL="${QWEN_EMBEDDING_MODEL:-Qwen/Qwen3-Embedding-8B}"
QWEN_DISPLAY_NAME="${QWEN_DISPLAY_NAME:-Qwen3-Embedding-8B}"
DENSE_QUERY_INSTRUCTION="${DENSE_QUERY_INSTRUCTION:-Given a user request, retrieve the tool schema that should handle it.}"
TRAIN_DEVICE="${TRAIN_DEVICE:-cuda:0}"
EMBEDDING_DEVICE="${EMBEDDING_DEVICE:-cuda:0}"
DENSE_DEVICE="${DENSE_DEVICE:-cuda:0}"
DENSE_DTYPE="${DENSE_DTYPE:-auto}"
EPOCHS="${EPOCHS:-25}"
BATCH_SIZE="${BATCH_SIZE:-32}"
DENSE_BATCH_SIZE="${DENSE_BATCH_SIZE:-8}"
DENSE_MAX_LENGTH="${DENSE_MAX_LENGTH:-8192}"
LIMIT="${LIMIT:-0}"

args=(
  -m benchmark.retrieval_ablation
  --dataset-root "${DATASET_ROOT}"
  --dataset-path "${DATASET_ROOT}/tool_embedding_dataset.jsonl"
  --train-dataset-path "${DATASET_ROOT}/tool_embedding_dataset_train.jsonl"
  --test-dataset-path "${DATASET_ROOT}/tool_embedding_dataset_test.jsonl"
  --tools-path "${DATASET_ROOT}/tools.json"
  --ablation-root "${ABLATION_ROOT}"
  --model-root "${MODEL_ROOT}"
  --output-root "${OUTPUT_ROOT}"
  --qwen-embedding-model "${QWEN_EMBEDDING_MODEL}"
  --qwen-display-name "${QWEN_DISPLAY_NAME}"
  --dense-query-instruction "${DENSE_QUERY_INSTRUCTION}"
  --train-device "${TRAIN_DEVICE}"
  --embedding-device "${EMBEDDING_DEVICE}"
  --dense-device "${DENSE_DEVICE}"
  --dense-dtype "${DENSE_DTYPE}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --dense-batch-size "${DENSE_BATCH_SIZE}"
  --dense-max-length "${DENSE_MAX_LENGTH}"
  --limit "${LIMIT}"
)

if [[ "${TRAIN_MISSING}" == "1" ]]; then
  args+=(--train-missing)
fi
if [[ "${FORCE_TRAIN}" == "1" ]]; then
  args+=(--force-train)
fi
if [[ "${SKIP_DENSE}" == "1" ]]; then
  args+=(--skip-dense)
fi
if [[ "${SKIP_CHECKPOINTS}" == "1" ]]; then
  args+=(--skip-checkpoints)
fi
if [[ "${DENSE_LOCAL_FILES_ONLY}" == "1" ]]; then
  args+=(--dense-local-files-only)
fi

"${PYTHON_BIN}" "${args[@]}" "$@"
