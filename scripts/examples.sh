#!/usr/bin/env bash

set -euo pipefail

# Input Directory
DATASET="ToolBench"
SPLIT_EXAMPLES_PER_TOOL=20
SPLIT_TEST_PER_TOOL=4
TOOL_BATCH_SIZE=4
MAX_NEW_TOKENS=2048
TOOLS_CSV_PATH="/scratch4/home/akrik/NTILC/data/${DATASET}/tools.csv"
TOOLS_JSON_PATH="/scratch4/home/akrik/NTILC/data/${DATASET}/tools.json"
TOOL_EMBEDDING_DATASET_PATH="/scratch4/home/akrik/NTILC/data/${DATASET}/tool_embedding_dataset.jsonl"
TOOL_EMBEDDING_DATASET_SUMMARY_PATH="/scratch4/home/akrik/NTILC/data/${DATASET}/tool_embedding_dataset_summary.json"
TOOL_EMBEDDING_TRAIN_PATH="/scratch4/home/akrik/NTILC/data/${DATASET}/tool_embedding_dataset_train.jsonl"
TOOL_EMBEDDING_TEST_PATH="/scratch4/home/akrik/NTILC/data/${DATASET}/tool_embedding_dataset_test.jsonl"
TOOL_EMBEDDING_SPLIT_SUMMARY_PATH="/scratch4/home/akrik/NTILC/data/${DATASET}/tool_embedding_dataset_split_summary.json"
HIERARCHY_PATH="/scratch4/home/akrik/NTILC/data/${DATASET}/tool_embedding_dataset_hierarchy.json"
BENCHMARK_PATH="/scratch4/home/akrik/NTILC/data/${DATASET}/benchmark.json"
OUTPUT_DIR="/scratch4/home/akrik/NTILC/data/${DATASET}/output"
BENCHMARK_OUTPUT_ROOT="/scratch4/home/akrik/NTILC/benchmark/output"
BENCHMARK_RUN_NAME="${DATASET,,}-full-benchmark"
BASE_PYTHON="/scratch4/home/akrik/base/bin/python"
export CUDA_VISIBLE_DEVICES=7
# Create Tool Schema (If Needed)
# python utils/create_tool_schemas.py --tools-path=${TOOLS_CSV_PATH} --output-path=${TOOLS_JSON_PATH}

# Create Tool Embedding Dataset (If Needed)
# python utils/create_dataset.py --tools-path=${TOOLS_JSON_PATH} --output-path=${TOOL_EMBEDDING_DATASET_PATH} --summary-path=${TOOL_EMBEDDING_DATASET_SUMMARY_PATH} --examples-per-tool=${SPLIT_EXAMPLES_PER_TOOL} --tool-batch-size=${TOOL_BATCH_SIZE} --max-new-tokens=${MAX_NEW_TOKENS}

# Create Explicit Train/Test Split
# Requires at least ${SPLIT_EXAMPLES_PER_TOOL} examples per tool in ${TOOL_EMBEDDING_DATASET_PATH}.
python scripts/split_tool_embedding_dataset.py \
  --dataset-path=${TOOL_EMBEDDING_DATASET_PATH} \
  --train-output-path=${TOOL_EMBEDDING_TRAIN_PATH} \
  --test-output-path=${TOOL_EMBEDDING_TEST_PATH} \
  --summary-path=${TOOL_EMBEDDING_SPLIT_SUMMARY_PATH} \
  --examples-per-tool=${SPLIT_EXAMPLES_PER_TOOL} \
  --test-per-tool=${SPLIT_TEST_PER_TOOL}

# Train One Normal Embedding Variant With Explicit Train/Test Split
# python -m training.train_embedding_space \
#   --dataset-path=${TOOL_EMBEDDING_DATASET_PATH} \
#   --train-dataset-path=${TOOL_EMBEDDING_TRAIN_PATH} \
#   --test-dataset-path=${TOOL_EMBEDDING_TEST_PATH} \
#   --output-dir=${OUTPUT_DIR} \
#   --loss-type=functional_margin

# Train One Hierarchical Embedding Variant With Explicit Train/Test Split
# python -m training.train_hierarchical_embedding_space \
#   --dataset-path=${TOOL_EMBEDDING_DATASET_PATH} \
#   --train-dataset-path=${TOOL_EMBEDDING_TRAIN_PATH} \
#   --test-dataset-path=${TOOL_EMBEDDING_TEST_PATH} \
#   --hierarchy-path=${HIERARCHY_PATH} \
#   --output-dir=${OUTPUT_DIR} \
#   --loss-type=functional_margin

# Train All Normal + Hierarchical Variants With Explicit Train/Test Split
# DATASET_PATH=${TOOL_EMBEDDING_DATASET_PATH} \
# TRAIN_DATASET_PATH=${TOOL_EMBEDDING_TRAIN_PATH} \
# TEST_DATASET_PATH=${TOOL_EMBEDDING_TEST_PATH} \
# HIERARCHY_PATH=${HIERARCHY_PATH} \
# OUTPUT_DIR=${OUTPUT_DIR} \
# bash scripts/train_all_embedding_spaces.sh

# Evaluate ToolCall15 Agent Pipeline
# python evals/ToolCall15/runEval.py \
#   --checkpoint-root="/scratch4/home/akrik/NTILC/data/ToolCall15/output" \
#   --checkpoint-filename="best.pt" \
#   --benchmark-path="/scratch4/home/akrik/NTILC/data/ToolCall15/benchmark.json" \
#   --tools-path="/scratch4/home/akrik/NTILC/data/ToolCall15/tools.json" \
#   --output-path="/scratch4/home/akrik/NTILC/data/ToolCall15/output/eval/eval_summary.json"

# Benchmark All Embedding Spaces On The Held-Out Test Split
# Use ${BASE_PYTHON} so the benchmark runner has access to torch/transformers.
# The benchmark runner auto-loads API keys from /scratch4/home/akrik/NTILC/.env by default.
# Start from: cp /scratch4/home/akrik/NTILC/.env.example /scratch4/home/akrik/NTILC/.env
# python -m benchmark.run_all \
#   --dataset-path=${TOOL_EMBEDDING_TEST_PATH} \
#   --tools-path=${TOOLS_JSON_PATH} \
#   --embedding-root=${OUTPUT_DIR} \
#   --output-root=${BENCHMARK_OUTPUT_ROOT} \
#   --run-name="${DATASET,,}-embedding-benchmark" \
#   --no-hybrid

# Benchmark Embeddings + Hybrid Qwen 3.5 27B Reranker + Frontier APIs
# python -m benchmark.run_all \
#   --dataset-path=${TOOL_EMBEDDING_TEST_PATH} \
#   --tools-path=${TOOLS_JSON_PATH} \
#   --embedding-root=${OUTPUT_DIR} \
#   --output-root=${BENCHMARK_OUTPUT_ROOT} \
#   --run-name="${BENCHMARK_RUN_NAME}" \
#   --hf-device=auto \
#   --hf-model="Qwen/Qwen3.5-27B" \
#   --hybrid-reranker-model="Qwen/Qwen3.5-27B" \
#   --openai-model="gpt-5.2" \
#   --anthropic-model="claude-opus-4-1-20250805" \
#   --gemini-model="gemini-2.5-pro"
