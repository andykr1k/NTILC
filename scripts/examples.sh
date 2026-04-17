#!/usr/bin/env bash

set -euo pipefail

# Input Directory
DATASET="OSS"
SPLIT_EXAMPLES_PER_TOOL=20
SPLIT_TEST_PER_TOOL=4
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
BASE_PYTHON="/scratch4/home/akrik/base/bin/python"
# Create Tool Schema (If Needed)
# python utils/create_tool_schema.py --tools-path=${TOOLS_CSV_PATH} --output-path=${TOOLS_JSON_PATH}

# Create Tool Embedding Dataset (If Needed)
# python utils/create_dataset.py --tools-path=${TOOLS_JSON_PATH} --output-path=${TOOL_EMBEDDING_DATASET_PATH} --summary-path=${TOOL_EMBEDDING_DATASET_SUMMARY_PATH} --examples-per-tool=${SPLIT_EXAMPLES_PER_TOOL}

# Create Explicit Train/Test Split
# Requires at least ${SPLIT_EXAMPLES_PER_TOOL} examples per tool in ${TOOL_EMBEDDING_DATASET_PATH}.
# python scripts/split_tool_embedding_dataset.py \
#   --dataset-path=${TOOL_EMBEDDING_DATASET_PATH} \
#   --train-output-path=${TOOL_EMBEDDING_TRAIN_PATH} \
#   --test-output-path=${TOOL_EMBEDDING_TEST_PATH} \
#   --summary-path=${TOOL_EMBEDDING_SPLIT_SUMMARY_PATH} \
#   --examples-per-tool=${SPLIT_EXAMPLES_PER_TOOL} \
#   --test-per-tool=${SPLIT_TEST_PER_TOOL}

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
# ${BASE_PYTHON} -m benchmark.run_all \
#   --dataset-path=${TOOL_EMBEDDING_TEST_PATH} \
#   --tools-path=${TOOLS_JSON_PATH} \
#   --embedding-root=${OUTPUT_DIR} \
#   --output-root=${BENCHMARK_OUTPUT_ROOT} \
#   --run-name="${DATASET,,}-embedding-benchmark" \
#   --no-hybrid

# Benchmark Embeddings + Hybrid Qwen 3.5 27B Reranker + Frontier APIs
# Current strong OSS picks for tool selection / reranking:
#   Qwen/Qwen3.5-4B
#   Qwen/Qwen3.5-9B
#   Qwen/Qwen3.5-27B
#   Qwen/Qwen3.5-35B-A3B
#   moonshotai/Kimi-Linear-48B-A3B-Base
${BASE_PYTHON} -m benchmark.run_all \
  --dataset-path=${TOOL_EMBEDDING_TEST_PATH} \
  --tools-path=${TOOLS_JSON_PATH} \
  --embedding-root=${OUTPUT_DIR} \
  --output-root=${BENCHMARK_OUTPUT_ROOT} \
  --run-name="${BENCHMARK_RUN_NAME}" \
  --hf-device=cuda:0 \
  --hf-model="Qwen/Qwen3.5-4B" \
  --hf-model="Qwen/Qwen3.5-9B" \
  --hf-model="Qwen/Qwen3.5-27B" \
  --hf-model="Qwen/Qwen3.5-35B-A3B" \
  --hf-model="moonshotai/Kimi-Linear-48B-A3B-Base" \
  --hybrid-reranker-model="Qwen/Qwen3.5-27B" \
  --openai-model="gpt-5.2" \
  --anthropic-model="claude-opus-4-1-20250805" \
  --gemini-model="gemini-2.5-pro"