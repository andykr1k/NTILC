#!/usr/bin/env bash

set -euo pipefail

# Input Directory
DATASET="API-Bank"
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

python utils/create_dataset.py --tools-path=${TOOLS_JSON_PATH} --output-path=${TOOL_EMBEDDING_DATASET_PATH} --summary-path=${TOOL_EMBEDDING_DATASET_SUMMARY_PATH} --examples-per-tool=${SPLIT_EXAMPLES_PER_TOOL} --tool-batch-size=${TOOL_BATCH_SIZE} --max-new-tokens=${MAX_NEW_TOKENS}
