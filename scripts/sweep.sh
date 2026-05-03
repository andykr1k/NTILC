#!/usr/bin/env bash

set -euo pipefail

# Input Directory
DATASET="${DATASET:-MetaTool}"
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
BENCHMARK_OUTPUT_ROOT="${BENCHMARK_OUTPUT_ROOT:-/scratch4/home/akrik/NTILC/benchmark/output}"
BENCHMARK_RUN_NAME="${DATASET,,}-functional-margin-compatibility-sweep"
BASE_PYTHON="${BASE_PYTHON:-/scratch4/home/akrik/base/bin/python}"
PYTHON_BIN="${PYTHON_BIN:-${BASE_PYTHON}}"
TRAIN_DEVICE="${TRAIN_DEVICE:-cuda:0}"
EMBEDDING_DEVICE="${EMBEDDING_DEVICE:-cuda:0}"
BENCHMARK_LIMIT="${BENCHMARK_LIMIT:-0}"
RANKING_LIMIT="${RANKING_LIMIT:-5}"
RUN_DATE="${RUN_DATE:-$(date +%F)}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"

WANDB_ENABLED="${WANDB_ENABLED:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-ntilc}"
WANDB_ENTITY="${WANDB_ENTITY:-andykr1k}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_TAGS="${WANDB_TAGS:-embedding-space,functional-margin,sweep}"
WANDB_NOTES="${WANDB_NOTES:-functional margin compatibility weight sweep}"
WANDB_GROUP_VALUE="${WANDB_GROUP:-functional-margin-compatibility-${DATASET}-${RUN_DATE}}"

DEFAULT_COMPATIBILITY_WEIGHTS="0.1 1 2 5 10 20 25 50 100 250 500 1000"
read -r -a COMPATIBILITY_WEIGHTS <<< "${COMPATIBILITY_WEIGHTS:-${DEFAULT_COMPATIBILITY_WEIGHTS}}"

weight_slug() {
  local value="$1"
  value="${value//./_}"
  value="${value//-/_neg_}"
  value="${value//+/_}"
  echo "${value}"
}

if [[ ! -f "${TOOL_EMBEDDING_DATASET_PATH}" ]]; then
  echo "Dataset not found: ${TOOL_EMBEDDING_DATASET_PATH}" >&2
  exit 1
fi

if [[ ! -f "${TOOL_EMBEDDING_TRAIN_PATH}" ]]; then
  echo "Train split not found: ${TOOL_EMBEDDING_TRAIN_PATH}" >&2
  exit 1
fi

if [[ ! -f "${TOOL_EMBEDDING_TEST_PATH}" ]]; then
  echo "Test split not found: ${TOOL_EMBEDDING_TEST_PATH}" >&2
  exit 1
fi

if [[ ! -f "${TOOLS_JSON_PATH}" ]]; then
  echo "Tools catalog not found: ${TOOLS_JSON_PATH}" >&2
  exit 1
fi

WANDB_ARGS=()
if [[ "${WANDB_ENABLED}" == "1" ]]; then
  WANDB_ARGS+=(
    "--wandb"
    "--wandb-group" "${WANDB_GROUP_VALUE}"
  )

  if [[ -n "${WANDB_PROJECT:-}" ]]; then
    WANDB_ARGS+=("--wandb-project" "${WANDB_PROJECT}")
  fi
  if [[ -n "${WANDB_ENTITY:-}" ]]; then
    WANDB_ARGS+=("--wandb-entity" "${WANDB_ENTITY}")
  fi
  if [[ -n "${WANDB_TAGS:-}" ]]; then
    WANDB_ARGS+=("--wandb-tags" "${WANDB_TAGS}")
  fi
  if [[ -n "${WANDB_NOTES:-}" ]]; then
    WANDB_ARGS+=("--wandb-notes" "${WANDB_NOTES}")
  fi
  if [[ -n "${WANDB_MODE:-}" ]]; then
    WANDB_ARGS+=("--wandb-mode" "${WANDB_MODE}")
  fi
fi

TRAINER_DATASET_ARGS=(
  "--dataset-path" "${TOOL_EMBEDDING_DATASET_PATH}"
  "--train-dataset-path" "${TOOL_EMBEDDING_TRAIN_PATH}"
  "--test-dataset-path" "${TOOL_EMBEDDING_TEST_PATH}"
  "--tools-path" "${TOOLS_JSON_PATH}"
)

BENCHMARK_LIMIT_ARGS=()
if [[ -n "${BENCHMARK_LIMIT}" && "${BENCHMARK_LIMIT}" != "0" ]]; then
  BENCHMARK_LIMIT_ARGS+=("--limit" "${BENCHMARK_LIMIT}")
fi

echo "Dataset: ${DATASET}"
echo "Train split: ${TOOL_EMBEDDING_TRAIN_PATH}"
echo "Test split: ${TOOL_EMBEDDING_TEST_PATH}"
echo "Training output base: ${OUTPUT_DIR}"
echo "Benchmark output: ${BENCHMARK_OUTPUT_ROOT}/${DATASET}/normal/functional_margin"
echo "Compatibility weights: ${COMPATIBILITY_WEIGHTS[*]}"
echo "Python: ${PYTHON_BIN}"
echo "Train device: ${TRAIN_DEVICE}"
echo "Embedding device: ${EMBEDDING_DEVICE}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
if [[ "${WANDB_ENABLED}" == "1" ]]; then
  echo "W&B group: ${WANDB_GROUP_VALUE}"
fi

for compatibility_weight in "${COMPATIBILITY_WEIGHTS[@]}"; do
  slug="$(weight_slug "${compatibility_weight}")"
  variant_name="compatibility_weight_${slug}"
  checkpoint_path="${OUTPUT_DIR}/normal/functional_margin/${variant_name}/best.pt"

  echo
  echo "=== Training normal functional_margin with compatibility_weight=${compatibility_weight} ==="
  "${PYTHON_BIN}" -m training.train_embedding_space \
    "${TRAINER_DATASET_ARGS[@]}" \
    --output-dir "${OUTPUT_DIR}" \
    --variant-name "${variant_name}" \
    --loss-type functional_margin \
    --compatibility-weight "${compatibility_weight}" \
    --device "${TRAIN_DEVICE}" \
    "${WANDB_ARGS[@]}" \
    --wandb-run-name "normal-functional_margin-${variant_name}-${RUN_DATE}"

  if [[ ! -f "${checkpoint_path}" ]]; then
    echo "Expected checkpoint was not created: ${checkpoint_path}" >&2
    exit 1
  fi

  echo
  echo "=== Benchmarking ${variant_name} ==="
  "${PYTHON_BIN}" -m benchmark.run_one \
    --dataset-path "${TOOL_EMBEDDING_TEST_PATH}" \
    --tools-path "${TOOLS_JSON_PATH}" \
    --checkpoint-path "${checkpoint_path}" \
    --output-root "${BENCHMARK_OUTPUT_ROOT}" \
    --dataset-name "${DATASET}" \
    --architecture normal \
    --loss-name functional_margin \
    --variant-name "${variant_name}" \
    --compatibility-weight "${compatibility_weight}" \
    --embedding-device "${EMBEDDING_DEVICE}" \
    --ranking-limit "${RANKING_LIMIT}" \
    "${BENCHMARK_LIMIT_ARGS[@]}"
done

echo
echo "Finished ${BENCHMARK_RUN_NAME}."
