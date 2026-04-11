#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"

export DATASET_PATH="${DATASET_PATH:-registry/generated/tool_embedding_dataset.jsonl}"
export HIERARCHY_PATH="${HIERARCHY_PATH:-registry/generated/hierarchy.json}"
export OUTPUT_DIR="${OUTPUT_DIR:-output/registry_embeddings}"

echo "Training from registry snapshot"
echo "  DATASET_PATH=${DATASET_PATH}"
echo "  HIERARCHY_PATH=${HIERARCHY_PATH}"
echo "  OUTPUT_DIR=${OUTPUT_DIR}"

exec bash scripts/train_all_embedding_spaces.sh
