#!/usr/bin/env bash

set -euo pipefail

# Fill in this one line.
DATASET_PATH="data/Vigil/tool_embedding_dataset.jsonl"

# Optional W&B defaults. Edit these if you want the script to run with W&B
# without needing extra environment variables every time.
WANDB_ENABLED="${WANDB_ENABLED:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-ntilc}"
WANDB_ENTITY="${WANDB_ENTITY:-andykr1k}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_TAGS="${WANDB_TAGS:-embedding-space,sweep}"
WANDB_NOTES="${WANDB_NOTES:-all losses, normal + hierarchical}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"

DATASET_STEM="${DATASET_PATH%.jsonl}"
DEFAULT_HIERARCHY_PATH="${DATASET_STEM}_hierarchy.json"
LEGACY_HEIRARCHY_PATH="${DATASET_STEM}_heirarchy.json"

if [[ -n "${HIERARCHY_PATH:-}" ]]; then
  RESOLVED_HIERARCHY_PATH="${HIERARCHY_PATH}"
elif [[ -f "${DEFAULT_HIERARCHY_PATH}" ]]; then
  RESOLVED_HIERARCHY_PATH="${DEFAULT_HIERARCHY_PATH}"
elif [[ -f "${LEGACY_HEIRARCHY_PATH}" ]]; then
  RESOLVED_HIERARCHY_PATH="${LEGACY_HEIRARCHY_PATH}"
else
  RESOLVED_HIERARCHY_PATH="${DEFAULT_HIERARCHY_PATH}"
fi

HIERARCHY_PATH="${RESOLVED_HIERARCHY_PATH}"
OUTPUT_DIR="${OUTPUT_DIR:-$(dirname "${DATASET_PATH}")/output}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_DATE="${RUN_DATE:-$(date +%F)}"
WANDB_GROUP_VALUE="${WANDB_GROUP:-embedding-sweep-${RUN_DATE}}"

LOSSES=(
  "prototype_ce"
  "contrastive"
  "circle"
)

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

if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "Dataset not found: ${DATASET_PATH}" >&2
  exit 1
fi

if [[ ! -f "${HIERARCHY_PATH}" ]]; then
  echo "Hierarchy mapping not found: ${HIERARCHY_PATH}" >&2
  echo "Expected a JSON file that maps tool_name -> parent_name." >&2
  exit 1
fi

echo "Root directory: ${ROOT_DIR}"
echo "Dataset: ${DATASET_PATH}"
echo "Hierarchy: ${HIERARCHY_PATH}"
echo "Output base: ${OUTPUT_DIR}"
if [[ "${WANDB_ENABLED}" == "1" ]]; then
  echo "W&B group: ${WANDB_GROUP_VALUE}"
fi

for loss in "${LOSSES[@]}"; do
  echo
  echo "=== Training normal embedding space with loss=${loss} ==="
  "${PYTHON_BIN}" -m training.train_embedding_space \
    --dataset-path "${DATASET_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --loss-type "${loss}" \
    "${WANDB_ARGS[@]}" \
    --wandb-run-name "normal-${loss}-${RUN_DATE}"
done

for loss in "${LOSSES[@]}"; do
  echo
  echo "=== Training hierarchical embedding space with loss=${loss} ==="
  "${PYTHON_BIN}" -m training.train_hierarchical_embedding_space \
    --dataset-path "${DATASET_PATH}" \
    --hierarchy-path "${HIERARCHY_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --loss-type "${loss}" \
    "${WANDB_ARGS[@]}" \
    --wandb-run-name "hierarchical-${loss}-${RUN_DATE}"
done

echo
echo "Finished training all embedding space variants."
