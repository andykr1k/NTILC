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

DATASETS="${DATASETS:-ToolBench BFCL API-Bank MetaTool ToolEyes}"
METHODS="${METHODS:-qwen_ict openai_ict gemini_ict anthropic_ict ntilc}"
RUN_NAME="${RUN_NAME:-main-inference-time-comparison}"
OUTPUT_ROOT="${OUTPUT_ROOT:-benchmark/output}"
DATA_ROOT="${DATA_ROOT:-data}"
LIMIT="${LIMIT:-100}"
RANKING_LIMIT="${RANKING_LIMIT:-5}"

QWEN_MODEL="${QWEN_MODEL:-Qwen/Qwen3.5-27B}"
QWEN_LABEL="${QWEN_LABEL:-Qwen3-27B (ICT)}"
OPENAI_MODEL="${OPENAI_MODEL:-gpt-5.2-2025-12-11}"
OPENAI_LABEL="${OPENAI_LABEL:-ChatGPT 5 (ICT)}"
GEMINI_MODEL="${GEMINI_MODEL:-gemini-2.5-flash}"
GEMINI_LABEL="${GEMINI_LABEL:-Gemini 2.5 Flash (ICT)}"
ANTHROPIC_MODEL="${ANTHROPIC_MODEL:-claude-sonnet-4-6}"
ANTHROPIC_LABEL="${ANTHROPIC_LABEL:-Claude Sonnet 4.6 (ICT)}"
NTILC_LABEL="${NTILC_LABEL:-NTILC}"
NTILC_CHECKPOINT_GLOB="${NTILC_CHECKPOINT_GLOB:-output/normal/functional_margin/**/best.pt}"

HF_DEVICE="${HF_DEVICE:-cuda:7}"
HF_DTYPE="${HF_DTYPE:-auto}"
HF_MAX_NEW_TOKENS="${HF_MAX_NEW_TOKENS:-160}"
EMBEDDING_DEVICE="${EMBEDDING_DEVICE:-cuda:7}"
API_MAX_OUTPUT_TOKENS="${API_MAX_OUTPUT_TOKENS:-0}"
API_TIMEOUT_SECONDS="${API_TIMEOUT_SECONDS:-120}"
API_PARALLEL_WORKERS="${API_PARALLEL_WORKERS:-3}"
REGISTRY_TOKENIZER_MODEL="${REGISTRY_TOKENIZER_MODEL:-${QWEN_MODEL}}"
DOTENV_PATH="${DOTENV_PATH:-.env}"

read -r -a DATASET_ARRAY <<< "${DATASETS}"
read -r -a METHOD_ARRAY <<< "${METHODS}"

ARGS=(
  -m benchmark.main_inference_time_comparison
  --data-root "${DATA_ROOT}"
  --datasets "${DATASET_ARRAY[@]}"
  --methods "${METHOD_ARRAY[@]}"
  --output-root "${OUTPUT_ROOT}"
  --run-name "${RUN_NAME}"
  --dotenv-path "${DOTENV_PATH}"
  --limit "${LIMIT}"
  --ranking-limit "${RANKING_LIMIT}"
  --qwen-model "${QWEN_MODEL}"
  --qwen-label "${QWEN_LABEL}"
  --hf-device "${HF_DEVICE}"
  --hf-dtype "${HF_DTYPE}"
  --hf-max-new-tokens "${HF_MAX_NEW_TOKENS}"
  --openai-model "${OPENAI_MODEL}"
  --openai-label "${OPENAI_LABEL}"
  --gemini-model "${GEMINI_MODEL}"
  --gemini-label "${GEMINI_LABEL}"
  --anthropic-model "${ANTHROPIC_MODEL}"
  --anthropic-label "${ANTHROPIC_LABEL}"
  --api-max-output-tokens "${API_MAX_OUTPUT_TOKENS}"
  --api-timeout-seconds "${API_TIMEOUT_SECONDS}"
  --api-parallel-workers "${API_PARALLEL_WORKERS}"
  --ntilc-label "${NTILC_LABEL}"
  --ntilc-checkpoint-glob "${NTILC_CHECKPOINT_GLOB}"
  --embedding-device "${EMBEDDING_DEVICE}"
  --registry-tokenizer-model "${REGISTRY_TOKENIZER_MODEL}"
)

if [[ "${HF_LOCAL_FILES_ONLY:-0}" == "1" ]]; then
  ARGS+=(--hf-local-files-only)
fi

if [[ "${REGISTRY_TOKENIZER_LOCAL_FILES_ONLY:-0}" == "1" ]]; then
  ARGS+=(--registry-tokenizer-local-files-only)
fi

if [[ "${SKIP_EXISTING:-0}" == "1" ]]; then
  ARGS+=(--skip-existing)
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  ARGS+=(--dry-run)
fi

echo "Root directory: ${ROOT_DIR}"
echo "Python: ${PYTHON_BIN}"
echo "Datasets: ${DATASET_ARRAY[*]}"
echo "Methods: ${METHOD_ARRAY[*]}"
echo "Run name: ${RUN_NAME}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Limit per dataset: ${LIMIT}"

"${PYTHON_BIN}" "${ARGS[@]}" "$@"
