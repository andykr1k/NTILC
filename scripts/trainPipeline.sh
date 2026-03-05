#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3

LOG_DIR="logs/main"
mkdir -p "$LOG_DIR"

timestamp=$(date +"%Y%m%d_%H%M%S")

run_step () {
    local name="$1"
    shift

    log_file="${LOG_DIR}/${timestamp}_${name}.log"

    echo
    echo "=== Running: ${name} ==="
    echo "Log: ${log_file}"
    echo

    "$@" 2>&1 | tee "$log_file"
}

run_step "generate_pairs" \
    python -m utils.generate_man_nl_command_pairs

run_step "clean_pairs" \
    bash scripts/cleanPairs.sh

run_step "train_intent_embeddings" \
    bash scripts/trainIE.sh

run_step "train_cluster_retrieval" \
    bash scripts/trainCR.sh

run_step "train_lora" \
    bash scripts/trainLora.sh

echo
echo "Pipeline complete."