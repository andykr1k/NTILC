#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${GPU_IDS:-0,1,2,3}"

LOG_DIR="logs/protocol"
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

run_step "generate_protocol_sft" \
    bash scripts/generateSFTData.sh

run_step "train_lora_protocol" \
    bash scripts/trainLoraProtocol.sh

echo
echo "Protocol pipeline complete."
