#!/usr/bin/env bash
set -euo pipefail

echo "[1/4] Cleaning NL-command pairs"
bash scripts/cleanPairs.sh

echo "[2/4] Training intent embeddings (Phase 1)"
bash scripts/trainIE.sh

echo "[3/4] Training cluster retrieval (Phase 2)"
bash scripts/trainCR.sh

echo "[4/4] Training LoRA command model"
bash scripts/trainLora.sh

echo "Pipeline complete."
