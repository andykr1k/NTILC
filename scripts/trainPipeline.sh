#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible wrapper for legacy full pipeline.
bash scripts/pipelines/legacy/trainPipeline.sh "$@"
