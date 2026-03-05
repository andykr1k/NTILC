#!/usr/bin/env bash
set -euo pipefail

# Wrapper for protocol full pipeline.
bash scripts/pipelines/protocol/trainPipeline.sh "$@"
