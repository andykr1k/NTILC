#!/usr/bin/env bash
set -euo pipefail

python utils/clean_nl_command_pairs.py \
  --input-json data/man/nl_command_pairs.json \
  --raw-tools-json data/man/raw_ai.json \
  --strict-first-token --strict-flags --dedupe-tool-command
