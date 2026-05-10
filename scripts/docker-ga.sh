#!/usr/bin/env bash
set -euo pipefail
POPULATION="${1:-3}"
GENERATIONS="${2:-1}"
INITIAL_CONFIG="${3:-/app/runtime/best_ai_config.json}"

docker compose --env-file .env run --rm tuner-service \
  python -m tsdb_tuner.cli ga-optimize --config /app/config/tuner.yml \
   --population "$POPULATION" --generations "$GENERATIONS" --initial-config "$INITIAL_CONFIG"
