#!/usr/bin/env bash
set -euo pipefail
EXP_ID="${1}"

docker compose --env-file .env run --rm tuner-service   python -m tsdb_tuner.cli show-monitoring --config /app/config/tuner.yml "$EXP_ID"