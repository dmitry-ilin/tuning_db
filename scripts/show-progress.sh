#!/usr/bin/env bash
set -euo pipefail

docker compose --env-file .env run --rm tuner-service \
  python -m tsdb_tuner.cli show-progress --config /app/config/tuner.yml