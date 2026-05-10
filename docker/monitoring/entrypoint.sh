#!/bin/bash
# Копируем дашборды из образа в writable volume /data
# Это нужно потому что /data монтируется как volume и затирает содержимое образа

mkdir -p /data/grafana-tuner-dashboards

# Копируем только если дашборд ещё не скопирован или обновился
for f in /otel-lgtm/tuner-dashboards/*.json; do
    fname=$(basename "$f")
    echo "[entrypoint] Копирую дашборд: $fname"
    cp "$f" "/data/grafana-tuner-dashboards/$fname"
done

# Запускаем оригинальный скрипт LGTM
exec /otel-lgtm/run-all.sh "$@"
